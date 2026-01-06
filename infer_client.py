#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import time
import json
import base64
import numpy as np
import cv2
import requests

import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

import PyKDL
import crtk
import dvrk

bridge = CvBridge()


# ============================================================
# HARD-CODE CONFIG (EDIT THESE)
# ============================================================
SERVER_URL = "http://<SERVER_IP>:8000"   # e.g. "http://192.168.1.10:8000"

# ROS topics
RGB_TOPIC     = "/rgb/image_rect_color"
DEPTH_TOPIC   = "/depth_to_rgb/hw_registered/image_rect/"
CAMINFO_TOPIC = "/rgb/camera_info"

# preprocessing (must match training as close as possible)
CROP       = [0, 480, 0, 640]     # y0 y1 x0 x1
RESIZE     = [224, 224]           # H W
DEPTH_UNIT = "auto"               # auto|mm|m
NUM_POINTS = 8192

# control
ARM        = "PSM1"
RATE_HZ    = 2.0
SLEEP_DT   = 0.1
EXEC_HORIZON = 50

# action postprocess
POS_SCALE   = 0.001               # if model outputs mm -> meters
MOTION_MODE = "hybrid"            # "hybrid" or "tool_centric"
REGISTRATION_JSON = ""            # optional, else ""
DRY_RUN = False                   # True: no robot commands
# ============================================================


# --------------------------
# Same math helpers
# --------------------------
def orthonormalize(rot_elements):
    c1 = rot_elements[:3]
    c2 = rot_elements[3:]
    c1 = c1 / (np.linalg.norm(c1) + 1e-8)
    c2 = c2 - np.dot(c2, c1) * c1
    c2 = c2 / (np.linalg.norm(c2) + 1e-8)
    c3 = np.cross(c1, c2)
    return np.column_stack((c1, c2, c3))

def get_rotation_matrix(rot6d):
    col1 = rot6d[0:3]
    col2 = rot6d[3:6]
    col3 = np.cross(col1, col2)
    return np.column_stack((col1, col2, col3))

def rotmat_to_rot6d(R):
    return np.concatenate([R[:, 0], R[:, 1]], axis=0)

def read_camera_info(topic, timeout=3.0):
    msg = rospy.wait_for_message(topic, CameraInfo, timeout=timeout)
    return np.array(msg.K, dtype=np.float32).reshape(3, 3)

def get_current_robot_state(psm):
    frame = psm.measured_cp()
    pos = np.array([frame.p.x(), frame.p.y(), frame.p.z()], dtype=np.float32)

    rot_kdl = frame.M
    R = np.array([
        [rot_kdl[0,0], rot_kdl[0,1], rot_kdl[0,2]],
        [rot_kdl[1,0], rot_kdl[1,1], rot_kdl[1,2]],
        [rot_kdl[2,0], rot_kdl[2,1], rot_kdl[2,2]],
    ], dtype=np.float32)

    rot6d = np.concatenate([R[:, 0], R[:, 1]], axis=0).astype(np.float32)
    gripper = np.array([-0.65], dtype=np.float32)  # keep your convention
    return np.concatenate([pos, rot6d, gripper], axis=0).astype(np.float32)

def send_trajectory_batch(psm, batch_qpos, sleep_dt=0.1):
    for qpos in batch_qpos:
        pos = qpos[0:3]
        rot6d = qpos[3:9]
        gripper = qpos[9]

        R = get_rotation_matrix(rot6d)
        rot_kdl = PyKDL.Rotation(
            R[0,0], R[0,1], R[0,2],
            R[1,0], R[1,1], R[1,2],
            R[2,0], R[2,1], R[2,2],
        )
        frame = PyKDL.Frame(rot_kdl, PyKDL.Vector(pos[0], pos[1], pos[2]))

        psm.move_cp(frame)
        psm.jaw.move_jp(np.array([gripper], dtype=np.float32))
        time.sleep(sleep_dt)


# --------------------------
# Encode helpers
# --------------------------
def rgb_to_jpg_b64(bgr_u8: np.ndarray, quality=90) -> str:
    ok, enc = cv2.imencode(".jpg", bgr_u8, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("cv2.imencode jpg failed")
    return base64.b64encode(enc.tobytes()).decode("utf-8")

def np_to_b64_npy(arr: np.ndarray) -> str:
    bio = io.BytesIO()
    np.save(bio, arr, allow_pickle=False)
    return base64.b64encode(bio.getvalue()).decode("utf-8")

def b64_npy_to_np(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64.encode("utf-8"))
    bio = io.BytesIO(raw)
    return np.load(bio, allow_pickle=False)


def main():
    rospy.init_node("lift3d_act_client_ros", anonymous=True)

    # dVRK init
    ral = crtk.ral("lift3d_act_client_ros")
    psm = dvrk.psm(ral=ral, arm_name=ARM, expected_interval=SLEEP_DT)

    # registration (optional)
    R_BC = None
    if REGISTRATION_JSON:
        with open(REGISTRATION_JSON, "r") as f:
            reg = json.load(f)
        T = np.array(reg["base-frame"]["transform"], dtype=np.float32)
        R_BC = T[:3, :3]

    period = 1.0 / max(RATE_HZ, 1e-6)

    while not rospy.is_shutdown():
        t0 = time.time()

        # 1) Sense from ROS
        rgb_msg = rospy.wait_for_message(RGB_TOPIC, Image, timeout=5.0)
        bgr = bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")  # HWC BGR uint8

        depth_msg = rospy.wait_for_message(DEPTH_TOPIC, Image, timeout=5.0)
        depth = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        depth = np.array(depth)  # keep original dtype (often uint16)

        K = read_camera_info(CAMINFO_TOPIC, timeout=3.0)
        robot_state = get_current_robot_state(psm)

        # 2) Pack request
        payload = {
            "rgb_jpg_b64": rgb_to_jpg_b64(bgr, quality=90),
            "depth_npy_b64": np_to_b64_npy(depth),
            "K_9": K.reshape(-1).tolist(),
            "robot_state_10": robot_state.tolist(),
            "crop": CROP,
            "resize": RESIZE,
            "depth_unit": DEPTH_UNIT,
            "num_points": int(NUM_POINTS),
        }

        # 3) Call server
        r = requests.post(SERVER_URL.rstrip("/") + "/infer", json=payload, timeout=30)
        r.raise_for_status()
        resp = r.json()

        actions = b64_npy_to_np(resp["actions_npy_b64"]).astype(np.float32)  # [K,A]
        Kchunk = actions.shape[0]
        horizon = min(EXEC_HORIZON, Kchunk)

        # 4) Integrate + execute
        predicted_qpos = [robot_state.copy()]
        batch_qpos = []

        for i in range(horizon):
            raw = actions[i].astype(np.float32)
            raw[0:3] *= float(POS_SCALE)

            dR = orthonormalize(raw[3:9])
            drot6d = np.concatenate([dR[:, 0], dR[:, 1]], axis=0).astype(np.float32)
            action = np.concatenate([raw[0:3], drot6d, [raw[9]]], axis=0).astype(np.float32)

            curr = predicted_qpos[-1]
            curr_pos = curr[0:3]
            curr_R = get_rotation_matrix(curr[3:9])
            curr_grip = curr[9]

            if MOTION_MODE == "tool_centric":
                delta_pos_world = curr_R @ action[0:3]
            else:
                delta_pos_end = action[0:3]
                delta_pos_world = (R_BC @ delta_pos_end) if (R_BC is not None) else delta_pos_end

            next_pos = curr_pos + delta_pos_world

            delta_R = get_rotation_matrix(action[3:9])
            next_R = curr_R @ delta_R

            next_grip = curr_grip + action[9]
            next_qpos = np.concatenate([next_pos, rotmat_to_rot6d(next_R), [next_grip]], axis=0).astype(np.float32)

            predicted_qpos.append(next_qpos)
            batch_qpos.append(next_qpos)

        if DRY_RUN:
            print(f"[DRY_RUN] got actions {actions.shape}, exec={len(batch_qpos)}, first_pos={batch_qpos[0][0:3]}")
        else:
            send_trajectory_batch(psm, batch_qpos, sleep_dt=SLEEP_DT)

        # pace
        dt = time.time() - t0
        if dt < period:
            time.sleep(period - dt)


if __name__ == "__main__":
    main()