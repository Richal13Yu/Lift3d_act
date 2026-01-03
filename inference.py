#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Online inference for: Lift3D visual encoder + ACT (Lift3DActActor) action head
Input: ROS RGB + Depth
Output: dVRK PSM motion commands

Assumptions:
- Your training checkpoint is a plain state_dict saved by torch.save(model.state_dict()).
- Your Hydra run saved config at: <run_dir>/.hydra/config.yaml
- Action format matches your previous ACT: [dx, dy, dz, rot6d(6), d_gripper] OR absolute in your own convention.
  (If your model outputs mm, set --pos_scale 0.001)
"""

import os
import time
import json
import argparse
from dataclasses import asdict

import numpy as np
import cv2
import torch
import torch.nn as nn

import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

import PyKDL
import crtk
import dvrk

from omegaconf import OmegaConf
from hydra.utils import instantiate

# ---- If your Lift3DActActor is in this path (adjust if needed) ----
from lift3d.models.act.act_actor import Lift3DActActor, ActOutput  # your file


bridge = CvBridge()


# --------------------------
# Math helpers (same as yours)
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
    R = np.column_stack((col1, col2, col3))
    return R

def rotmat_to_rot6d(R):
    return np.concatenate([R[:, 0], R[:, 1]], axis=0)

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

    # NOTE: 这里沿用你原来的 gripper（你也可以换成真实 jaw 状态）
    gripper = np.array([-0.65], dtype=np.float32)

    state = np.concatenate([pos, rot6d, gripper], axis=0).astype(np.float32)
    return state


# --------------------------
# Camera + point cloud
# --------------------------
def read_camera_info(topic, timeout=3.0):
    msg = rospy.wait_for_message(topic, CameraInfo, timeout=timeout)
    K = np.array(msg.K, dtype=np.float32).reshape(3, 3)
    return K

def crop_resize(img, crop, size, is_depth=False):
    y0, y1, x0, x1 = crop
    img = img[y0:y1, x0:x1]
    interp = cv2.INTER_NEAREST if is_depth else cv2.INTER_LINEAR
    return cv2.resize(img, size, interpolation=interp)

def depth_to_point_cloud(depth_m, K, num_points=8192, depth_min=0.02, depth_max=2.0):
    """
    depth_m: HxW float32 in meters
    K: 3x3 intrinsics
    return: Nx3 float32 in camera frame
    """
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    H, W = depth_m.shape
    z = depth_m

    valid = (z > depth_min) & (z < depth_max) & np.isfinite(z)
    if valid.sum() < 50:
        # too few points -> return zeros (model still runs but policy likely bad)
        return np.zeros((num_points, 3), dtype=np.float32)

    v, u = np.where(valid)
    z = z[v, u]

    x = (u.astype(np.float32) - cx) / fx * z
    y = (v.astype(np.float32) - cy) / fy * z
    pts = np.stack([x, y, z], axis=1).astype(np.float32)  # [M,3]

    # sample to fixed num_points
    M = pts.shape[0]
    if M >= num_points:
        idx = np.random.choice(M, num_points, replace=False)
        pts = pts[idx]
    else:
        idx = np.random.choice(M, num_points, replace=True)
        pts = pts[idx]

    return pts

def get_rgb_depth_pointcloud(
    rgb_topic: str,
    depth_topic: str,
    caminfo_topic: str,
    crop=(0, 480, 0, 640),
    resize=(224, 224),
    depth_unit="auto",         # "auto" | "mm" | "m"
    num_points=8192,
):
    # ---- RGB ----
    rgb_msg = rospy.wait_for_message(rgb_topic, Image, timeout=5.0)
    rgb_bgr = bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
    rgb_bgr = crop_resize(rgb_bgr, crop=crop, size=(resize[1], resize[0]), is_depth=False)
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0  # [H,W,3]

    # ---- Depth ----
    depth_msg = rospy.wait_for_message(depth_topic, Image, timeout=5.0)
    depth = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
    depth = depth.astype(np.float32)
    depth[np.isnan(depth)] = 0.0
    depth[np.isinf(depth)] = 0.0
    depth = crop_resize(depth, crop=crop, size=(resize[1], resize[0]), is_depth=True)

    # unit normalize to meters
    if depth_unit == "mm":
        depth_m = depth / 1000.0
    elif depth_unit == "m":
        depth_m = depth
    else:
        # auto: common case is uint16 in mm scale
        if depth.max() > 10.0:  # likely mm
            depth_m = depth / 1000.0
        else:
            depth_m = depth

    # ---- intrinsics ----
    K = read_camera_info(caminfo_topic, timeout=3.0)

    # ---- point cloud ----
    pts = depth_to_point_cloud(depth_m, K, num_points=num_points)

    # Torch tensors
    img_t = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()     # [3,H,W]
    pc_t  = torch.from_numpy(pts).contiguous()                      # [N,3]

    return img_t, pc_t, K


# --------------------------
# dVRK sending
# --------------------------
def send_trajectory_batch(psm, batch_qpos, sleep_dt=0.1):
    for idx, qpos in enumerate(batch_qpos):
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
# Build model from Hydra config + checkpoint
# --------------------------
def build_model_from_hydra(hydra_config_path: str, ckpt_path: str, device: str, robot_state_dim: int, action_dim: int):
    cfg = OmegaConf.load(hydra_config_path)

    # instantiate your agent (should point to Lift3DActActor + lift3d encoder)
    model: nn.Module = instantiate(
        cfg.agent.instantiate_config,
        robot_state_dim=robot_state_dim,
        action_dim=action_dim,
    )

    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and all(isinstance(k, str) for k in sd.keys()):
        model.load_state_dict(sd, strict=False)
    else:
        raise RuntimeError("Checkpoint format not recognized. Expected a state_dict dict.")

    model.to(device)
    model.eval()
    return model


# --------------------------
# Main loop
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hydra_config", type=str, required=True, help="Path to <run_dir>/.hydra/config.yaml")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to best_model.pth (state_dict)")

    # ROS topics
    parser.add_argument("--rgb_topic", type=str, default="/rgb/image_rect_color")
    parser.add_argument("--depth_topic", type=str, default="/depth_to_rgb/hw_registered/image_rect/")
    parser.add_argument("--caminfo_topic", type=str, default="/rgb/camera_info")

    # preprocessing (MUST match training as close as possible)
    parser.add_argument("--crop", type=int, nargs=4, default=[0, 480, 0, 640], help="y0 y1 x0 x1")
    parser.add_argument("--resize", type=int, nargs=2, default=[224, 224], help="H W")
    parser.add_argument("--depth_unit", type=str, default="auto", choices=["auto", "mm", "m"])
    parser.add_argument("--num_points", type=int, default=8192)

    # control
    parser.add_argument("--arm", type=str, default="PSM1")
    parser.add_argument("--rate_hz", type=float, default=2.0, help="How often to query policy and send a chunk")
    parser.add_argument("--sleep_dt", type=float, default=0.1, help="Sleep between steps inside chunk execution")
    parser.add_argument("--exec_horizon", type=int, default=50, help="How many steps from chunk to execute (<=K)")

    # action postprocess (match your training convention)
    parser.add_argument("--pos_scale", type=float, default=0.001, help="If model outputs mm, use 0.001 to convert to meters")
    parser.add_argument("--motion_mode", type=str, default="hybrid", choices=["hybrid", "tool_centric"])
    parser.add_argument("--registration_json", type=str, default="", help="Optional: camera->base registration json (same as your old script)")
    parser.add_argument("--dry_run", action="store_true", help="Do not send commands to robot")

    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    rospy.init_node("lift3d_act_infer", anonymous=True)

    # dVRK init
    ral = crtk.ral("lift3d_act_infer")
    psm = dvrk.psm(ral=ral, arm_name=args.arm, expected_interval=args.sleep_dt)

    # registration (optional)
    R_BC = None
    if args.registration_json:
        with open(args.registration_json, "r") as f:
            reg = json.load(f)
        T = np.array(reg["base-frame"]["transform"], dtype=np.float32)
        R_BC = T[:3, :3]  # (endoscope/camera)->base rotation like your old code

    # dims (keep same as your old pipeline)
    robot_state_dim = 10  # pos3 + rot6d6 + gripper1
    action_dim = 10       # dx3 + rot6d6 + dgripper1

    # build model
    model = build_model_from_hydra(
        hydra_config_path=args.hydra_config,
        ckpt_path=args.ckpt,
        device=args.device,
        robot_state_dim=robot_state_dim,
        action_dim=action_dim,
    )
    print("[OK] Model loaded.")

    # loop
    period = 1.0 / max(args.rate_hz, 1e-6)
    while not rospy.is_shutdown():
        t0 = time.time()

        # 1) sense
        img_t, pc_t, _K = get_rgb_depth_pointcloud(
            rgb_topic=args.rgb_topic,
            depth_topic=args.depth_topic,
            caminfo_topic=args.caminfo_topic,
            crop=tuple(args.crop),
            resize=tuple(args.resize),
            depth_unit=args.depth_unit,
            num_points=args.num_points,
        )
        robot_state = get_current_robot_state(psm)

        # 2) to torch batch
        images = img_t.unsqueeze(0).to(args.device)                  # [1,3,H,W]
        point_clouds = pc_t.unsqueeze(0).to(args.device)             # [1,N,3]
        robot_states = torch.from_numpy(robot_state).unsqueeze(0).to(args.device)  # [1,10]

        # 3) policy forward (inference: actions=None)
        with torch.inference_mode():
            out = model(images, point_clouds, robot_states, texts=None, actions=None, is_pad=None)

        if isinstance(out, ActOutput):
            actions_hat = out.actions
        else:
            actions_hat = out
        actions_hat = actions_hat[0].detach().cpu().numpy()  # [K,A]

        # 4) integrate + execute
        initial_state = robot_state.copy()
        predicted_qpos = [initial_state]
        batch_qpos = []

        K = actions_hat.shape[0]
        horizon = min(args.exec_horizon, K)

        for i in range(horizon):
            raw = actions_hat[i].astype(np.float32)

            # position scale (mm->m if needed)
            raw[0:3] *= float(args.pos_scale)

            # make delta rotation valid
            dR = orthonormalize(raw[3:9])
            drot6d = np.concatenate([dR[:, 0], dR[:, 1]], axis=0).astype(np.float32)
            action = np.concatenate([raw[0:3], drot6d, [raw[9]]], axis=0).astype(np.float32)

            curr = predicted_qpos[-1]
            curr_pos = curr[0:3]
            curr_R = get_rotation_matrix(curr[3:9])
            curr_grip = curr[9]

            # Δpos interpretation (same idea as你原脚本)
            if args.motion_mode == "tool_centric":
                delta_pos_world = curr_R @ action[0:3]
            else:
                delta_pos_end = action[0:3]
                if R_BC is not None:
                    delta_pos_world = R_BC @ delta_pos_end
                else:
                    delta_pos_world = delta_pos_end

            next_pos = curr_pos + delta_pos_world

            # ΔR apply in tool frame
            delta_R = get_rotation_matrix(action[3:9])
            next_R = curr_R @ delta_R

            next_grip = curr_grip + action[9]
            next_qpos = np.concatenate([next_pos, rotmat_to_rot6d(next_R), [next_grip]], axis=0).astype(np.float32)

            predicted_qpos.append(next_qpos)
            batch_qpos.append(next_qpos)

        if args.dry_run:
            print(f"[DRY_RUN] computed {len(batch_qpos)} steps, first pos={batch_qpos[0][0:3]}")
        else:
            send_trajectory_batch(psm, batch_qpos, sleep_dt=args.sleep_dt)

        # pace
        dt = time.time() - t0
        if dt < period:
            time.sleep(period - dt)


if __name__ == "__main__":
    main()