#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert your peg_recover ACT-style dataset into Lift3D-style Zarr.

Output structure:
/
 ├── data
 │   ├── actions      (N, 10) float32
 │   ├── images       (N, 224, 224, 3) uint8
 │   ├── point_clouds (N, 1024, 6) float32
 │   ├── robot_states (N, 10) float32
 │   └── texts        (N,) <Uxx (fixed-length unicode)
 └── meta
     └── episode_ends (num_datasets,) int64

Notes:
- zarr v3 requires create_dataset(..., shape=...) and does NOT accept numcodecs.Blosc as compressor.
- This script:
  * uses Blosc compression only on zarr v2
  * on zarr v3, skips compression args to guarantee compatibility
"""

import argparse
import json
import pathlib
from typing import Dict, List, Tuple

import numpy as np
import zarr
from tqdm import tqdm

try:
    import cv2
except Exception as e:
    raise RuntimeError("Need opencv-python (cv2). Install: pip install opencv-python") from e


# =========================
# FIXED CAMERA INTRINSICS (1080p)
# =========================
FIXED_K = np.array(
    [
        [903.95819092, 0.0, 956.72424316],
        [0.0, 904.0100708, 548.60406494],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


# -------------------------
# ACT-compatible math utils
# -------------------------

def quaternion_to_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    """quat: [x,y,z,w] -> 3x3"""
    x, y, z, w = quat_xyzw.astype(np.float32)
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float32,
    )


def get_pose_data(robot_data: List[Dict]) -> Dict:
    for d in robot_data:
        if "pose" in d:
            return d["pose"]
    raise ValueError("robot_data missing 'pose'")


def get_gripper_data(robot_data: List[Dict]) -> Dict:
    for d in robot_data:
        if "gripper" in d:
            return d["gripper"]
    raise ValueError("robot_data missing 'gripper'")


def transform_point(p_xyz: np.ndarray, T_4x4: np.ndarray) -> np.ndarray:
    p_h = np.concatenate([p_xyz.astype(np.float32), np.array([1.0], np.float32)], axis=0)
    out = T_4x4.astype(np.float32) @ p_h
    return (out[:3] / out[3]).astype(np.float32)


def transform_rotation(R_3x3: np.ndarray, T_4x4: np.ndarray) -> np.ndarray:
    return (T_4x4[:3, :3].astype(np.float32) @ R_3x3.astype(np.float32)).astype(np.float32)


def compute_tool_centric_action(meta_curr: Dict, meta_next: Dict, T_reg: np.ndarray) -> np.ndarray:
    # NOTE: registration not used here, kept to match your signature
    curr_pose = get_pose_data(meta_curr["robot_data"])
    next_pose = get_pose_data(meta_next["robot_data"])
    curr_grip = get_gripper_data(meta_curr["robot_data"])
    next_grip = get_gripper_data(meta_next["robot_data"])

    curr_pos = np.array(
        [curr_pose["position"]["x"], curr_pose["position"]["y"], curr_pose["position"]["z"]],
        np.float32,
    )
    next_pos = np.array(
        [next_pose["position"]["x"], next_pose["position"]["y"], next_pose["position"]["z"]],
        np.float32,
    )

    curr_quat = np.array(
        [
            curr_pose["orientation"]["x"],
            curr_pose["orientation"]["y"],
            curr_pose["orientation"]["z"],
            curr_pose["orientation"]["w"],
        ],
        np.float32,
    )
    next_quat = np.array(
        [
            next_pose["orientation"]["x"],
            next_pose["orientation"]["y"],
            next_pose["orientation"]["z"],
            next_pose["orientation"]["w"],
        ],
        np.float32,
    )

    R0 = quaternion_to_matrix(curr_quat)
    R1 = quaternion_to_matrix(next_quat)

    delta_pos_world = next_pos - curr_pos
    delta_pos_tool = (R0.T @ delta_pos_world).astype(np.float32)

    dR = (R0.T @ R1).astype(np.float32)
    delta_rot = np.concatenate([dR[:, 0], dR[:, 1]], axis=0).astype(np.float32)  # 6

    dgrip = np.array([float(next_grip["position"][0] - curr_grip["position"][0])], np.float32)

    out = np.concatenate([delta_pos_tool, delta_rot, dgrip], axis=0).astype(np.float32)
    assert out.shape == (10,)
    return out


def compute_hybrid_relative_action(meta_curr: Dict, meta_next: Dict, T_reg: np.ndarray) -> np.ndarray:
    curr_pose = get_pose_data(meta_curr["robot_data"])
    next_pose = get_pose_data(meta_next["robot_data"])
    curr_grip = get_gripper_data(meta_curr["robot_data"])
    next_grip = get_gripper_data(meta_next["robot_data"])

    curr_pos = np.array(
        [curr_pose["position"]["x"], curr_pose["position"]["y"], curr_pose["position"]["z"]],
        np.float32,
    )
    next_pos = np.array(
        [next_pose["position"]["x"], next_pose["position"]["y"], next_pose["position"]["z"]],
        np.float32,
    )

    curr_pos_endo = transform_point(curr_pos, T_reg)
    next_pos_endo = transform_point(next_pos, T_reg)
    delta_pos = (next_pos_endo - curr_pos_endo).astype(np.float32)

    curr_quat = np.array(
        [
            curr_pose["orientation"]["x"],
            curr_pose["orientation"]["y"],
            curr_pose["orientation"]["z"],
            curr_pose["orientation"]["w"],
        ],
        np.float32,
    )
    next_quat = np.array(
        [
            next_pose["orientation"]["x"],
            next_pose["orientation"]["y"],
            next_pose["orientation"]["z"],
            next_pose["orientation"]["w"],
        ],
        np.float32,
    )
    R0 = quaternion_to_matrix(curr_quat)
    R1 = quaternion_to_matrix(next_quat)

    dR = (R0.T @ R1).astype(np.float32)
    delta_rot = np.concatenate([dR[:, 0], dR[:, 1]], axis=0).astype(np.float32)  # 6

    dgrip = np.array([float(next_grip["position"][0] - curr_grip["position"][0])], np.float32)

    out = np.concatenate([delta_pos, delta_rot, dgrip], axis=0).astype(np.float32)
    assert out.shape == (10,)
    return out


def get_proprioception(meta: Dict, T_reg: np.ndarray) -> np.ndarray:
    pose = get_pose_data(meta["robot_data"])
    grip = get_gripper_data(meta["robot_data"])

    pos = np.array(
        [pose["position"]["x"], pose["position"]["y"], pose["position"]["z"]],
        np.float32,
    )
    quat = np.array(
        [
            pose["orientation"]["x"],
            pose["orientation"]["y"],
            pose["orientation"]["z"],
            pose["orientation"]["w"],
        ],
        np.float32,
    )
    R = quaternion_to_matrix(quat)

    pos_t = transform_point(pos, T_reg)
    R_t = transform_rotation(R, T_reg)

    orient = np.concatenate([R_t[:, 0], R_t[:, 1]], axis=0).astype(np.float32)  # 6
    g = np.array([float(grip["position"][0])], np.float32)

    out = np.concatenate([pos_t, orient, g], axis=0).astype(np.float32)
    assert out.shape == (10,)
    return out


# -------------------------
# Camera + point cloud
# -------------------------

def update_K_for_crop_and_resize(
    K: np.ndarray,
    crop_rect: Tuple[int, int, int, int],
    out_w: int,
    out_h: int,
) -> np.ndarray:
    x1, y1, x2, y2 = crop_rect
    crop_w = float(x2 - x1)
    crop_h = float(y2 - y1)
    if crop_w <= 0 or crop_h <= 0:
        raise ValueError(f"Invalid crop rect {crop_rect}")

    sx = float(out_w) / crop_w
    sy = float(out_h) / crop_h

    K2 = K.astype(np.float32).copy()
    K2[0, 2] -= float(x1)
    K2[1, 2] -= float(y1)
    K2[0, 0] *= sx
    K2[1, 1] *= sy
    K2[0, 2] *= sx
    K2[1, 2] *= sy
    return K2


def crop_and_resize(
    img: np.ndarray,
    crop_rect: Tuple[int, int, int, int],
    out_w: int,
    out_h: int,
    interp,
) -> np.ndarray:
    x1, y1, x2, y2 = crop_rect
    h, w = img.shape[:2]
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Crop rect out of bounds or empty: {crop_rect} for image {w}x{h}")
    cropped = img[y1:y2, x1:x2]
    return cv2.resize(cropped, (out_w, out_h), interpolation=interp)


def backproject_rgbd_to_pc(
    rgb_u8: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    depth_scale: float,
    max_depth_m: float,
) -> np.ndarray:
    if depth.ndim == 3:
        depth = depth[..., 0]
    z = depth.astype(np.float32) * float(depth_scale)
    valid = (z > 0) & (z < float(max_depth_m))
    if valid.sum() == 0:
        return np.zeros((0, 6), dtype=np.float32)

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    ys, xs = np.where(valid)
    zvals = z[ys, xs]
    xvals = (xs.astype(np.float32) - cx) * zvals / fx
    yvals = (ys.astype(np.float32) - cy) * zvals / fy

    cols = rgb_u8[ys, xs].astype(np.float32)
    pc = np.concatenate([xvals[:, None], yvals[:, None], zvals[:, None], cols], axis=1).astype(np.float32)
    return pc


def sample_pc(pc: np.ndarray, num_points: int) -> np.ndarray:
    if pc.shape[0] == 0:
        return np.zeros((num_points, 6), dtype=np.float32)
    if pc.shape[0] >= num_points:
        idx = np.random.choice(pc.shape[0], num_points, replace=False)
    else:
        idx = np.random.choice(pc.shape[0], num_points, replace=True)
    return pc[idx].astype(np.float32)


# -------------------------
# IO helpers
# -------------------------

def read_json(p: pathlib.Path) -> Dict:
    with open(p, "r") as f:
        return json.load(f)


def list_sorted_dirs(root: pathlib.Path, prefix: str) -> List[pathlib.Path]:
    ds = [p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)]

    def key(p: pathlib.Path):
        try:
            return int(p.name.split("_")[-1])
        except Exception:
            return p.name

    return sorted(ds, key=key)


# -------------------------
# zarr write helpers (v2/v3 compatible)
# -------------------------

def _zarr_major_version() -> int:
    v = getattr(zarr, "__version__", "2.0.0")
    try:
        return int(v.split(".")[0])
    except Exception:
        return 2


def _get_v2_compressor():
    # Only for zarr v2
    try:
        from numcodecs import Blosc
        return Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE)
    except Exception:
        return None


def create_dataset_compat(group, name: str, data: np.ndarray, dtype, chunks, compressor_v2):
    """
    zarr v3: must pass shape=... and MUST NOT pass numcodecs compressor (codec mismatch).
    zarr v2: can pass compressor=numcodecs.Blosc.
    """
    zmaj = _zarr_major_version()
    if zmaj >= 3:
        # v3: no compressor arg (avoid BytesBytesCodec error)
        return group.create_dataset(
            name,
            shape=data.shape,
            data=data,
            dtype=dtype,
            chunks=chunks,
        )
    else:
        # v2
        kwargs = dict(
            name=name,
            data=data,
            dtype=dtype,
            chunks=chunks,
        )
        if compressor_v2 is not None:
            kwargs["compressor"] = compressor_v2
        return group.create_dataset(**kwargs)


# -------------------------
# main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="root containing dataset_*/")
    ap.add_argument("--output-zarr", type=str, required=True)
    ap.add_argument("--registration-json", type=str, required=True, help="PSM1-registration-open-cv.json")
    ap.add_argument("--action-type", type=str, default="hybrid_relative",
                    choices=["hybrid_relative", "tool_centric"])
    ap.add_argument("--text", type=str, default="peg recover")

    ap.add_argument("--rgb-name", type=str, default="rgb.jpg")
    ap.add_argument("--depth-name", type=str, default="depth.png")
    ap.add_argument("--meta-name", type=str, default="metadata.json")

    ap.add_argument("--crop", type=int, nargs=4, required=True,
                    help="crop rect on original image: x1 y1 x2 y2")

    ap.add_argument("--pc-size", type=int, nargs=2, default=[640, 480])
    ap.add_argument("--img-size", type=int, default=224)

    ap.add_argument("--num-points", type=int, default=1024)
    ap.add_argument("--depth-scale", type=float, default=0.001)
    ap.add_argument("--max-depth-m", type=float, default=2.0)

    args = ap.parse_args()

    root = pathlib.Path(args.root).expanduser().resolve()
    out = pathlib.Path(args.output_zarr).expanduser().resolve()
    reg_path = pathlib.Path(args.registration_json).expanduser().resolve()

    reg = read_json(reg_path)
    T_reg = np.array(reg["base-frame"]["transform"], dtype=np.float32)
    if T_reg.shape != (4, 4):
        raise ValueError("registration_json base-frame.transform must be 4x4")

    crop_rect = tuple(int(x) for x in args.crop)
    pc_w, pc_h = int(args.pc_size[0]), int(args.pc_size[1])

    K_pc = update_K_for_crop_and_resize(FIXED_K, crop_rect, pc_w, pc_h)

    dataset_dirs = list_sorted_dirs(root, "dataset")
    if not dataset_dirs:
        raise RuntimeError(f"No dataset_* dirs under {root}")

    images_all: List[np.ndarray] = []
    pcs_all: List[np.ndarray] = []
    robot_states_all: List[np.ndarray] = []
    actions_all: List[np.ndarray] = []
    texts_all: List[str] = []
    episode_ends: List[int] = []
    total_steps = 0

    for dset in tqdm(dataset_dirs, desc="Datasets", unit="dataset"):
        frame_dirs = list_sorted_dirs(dset, "frame")
        if len(frame_dirs) < 2:
            continue

        metas = [read_json(fd / args.meta_name) for fd in frame_dirs]

        for t in tqdm(
            range(len(frame_dirs) - 1),
            desc=f"Frames ({dset.name})",
            unit="step",
            leave=False,
        ):
            fd0 = frame_dirs[t]
            m0 = metas[t]
            m1 = metas[t + 1]

            rs = get_proprioception(m0, T_reg)
            if args.action_type == "hybrid_relative":
                act = compute_hybrid_relative_action(m0, m1, T_reg)
            else:
                act = compute_tool_centric_action(m0, m1, T_reg)

            rgb_bgr = cv2.imread(str(fd0 / args.rgb_name), cv2.IMREAD_COLOR)
            if rgb_bgr is None:
                raise FileNotFoundError(fd0 / args.rgb_name)
            rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

            depth = cv2.imread(str(fd0 / args.depth_name), cv2.IMREAD_UNCHANGED)
            if depth is None:
                raise FileNotFoundError(fd0 / args.depth_name)

            rgb_pc = crop_and_resize(rgb, crop_rect, pc_w, pc_h, interp=cv2.INTER_AREA)
            depth_pc = crop_and_resize(depth, crop_rect, pc_w, pc_h, interp=cv2.INTER_NEAREST)

            pc_full = backproject_rgbd_to_pc(
                rgb_u8=rgb_pc,
                depth=depth_pc,
                K=K_pc,
                depth_scale=args.depth_scale,
                max_depth_m=args.max_depth_m,
            )
            pc = sample_pc(pc_full, args.num_points)

            rgb_img = cv2.resize(rgb_pc, (args.img_size, args.img_size), interpolation=cv2.INTER_AREA)

            images_all.append(rgb_img.astype(np.uint8))
            pcs_all.append(pc.astype(np.float32))
            robot_states_all.append(rs.astype(np.float32))
            actions_all.append(act.astype(np.float32))
            texts_all.append(args.text)

            total_steps += 1

        episode_ends.append(total_steps)

    if total_steps == 0:
        raise RuntimeError("No steps written (check dataset/frame dirs).")

    images = np.stack(images_all, axis=0).astype(np.uint8)
    pcs = np.stack(pcs_all, axis=0).astype(np.float32)
    robot_states = np.stack(robot_states_all, axis=0).astype(np.float32)
    actions = np.stack(actions_all, axis=0).astype(np.float32)

    # texts: fixed-length unicode (stable for zarr v2/v3; avoids object_codec)
    max_len = max(len(str(t)) for t in texts_all) if len(texts_all) > 0 else 1
    texts = np.array([str(t) for t in texts_all], dtype=f"<U{max_len}")

    episode_ends = np.array(episode_ends, dtype=np.int64)

    out.parent.mkdir(parents=True, exist_ok=True)
    zroot = zarr.group(str(out))
    data = zroot.create_group("data", overwrite=True)
    meta = zroot.create_group("meta", overwrite=True)

    compressor_v2 = _get_v2_compressor()

    create_dataset_compat(
        data,
        "images",
        images,
        dtype="uint8",
        chunks=(100, images.shape[1], images.shape[2], images.shape[3]),
        compressor_v2=compressor_v2,
    )
    create_dataset_compat(
        data,
        "point_clouds",
        pcs,
        dtype="float32",
        chunks=(100, pcs.shape[1], pcs.shape[2]),
        compressor_v2=compressor_v2,
    )
    create_dataset_compat(
        data,
        "robot_states",
        robot_states,
        dtype="float32",
        chunks=(100, robot_states.shape[1]),
        compressor_v2=compressor_v2,
    )
    create_dataset_compat(
        data,
        "actions",
        actions,
        dtype="float32",
        chunks=(100, actions.shape[1]),
        compressor_v2=compressor_v2,
    )
    create_dataset_compat(
        data,
        "texts",
        texts,
        dtype=texts.dtype,
        chunks=(100,),
        compressor_v2=compressor_v2,
    )
    create_dataset_compat(
        meta,
        "episode_ends",
        episode_ends,
        dtype="int64",
        chunks=(max(1, min(episode_ends.shape[0], 100)),),
        compressor_v2=compressor_v2,
    )

    print("zarr version:", getattr(zarr, "__version__", "unknown"))
    print("Fixed K (1080p):\n", FIXED_K)
    print("crop:", crop_rect, "pc_size:", (pc_w, pc_h))
    print("K after crop+resize (for point cloud):\n", K_pc)
    print(zroot.tree())
    print("Saved:", out)


if __name__ == "__main__":
    main()