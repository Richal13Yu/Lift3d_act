#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import os
import base64
from typing import List, Optional, Tuple

import numpy as np
import cv2
import torch
import torch.nn as nn

from fastapi import FastAPI
from pydantic import BaseModel, Field
from omegaconf import OmegaConf
from hydra.utils import instantiate

# ---- adjust import path if needed ----
from lift3d.models.act.act_actor import ActOutput  # keep for output parsing


# ============================================================
# HARD-CODE CONFIG (EDIT THESE)
# ============================================================
HYDRA_CONFIG_PATH = "/path/to/run_dir/.hydra/config.yaml"
CKPT_PATH         = "/path/to/best_model.pth"
DEVICE            = "cuda"   # "cuda" or "cpu"

HOST = "0.0.0.0"
PORT = 8000

# These should match your training (often both 10 for xyz+rot6+grip)
ROBOT_STATE_DIM = 10
ACTION_DIM      = 10

# Match training defaults more closely
DEFAULT_NUM_POINTS = 1024     # your zarr script used 1024
DEFAULT_IMG_SIZE   = 224
DEFAULT_PC_CHANNELS = 6       # xyzrgb (your zarr is (N,1024,6))
# ============================================================


# --------------------------
# Utils: base64 <-> numpy
# --------------------------
def np_to_b64_npy(arr: np.ndarray) -> str:
    bio = io.BytesIO()
    np.save(bio, arr, allow_pickle=False)
    return base64.b64encode(bio.getvalue()).decode("utf-8")

def b64_npy_to_np(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64.encode("utf-8"))
    bio = io.BytesIO(raw)
    return np.load(bio, allow_pickle=False)

def jpg_b64_to_rgb_u8(b64jpg: str) -> np.ndarray:
    raw = base64.b64decode(b64jpg.encode("utf-8"))
    buf = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # HWC BGR uint8
    if bgr is None:
        raise ValueError("Failed to decode jpg")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # uint8
    return rgb


# --------------------------
# Crop/resize + K adjustment (CRITICAL)
# --------------------------
def crop_resize(img: np.ndarray, crop_y0y1x0x1: Tuple[int, int, int, int], out_hw: Tuple[int, int], is_depth: bool = False) -> np.ndarray:
    y0, y1, x0, x1 = crop_y0y1x0x1
    h, w = img.shape[:2]
    y0 = int(np.clip(y0, 0, h))
    y1 = int(np.clip(y1, 0, h))
    x0 = int(np.clip(x0, 0, w))
    x1 = int(np.clip(x1, 0, w))
    if y1 <= y0 or x1 <= x0:
        raise ValueError(f"Invalid crop {crop_y0y1x0x1} for image {h}x{w}")

    cropped = img[y0:y1, x0:x1]
    interp = cv2.INTER_NEAREST if is_depth else cv2.INTER_AREA
    out_h, out_w = out_hw
    return cv2.resize(cropped, (out_w, out_h), interpolation=interp)

def adjust_K_for_crop_resize(
    K: np.ndarray,
    crop_y0y1x0x1: Tuple[int, int, int, int],
    out_hw: Tuple[int, int],
) -> np.ndarray:
    """
    If K corresponds to the ORIGINAL image, and you crop then resize,
    you MUST update intrinsics accordingly.

    crop is (y0, y1, x0, x1)
    out_hw is (H, W)
    """
    y0, y1, x0, x1 = crop_y0y1x0x1
    out_h, out_w = out_hw
    crop_h = float(y1 - y0)
    crop_w = float(x1 - x0)
    if crop_h <= 0 or crop_w <= 0:
        raise ValueError("Empty crop")

    sx = float(out_w) / crop_w
    sy = float(out_h) / crop_h

    K2 = K.astype(np.float32).copy()
    # shift principal point due to crop
    K2[0, 2] -= float(x0)
    K2[1, 2] -= float(y0)
    # scale due to resize
    K2[0, 0] *= sx
    K2[1, 1] *= sy
    K2[0, 2] *= sx
    K2[1, 2] *= sy
    return K2


# --------------------------
# Depth -> Point cloud (xyzrgb, then sample)
# --------------------------
def backproject_rgbd_to_pc_xyzrgb(
    rgb_u8: np.ndarray,         # HxWx3 uint8
    depth_m: np.ndarray,        # HxW float32 meters
    K: np.ndarray,              # 3x3 float32, MUST match rgb/depth resolution
    depth_min: float = 0.02,
    depth_max: float = 2.0,
) -> np.ndarray:
    if depth_m.ndim == 3:
        depth_m = depth_m[..., 0]
    if rgb_u8.ndim != 3 or rgb_u8.shape[2] != 3:
        raise ValueError(f"rgb_u8 must be HxWx3, got {rgb_u8.shape}")
    if depth_m.shape[:2] != rgb_u8.shape[:2]:
        raise ValueError(f"rgb/depth shape mismatch: rgb={rgb_u8.shape}, depth={depth_m.shape}")

    z = depth_m.astype(np.float32)
    valid = (z > float(depth_min)) & (z < float(depth_max)) & np.isfinite(z)
    if valid.sum() < 50:
        return np.zeros((0, 6), dtype=np.float32)

    fx = float(K[0, 0]); fy = float(K[1, 1])
    cx = float(K[0, 2]); cy = float(K[1, 2])

    v, u = np.where(valid)
    zvals = z[v, u]
    x = (u.astype(np.float32) - cx) * zvals / fx
    y = (v.astype(np.float32) - cy) * zvals / fy

    cols = rgb_u8[v, u].astype(np.float32)  # 0..255
    pc = np.concatenate([x[:, None], y[:, None], zvals[:, None], cols], axis=1).astype(np.float32)  # Nx6
    return pc

def sample_pc(pc: np.ndarray, num_points: int) -> np.ndarray:
    if pc.shape[0] == 0:
        return np.zeros((num_points, pc.shape[1] if pc.ndim == 2 else 6), dtype=np.float32)
    M = pc.shape[0]
    if M >= num_points:
        idx = np.random.choice(M, num_points, replace=False)
    else:
        idx = np.random.choice(M, num_points, replace=True)
    return pc[idx].astype(np.float32)


# --------------------------
# Model IO helpers
# --------------------------
def _strip_prefix_if_present(state_dict, prefixes=("module.", "model.")):
    if not isinstance(state_dict, dict):
        return state_dict
    keys = list(state_dict.keys())
    for p in prefixes:
        if len(keys) > 0 and all(k.startswith(p) for k in keys):
            return {k[len(p):]: v for k, v in state_dict.items()}
    return state_dict

def build_model_from_hydra(
    hydra_config_path: str,
    ckpt_path: str,
    device: str,
    robot_state_dim: int,
    action_dim: int,
) -> nn.Module:
    cfg = OmegaConf.load(hydra_config_path)

    model: nn.Module = instantiate(
        cfg.agent.instantiate_config,
        robot_state_dim=robot_state_dim,
        action_dim=action_dim,
    )

    obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        sd = obj["state_dict"]
    elif isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        sd = obj["model"]
    elif isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()):
        sd = obj
    else:
        raise RuntimeError("Checkpoint format not recognized. Expected a state_dict-like dict.")

    sd = _strip_prefix_if_present(sd, prefixes=("module.", "model."))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[WARN] missing keys: {len(missing)} (show up to 20)\n{missing[:20]}")
    if unexpected:
        print(f"[WARN] unexpected keys: {len(unexpected)} (show up to 20)\n{unexpected[:20]}")

    model.to(device)
    model.eval()
    return model

def parse_actions(out) -> torch.Tensor:
    # out can be Tensor, dict, or ActOutput-like
    if torch.is_tensor(out):
        return out
    if isinstance(out, dict):
        if "actions" in out and torch.is_tensor(out["actions"]):
            return out["actions"]
        if "a_hat" in out and torch.is_tensor(out["a_hat"]):
            return out["a_hat"]
        raise ValueError(f"Dict output has no tensor 'actions'/'a_hat'. keys={list(out.keys())}")
    if isinstance(out, ActOutput) or hasattr(out, "actions"):
        v = out.actions
        if torch.is_tensor(v):
            return v
    raise ValueError(f"Unsupported model output type: {type(out)}")


# --------------------------
# Request / Response schema
# --------------------------
class InferRequest(BaseModel):
    rgb_jpg_b64: str
    depth_npy_b64: str

    # K is assumed to correspond to the ORIGINAL rgb/depth (before crop/resize),
    # unless you set k_is_for_cropped_resized=True
    K_9: List[float] = Field(..., description="3x3 intrinsics flattened row-major")
    robot_state_10: List[float]

    crop: List[int] = Field(default_factory=lambda: [0, 480, 0, 640])   # y0 y1 x0 x1
    resize: List[int] = Field(default_factory=lambda: [DEFAULT_IMG_SIZE, DEFAULT_IMG_SIZE])  # H W

    depth_unit: str = "auto"   # auto|mm|m
    num_points: int = DEFAULT_NUM_POINTS
    pc_channels: int = DEFAULT_PC_CHANNELS  # 6 (xyzrgb) recommended

    k_is_for_cropped_resized: bool = False  # if True, don't adjust K

class InferResponse(BaseModel):
    actions_npy_b64: str
    K: int
    A: int


# --------------------------
# FastAPI app
# --------------------------
app = FastAPI()
MODEL: Optional[nn.Module] = None


@app.on_event("startup")
def _startup():
    global MODEL
    MODEL = build_model_from_hydra(
        hydra_config_path=HYDRA_CONFIG_PATH,
        ckpt_path=CKPT_PATH,
        device=DEVICE,
        robot_state_dim=ROBOT_STATE_DIM,
        action_dim=ACTION_DIM,
    )
    print("[OK] Server model loaded.")


@app.get("/health")
def health():
    return {"ok": True, "device": DEVICE, "model_loaded": MODEL is not None}


@app.post("/infer", response_model=InferResponse)
def infer(req: InferRequest):
    if MODEL is None:
        raise RuntimeError("Model not loaded")

    # --------------------------
    # Decode inputs
    # --------------------------
    rgb_u8 = jpg_b64_to_rgb_u8(req.rgb_jpg_b64)  # HWC uint8
    depth = b64_npy_to_np(req.depth_npy_b64).astype(np.float32)
    depth[~np.isfinite(depth)] = 0.0

    crop = tuple(int(x) for x in req.crop)     # (y0,y1,x0,x1)
    out_hw = tuple(int(x) for x in req.resize) # (H,W)

    # Crop+resize both rgb and depth to SAME resolution used by model
    rgb_u8 = crop_resize(rgb_u8, crop_y0y1x0x1=crop, out_hw=out_hw, is_depth=False)
    depth  = crop_resize(depth,  crop_y0y1x0x1=crop, out_hw=out_hw, is_depth=True)

    # Intrinsics
    K = np.array(req.K_9, dtype=np.float32).reshape(3, 3)
    if not req.k_is_for_cropped_resized:
        K = adjust_K_for_crop_resize(K, crop_y0y1x0x1=crop, out_hw=out_hw)

    # depth -> meters
    if req.depth_unit == "mm":
        depth_m = depth / 1000.0
    elif req.depth_unit == "m":
        depth_m = depth
    else:
        # heuristic: if max looks like mm
        depth_m = depth / 1000.0 if float(np.nanmax(depth)) > 10.0 else depth

    # --------------------------
    # Build point cloud consistent with training (xyzrgb, then sample)
    # --------------------------
    pc_full = backproject_rgbd_to_pc_xyzrgb(
        rgb_u8=rgb_u8,
        depth_m=depth_m,
        K=K,
        depth_min=0.02,
        depth_max=2.0,
    )
    pc = sample_pc(pc_full, int(req.num_points))  # [N,6]

    # If model expects only xyz, drop rgb
    if int(req.pc_channels) == 3:
        pc = pc[:, :3]
    elif int(req.pc_channels) == 6:
        pass
    else:
        raise ValueError(f"pc_channels must be 3 or 6, got {req.pc_channels}")

    # --------------------------
    # Prepare tensors
    # IMPORTANT: keep rgb in 0..255 range (float32),
    # to better match common Lift3D dataset pipelines.
    # --------------------------
    rgb_f = rgb_u8.astype(np.float32)  # 0..255
    img_t = torch.from_numpy(rgb_f).permute(2, 0, 1).contiguous().unsqueeze(0).to(DEVICE)  # [1,3,H,W]
    pc_t  = torch.from_numpy(pc.astype(np.float32)).contiguous().unsqueeze(0).to(DEVICE)   # [1,N,C]
    rs = np.asarray(req.robot_state_10, dtype=np.float32)
    if rs.shape[0] != ROBOT_STATE_DIM:
        raise ValueError(f"robot_state_10 must have len={ROBOT_STATE_DIM}, got {rs.shape[0]}")
    rs_t  = torch.from_numpy(rs).unsqueeze(0).to(DEVICE)                                    # [1,10]

    # --------------------------
    # Inference (NO conditional prior; no GT passed)
    # --------------------------
    with torch.inference_mode():
        out = MODEL(img_t, pc_t, rs_t, texts=None, actions=None, is_pad=None)

    actions_hat = parse_actions(out)        # Tensor
    # actions_hat is usually [B,K,A] for ACT; return [K,A]
    if actions_hat.dim() == 3:
        actions = actions_hat[0].detach().cpu().numpy().astype(np.float32)
    elif actions_hat.dim() == 2:
        actions = actions_hat.detach().cpu().numpy().astype(np.float32)
    else:
        raise ValueError(f"Unexpected actions_hat shape: {tuple(actions_hat.shape)}")

    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        # still return, but make it explicit
        raise ValueError(f"Expected actions shape [K,{ACTION_DIM}], got {actions.shape}")

    return InferResponse(
        actions_npy_b64=np_to_b64_npy(actions),
        K=int(actions.shape[0]),
        A=int(actions.shape[1]),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")