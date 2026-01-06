#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import base64
from typing import List

import numpy as np
import cv2
import torch
import torch.nn as nn

from fastapi import FastAPI
from pydantic import BaseModel
from omegaconf import OmegaConf
from hydra.utils import instantiate

# ---- adjust import path if needed ----
from lift3d.models.act.act_actor import Lift3DActActor, ActOutput  # your file


# ============================================================
# HARD-CODE CONFIG (EDIT THESE)
# ============================================================
HYDRA_CONFIG_PATH = "/path/to/run_dir/.hydra/config.yaml"
CKPT_PATH         = "/path/to/best_model.pth"
DEVICE            = "cuda"   # "cuda" or "cpu"

HOST = "0.0.0.0"
PORT = 8000

ROBOT_STATE_DIM = 10
ACTION_DIM      = 10
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

def jpg_b64_to_rgb_float01(b64jpg: str) -> np.ndarray:
    raw = base64.b64decode(b64jpg.encode("utf-8"))
    buf = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # HWC BGR uint8
    if bgr is None:
        raise ValueError("Failed to decode jpg")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb


# --------------------------
# Same helpers as yours
# --------------------------
def crop_resize(img, crop, size, is_depth=False):
    y0, y1, x0, x1 = crop
    img = img[y0:y1, x0:x1]
    interp = cv2.INTER_NEAREST if is_depth else cv2.INTER_LINEAR
    return cv2.resize(img, (size[1], size[0]), interpolation=interp)  # size=(H,W)

def depth_to_point_cloud(depth_m, K, num_points=8192, depth_min=0.02, depth_max=2.0):
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    z = depth_m
    valid = (z > depth_min) & (z < depth_max) & np.isfinite(z)
    if valid.sum() < 50:
        return np.zeros((num_points, 3), dtype=np.float32)

    v, u = np.where(valid)
    z = z[v, u]

    x = (u.astype(np.float32) - cx) / fx * z
    y = (v.astype(np.float32) - cy) / fy * z
    pts = np.stack([x, y, z], axis=1).astype(np.float32)

    M = pts.shape[0]
    if M >= num_points:
        idx = np.random.choice(M, num_points, replace=False)
    else:
        idx = np.random.choice(M, num_points, replace=True)
    return pts[idx]


# --------------------------
# Request / Response schema
# --------------------------
class InferRequest(BaseModel):
    rgb_jpg_b64: str
    depth_npy_b64: str
    K_9: List[float]
    robot_state_10: List[float]

    crop: List[int] = [0, 480, 0, 640]    # y0 y1 x0 x1
    resize: List[int] = [224, 224]        # H W
    depth_unit: str = "auto"              # auto|mm|m
    num_points: int = 8192

class InferResponse(BaseModel):
    actions_npy_b64: str
    K: int
    A: int


# --------------------------
# Model loader
# --------------------------
def build_model_from_hydra(hydra_config_path: str, ckpt_path: str, device: str, robot_state_dim: int, action_dim: int):
    cfg = OmegaConf.load(hydra_config_path)

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
# FastAPI app
# --------------------------
app = FastAPI()
MODEL: nn.Module = None


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

    rgb = jpg_b64_to_rgb_float01(req.rgb_jpg_b64)  # HWC float32
    depth = b64_npy_to_np(req.depth_npy_b64).astype(np.float32)
    depth[np.isnan(depth)] = 0.0
    depth[np.isinf(depth)] = 0.0

    crop = tuple(req.crop)
    resize = tuple(req.resize)

    rgb = crop_resize(rgb, crop=crop, size=resize, is_depth=False)
    depth = crop_resize(depth, crop=crop, size=resize, is_depth=True)

    K = np.array(req.K_9, dtype=np.float32).reshape(3, 3)

    # depth -> meters
    if req.depth_unit == "mm":
        depth_m = depth / 1000.0
    elif req.depth_unit == "m":
        depth_m = depth
    else:
        depth_m = depth / 1000.0 if depth.max() > 10.0 else depth

    pts = depth_to_point_cloud(depth_m, K, num_points=req.num_points)

    img_t = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().unsqueeze(0).to(DEVICE)
    pc_t  = torch.from_numpy(pts).contiguous().unsqueeze(0).to(DEVICE)
    rs_t  = torch.tensor(req.robot_state_10, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        out = MODEL(img_t, pc_t, rs_t, texts=None, actions=None, is_pad=None)

    actions_hat = out.actions if isinstance(out, ActOutput) else out
    actions = actions_hat[0].detach().cpu().numpy().astype(np.float32)  # [K,A]

    return InferResponse(
        actions_npy_b64=np_to_b64_npy(actions),
        K=int(actions.shape[0]),
        A=int(actions.shape[1]),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")