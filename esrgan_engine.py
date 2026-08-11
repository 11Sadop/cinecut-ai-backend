"""
esrgan_engine.py
─────────────────────────────────────────────────────────────────────────
REAL AI super-resolution engine using Real-ESRGAN (x4plus / RRDBNet),
replacing the previous "fake" upscaler that was just FFmpeg lanczos
scaling + sharpen filters dressed up as "AI 4K".

This module implements the RRDBNet generator architecture directly
(instead of depending on the `realesrgan`/`basicsr` pip packages, which
have a well-known broken import against modern torchvision releases)
and loads the official `weights/RealESRGAN_x4plus.pth` checkpoint that
already ships with this repo. Inference is tiled so it works within
limited VRAM (important on smaller GPUs / RunPod entry-level pods).

Reference architecture: Wang et al., "Real-ESRGAN: Training Real-World
Blind Super-Resolution with Pure Synthetic Data" (RRDBNet generator,
23 residual-in-residual dense blocks, 4x scale) — this is the exact
architecture the shipped `RealESRGAN_x4plus.pth` checkpoint was trained
with, so the state_dict keys line up with no translation needed.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "RealESRGAN_x4plus.pth")


# ─────────────────────────────────────────────────────────────────────────
#  RRDBNet architecture (matches BasicSR's rrdbnet_arch.py exactly so the
#  official Real-ESRGAN_x4plus.pth state_dict loads with strict=True)
# ─────────────────────────────────────────────────────────────────────────
class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


def _make_layer(block, n_layers, **kwargs):
    return nn.Sequential(*[block(**kwargs) for _ in range(n_layers)])


class RRDBNet(nn.Module):
    """4x super-resolution generator (matches RealESRGAN_x4plus.pth)."""
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4):
        super().__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = _make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


# ─────────────────────────────────────────────────────────────────────────
#  Model loading (lazy, cached)
# ─────────────────────────────────────────────────────────────────────────
_model = None
_device = None


def _get_model():
    global _model, _device
    if _model is not None:
        return _model, _device
    if not os.path.isfile(WEIGHTS_PATH):
        print(f"⚠️ Real-ESRGAN weights not found at {WEIGHTS_PATH} — real AI upscale unavailable.")
        return None, None
    try:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        state = torch.load(WEIGHTS_PATH, map_location=_device)
        state = state.get("params_ema", state.get("params", state))
        net.load_state_dict(state, strict=True)
        net.eval()
        net.to(_device)
        if _device.type == "cuda":
            net.half()  # fp16 for speed/VRAM on GPU
        _model = net
        print(f"✅ Real-ESRGAN x4plus loaded on {_device.type.upper()}")
        return _model, _device
    except Exception as e:
        print("❌ Real-ESRGAN load failed:", e)
        return None, None


def is_available() -> bool:
    m, _ = _get_model()
    return m is not None


# ─────────────────────────────────────────────────────────────────────────
#  Tiled inference — avoids OOM on large frames / small GPUs
# ─────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def upscale_frame_4x(frame_rgb: np.ndarray, tile: int = 640, tile_pad: int = 16) -> np.ndarray:
    """Upscales a single HxWx3 uint8 RGB frame by exactly 4x using the real
    Real-ESRGAN RRDBNet generator. Processes in overlapping tiles so large
    video frames don't blow out VRAM."""
    model, device = _get_model()
    if model is None:
        raise RuntimeError("Real-ESRGAN model not available")

    h, w, _ = frame_rgb.shape
    img = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
    if device.type == "cuda":
        img = img.half()

    scale = 4
    output = torch.zeros((1, 3, h * scale, w * scale), dtype=img.dtype, device=device)

    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            py0, px0 = max(0, y0 - tile_pad), max(0, x0 - tile_pad)
            py1, px1 = min(h, y1 + tile_pad), min(w, x1 + tile_pad)

            patch = img[:, :, py0:py1, px0:px1]
            out_patch = model(patch)

            # Crop back to the un-padded region, accounting for the 4x scale
            top = (y0 - py0) * scale
            left = (x0 - px0) * scale
            out_h = (y1 - y0) * scale
            out_w = (x1 - x0) * scale
            output[:, :, y0 * scale:y1 * scale, x0 * scale:x1 * scale] = out_patch[:, :, top:top + out_h, left:left + out_w]

    out_np = output.float().clamp(0, 1).mul(255.0).round().byte()
    out_np = out_np.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return out_np


def upscale_frame_to_size(frame_rgb: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Real-ESRGAN gives a fixed 4x. If the requested target resolution
    isn't exactly 4x the source, do the 4x AI pass first (so all fine
    detail is genuinely reconstructed by the network) then a final
    high-quality Lanczos resize down/up to the exact requested pixels."""
    import cv2
    up = upscale_frame_4x(frame_rgb)
    uh, uw = up.shape[:2]
    if uw == target_w and uh == target_h:
        return up
    interp = cv2.INTER_AREA if (target_w < uw) else cv2.INTER_LANCZOS4
    return cv2.resize(up, (target_w, target_h), interpolation=interp)
