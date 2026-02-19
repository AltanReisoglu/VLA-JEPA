"""
inference.py  –  Mock smoke-test for VLA_JEPA
Builds the model from config, feeds it **fully synthetic** data,
and verifies that both forward() and predict_action() run without error.
No real dataset or checkpoint is required.
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import os
import sys
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from omegaconf import OmegaConf

# ---- env ----------------------------------------------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    from huggingface_hub import login
    login(token=HF_TOKEN)

# ---- local imports -------------------------------------------------------
from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args
from starVLA.model.framework import build_framework


# =========================================================================
#  Helpers
# =========================================================================

def make_fake_batch(
    batch_size: int = 1,
    num_views: int = 2,
    num_frames: int = 8,
    img_h: int = 224,
    img_w: int = 224,
    vid_h: int = 256,
    vid_w: int = 256,
    action_chunk: int = 7,
    action_dim: int = 7,
    state_dim: int = 8,
):
    """Return a list[dict] that matches VLA_JEPA.forward() input format."""
    batch = []
    for _ in range(batch_size):
        # Two-view PIL images
        images = [
            Image.fromarray(np.random.randint(0, 255, (img_h, img_w, 3), dtype=np.uint8))
            for _ in range(num_views)
        ]
        # Video tensor: [V, T, H, W, 3]
        video = np.random.randint(0, 255, (num_views, num_frames, vid_h, vid_w, 3), dtype=np.uint8)
        # Continuous action label: [action_chunk, action_dim]
        action = np.random.uniform(-1, 1, (action_chunk, action_dim)).astype(np.float32)
        # State: [1, state_dim]
        state = np.random.uniform(-1, 1, (1, state_dim)).astype(np.float32)

        batch.append({
            "image": images,
            "video": video,
            "lang": "pick up the red block and place it on the blue target",
            "action": action,
            "state": state,
        })
    return batch


def safe_to_device(module, device):
    """Move a module to *device*, handling meta-tensor modules gracefully."""
    try:
        module.to(device)
    except NotImplementedError:
        print(f"  ⚠ {type(module).__name__} on meta device → to_empty()")
        module.to_empty(device=device)
        def _init(m):
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
            elif hasattr(m, "weight") and m.weight is not None and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight)
        module.apply(_init)


# =========================================================================
#  Main
# =========================================================================

def run_mock_test(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"▸ Device: {device}")
    print(f"▸ CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if device.type == "cuda" else "")

    # ── 1. Build model ────────────────────────────────────────────────
    print("\n[1/4] Building VLA_JEPA model …")
    model = build_framework(cfg)
    print(f"  ✓ Model class: {type(model).__name__}")

    # Move trainable sub-modules to device
    for name in ["vj_predictor", "action_model", "slot_attention",
                  "teacher_slot_attention", "slot_proj", "teacher_slot_proj",
                  "cjepa_slot_masking", "cjepa_frozen"]:
        mod = getattr(model, name, None)
        if mod is not None:
            safe_to_device(mod, device)

    try:
        model.gemma_interface.to(device)
    except Exception:
        pass  # already on device or managed elsewhere

    # ── 2. Create fake data ───────────────────────────────────────────
    print("\n[2/4] Creating fake batch …")
    action_chunk = cfg.framework.action_model.get("action_horizon", 7)
    action_dim = cfg.framework.action_model.get("action_dim", 7)
    state_dim = cfg.framework.action_model.get("state_dim", 8)
    num_frames = cfg.framework.vj2_model.get("num_frames", 8)
    img_res = cfg.datasets.vla_data.get("resolution_size", 224)
    vid_res = cfg.datasets.vla_data.get("video_resolution_size", 256)

    batch = make_fake_batch(
        batch_size=1,
        num_frames=num_frames,
        img_h=img_res, img_w=img_res,
        vid_h=vid_res, vid_w=vid_res,
        action_chunk=action_chunk,
        action_dim=action_dim,
        state_dim=state_dim,
    )
    print(f"  ✓ batch_size=1, frames={num_frames}, action=[{action_chunk},{action_dim}]")

    # ── 3. Forward pass (training mode) ───────────────────────────────
    print("\n[3/4] Running forward() …")
    model.train()
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        output = model(batch)

    print("  ✓ forward() succeeded")
    for k, v in output.items():
        val = v.item() if hasattr(v, "item") else v
        print(f"    {k}: {val}")

    # ── 4. Predict action (inference mode) ────────────────────────────
    print("\n[4/4] Running predict_action() …")
    model.eval()
    with torch.no_grad():
        pred = model.predict_action(
            batch_images=[batch[0]["image"]],
            instructions=[batch[0]["lang"]],
            state=[batch[0]["state"]],
        )

    actions = pred["normalized_actions"]  # [B, T, action_dim]
    print(f"  ✓ predict_action() succeeded")
    print(f"    output shape : {actions.shape}")
    print(f"    action range : [{actions.min():.4f}, {actions.max():.4f}]")
    print(f"    action mean  : {actions.mean():.4f}")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  ✅  All mock tests PASSED — model is functional!")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLA_JEPA mock smoke-test")
    parser.add_argument(
        "--config_yaml", type=str,
        default="scripts/config/vlajepa_cotrain.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    run_mock_test(cfg)
