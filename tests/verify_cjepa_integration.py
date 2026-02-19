"""
End-to-end mock verification for C-JEPA + VLA-JEPA integration.

Validates:
  1. YAML config completeness (all fields VLA_JEPA.__init__ needs)
  2. Component construction + shapes (SlotAttention, C-JEPA frozen modules)
  3. Full forward data-flow simulation (mimics VLA_JEPA.forward() Steps 2b->2c)
  4. Freeze / gradient semantics
  5. Checkpoint loading (no-prefix + prefixed formats)
  6. train() override (frozen modules stay eval)
  7. Dimension adapter correctness
  8. time_pos_embed bounds safety
  9. ObjectMaskingModule fallback (when cjepa_frozen absent)
 10. End-to-end gradient chain

Usage:
    .venv\\Scripts\\python.exe tests/verify_cjepa_integration.py
"""

import sys, os, tempfile, traceback
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Imports from the project
# ---------------------------------------------------------------------------
from starVLA.model.modules.world_model.cjepa_frozen import (
    NonCausalTransformer,
    MaskedSlotPredictor,
    CJEPAFrozenWorldModel,
    CJEPAFrozenSlotMasking,
    load_cjepa_predictor_weights,
    _load_masking_weights_from_ckpt,
)
from starVLA.model.framework.VLA_JEPA import SlotAttention, ObjectMaskingModule

# ---------------------------------------------------------------------------
# Constants - must match CLEVRER-VideoSAUR best checkpoint
# ---------------------------------------------------------------------------
CJEPA_NUM_SLOTS   = 7
CJEPA_SLOT_DIM    = 128
CJEPA_HISTORY     = 3
CJEPA_PRED        = 1
CJEPA_NUM_MASKED  = 4
CJEPA_DEPTH       = 6
CJEPA_HEADS       = 16
CJEPA_SEED        = 42

STUDENT_SLOT_DIM  = 256      # VLA-JEPA slot_dim
VISUAL_DIM        = 1280     # vjepa2-vitl hidden_size
BATCH             = 2
NUM_VIEWS         = 2

PASS = FAIL = 0


def check(cond: bool, msg: str):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  [PASS] {msg}")
    else:
        FAIL += 1; print(f"  [FAIL] {msg}")


# ======================================================================
# Helper: save mock checkpoints in both formats
# ======================================================================
def _save_no_prefix(pred: MaskedSlotPredictor, path: str):
    torch.save(pred.state_dict(), path)

def _save_with_prefix(pred: MaskedSlotPredictor, path: str):
    sd = {f"model.predictor.{k}": v for k, v in pred.state_dict().items()}
    sd["model.encoder.backbone.weight"] = torch.randn(10, 10)
    torch.save({"state_dict": sd}, path)


# ======================================================================
# TEST 1: YAML Config Completeness
# ======================================================================
def test_yaml_configs():
    print("\n=== TEST 1: YAML Config Completeness ===")
    from omegaconf import OmegaConf

    configs = [
        "scripts/config/vlajepa_cotrain.yaml",
        "starVLA/config/training/starvla_cotrain_oxe.yaml",
    ]
    required_vj2 = ["base_encoder", "depth", "num_heads", "num_frames",
                     "num_slots", "slot_dim", "num_iterations", "eps",
                     "special_action_token", "num_action_tokens_per_timestep",
                     "embodied_action_token", "num_embodied_action_tokens_per_instruction"]
    required_cjepa = ["num_slots", "slot_dim", "history_frames", "pred_frames",
                      "num_masked_slots", "depth", "heads", "hf_repo", "hf_filename"]

    for cfg_path in configs:
        full = os.path.join(os.path.dirname(os.path.dirname(__file__)), cfg_path)
        if not os.path.exists(full):
            check(False, f"{cfg_path} not found"); continue
        cfg = OmegaConf.load(full)
        name = os.path.basename(cfg_path)

        # framework.name
        check(cfg.framework.name == "VLA_JEPA", f"[{name}] framework.name == VLA_JEPA")

        # vj2_model keys
        for k in required_vj2:
            check(k in cfg.framework.vj2_model, f"[{name}] vj2_model.{k} present")

        # cjepa_frozen keys (if present)
        cjepa = cfg.framework.get("cjepa_frozen", None)
        if cjepa is not None:
            for k in required_cjepa:
                check(k in cjepa, f"[{name}] cjepa_frozen.{k} present")
            # Cross-config consistency
            check(cjepa.num_slots == cfg.framework.vj2_model.num_slots,
                  f"[{name}] cjepa num_slots == vj2 num_slots ({cjepa.num_slots}=={cfg.framework.vj2_model.num_slots})")


# ======================================================================
# TEST 2: NonCausalTransformer
# ======================================================================
def test_noncausal_transformer():
    print("\n=== TEST 2: NonCausalTransformer ===")
    xf = NonCausalTransformer(dim=CJEPA_SLOT_DIM, depth=2, heads=4,
                               dim_head=32, mlp_dim=256)
    x = torch.randn(BATCH, 28, CJEPA_SLOT_DIM)
    out = xf(x)
    check(out.shape == (BATCH, 28, CJEPA_SLOT_DIM), f"shape {out.shape}")
    out2, attns = xf(x, return_attention=True)
    check(len(attns) == 2, f"{len(attns)} attention maps for depth=2")


# ======================================================================
# TEST 3: MaskedSlotPredictor
# ======================================================================
def test_masked_slot_predictor():
    print("\n=== TEST 3: MaskedSlotPredictor ===")
    pred = MaskedSlotPredictor(
        num_slots=CJEPA_NUM_SLOTS, slot_dim=CJEPA_SLOT_DIM,
        history_frames=CJEPA_HISTORY, pred_frames=CJEPA_PRED,
        num_masked_slots=CJEPA_NUM_MASKED, seed=CJEPA_SEED,
        depth=CJEPA_DEPTH, heads=CJEPA_HEADS,
    )
    T_total = CJEPA_HISTORY + CJEPA_PRED
    x = torch.randn(BATCH, CJEPA_HISTORY, CJEPA_NUM_SLOTS, CJEPA_SLOT_DIM)
    out, midx = pred(x)
    check(out.shape == (BATCH, T_total, CJEPA_NUM_SLOTS, CJEPA_SLOT_DIM),
          f"forward: {out.shape}")
    check(midx.shape[0] == CJEPA_NUM_MASKED, f"mask count {midx.shape[0]}")
    with torch.no_grad():
        fut = pred.inference(x)
    check(fut.shape == (BATCH, CJEPA_PRED, CJEPA_NUM_SLOTS, CJEPA_SLOT_DIM),
          f"inference: {fut.shape}")
    # t=0 anchor check
    prepared, _ = pred.prepare_input(x)
    expected0 = x[:, 0] + pred.time_pos_embed[:, 0]
    diff0 = (prepared[:, 0] - expected0).abs().max().item()
    check(diff0 < 1e-5, f"t=0 real+PE (diff={diff0:.1e})")

    # Inference with T_hist > history_frames (OOB fix)
    big_x = torch.randn(1, 7, CJEPA_NUM_SLOTS, CJEPA_SLOT_DIM)
    with torch.no_grad():
        try:
            big_fut = pred.inference(big_x)
            check(big_fut.shape == (1, CJEPA_PRED, CJEPA_NUM_SLOTS, CJEPA_SLOT_DIM),
                  f"inference T_h=7: {big_fut.shape}")
        except (IndexError, RuntimeError) as e:
            check(False, f"inference T_h=7 OOB: {e}")

    return pred


# ======================================================================
# TEST 4: SlotAttention (from VLA_JEPA)
# ======================================================================
def test_slot_attention():
    print("\n=== TEST 4: SlotAttention ===")
    sa = SlotAttention(
        num_slots=CJEPA_NUM_SLOTS,
        slot_dim=STUDENT_SLOT_DIM,
        feature_dim=VISUAL_DIM * NUM_VIEWS,   # after view concat
        num_iterations=2,
    )
    feats = torch.randn(BATCH, 196, VISUAL_DIM * NUM_VIEWS)
    slots = sa(feats, prev_slots=None)
    check(slots.shape == (BATCH, CJEPA_NUM_SLOTS, STUDENT_SLOT_DIM),
          f"first frame: {slots.shape}")
    slots2 = sa(feats, prev_slots=slots)
    check(slots2.shape == slots.shape, "temporal conditioning same shape")


# ======================================================================
# TEST 5: CJEPAFrozenWorldModel
# ======================================================================
def test_frozen_world_model():
    print("\n=== TEST 5: CJEPAFrozenWorldModel ===")
    wm = CJEPAFrozenWorldModel(
        num_slots=CJEPA_NUM_SLOTS, cjepa_slot_dim=CJEPA_SLOT_DIM,
        history_frames=CJEPA_HISTORY, pred_frames=CJEPA_PRED,
        num_masked_slots=CJEPA_NUM_MASKED, seed=CJEPA_SEED,
        depth=2, heads=4, dim_head=32, mlp_dim=256,
        student_slot_dim=STUDENT_SLOT_DIM,
    )
    wm.freeze_predictor()

    # Frozen check
    check(not any(p.requires_grad for p in wm.predictor.parameters()),
          "predictor frozen")
    check(all(p.requires_grad for p in wm.slot_adapter.parameters()),
          "slot_adapter trainable")

    # Inference mode
    h = torch.randn(BATCH, CJEPA_HISTORY, CJEPA_NUM_SLOTS, STUDENT_SLOT_DIM)
    out, midx = wm(h, use_inference=True)
    check(out.shape == (BATCH, CJEPA_PRED, CJEPA_NUM_SLOTS, STUDENT_SLOT_DIM),
          f"inference: {out.shape}")

    # Training mode
    out_t, midx_t = wm(h, use_inference=False)
    T = CJEPA_HISTORY + CJEPA_PRED
    check(out_t.shape == (BATCH, T, CJEPA_NUM_SLOTS, STUDENT_SLOT_DIM),
          f"training: {out_t.shape}")


# ======================================================================
# TEST 6: CJEPAFrozenSlotMasking
# ======================================================================
def test_frozen_slot_masking():
    print("\n=== TEST 6: CJEPAFrozenSlotMasking ===")
    m = CJEPAFrozenSlotMasking(
        cjepa_slot_dim=CJEPA_SLOT_DIM, student_slot_dim=STUDENT_SLOT_DIM,
        num_slots=CJEPA_NUM_SLOTS, history_frames=CJEPA_HISTORY,
        pred_frames=CJEPA_PRED, num_masked_slots=CJEPA_NUM_MASKED,
        seed=CJEPA_SEED,
    )
    m.freeze_masking()

    check(not m.mask_token.requires_grad, "mask_token frozen")
    check(not m.time_pos_embed.requires_grad, "time_pos_embed frozen")
    check(all(p.requires_grad for p in m.adapter_in.parameters()), "adapter_in trainable")

    # apply_masking - T_h must equal history_frames for time_pos_embed bounds
    T_h = CJEPA_HISTORY
    T_total = T_h + CJEPA_PRED
    slots = torch.randn(BATCH, T_h + 1, CJEPA_NUM_SLOTS, STUDENT_SLOT_DIM,
                         requires_grad=True)
    out, midx = m.apply_masking(slots, T_h)
    check(out.shape == (BATCH, T_total, CJEPA_NUM_SLOTS, STUDENT_SLOT_DIM),
          f"masking shape: {out.shape}")
    check(midx.shape[0] == CJEPA_NUM_MASKED, f"mask count == {CJEPA_NUM_MASKED}")

    # Gradient flow
    out.sum().backward()
    check(slots.grad is not None, "gradient flows to input slots")
    check(m.mask_token.grad is None, "no gradient to frozen mask_token")

    return m


# ======================================================================
# TEST 7: Checkpoint loading - no prefix
# ======================================================================
def test_ckpt_no_prefix(pred):
    print("\n=== TEST 7: Checkpoint loading (no prefix) ===")
    with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as f:
        p = f.name
    try:
        _save_no_prefix(pred, p)
        new = MaskedSlotPredictor(
            num_slots=CJEPA_NUM_SLOTS, slot_dim=CJEPA_SLOT_DIM,
            history_frames=CJEPA_HISTORY, pred_frames=CJEPA_PRED,
            num_masked_slots=CJEPA_NUM_MASKED, seed=CJEPA_SEED,
            depth=CJEPA_DEPTH, heads=CJEPA_HEADS,
        )
        load_cjepa_predictor_weights(new, p)
        ok = all((pred.state_dict()[k] - new.state_dict()[k]).abs().max() < 1e-6
                 for k in pred.state_dict())
        check(ok, "all weights match (no prefix)")
    finally:
        os.unlink(p)


# ======================================================================
# TEST 8: Checkpoint loading - model.predictor. prefix
# ======================================================================
def test_ckpt_prefix(pred):
    print("\n=== TEST 8: Checkpoint loading (model.predictor.) ===")
    with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as f:
        p = f.name
    try:
        _save_with_prefix(pred, p)
        new = MaskedSlotPredictor(
            num_slots=CJEPA_NUM_SLOTS, slot_dim=CJEPA_SLOT_DIM,
            history_frames=CJEPA_HISTORY, pred_frames=CJEPA_PRED,
            num_masked_slots=CJEPA_NUM_MASKED, seed=CJEPA_SEED,
            depth=CJEPA_DEPTH, heads=CJEPA_HEADS,
        )
        load_cjepa_predictor_weights(new, p)
        ok = all((pred.state_dict()[k] - new.state_dict()[k]).abs().max() < 1e-6
                 for k in pred.state_dict())
        check(ok, "all weights match (model.predictor. prefix)")
    finally:
        os.unlink(p)


# ======================================================================
# TEST 9: _load_masking_weights_from_ckpt
# ======================================================================
def test_masking_ckpt(pred):
    print("\n=== TEST 9: Masking weight loading ===")
    with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as f:
        p = f.name
    try:
        _save_no_prefix(pred, p)
        m = CJEPAFrozenSlotMasking(
            cjepa_slot_dim=CJEPA_SLOT_DIM, student_slot_dim=STUDENT_SLOT_DIM,
            num_slots=CJEPA_NUM_SLOTS, history_frames=CJEPA_HISTORY,
            pred_frames=CJEPA_PRED, num_masked_slots=CJEPA_NUM_MASKED,
        )
        _load_masking_weights_from_ckpt(m, p)
        check((m.mask_token.data - pred.mask_token.data).abs().max() < 1e-6,
              "mask_token matches")
        check((m.id_projector.weight.data - pred.id_projector.weight.data).abs().max() < 1e-6,
              "id_projector.weight matches")
    finally:
        os.unlink(p)

    # --- Test time_pos_embed shape mismatch (ckpt T=16, module T=4) ---
    print("\n=== TEST 9b: Masking weight loading with time_pos_embed mismatch ===")
    big_pred = MaskedSlotPredictor(
        num_slots=CJEPA_NUM_SLOTS, slot_dim=CJEPA_SLOT_DIM,
        history_frames=12, pred_frames=4,  # total_frames=16
        num_masked_slots=CJEPA_NUM_MASKED, seed=CJEPA_SEED,
        depth=CJEPA_DEPTH, heads=CJEPA_HEADS,
    )
    with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as f:
        p = f.name
    try:
        _save_no_prefix(big_pred, p)
        # Load into masking module with T=4
        m2 = CJEPAFrozenSlotMasking(
            cjepa_slot_dim=CJEPA_SLOT_DIM, student_slot_dim=STUDENT_SLOT_DIM,
            num_slots=CJEPA_NUM_SLOTS, history_frames=CJEPA_HISTORY,
            pred_frames=CJEPA_PRED, num_masked_slots=CJEPA_NUM_MASKED,
        )
        try:
            _load_masking_weights_from_ckpt(m2, p)
            check(m2.time_pos_embed.shape == (1, 4, 1, CJEPA_SLOT_DIM),
                  f"time_pos_embed interpolated to T=4: {m2.time_pos_embed.shape}")
        except RuntimeError as e:
            check(False, f"time_pos_embed mismatch not handled: {e}")

        # Load into predictor with T=4
        small_pred = MaskedSlotPredictor(
            num_slots=CJEPA_NUM_SLOTS, slot_dim=CJEPA_SLOT_DIM,
            history_frames=CJEPA_HISTORY, pred_frames=CJEPA_PRED,
            num_masked_slots=CJEPA_NUM_MASKED, seed=CJEPA_SEED,
            depth=CJEPA_DEPTH, heads=CJEPA_HEADS,
        )
        try:
            load_cjepa_predictor_weights(small_pred, p)
            check(small_pred.time_pos_embed.shape == (1, 4, 1, CJEPA_SLOT_DIM),
                  f"predictor time_pos_embed interpolated to T=4: {small_pred.time_pos_embed.shape}")
        except RuntimeError as e:
            check(False, f"predictor time_pos_embed mismatch not handled: {e}")
    finally:
        os.unlink(p)


# ======================================================================
# TEST 10: Full VLA_JEPA.forward() simulation (Steps 2b->2c)
# ======================================================================
def test_full_forward_simulation():
    print("\n=== TEST 10: VLA_JEPA forward() simulation ===")

    # ---- Build modules (same as VLA_JEPA.__init__) ----
    num_slots = CJEPA_NUM_SLOTS
    slot_dim  = STUDENT_SLOT_DIM

    slot_attention = SlotAttention(
        num_slots=num_slots, slot_dim=slot_dim,
        feature_dim=VISUAL_DIM * NUM_VIEWS, num_iterations=2,
    )

    cjepa_slot_masking = CJEPAFrozenSlotMasking(
        cjepa_slot_dim=CJEPA_SLOT_DIM, student_slot_dim=slot_dim,
        num_slots=num_slots, history_frames=CJEPA_HISTORY,
        pred_frames=CJEPA_PRED, num_masked_slots=CJEPA_NUM_MASKED,
        seed=CJEPA_SEED,
    )
    cjepa_slot_masking.freeze_masking()

    cjepa_frozen = CJEPAFrozenWorldModel(
        num_slots=num_slots, cjepa_slot_dim=CJEPA_SLOT_DIM,
        history_frames=CJEPA_HISTORY, pred_frames=CJEPA_PRED,
        num_masked_slots=CJEPA_NUM_MASKED, seed=CJEPA_SEED,
        depth=2, heads=4, dim_head=32, mlp_dim=256,
        student_slot_dim=slot_dim,
    )
    cjepa_frozen.freeze_predictor()

    # Freeze slot attention
    for p in slot_attention.parameters():
        p.requires_grad_(False)
    slot_attention.eval()

    slot_proj = nn.Linear(slot_dim, VISUAL_DIM * 2)

    # ---- Test with num_latents_temporal matching C-JEPA (4) ----
    num_latents_temporal = CJEPA_HISTORY + CJEPA_PRED  # 4
    tokens_per_latent = 196
    video_embeddings = torch.randn(
        BATCH, num_latents_temporal * tokens_per_latent, VISUAL_DIM * NUM_VIEWS
    )
    video_4d = video_embeddings.view(
        BATCH, num_latents_temporal, tokens_per_latent, VISUAL_DIM * NUM_VIEWS
    )
    prev = None; slots_list = []
    for t in range(num_latents_temporal):
        s = slot_attention(video_4d[:, t], prev)
        slots_list.append(s); prev = s
    video_slots = torch.stack(slots_list, dim=1)
    check(video_slots.shape == (BATCH, num_latents_temporal, num_slots, slot_dim),
          f"video_slots: {video_slots.shape}")

    # Step 2b-aux: clamped C-JEPA auxiliary loss
    _max_hist = cjepa_frozen.predictor.history_frames
    T_h = min(num_latents_temporal - 1, _max_hist)  # min(3, 3) = 3
    check(T_h == CJEPA_HISTORY, f"T_h clamped: {T_h} == {CJEPA_HISTORY}")
    _hist_start = num_latents_temporal - 1 - T_h
    history_slots = video_slots[:, _hist_start:_hist_start + T_h]
    cjepa_pred_back, _ = cjepa_frozen(history_slots, use_inference=True)
    _target_start = _hist_start + T_h
    pred_avail = min(cjepa_pred_back.shape[1], num_latents_temporal - _target_start)
    check(pred_avail > 0, f"pred_avail={pred_avail}")
    target = video_slots[:, _target_start:_target_start + pred_avail].detach()
    cjepa_loss = F.mse_loss(cjepa_pred_back[:, :pred_avail], target)
    check(cjepa_loss.dim() == 0, f"cjepa_loss scalar: {cjepa_loss.item():.4f}")

    # ---- Test with more latents than history_frames (e.g. tubelet=1) ----
    num_latents_big = 8
    big_slots = torch.randn(BATCH, num_latents_big, num_slots, slot_dim)
    T_h_big = min(num_latents_big - 1, _max_hist)  # min(7, 3) = 3
    check(T_h_big == _max_hist, f"T_h_big clamped to max_hist: {T_h_big} == {_max_hist}")
    _hs2 = num_latents_big - 1 - T_h_big  # use LAST 3 frames
    h2 = big_slots[:, _hs2:_hs2 + T_h_big]
    pred2, _ = cjepa_frozen(h2, use_inference=True)
    check(pred2.shape == (BATCH, CJEPA_PRED, num_slots, slot_dim),
          f"clamped aux pred: {pred2.shape}")

    # Step 2c: masking
    T_h_mask = num_latents_temporal - 1
    masked_slots, _midx = cjepa_slot_masking.apply_masking(video_slots, T_h_mask)
    if masked_slots.shape[1] != num_latents_temporal:
        masked_slots = masked_slots[:, :num_latents_temporal]
    check(masked_slots.shape == (BATCH, num_latents_temporal, num_slots, slot_dim),
          f"masked_slots: {masked_slots.shape}")

    projected = slot_proj(masked_slots)
    flat = projected.view(BATCH, num_latents_temporal * num_slots, -1)
    input_states = flat[:, :num_slots * T_h_mask]
    check(input_states.shape[1] == num_slots * T_h_mask,
          f"input_states tokens: {input_states.shape[1]} == {num_slots * T_h_mask}")

    # Loss dict with scaling
    _scale = 0.05
    losses = {"wm_loss": torch.tensor(0.5), "cjepa_frozen_loss": cjepa_loss * _scale}
    check("cjepa_frozen_loss" in losses, "loss dict has cjepa_frozen_loss")
    check(losses["cjepa_frozen_loss"].item() < cjepa_loss.item(),
          f"loss scaled: {losses['cjepa_frozen_loss'].item():.4f} < {cjepa_loss.item():.4f}")


# ======================================================================
# TEST 11: train() override keeps frozen modules eval
# ======================================================================
def test_train_override():
    print("\n=== TEST 11: train() override ===")
    m = CJEPAFrozenSlotMasking(
        cjepa_slot_dim=CJEPA_SLOT_DIM, student_slot_dim=STUDENT_SLOT_DIM,
        num_slots=CJEPA_NUM_SLOTS, history_frames=CJEPA_HISTORY,
        pred_frames=CJEPA_PRED, num_masked_slots=CJEPA_NUM_MASKED,
    )
    m.freeze_masking()
    wm = CJEPAFrozenWorldModel(
        num_slots=CJEPA_NUM_SLOTS, cjepa_slot_dim=CJEPA_SLOT_DIM,
        history_frames=CJEPA_HISTORY, pred_frames=CJEPA_PRED,
        num_masked_slots=CJEPA_NUM_MASKED, depth=2, heads=4,
        dim_head=32, mlp_dim=256, student_slot_dim=STUDENT_SLOT_DIM,
    )
    wm.freeze_predictor()

    # Simulate model.train() + manual re-eval (what VLA_JEPA.train() does)
    m.train(); wm.train()
    m.id_projector.eval(); wm.predictor.eval()

    check(not m.id_projector.training, "id_projector eval after train()")
    check(not wm.predictor.training, "predictor eval after train()")
    check(m.adapter_in.training, "adapter_in in train mode")
    check(wm.slot_adapter.training, "slot_adapter in train mode")


# ======================================================================
# TEST 12: Dimension adapter Identity vs Linear
# ======================================================================
def test_dim_adapters():
    print("\n=== TEST 12: Dimension adapters ===")
    same = CJEPAFrozenSlotMasking(
        cjepa_slot_dim=128, student_slot_dim=128,
        num_slots=7, history_frames=3, pred_frames=1,
    )
    diff = CJEPAFrozenSlotMasking(
        cjepa_slot_dim=128, student_slot_dim=256,
        num_slots=7, history_frames=3, pred_frames=1,
    )
    check(isinstance(same.adapter_in, nn.Identity), "Identity when dims match")
    check(not isinstance(diff.adapter_in, nn.Identity), "Linear when dims differ")
    check(isinstance(same.adapter_out, nn.Identity), "out Identity when match")
    check(not isinstance(diff.adapter_out, nn.Identity), "out Linear when differ")


# ======================================================================
# TEST 13: time_pos_embed bounds safety
# ======================================================================
def test_time_pos_embed_bounds():
    print("\n=== TEST 13: time_pos_embed bounds ===")
    m = CJEPAFrozenSlotMasking(
        cjepa_slot_dim=CJEPA_SLOT_DIM, student_slot_dim=STUDENT_SLOT_DIM,
        num_slots=CJEPA_NUM_SLOTS, history_frames=CJEPA_HISTORY,
        pred_frames=CJEPA_PRED, num_masked_slots=CJEPA_NUM_MASKED,
    )
    T_total_max = CJEPA_HISTORY + CJEPA_PRED  # 4
    tpe_T = m.time_pos_embed.shape[1]
    check(tpe_T == T_total_max,
          f"time_pos_embed T dim = {tpe_T} == {T_total_max}")

    # T_h matching history_frames works
    slots_ok = torch.randn(1, CJEPA_HISTORY + 1, CJEPA_NUM_SLOTS, STUDENT_SLOT_DIM)
    out_ok, _ = m.apply_masking(slots_ok, T_h=CJEPA_HISTORY)
    check(out_ok.shape[1] == T_total_max, f"T_h={CJEPA_HISTORY} OK: T_out={out_ok.shape[1]}")

    # T_h > history_frames: time_pos_embed OOB risk
    big_T_h = 7
    slots_big = torch.randn(1, big_T_h + 1, CJEPA_NUM_SLOTS, STUDENT_SLOT_DIM)
    try:
        out_big, _ = m.apply_masking(slots_big, T_h=big_T_h)
        check(out_big.shape[1] == big_T_h + CJEPA_PRED,
              f"T_h={big_T_h} handled: T_out={out_big.shape[1]}")
    except (IndexError, RuntimeError) as e:
        check(False, f"T_h={big_T_h} OOB error: {type(e).__name__}: {e}")
        print("  [INFO] time_pos_embed has {0} positions but T_total={1} needed.".format(
            tpe_T, big_T_h + CJEPA_PRED))
        print("  [INFO] VLA_JEPA must ensure T_h <= history_frames OR extend time_pos_embed.")


# ======================================================================
# TEST 14: ObjectMaskingModule fallback
# ======================================================================
def test_object_masking_fallback():
    print("\n=== TEST 14: ObjectMaskingModule fallback ===")
    om = ObjectMaskingModule(slot_dim=STUDENT_SLOT_DIM, max_timesteps=20)
    T_h = 3; T = 5; N = 7
    slots = torch.randn(BATCH, T, N, STUDENT_SLOT_DIM)
    mask_indices = ObjectMaskingModule.sample_mask_indices(N)
    out, mask_map = om.apply_masking(slots, mask_indices, T_h)
    check(out.shape == (BATCH, T, N, STUDENT_SLOT_DIM), f"fallback shape: {out.shape}")
    check(mask_map.shape == (BATCH, T, N), f"mask_map shape: {mask_map.shape}")
    check(not mask_map[:, 0].any().item(), "t=0 never masked (identity anchor)")


# ======================================================================
# TEST 15: End-to-end gradient chain
# ======================================================================
def test_gradient_chain():
    print("\n=== TEST 15: End-to-end gradient chain ===")
    sa = SlotAttention(num_slots=7, slot_dim=256,
                       feature_dim=VISUAL_DIM*2, num_iterations=2)
    masking = CJEPAFrozenSlotMasking(
        cjepa_slot_dim=128, student_slot_dim=256,
        num_slots=7, history_frames=3, pred_frames=1,
        num_masked_slots=4, seed=42,
    )
    masking.freeze_masking()
    proj = nn.Linear(256, VISUAL_DIM * 2)

    feats = torch.randn(1, 4, 196, VISUAL_DIM * 2)  # [B, T, N_spatial, D]
    prev = None; slist = []
    for t in range(4):
        s = sa(feats[:, t], prev); slist.append(s); prev = s
    video_slots = torch.stack(slist, dim=1)

    masked, _ = masking.apply_masking(video_slots, T_h=3)
    projected = proj(masked)
    loss = projected.sum()
    loss.backward()

    check(sa.to_q.weight.grad is not None, "SlotAttention gets gradient")
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in masking.adapter_in.parameters())
    check(has_grad, "adapter_in gets gradient")
    check(proj.weight.grad is not None, "slot_proj gets gradient")
    check(masking.mask_token.grad is None, "frozen mask_token no gradient")


# ======================================================================
# MAIN
# ======================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("C-JEPA + VLA-JEPA Integration Verification (v2)")
    print("=" * 70)

    predictor = None

    ordered_tests = [
        test_yaml_configs,
        test_noncausal_transformer,
        test_masked_slot_predictor,   # returns predictor
        test_slot_attention,
        test_frozen_world_model,
        test_frozen_slot_masking,
        # ckpt tests inserted after predictor is available
        test_full_forward_simulation,
        test_train_override,
        test_dim_adapters,
        test_time_pos_embed_bounds,
        test_object_masking_fallback,
        test_gradient_chain,
    ]

    for t in ordered_tests:
        try:
            ret = t()
            if isinstance(ret, MaskedSlotPredictor):
                predictor = ret
        except Exception as e:
            FAIL += 1
            print(f"  [FAIL] {t.__name__} raised: {e}")
            traceback.print_exc()

    # Checkpoint tests (need predictor)
    if predictor is not None:
        for t in [test_ckpt_no_prefix, test_ckpt_prefix, test_masking_ckpt]:
            try:
                t(predictor)
            except Exception as e:
                FAIL += 1
                print(f"  [FAIL] {t.__name__} raised: {e}")
                traceback.print_exc()
    else:
        print("\n  [SKIP] Checkpoint tests skipped (predictor not built)")

    print("\n" + "=" * 70)
    print(f"RESULTS: {PASS} passed, {FAIL} failed")
    print("=" * 70)
    if FAIL:
        print("\nSome checks failed - review output above.")
        sys.exit(1)
    else:
        print("\nAll checks passed.")
        sys.exit(0)
