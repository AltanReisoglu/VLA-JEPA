"""
Frozen C-JEPA World Model Module.

Integrates the pre-trained MaskedSlotPredictor from:
    galilai-group/cjepa  (https://github.com/galilai-group/cjepa)
    HuggingFace weights: HazelNam/CJEPA

Architecture Reference  (src/cjepa_predictor.py  in the cjepa repo):
    - NonCausalTransformer : standard ViT encoder with full (non-causal) attention
    - MaskedSlotPredictor  : object-level masked slot prediction (V-JEPA style, over slot sequences)

How it is used here (frozen / no-grad):
    1. Load pre-trained weights from HuggingFace (HazelNam/CJEPA).
    2. Freeze all parameters (eval mode, no gradient).
    3. Optionally project VLA-JEPA slot features into cjepa's slot_dim via a learnable adapter.
    4. Run the frozen predictor to obtain pseudo-target slot predictions.
    5. Compute an auxiliary distillation loss against the main vj_predictor output.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NonCausalTransformer  (mirrored from cjepa src/cjepa_predictor.py)
# ---------------------------------------------------------------------------

class NonCausalTransformer(nn.Module):
    """Standard Transformer Encoder with full (non-causal) self-attention."""

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True),
                nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, mlp_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(mlp_dim, dim),
                    nn.Dropout(dropout),
                ),
            ]))

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        Args:
            x: (B, SeqLen, D)
        Returns:
            (B, SeqLen, D) — or ((B, SeqLen, D), list[attn_weights]) if return_attention
        """
        attn_weights_list: Optional[List] = [] if return_attention else None
        for attn_module, ff in self.layers:
            if return_attention:
                attn_out, attn_wts = attn_module(
                    x, x, x, need_weights=True, average_attn_weights=True
                )
                attn_weights_list.append(attn_wts)
            else:
                attn_out, _ = attn_module(x, x, x)
            x = x + attn_out
            x = x + ff(x)

        out = self.norm(x)
        if return_attention:
            return out, attn_weights_list
        return out


# ---------------------------------------------------------------------------
# MaskedSlotPredictor  (mirrored from cjepa src/cjepa_predictor.py)
# ---------------------------------------------------------------------------

class MaskedSlotPredictor(nn.Module):
    """
    C-JEPA predictor operating over object-slot sequences.

    Args:
        num_slots:       Total number of slots per frame (S).
        slot_dim:        Dimension of each slot vector (D).
        history_frames:  Number of observed (history) frames (T_hist).
        pred_frames:     Number of future frames to predict.
        num_masked_slots: Number of slots to mask in history frames.
        seed:            RNG seed for reproducible masking.
        depth / heads / dim_head / mlp_dim / dropout: Transformer hyper-params.
    """

    def __init__(
        self,
        num_slots: int,
        slot_dim: int = 64,
        history_frames: int = 3,
        pred_frames: int = 1,
        num_masked_slots: int = 2,
        seed: int = 42,
        depth: int = 6,
        heads: int = 8,
        dim_head: int = 64,
        mlp_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.history_frames = history_frames
        self.pred_frames = pred_frames
        self.total_frames = history_frames + pred_frames
        self.num_masked_slots = num_masked_slots
        self.seed = seed

        # 1. Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, slot_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # 2. Temporal positional embedding  (1, T_total, 1, D)
        self.time_pos_embed = nn.Parameter(
            torch.randn(1, self.total_frames, 1, slot_dim)
        )

        # 3. Identity anchor projector
        self.id_projector = nn.Linear(slot_dim, slot_dim)

        # 4. Non-causal Transformer backbone
        self.transformer = NonCausalTransformer(
            dim=slot_dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )

        # 5. Output head
        self.to_out = nn.Linear(slot_dim, slot_dim)

    # ------------------------------------------------------------------
    def get_mask_indices(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (is_slot_masked bool tensor, masked_indices long tensor)."""
        rng = np.random.RandomState(self.seed)
        masked_indices = rng.choice(self.num_slots, self.num_masked_slots, replace=False)
        is_slot_masked = torch.zeros(self.num_slots, dtype=torch.bool, device=device)
        is_slot_masked[masked_indices] = True
        return is_slot_masked, torch.from_numpy(masked_indices).to(device)

    # ------------------------------------------------------------------
    def prepare_input(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Construct the full transformer input (history + future query tokens).

        Args:
            x: (B, T_hist, S, D)

        Returns:
            final_input: (B, T_total, S, D)
            masked_indices: 1-D long tensor of masked slot indices
        """
        B, T_hist, S, D = x.shape
        T_total = self.total_frames
        device = x.device

        if self.num_masked_slots > 0:
            is_slot_masked, masked_indices = self.get_mask_indices(B, device)
        else:
            masked_indices = torch.tensor([], dtype=torch.long, device=device)

        # Anchor query = id_projector(first frame of each slot)
        anchors = x[:, 0, :, :]                        # (B, S, D)
        anchor_queries = self.id_projector(anchors)     # (B, S, D)

        # Base query grid = mask_token + time_PE + anchor_query
        tokens_grid = self.mask_token.expand(B, T_total, S, D)
        pos_grid    = self.time_pos_embed.expand(B, T_total, S, D)
        anchor_grid = anchor_queries.unsqueeze(1).expand(B, T_total, S, D)
        query_input = tokens_grid + pos_grid + anchor_grid          # (B, T, S, D)

        final_input = query_input.clone()

        # (A) Always overwrite t=0 with real data + time_PE(0)
        final_input[:, 0, :, :] = x[:, 0, :, :] + self.time_pos_embed[:, 0, :, :]

        # (B) Overwrite history (t=1..T_hist-1) for UNMASKED slots
        if self.num_masked_slots > 0:
            unmasked_indices = torch.where(~is_slot_masked)[0]
        else:
            unmasked_indices = torch.arange(S, device=device)

        if len(unmasked_indices) > 0 and T_hist > 1:
            real_history = x[:, 1:, unmasked_indices, :]   # (B, T_hist-1, n_unmasked, D)
            history_pos  = self.time_pos_embed[:, 1:T_hist, :, :].expand(B, T_hist - 1, S, D)
            history_pos_unmasked = history_pos[:, :, unmasked_indices, :]
            final_input[:, 1:T_hist, unmasked_indices, :] = real_history + history_pos_unmasked

        return final_input, masked_indices

    # ------------------------------------------------------------------
    @torch.no_grad()
    def inference(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference (no masking of history).

        Args:
            x: (B, T_hist, S, D) — fully visible history

        Returns:
            (B, T_pred, S, D) — predicted future slots
        """
        B, T_hist, S, D = x.shape
        T_pred  = self.pred_frames
        T_total = T_hist + T_pred

        # Interpolate time_pos_embed when T_total exceeds stored size
        tpe = self.time_pos_embed  # (1, total_frames, 1, D)
        if T_total <= tpe.shape[1]:
            inf_pos = tpe[:, -T_total:, :, :]
        else:
            _t = tpe.squeeze(2).permute(0, 2, 1)  # (1, D, stored_T)
            _t = F.interpolate(_t, size=T_total, mode="linear", align_corners=True)
            inf_pos = _t.permute(0, 2, 1).unsqueeze(2)  # (1, T_total, 1, D)

        anchors       = x[:, 0, :, :]
        anchor_queries = self.id_projector(anchors)

        input_history = x + inf_pos[:, :T_hist, :, :]

        tokens_grid = self.mask_token.expand(B, T_pred, S, D)
        pos_grid    = inf_pos[:, T_hist:T_total, :, :].expand(B, T_pred, S, D)
        anchor_grid = anchor_queries.unsqueeze(1).expand(B, T_pred, S, D)
        input_future = tokens_grid + pos_grid + anchor_grid

        full_input = torch.cat([input_history, input_future], dim=1)
        x_flat  = rearrange(full_input, "b t s d -> b (t s) d")
        out_flat = self.transformer(x_flat)
        out = rearrange(out_flat, "b (t s) d -> b t s d", t=T_total, s=S)
        out = self.to_out(out)
        return out[:, T_hist:, :, :]          # (B, T_pred, S, D)

    # ------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Training forward with masking.

        Args:
            x: (B, T_hist, S, D)

        Returns:
            out:            (B, T_total, S, D)
            masked_indices: 1-D long tensor
        """
        x_input, masked_indices = self.prepare_input(x)
        x_flat   = rearrange(x_input, "b t s d -> b (t s) d")
        out_flat = self.transformer(x_flat)
        out = rearrange(out_flat, "b (t s) d -> b t s d", t=self.total_frames, s=self.num_slots)
        out = self.to_out(out)
        return out, masked_indices


# ---------------------------------------------------------------------------
# Checkpoint helper
# ---------------------------------------------------------------------------

def load_cjepa_predictor_weights(
    predictor: MaskedSlotPredictor,
    ckpt_path: str,
    map_location: str = "cpu",
) -> MaskedSlotPredictor:
    """
    Load *only* the predictor sub-module weights from a cjepa checkpoint.

    The cjepa checkpoint (stable_pretraining / PyTorch Lightning) stores the
    full CausalWM model.  The predictor weights live under the key prefix
    ``model.predictor.`` inside the ``state_dict``.

    Args:
        predictor:    An *already constructed* MaskedSlotPredictor instance
                      whose hyper-params match those used when the checkpoint
                      was trained.
        ckpt_path:    Local path (or HuggingFace hub path) to the ``.ckpt``
                      file produced by cjepa training.
        map_location: Where to load the tensors (``"cpu"`` recommended).

    Returns:
        The predictor with weights loaded.
    """
    raw = torch.load(ckpt_path, map_location=map_location, weights_only=False)

    # Support both Lightning CheckpointConnector format and plain state-dict
    if isinstance(raw, dict) and "state_dict" in raw:
        full_sd = raw["state_dict"]
    elif isinstance(raw, dict):
        full_sd = raw
    else:
        raise ValueError(f"Unrecognised checkpoint format: {type(raw)}")

    # Extract predictor weights — try several key prefixes
    # C-JEPA training code saves `predictor.state_dict()` directly (no prefix)
    # but Lightning wrappers may add `model.predictor.` or similar.
    predictor_sd: dict = {}
    for prefix in ("model.predictor.", "module.model.predictor.", "predictor.", "model."):
        filtered = {
            k[len(prefix):]: v
            for k, v in full_sd.items()
            if k.startswith(prefix)
        }
        if filtered:
            predictor_sd = filtered
            logger.info(f"[CJEPAFrozen] Extracted predictor weights with prefix '{prefix}' "
                        f"({len(predictor_sd)} keys)")
            break

    # Fallback: no prefix at all (plain state_dict from predictor.state_dict())
    if not predictor_sd:
        # Check if top-level keys look like predictor keys
        predictor_key_hints = {"mask_token", "time_pos_embed", "id_projector.weight",
                               "transformer.norm.weight", "to_out.weight"}
        if predictor_key_hints & set(full_sd.keys()):
            predictor_sd = full_sd
            logger.info(f"[CJEPAFrozen] Using checkpoint as plain predictor state_dict "
                        f"(no prefix, {len(predictor_sd)} keys)")

    if not predictor_sd:
        available_prefixes = sorted({k.split(".")[0] for k in full_sd.keys()})
        raise KeyError(
            f"Could not find predictor keys in checkpoint.\n"
            f"Top-level keys found: {available_prefixes}"
        )

    # --- Handle time_pos_embed temporal dimension mismatch ---
    # Checkpoint may have been trained with more total_frames (e.g. 16)
    # than our predictor's total_frames (e.g. 4). Interpolate if needed.
    if "time_pos_embed" in predictor_sd:
        ckpt_tpe = predictor_sd["time_pos_embed"]
        mod_tpe = predictor.time_pos_embed
        if ckpt_tpe.shape[1] != mod_tpe.shape[1]:
            ckpt_T = ckpt_tpe.shape[1]
            mod_T = mod_tpe.shape[1]
            logger.info(
                f"[CJEPAFrozen] time_pos_embed mismatch: ckpt T={ckpt_T}, "
                f"predictor T={mod_T}. Interpolating."
            )
            p_flat = ckpt_tpe.squeeze(2).permute(0, 2, 1)  # (1, D, ckpt_T)
            p_interp = F.interpolate(p_flat, size=mod_T, mode="linear", align_corners=False)
            predictor_sd["time_pos_embed"] = p_interp.permute(0, 2, 1).unsqueeze(2)

    missing, unexpected = predictor.load_state_dict(predictor_sd, strict=True)
    if missing:
        logger.warning(f"[CJEPAFrozen] Missing keys: {missing}")
    if unexpected:
        logger.warning(f"[CJEPAFrozen] Unexpected keys: {unexpected}")

    logger.info("[CJEPAFrozen] Predictor weights loaded successfully.")
    return predictor


def download_cjepa_checkpoint(
    hf_repo: str = "HazelNam/CJEPA",
    filename: str = "cjepa-ckpts/clevrer_videosaur_4_epoch_30_object.ckpt",
    cache_dir: Optional[str] = None,
) -> str:
    """
    Download a cjepa checkpoint from HuggingFace Hub.

    Available checkpoints (as of Feb 2026 — filenames were recently renamed):

        PushT  (VideoSAUR, NUM_SLOTS=4, SLOT_DIM=128):
            cjepa-ckpts/pusht_videosaur_0_epoch_30_object.ckpt
            cjepa-ckpts/pusht_videosaur_1_epoch_30_object.ckpt  ← best (*)
            cjepa-ckpts/pusht_videosaur_2_epoch_30_object.ckpt

        CLEVRER (VideoSAUR, NUM_SLOTS=7, SLOT_DIM=128):
            cjepa-ckpts/clevrer_videosaur_0_epoch_30_object.ckpt
            cjepa-ckpts/clevrer_videosaur_1_epoch_30_object.ckpt
            cjepa-ckpts/clevrer_videosaur_2_epoch_30_object.ckpt
            cjepa-ckpts/clevrer_videosaur_3_epoch_30_object.ckpt
            cjepa-ckpts/clevrer_videosaur_4_epoch_30_object.ckpt  ← best (*)

        CLEVRER (SAVi, NUM_SLOTS=7, SLOT_DIM=128):
            cjepa-ckpts/clevrer_savi_0_epoch_30_object.ckpt
            cjepa-ckpts/clevrer_savi_1_epoch_30_object.ckpt
            cjepa-ckpts/clevrer_savi_2_epoch_30_object.ckpt  ← best (*)
            cjepa-ckpts/clevrer_savi_3_epoch_30_object.ckpt
            cjepa-ckpts/clevrer_savi_4_epoch_30_object.ckpt

        Full model (encoder + slot attention — NOT predictor-only):
            clevrer_videosaur_model.ckpt   (139 MB)
            pusht_videosaur_model.ckpt     (139 MB)
            clevrer_savi_model.pth         (14.3 MB)

        Pre-extracted slot representations (NOT model weights):
            clevrer_videosaur_slots.pkl    (9.18 GB)
            pusht_videosaur_slots.pkl      (4.79 GB)
            clevrer_savi_slots.pkl         (13.8 GB)

    Returns:
        Local path to the downloaded checkpoint file.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "huggingface_hub is required to download cjepa checkpoints. "
            "Install with: pip install huggingface_hub"
        ) from exc

    local_path = hf_hub_download(
        repo_id=hf_repo,
        filename=filename,
        cache_dir=cache_dir,
    )
    logger.info(f"[CJEPAFrozen] Downloaded checkpoint to: {local_path}")
    return local_path


# ---------------------------------------------------------------------------
# CJEPAFrozenWorldModel — the high-level wrapper used inside VLA_JEPA
# ---------------------------------------------------------------------------

class CJEPAFrozenWorldModel(nn.Module):
    """
    Frozen C-JEPA world-model component for VLA-JEPA.

    Workflow inside VLA_JEPA.forward():
        1. Student's slot features  (B, T, S, student_slot_dim)
           are projected to cjepa's slot_dim via ``slot_adapter``
           (a simple 1-layer MLP, the *only* trainable part of this module).
        2. The adapted slots are fed into the frozen ``predictor``.
        3. The frozen predictor outputs pseudo-target predictions
           (B, T_total, S, cjepa_slot_dim).
        4. These are projected *back* to student_slot_dim by
           ``output_back_proj`` (also trainable) so the VLA-JEPA predictor
           can be distilled against them.

    The predictor itself is **fully frozen** (eval mode, no gradient).
    """

    def __init__(
        self,
        # Hyper-params that must match the pre-trained checkpoint
        num_slots: int,
        cjepa_slot_dim: int,
        history_frames: int,
        pred_frames: int,
        num_masked_slots: int,
        seed: int = 42,
        depth: int = 6,
        heads: int = 8,
        dim_head: int = 64,
        mlp_dim: int = 2048,
        dropout: float = 0.1,
        # Adapter: bridge between VLA-JEPA's slot_dim and cjepa's slot_dim
        student_slot_dim: Optional[int] = None,
    ):
        super().__init__()

        self.cjepa_slot_dim   = cjepa_slot_dim
        self.student_slot_dim = student_slot_dim or cjepa_slot_dim
        self.num_slots        = num_slots

        # --- Frozen predictor ---
        self.predictor = MaskedSlotPredictor(
            num_slots=num_slots,
            slot_dim=cjepa_slot_dim,
            history_frames=history_frames,
            pred_frames=pred_frames,
            num_masked_slots=num_masked_slots,
            seed=seed,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )

        # --- Trainable adapters (learn to align feature spaces) ---
        if self.student_slot_dim != cjepa_slot_dim:
            self.slot_adapter = nn.Sequential(
                nn.LayerNorm(self.student_slot_dim),
                nn.Linear(self.student_slot_dim, cjepa_slot_dim),
            )
            self.output_back_proj = nn.Linear(cjepa_slot_dim, self.student_slot_dim)
        else:
            self.slot_adapter     = nn.Identity()
            self.output_back_proj = nn.Identity()

    # ------------------------------------------------------------------
    def freeze_predictor(self) -> None:
        """Freeze predictor weights (no gradient, eval mode)."""
        for p in self.predictor.parameters():
            p.requires_grad_(False)
        self.predictor.eval()
        logger.info("[CJEPAFrozen] Predictor frozen.")

    # ------------------------------------------------------------------
    def load_and_freeze(
        self,
        ckpt_path: str,
        map_location: str = "cpu",
    ) -> None:
        """Load checkpoint weights and freeze the predictor."""
        load_cjepa_predictor_weights(
            self.predictor, ckpt_path, map_location=map_location
        )
        self.freeze_predictor()

    # ------------------------------------------------------------------
    def forward(
        self,
        student_slots: torch.Tensor,
        use_inference: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run frozen predictor on student slot features.

        Args:
            student_slots: (B, T_hist, S, student_slot_dim)
            use_inference:  If True, use the no-mask inference path
                            (returns only future frames).
                            If False, use training path with masking.

        Returns:
            back_projected: (B, T_total or T_pred, S, student_slot_dim)
                            Predictor outputs projected back to student space.
            masked_indices:  1-D long tensor of slot indices that were masked
                             (empty if use_inference=True).
        """
        # Project to cjepa feature space  (trainable)
        adapted = self.slot_adapter(student_slots)  # (B, T, S, cjepa_slot_dim)

        # Run frozen predictor
        self.predictor.eval()
        with torch.no_grad():
            if use_inference:
                pred_out = self.predictor.inference(adapted)   # (B, T_pred, S, D)
                masked_indices = torch.tensor([], dtype=torch.long, device=student_slots.device)
            else:
                pred_out, masked_indices = self.predictor(adapted)  # (B, T_total, S, D)

        # Project back to student space (trainable)
        back_projected = self.output_back_proj(pred_out)
        return back_projected, masked_indices


# ---------------------------------------------------------------------------
# Factory / convenience constructor
# ---------------------------------------------------------------------------

def build_cjepa_frozen_world_model(
    cfg,
    student_slot_dim: int,
    ckpt_path: Optional[str] = None,
    auto_download: bool = True,
) -> CJEPAFrozenWorldModel:
    """
    Build and optionally load a CJEPAFrozenWorldModel from config.

    Expected config keys (under ``framework.cjepa_frozen``):
        num_slots, slot_dim, history_frames, pred_frames,
        num_masked_slots, depth, heads, dim_head, mlp_dim, dropout,
        ckpt_path (optional), hf_filename (optional),
        hf_repo    (default: "HazelNam/CJEPA")

    Args:
        cfg:               OmegaConf / dict config node for ``cjepa_frozen``.
        student_slot_dim:  Slot dimension used by VLA-JEPA (for the adapter).
        ckpt_path:         Override checkpoint path (bypasses cfg.ckpt_path).
        auto_download:     If True and no local ckpt found, download from HF.

    Returns:
        Constructed CJEPAFrozenWorldModel with weights loaded and frozen.
    """
    model = CJEPAFrozenWorldModel(
        num_slots=cfg.num_slots,
        cjepa_slot_dim=cfg.slot_dim,
        history_frames=cfg.history_frames,
        pred_frames=cfg.pred_frames,
        num_masked_slots=cfg.num_masked_slots,
        seed=cfg.get("seed", 42),
        depth=cfg.get("depth", 6),
        heads=cfg.get("heads", 16),
        dim_head=cfg.get("dim_head", 64),
        mlp_dim=cfg.get("mlp_dim", 2048),
        dropout=cfg.get("dropout", 0.1),
        student_slot_dim=student_slot_dim,
    )

    # Determine checkpoint path
    ckpt = ckpt_path or cfg.get("ckpt_path", None)

    if ckpt is None and auto_download:
        hf_repo     = cfg.get("hf_repo",     "HazelNam/CJEPA")
        hf_filename = cfg.get("hf_filename",  "cjepa-ckpts/clevrer_videosaur_4_epoch_30_object.ckpt")
        cache_dir   = cfg.get("cache_dir",    None)
        ckpt = download_cjepa_checkpoint(
            hf_repo=hf_repo,
            filename=hf_filename,
            cache_dir=cache_dir,
        )

    if ckpt is not None:
        model.load_and_freeze(ckpt)
    else:
        logger.warning(
            "[CJEPAFrozen] No checkpoint path provided and auto_download=False. "
            "Predictor weights are RANDOM. Call model.load_and_freeze(ckpt_path) manually."
        )
        model.freeze_predictor()

    return model


# ---------------------------------------------------------------------------
# CJEPAFrozenSlotMasking — frozen C-JEPA masking pipeline for VLA-JEPA
# ---------------------------------------------------------------------------

class CJEPAFrozenSlotMasking(nn.Module):
    """
    Frozen C-JEPA-style object-level masking module.

    This replaces VLA-JEPA's trainable ObjectMaskingModule with the exact
    masking protocol learned during C-JEPA pre-training.

    Frozen parameters (from C-JEPA checkpoint):
        mask_token     : (1, 1, cjepa_slot_dim)  — base learnable query for masked positions
        time_pos_embed : (1, T_total, 1, cjepa_slot_dim)  — temporal positional encoding
        id_projector   : Linear(cjepa_slot_dim → cjepa_slot_dim)  — identity anchor projector

    Trainable adapters (bridging VLA-JEPA's slot_dim ↔ cjepa's slot_dim):
        adapter_in     : LayerNorm + Linear (student_slot_dim → cjepa_slot_dim)
        adapter_out    : Linear (cjepa_slot_dim → student_slot_dim)

    How masking works (C-JEPA's prepare_input logic):
        - t=0        : ALL slots = real data + time_PE(0)   (identity anchor, never masked)
        - t=1..T_h-1 : unmasked slots = real + time_PE(t),  masked slots = mask_token + time_PE(t) + id_proj(anchor)
        - t≥T_h      : ALL slots = mask_token + time_PE(t) + id_proj(anchor)   (future = fully masked)
    """

    def __init__(
        self,
        cjepa_slot_dim: int,
        student_slot_dim: int,
        num_slots: int,
        history_frames: int,
        pred_frames: int,
        num_masked_slots: int = 1,
        seed: int = 42,
    ):
        super().__init__()
        self.cjepa_slot_dim = cjepa_slot_dim
        self.student_slot_dim = student_slot_dim
        self.num_slots = num_slots
        self.history_frames = history_frames
        self.pred_frames = pred_frames
        self.total_frames = history_frames + pred_frames
        self.num_masked_slots = num_masked_slots
        self.seed = seed

        # --- Frozen masking parameters (loaded from C-JEPA checkpoint) ---
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cjepa_slot_dim))
        self.time_pos_embed = nn.Parameter(
            torch.randn(1, self.total_frames, 1, cjepa_slot_dim)
        )
        self.id_projector = nn.Linear(cjepa_slot_dim, cjepa_slot_dim)

        # --- Trainable dimension adapters ---
        if student_slot_dim != cjepa_slot_dim:
            self.adapter_in = nn.Sequential(
                nn.LayerNorm(student_slot_dim),
                nn.Linear(student_slot_dim, cjepa_slot_dim),
            )
            self.adapter_out = nn.Linear(cjepa_slot_dim, student_slot_dim)
        else:
            self.adapter_in = nn.Identity()
            self.adapter_out = nn.Identity()

    # ------------------------------------------------------------------
    def freeze_masking(self) -> None:
        """Freeze the masking parameters (mask_token, time_pos_embed, id_projector)."""
        self.mask_token.requires_grad_(False)
        self.time_pos_embed.requires_grad_(False)
        for p in self.id_projector.parameters():
            p.requires_grad_(False)
        self.id_projector.eval()
        logger.info("[CJEPAFrozenSlotMasking] Masking parameters frozen.")

    # ------------------------------------------------------------------
    def get_mask_indices(
        self, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Determine which slots to mask (deterministic via seed, matching C-JEPA)."""
        rng = np.random.RandomState(self.seed)
        masked_indices = rng.choice(
            self.num_slots, self.num_masked_slots, replace=False
        )
        is_slot_masked = torch.zeros(self.num_slots, dtype=torch.bool, device=device)
        is_slot_masked[masked_indices] = True
        return is_slot_masked, torch.from_numpy(masked_indices).to(device)

    # ------------------------------------------------------------------
    def apply_masking(
        self,
        slots: torch.Tensor,
        T_h: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply frozen C-JEPA-style object-level masking.

        This follows MaskedSlotPredictor.prepare_input() from the C-JEPA repo
        exactly, using the frozen mask_token / time_pos_embed / id_projector,
        but bridged to the student's slot dimension via trainable adapters.

        Args:
            slots: (B, T, S, student_slot_dim) — output of SlotAttention.
                   T must be >= T_h.  Only the first T_h frames are history.
            T_h:   Number of history frames.

        Returns:
            masked_output: (B, T_total, S, student_slot_dim)
                           Masked slot sequence in student feature space.
                           T_total = T_h + pred_frames.
            masked_indices: 1-D long tensor of which slot indices were masked.
        """
        B, T, S, _ = slots.shape
        T_total = T_h + self.pred_frames
        device = slots.device

        # 1. Project to C-JEPA dim  (trainable adapter — gradient flows here)
        adapted = self.adapter_in(slots[:, :T_h])  # (B, T_h, S, cjepa_slot_dim)
        D = self.cjepa_slot_dim

        # 2. Get mask indices  (frozen / deterministic)
        if self.num_masked_slots > 0:
            is_slot_masked, masked_indices = self.get_mask_indices(device)
        else:
            masked_indices = torch.tensor([], dtype=torch.long, device=device)
            is_slot_masked = torch.zeros(S, dtype=torch.bool, device=device)

        # 3. Build anchor queries from t=0 (frozen id_projector)
        anchors = adapted[:, 0, :, :]                           # (B, S, D)
        with torch.no_grad():
            anchor_queries = self.id_projector(anchors.detach())  # (B, S, D)  frozen

        # 4. Get time positional embeddings (interpolate if T_total > stored size)
        tpe = self.time_pos_embed  # (1, total_frames, 1, D)
        if T_total <= tpe.shape[1]:
            time_pe = tpe[:, :T_total, :, :]
        else:
            # Interpolate to cover more timesteps than the checkpoint was trained on.
            # Shape dance: (1, stored_T, 1, D) → (1, D, stored_T) → interp → back
            _t = tpe.squeeze(2).permute(0, 2, 1)          # (1, D, stored_T)
            _t = F.interpolate(_t, size=T_total, mode="linear", align_corners=True)
            time_pe = _t.permute(0, 2, 1).unsqueeze(2)    # (1, T_total, 1, D)

        # Construct base query grid (all-masked default)
        #    query = mask_token + time_pos_embed + anchor_queries
        mask_tok = self.mask_token.expand(B, T_total, S, D)
        pos_grid = time_pe.expand(B, T_total, S, D)
        anc_grid = anchor_queries.unsqueeze(1).expand(B, T_total, S, D)
        query_input = mask_tok + pos_grid + anc_grid              # (B, T_total, S, D)

        final_input = query_input.clone()

        # 5a. t=0 always uses real data + time_PE(0)  — identity anchor
        final_input[:, 0, :, :] = adapted[:, 0, :, :] + time_pe[:, 0, :, :]

        # 5b. t=1..T_h-1 : overwrite UNMASKED slots with real data + time_PE
        if self.num_masked_slots > 0:
            unmasked_indices = torch.where(~is_slot_masked)[0]
        else:
            unmasked_indices = torch.arange(S, device=device)

        if len(unmasked_indices) > 0 and T_h > 1:
            real_history = adapted[:, 1:T_h, unmasked_indices, :]     # (B, T_h-1, n_unmasked, D)
            hist_pos = time_pe[:, 1:T_h, :, :].expand(B, T_h - 1, S, D)
            hist_pos_unmasked = hist_pos[:, :, unmasked_indices, :]
            final_input[:, 1:T_h, unmasked_indices, :] = real_history + hist_pos_unmasked

        # 5c. t>=T_h : all slots remain as query tokens (future — fully masked)
        #     (already set by the query_input default above)

        # 6. Project back to student dim  (trainable adapter — gradient flows here)
        output = self.adapter_out(final_input)  # (B, T_total, S, student_slot_dim)

        return output, masked_indices


def _load_masking_weights_from_ckpt(
    masking_module: CJEPAFrozenSlotMasking,
    ckpt_path: str,
    map_location: str = "cpu",
) -> CJEPAFrozenSlotMasking:
    """
    Load frozen masking parameters (mask_token, time_pos_embed, id_projector)
    from a C-JEPA checkpoint into a CJEPAFrozenSlotMasking module.
    """
    raw = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    if isinstance(raw, dict) and "state_dict" in raw:
        full_sd = raw["state_dict"]
    elif isinstance(raw, dict):
        full_sd = raw
    else:
        raise ValueError(f"Unrecognised checkpoint format: {type(raw)}")

    # Find the predictor prefix (C-JEPA training saves predictor.state_dict()
    # directly, so keys may have NO prefix — e.g. just 'mask_token')
    predictor_prefix = None
    for pfx in ("model.predictor.", "module.model.predictor.", "predictor.", "model."):
        if any(k.startswith(pfx) for k in full_sd):
            predictor_prefix = pfx
            break

    # Fallback: no prefix (plain state_dict from predictor.state_dict())
    if predictor_prefix is None:
        if "mask_token" in full_sd or "time_pos_embed" in full_sd:
            predictor_prefix = ""  # direct keys
        else:
            raise KeyError(
                f"Could not find masking keys in checkpoint. "
                f"Top-level keys: {sorted(set(k.split('.')[0] for k in full_sd))}"
            )

    # Map checkpoint keys → masking module keys
    key_map = {
        "mask_token": "mask_token",
        "time_pos_embed": "time_pos_embed",
        "id_projector.weight": "id_projector.weight",
        "id_projector.bias": "id_projector.bias",
    }

    loaded = 0
    for ckpt_key, mod_key in key_map.items():
        full_key = predictor_prefix + ckpt_key
        if full_key in full_sd:
            param = full_sd[full_key]
            target = masking_module
            parts = mod_key.split(".")
            for p in parts[:-1]:
                target = getattr(target, p)
            # Handle Parameter vs buffer
            attr = getattr(target, parts[-1])

            # --- Handle time_pos_embed temporal dimension mismatch ---
            # Checkpoint may have been trained with more total_frames (e.g. 16)
            # than our module's total_frames (e.g. 4). Interpolate if needed.
            if mod_key == "time_pos_embed" and param.shape[1] != attr.shape[1]:
                ckpt_T = param.shape[1]
                mod_T = attr.shape[1]
                logger.info(
                    f"[CJEPAFrozenSlotMasking] time_pos_embed shape mismatch: "
                    f"ckpt has T={ckpt_T}, module expects T={mod_T}. Interpolating."
                )
                # param: (1, ckpt_T, 1, D) → rearrange for F.interpolate
                p_flat = param.squeeze(2).permute(0, 2, 1)  # (1, D, ckpt_T)
                p_interp = F.interpolate(p_flat, size=mod_T, mode="linear", align_corners=False)
                param = p_interp.permute(0, 2, 1).unsqueeze(2)  # (1, mod_T, 1, D)

            if isinstance(attr, nn.Parameter):
                attr.data.copy_(param)
            else:
                setattr(target, parts[-1], param)
            loaded += 1
            logger.info(f"[CJEPAFrozenSlotMasking] Loaded {full_key}")
        else:
            logger.warning(f"[CJEPAFrozenSlotMasking] Key not found: {full_key}")

    logger.info(f"[CJEPAFrozenSlotMasking] Loaded {loaded}/{len(key_map)} masking weights.")
    return masking_module


def build_cjepa_frozen_slot_masking(
    cfg,
    student_slot_dim: int,
    ckpt_path: Optional[str] = None,
    auto_download: bool = True,
) -> CJEPAFrozenSlotMasking:
    """
    Build a CJEPAFrozenSlotMasking module from config, load weights, and freeze.

    Expected config keys (under ``framework.cjepa_frozen``):
        num_slots, slot_dim, history_frames, pred_frames,
        num_masked_slots, seed,
        ckpt_path or hf_filename + hf_repo

    Args:
        cfg:               Config node for cjepa_frozen.
        student_slot_dim:  VLA-JEPA's slot dimension.
        ckpt_path:         Override checkpoint path.
        auto_download:     Download from HF if no local path.

    Returns:
        CJEPAFrozenSlotMasking with masking weights loaded and frozen,
        adapters trainable.
    """
    module = CJEPAFrozenSlotMasking(
        cjepa_slot_dim=cfg.slot_dim,
        student_slot_dim=student_slot_dim,
        num_slots=cfg.num_slots,
        history_frames=cfg.history_frames,
        pred_frames=cfg.pred_frames,
        num_masked_slots=cfg.get("num_masked_slots", 1),
        seed=cfg.get("seed", 42),
    )

    ckpt = ckpt_path or cfg.get("ckpt_path", None)
    if ckpt is None and auto_download:
        hf_repo = cfg.get("hf_repo", "HazelNam/CJEPA")
        hf_filename = cfg.get("hf_filename", "cjepa-ckpts/clevrer_videosaur_4_epoch_30_object.ckpt")
        cache_dir = cfg.get("cache_dir", None)
        ckpt = download_cjepa_checkpoint(
            hf_repo=hf_repo, filename=hf_filename, cache_dir=cache_dir
        )

    if ckpt is not None:
        _load_masking_weights_from_ckpt(module, ckpt)
        module.freeze_masking()
    else:
        logger.warning(
            "[CJEPAFrozenSlotMasking] No checkpoint provided. "
            "Masking weights are RANDOM."
        )

    return module
