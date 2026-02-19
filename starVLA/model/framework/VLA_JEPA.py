# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Junqiu YU / Fudan University] in [2025]. 
# Design and Merged by [Jinhui YE / HKUST University] in [2025].
"""
Qwen-GR00T Framework
A lightweight implementation that Qwen-VL + Flow-matching head to directly predict continuous actions
Flow-matching header is copyright from GR00T N1.5,
"""
from typing import List
from tqdm import tqdm
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import AutoVideoProcessor, AutoModel, AutoTokenizer, VJEPA2VideoProcessor

from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.action_model.GR00T_ActionHeader import get_action_model, FlowmatchingActionHead
from starVLA.model.modules.world_model.vj2_predictor import VisionTransformerPredictorAC
from starVLA.model.modules.world_model.cjepa_frozen import (
    build_cjepa_frozen_world_model,
    build_cjepa_frozen_slot_masking,
)
from starVLA.model.framework.AdapterX import MultiModalTargetEncoder
from starVLA.training.trainer_utils.trainer_tools import resize_images
from starVLA.model.tools import FRAMEWORK_REGISTRY

"""
Slot Attention module.
Paper Section 3: "Object-Centric Representation via Slot Attention"
Reference: Locatello et al., 2020
Video extension: Kipf et al., 2022 (SAVi)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SlotAttention(nn.Module):
    """
    Slot Attention for object-centric representation learning.
    
    Key properties (Paper Section 3):
    - Iteratively groups visual features into N slots via competitive attention
    - Permutation-equivariant: slot ordering is arbitrary
    - Video extension: slots conditioned on previous frame's slots
      for temporal consistency
    """
    
    def __init__(
        self,
        num_slots: int,
        slot_dim: int,
        feature_dim: int,
        num_iterations: int = 2,
        eps: float = 1e-8
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_iterations = num_iterations
        self.eps = eps
        self.scale = slot_dim ** -0.5
        
        # Slot initialization (when no previous slots available)
        # Paper: stochastic SAVi uses Gaussian prior
        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, 1, slot_dim))
        nn.init.xavier_uniform_(self.slots_mu)
        
        # Attention components
        self.norm_features = nn.LayerNorm(feature_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        
        # K, V from features; Q from slots
        self.to_k = nn.Linear(feature_dim, slot_dim, bias=False)
        self.to_v = nn.Linear(feature_dim, slot_dim, bias=False)
        self.to_q = nn.Linear(slot_dim, slot_dim, bias=False)
        
        # Slot update (GRU)
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        
        # MLP for slot refinement
        self.norm_pre_mlp = nn.LayerNorm(slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(slot_dim * 4, slot_dim)
        )
    
    def forward(
        self,
        features: torch.Tensor,
        prev_slots: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            features: [B, num_patches, feature_dim] - visual features
            prev_slots: [B, num_slots, slot_dim] - previous frame's slots
                        None for first frame (random init)
        
        Returns:
            slots: [B, num_slots, slot_dim]
        
        Paper: "conditioning slot updates on previous-frame slots"
        (Section 3, temporal consistency)
        """
        B, N, _ = features.shape
        features = self.norm_features(features)
        
        # Keys and Values from visual features
        k = self.to_k(features)  # [B, N, D]
        v = self.to_v(features)  # [B, N, D]
        
        # Initialize slots
        if prev_slots is None:
            # Paper: Gaussian initialization for first frame
            # "stochastic SAVi with a Gaussian prior of variance 0.01"
            mu = self.slots_mu.expand(B, self.num_slots, -1)
            sigma = self.slots_log_sigma.exp().expand(B, self.num_slots, -1)
            slots = mu + sigma * torch.randn_like(mu)
        else:
            # Paper: "conditioning slot updates on previous-frame slots"
            # This is the key to temporal consistency!
            slots = prev_slots
        
        # Iterative slot attention
        for _ in range(self.num_iterations):
            slots_prev = slots
            slots_norm = self.norm_slots(slots)
            
            # Queries from slots
            q = self.to_q(slots_norm)  # [B, num_slots, D]
            
            # Attention scores: slots compete for features
            # [B, num_slots, num_patches]
            dots = torch.einsum('bsd,bnd->bsn', q, k) * self.scale
            
            # Softmax over SLOTS dimension (competition!)
            # Each patch goes to the most "compatible" slot
            attn = dots.softmax(dim=1) + self.eps  # [B, num_slots, N]
            attn = attn / attn.sum(dim=-1, keepdim=True)  # Normalize
            
            # Weighted sum of values
            updates = torch.einsum('bsn,bnd->bsd', attn, v)  # [B, num_slots, D]
            
            # GRU update
            slots = self.gru(
                updates.reshape(B * self.num_slots, self.slot_dim),
                slots_prev.reshape(B * self.num_slots, self.slot_dim)
            ).reshape(B, self.num_slots, self.slot_dim)
            
            # MLP refinement
            slots = slots + self.mlp(self.norm_pre_mlp(slots))
        
        return slots  # [B, num_slots, slot_dim]

"""
Object-Level Masking Module.

Paper Section 4.2: "Object-Level Masking for Latent Interventions"

Key equations:
    Masked token: z̃_i^τ = φ(z_i^{t_0}) + e_τ
    
Where:
    φ: learnable linear projection (identity projector)
    z_i^{t_0}: identity anchor (slot at earliest/latest observed time)
    e_τ: learnable temporal positional encoding

Paper: "By applying object-level masking that requires an object's state
to be inferred from other objects, C-JEPA induces latent interventions
with counterfactual-like effects and prevents shortcut solutions"
"""

import torch
import torch.nn as nn
import random
from typing import List, Tuple, Optional


class ObjectMaskingModule(nn.Module):
    """
    Implements object-level masking for C-JEPA training.
    
    Three cases (Paper Section 4.2):
    1. Identity anchor (t_0): NEVER masked - provides object identity
    2. History masked slots: φ(z_i^{t_0}) + e_τ  (t_0 = first history frame)
    3. Future slots: ALL masked with φ(z_i^t) + e_τ  (t = last history frame)
    
    Design rationale:
    - Object-level (not patch-level) → entire object masked
    - Structural masking → model MUST use interactions
    - Identity anchor → tells predictor WHICH object to predict
    - Temporal embedding → tells predictor WHEN to predict
    """
    
    def __init__(
        self,
        slot_dim: int = 128,
        max_timesteps: int = 20
    ):
        super().__init__()
        self.slot_dim = slot_dim
        
        # Identity projector φ
        # Paper: "φ is a learnable linear projection"
        # Projects identity anchor into mask token space
        # NOT a full pass-through - creates intermediate representation
        self.phi = nn.Linear(slot_dim, slot_dim)
        
        # Temporal positional encoding e_τ
        # Paper: "temporal positional encoding e_τ"
        # Learnable, one embedding per timestep
        # NOTE: NO object positional encoding!
        # Paper: "we omit positional encodings along the entity dimension"
        self.temporal_embedding = nn.Embedding(max_timesteps, slot_dim)
        
        # Initialize phi near identity
        nn.init.eye_(self.phi.weight)
        nn.init.zeros_(self.phi.bias)
    
    def create_history_mask_token(
        self,
        identity_anchor: torch.Tensor,
        time_idx: int
    ) -> torch.Tensor:
        """
        Create masked token for a history slot.
        
        Formula: z̃_i^τ = φ(z_i^{t_0}) + e_τ
        
        Args:
            identity_anchor: [B, slot_dim] - slot at earliest history time (t_0)
            time_idx: int - absolute time index for temporal embedding
        
        Returns:
            mask_token: [B, slot_dim]
        
        Paper: "The identity anchor is the slot at the earliest time step t_0,
        which is always observable, to distinguish which entities are masked"
        """
        B = identity_anchor.shape[0]
        device = identity_anchor.device
        
        # φ(z_i^{t_0}) - project identity anchor
        identity_feat = self.phi(identity_anchor)  # [B, D]
        
        # e_τ - temporal embedding
        time_tensor = torch.tensor(time_idx, device=device)
        temporal_feat = self.temporal_embedding(time_tensor)  # [D]
        temporal_feat = temporal_feat.unsqueeze(0).expand(B, -1)  # [B, D]
        
        return identity_feat + temporal_feat  # [B, D]
    
    def create_future_mask_token(
        self,
        last_observed_slot: torch.Tensor,
        time_idx: int
    ) -> torch.Tensor:
        """
        Create masked token for a future slot.
        
        Formula: z̃_i^τ = φ(z_i^t) + e_τ  (τ > t)
        
        Key difference from history masking:
        - Identity anchor = LAST observed frame (t), not first (t_0)
        - ALL objects masked (not just selected ones)
        
        Args:
            last_observed_slot: [B, slot_dim] - slot at last history time (t)
            time_idx: int - absolute time index
        
        Returns:
            mask_token: [B, slot_dim]
        """
        B = last_observed_slot.shape[0]
        device = last_observed_slot.device
        
        # φ(z_i^t) - project last observed slot
        identity_feat = self.phi(last_observed_slot)  # [B, D]
        
        # e_τ - temporal embedding
        time_tensor = torch.tensor(time_idx, device=device)
        temporal_feat = self.temporal_embedding(time_tensor)  # [D]
        temporal_feat = temporal_feat.unsqueeze(0).expand(B, -1)  # [B, D]
        
        return identity_feat + temporal_feat  # [B, D]
    
    def apply_masking(
        self,
        slots: torch.Tensor,
        mask_indices: List[int],
        T_h: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply object-level masking to full slot sequence.
        
        Timeline:
            t=0 (t-T_h+1): Identity anchor - NEVER masked
            t=1,...,T_h-1: Selected slots masked with history formula
            t=T_h,...,T:   ALL slots masked with future formula
        
        Args:
            slots: [B, T, N, D] - full slot sequence (history + future)
            mask_indices: List[int] - which slots to mask in history
            T_h: int - history window size
        
        Returns:
            masked_slots: [B, T, N, D] - slots with masking applied
            mask_map: [B, T, N] bool - True where masked
        """
        B, T, N, D = slots.shape
        masked_slots = slots.clone()
        mask_map = torch.zeros(B, T, N, dtype=torch.bool, device=slots.device)
        
        # === IDENTITY ANCHOR: t=0 (first history frame) ===
        # NEVER masked - provides object identity
        # Paper: "The identity anchor is always observable"
        identity_anchors = slots[:, 0, :, :]  # [B, N, D] - t=0 slots
        
        # === HISTORY MASKING: t=1,...,T_h-1 ===
        # Only selected objects (mask_indices) are masked
        for t in range(1, T_h):
            for slot_idx in mask_indices:
                # Get identity anchor for this object
                anchor = identity_anchors[:, slot_idx, :]  # [B, D]
                
                # Create masked token: φ(z_i^{t_0}) + e_τ
                mask_token = self.create_history_mask_token(
                    identity_anchor=anchor,
                    time_idx=t
                )
                
                masked_slots[:, t, slot_idx, :] = mask_token
                mask_map[:, t, slot_idx] = True
        
        # === FUTURE MASKING: t=T_h,...,T-1 ===
        # ALL objects masked
        # Identity anchor = LAST history frame (t=T_h-1)
        # Paper Figure 2 (right encoder): future frame is separate target
        last_history_slots = slots[:, T_h - 1, :, :]  # [B, N, D]
        
        for t in range(T_h, T):
            for slot_idx in range(N):
                
                # Special case: if this slot was also masked in history,
                # use the original t_0 anchor (not the masked value at T_h-1)
                if slot_idx in mask_indices:
                    anchor = identity_anchors[:, slot_idx, :]  # Original t_0
                else:
                    anchor = last_history_slots[:, slot_idx, :]  # Last observed
                
                # Create future masked token: φ(z_i^t) + e_τ
                mask_token = self.create_future_mask_token(
                    last_observed_slot=anchor,
                    time_idx=t
                )
                
                masked_slots[:, t, slot_idx, :] = mask_token
                mask_map[:, t, slot_idx] = True
        
        return masked_slots, mask_map
    
    @staticmethod
    def sample_mask_indices(
        num_slots: int,
        min_masked: int = 1,
        max_masked: int = 4
    ) -> List[int]:
        """
        Randomly sample which slots to mask.
        
        Paper: "M ~ Uniform({1,...,N})"
        (Figure 1 caption)
        
        Note: We skip slot 0 (typically background)
        and sample from remaining slots.
        """
        num_to_mask = random.randint(min_masked, max_masked)
        # Sample from slots 1 to N-1 (skip background at 0)
        available = list(range(1, num_slots))
        num_to_mask = min(num_to_mask, len(available))
        return random.sample(available, num_to_mask)


@FRAMEWORK_REGISTRY.register("VLA_JEPA")
class VLA_JEPA(baseframework):
    """
    Multimodal vision-language-action model.

    Components:
      - Qwen VL interface for fused language/vision token embeddings
      - DiT diffusion head for future action sequence modeling
      - JEPA world model for future frame prediction

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """
        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        embodied_action_token = self.config.framework.vj2_model.get("embodied_action_token", "<|embodied_action|>")
        action_tokens, self.action_token_ids, self.embodied_action_token_id = self.expand_tokenizer(
            tokenizer=self.qwen_vl_interface.processor.tokenizer,
            special_action_token=self.config.framework.vj2_model.special_action_token,
            max_action_tokens=self.config.framework.action_model.action_horizon * 4,
            embodied_action_token=embodied_action_token
        )



        # TODO speical tokens

        # align dims --> we should put them to config or no?
        # align dims --> we should put them to config or no?
        hidden_size = getattr(self.qwen_vl_interface.model.config, "hidden_size", getattr(self.qwen_vl_interface.model.config, "d_model", 2048))

        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = hidden_size

        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)  # 修复后续引用

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        
        self.vj_encoder = AutoModel.from_pretrained(self.config.framework.vj2_model.base_encoder, device_map="cuda")
        self.vj_processor = VJEPA2VideoProcessor.from_pretrained(self.config.framework.vj2_model.base_encoder)

        # Text Encoder (Gemma) - Moved up to get hidden size
        from starVLA.model.modules.sub_system.embedding_gemma import EmbeddingGemmaInterface, EmbeddingGemmaConfig
        text_encoder_cfg = self.config.framework.get("text_encoder", {})
        self.gemma_cfg = EmbeddingGemmaConfig(
            model_name=text_encoder_cfg.get("model_name", "google/embeddinggemma-300m"),
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.gemma_interface = EmbeddingGemmaInterface(self.gemma_cfg)

        tubelet_size = self.vj_encoder.config.tubelet_size
        
        # Calculate predictor output dimension (Visual + Text) * Views
        # Assuming 2 views based on hardcoded * 2
        visual_dim = self.vj_encoder.config.hidden_size
        text_dim = self.gemma_interface.hidden_size
        predictor_output_dim = (visual_dim + text_dim) * 2

        # SlotAttention: operates on view-concatenated features (visual_dim * 2)
        num_slots = self.config.framework.vj2_model.num_slots
        slot_dim = self.config.framework.vj2_model.slot_dim
        self.slot_attention = SlotAttention(
            num_slots=num_slots,
            slot_dim=slot_dim,
            feature_dim=visual_dim * 2,  # after view concatenation
            num_iterations=self.config.framework.vj2_model.num_iterations,
            eps=self.config.framework.vj2_model.eps
        )

        # Teacher SlotAttention: operates on Multimodal features (Visual + Text) * Views
        # Dimensions differ from student (Visual only), so we need a separate module.
        self.teacher_slot_attention = SlotAttention(
            num_slots=num_slots,
            slot_dim=slot_dim, # Can share slot_dim with student
            feature_dim=predictor_output_dim, # (Visual + Text) * 2
            num_iterations=self.config.framework.vj2_model.num_iterations,
            eps=self.config.framework.vj2_model.eps
        )

        # Teacher Slot Projector: slot_dim (256) -> predictor_output_dim (3072)
        # We need the teacher targets to match the predictor output dimension.
        self.teacher_slot_proj = nn.Linear(slot_dim, predictor_output_dim)

        # ---------------------------------------------------------------
        # Frozen C-JEPA integration  (galilai-group/cjepa)
        # Weights from HuggingFace: HazelNam/CJEPA
        # Requires a 'cjepa_frozen' section in the framework config.
        # ---------------------------------------------------------------
        cjepa_cfg = self.config.framework.get("cjepa_frozen", None)

        # ---------- Frozen slot masking (replaces trainable ObjectMaskingModule) ----------
        # Uses C-JEPA's pre-trained mask_token / time_pos_embed / id_projector
        # as the masking protocol.  Only the dimension adapters are trainable.
        self.object_masking = None          # disabled; kept as attr for compat
        self.cjepa_slot_masking = None
        if cjepa_cfg is not None:
            self.cjepa_slot_masking = build_cjepa_frozen_slot_masking(
                cfg=cjepa_cfg,
                student_slot_dim=slot_dim,
            )
            logger.info(
                f"[VLA_JEPA] Frozen C-JEPA slot masking loaded "
                f"(cjepa_slot_dim={cjepa_cfg.slot_dim}, num_slots={cjepa_cfg.num_slots})."
            )
        else:
            # Fallback: use the trainable ObjectMaskingModule if cjepa_frozen is absent
            self.object_masking = ObjectMaskingModule(
                slot_dim=slot_dim,
                max_timesteps=self.config.framework.vj2_model.num_frames // tubelet_size
            )

        # ---------- Freeze SlotAttention ----------
        # When using frozen C-JEPA masking the slot attention is also frozen
        # so the full slot→masking pipeline is fixed, and only the predictor
        # + downstream modules are trained.
        if cjepa_cfg is not None and cjepa_cfg.get("freeze_slot_attention", True):
            for p in self.slot_attention.parameters():
                p.requires_grad_(False)
            self.slot_attention.eval()
            logger.info("[VLA_JEPA] SlotAttention frozen.")

        # ---------- Frozen C-JEPA auxiliary predictor (optional distillation loss) ----------
        self.cjepa_frozen = None
        if cjepa_cfg is not None:
            self.cjepa_frozen = build_cjepa_frozen_world_model(
                cfg=cjepa_cfg,
                student_slot_dim=slot_dim,
            )
            logger.info(
                f"[VLA_JEPA] Frozen C-JEPA auxiliary predictor loaded "
                f"(num_slots={cjepa_cfg.num_slots}, slot_dim={cjepa_cfg.slot_dim})."
            )

        # Project slot_dim → predictor embed_dim if they differ
        predictor_embed_dim = slot_dim
        self.slot_proj = None
        if slot_dim != visual_dim * 2:
            self.slot_proj = nn.Linear(slot_dim, visual_dim * 2)
            predictor_embed_dim = visual_dim * 2

        # Predictor: img_size set so grid_H * grid_W = num_slots
        # This makes the predictor's internal T×(grid_H*grid_W) reshape
        # work with slot-count tokens per timestep instead of spatial patches.
        self.vj_predictor = VisionTransformerPredictorAC(
            num_frames=self.config.framework.vj2_model.num_frames // tubelet_size,
            img_size=(num_slots, 1),
            patch_size=1,
            tubelet_size=1,
            depth=self.config.framework.vj2_model.depth,
            num_heads=self.config.framework.vj2_model.num_heads,
            embed_dim=predictor_embed_dim,  # slot output dim (possibly projected)
            action_embed_dim=hidden_size,
            num_add_tokens=self.config.framework.vj2_model.num_action_tokens_per_timestep,
            output_dim=predictor_output_dim,  # Output Visual + Text
        )

        # Multimodal Projector for World Model Target (Teacher)
        self.teacher_encoder = MultiModalTargetEncoder(
            vjepa_path=self.vj_encoder,
            text_model=self.gemma_interface.model,
            num_fusion_layers=4,
            freeze_vjepa=True,
            freeze_text=True
        )

        self.replace_prompt = "".join(
            [each * self.config.framework.vj2_model.num_action_tokens_per_timestep for each in
             action_tokens[:self.config.framework.vj2_model.num_frames // tubelet_size - 1]]
        )

        self.embodied_replace_prompt = "".join([embodied_action_token * self.config.framework.vj2_model.num_embodied_action_tokens_per_instruction])
    def expand_tokenizer(self, 
                         tokenizer: AutoTokenizer,
                         special_action_token: str = "<|action_{}|>",
                         max_action_tokens: int = 32,
                         embodied_action_token: str = "<|embodied_action|>"):
        action_tokens, action_token_ids = [], []
        for i in range(0, max_action_tokens):
            action_token_i = special_action_token.format(i)
            action_tokens.append(action_token_i)
            if action_token_i not in tokenizer.get_vocab():
                added = tokenizer.add_tokens([action_token_i], special_tokens=True)
                if added == 0:
                    logger.warning(f"Warning: 0 tokens added (they may already exist) action_token_i: {action_token_i}.")
            action_token_id = tokenizer.convert_tokens_to_ids(action_token_i)    
            action_token_ids.append(action_token_id)
        
        if embodied_action_token not in tokenizer.get_vocab():
            added = tokenizer.add_tokens([embodied_action_token], special_tokens=True)
            if added == 0:
                logger.warning(f"Warning: 0 tokens added (they may already exist) embodied_action_token: {embodied_action_token}.")
        embodied_action_token_id = tokenizer.convert_tokens_to_ids(embodied_action_token)

        vla_embedding_size = self.qwen_vl_interface.model.get_input_embeddings().weight.size(0)
        if vla_embedding_size < len(tokenizer):
            # 2) resize embeddings of vla
            self.qwen_vl_interface.model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Model embedding size: {vla_embedding_size} ;tokenizer.vocab_size: {len(tokenizer)}")
        return action_tokens, action_token_ids, embodied_action_token_id

    # ------------------------------------------------------------------
    # Override train() to keep frozen modules in eval mode
    # ------------------------------------------------------------------
    def train(self, mode: bool = True):
        super().train(mode)
        # Keep frozen SlotAttention in eval mode
        if hasattr(self, "slot_attention") and not any(
            p.requires_grad for p in self.slot_attention.parameters()
        ):
            self.slot_attention.eval()
        # Keep frozen C-JEPA masking in eval mode
        if hasattr(self, "cjepa_slot_masking") and self.cjepa_slot_masking is not None:
            # id_projector is frozen; adapters stay in train mode
            self.cjepa_slot_masking.id_projector.eval()
        # Keep frozen C-JEPA auxiliary predictor in eval mode
        if hasattr(self, "cjepa_frozen") and self.cjepa_frozen is not None:
            self.cjepa_frozen.predictor.eval()
        return self

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """

        """
        batch_images = [example["image"] for example in examples]  # [B, [PIL.Image]]
        batch_videos = [example["video"] for example in examples]  #  [B, V, T, H, W, 3]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"]for example in examples] if "action" in examples[0] else None # label [B， len, 7]
        
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]

        """
        if self.action_model.device == torch.device("cuda:0") and "action" in examples[0]:
            print(batch_videos[0].shape) #[V, T, H, W, 3]
            print(instructions[0])
            print(actions[0].shape) # [T-1, action_dim]
            print(state[0].shape) if state is not None else print("No state") #[state_dim]
            print(len(batch_videos), len(instructions), len(actions), len(state) if state is not None else "No state")
            from diffusers.utils import export_to_video
            export_to_video(batch_videos[0][0]/255.0, "data_view_0.mp4")
            export_to_video(batch_videos[0][1]/255.0, "data_view_1.mp4")
            batch_images[0][0].save("data_image_view_0.png")
            batch_images[0][1].save("data_image_view_1.png")
            #print(self.action_tokens)
            print(self.replace_prompt)
            print(self.action_token_ids)
        elif self.action_model.device == torch.device("cuda:0") and "action" not in examples[0]:
            print(batch_videos[0].shape) #[V, T, H, W, 3]
            print(instructions[0])
            print(len(batch_videos), len(instructions))
            from diffusers.utils import export_to_video
            export_to_video(batch_videos[0][0]/255.0, "video_view_0.mp4")
            export_to_video(batch_videos[0][1]/255.0, "video_view_1.mp4")
            batch_images[0][0].save("video_image_view_0.png")
        exit()
        """
        
        

        #[print(each.shape, end=";") for each in batch_videos]
        batch_videos = np.stack(batch_videos)  #  [B, V, T, H, W, 3]
        batch_videos = batch_videos.transpose(0,1,2,5,3,4)  # [B, V, T, 3, H, W]

        # Step 1: QWenVL input format
        if actions is not None:
            qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
                images=batch_images, 
                instructions=instructions,
                prompt_replace_dict={"{actions}":self.replace_prompt, "{e_actions}":self.embodied_replace_prompt},
                prompt_template=self.config.datasets.vla_data.get("CoT_prompt", "")) 
        else:
            qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
                images=batch_images, 
                instructions=instructions,
                prompt_replace_dict={"{actions}":self.replace_prompt},
                prompt_template=self.config.datasets.video_data.get("CoT_prompt", ""))

        

        
        action_indices = torch.isin(qwen_inputs['input_ids'], torch.tensor(self.action_token_ids, device=qwen_inputs['input_ids'].device))
        action_indices = action_indices.nonzero(as_tuple=True)

        # TODO action condition tokens
        #embodied_action_indices = torch.isin(qwen_inputs['input_ids'], torch.tensor([self.embodied_action_token_id], device=qwen_inputs['input_ids'].device))
        embodied_action_indices = torch.isin(qwen_inputs['input_ids'], torch.tensor([self.embodied_action_token_id], device=qwen_inputs['input_ids'].device))
        embodied_action_indices = embodied_action_indices.nonzero(as_tuple=True)

        # Ensure all sub-modules are on CUDA (frozen modules may stay on CPU after accelerate.prepare)
        _dev = torch.device("cuda")
        for _mod_name in ["slot_attention", "teacher_slot_attention", "slot_proj",
                          "teacher_slot_proj", "action_model", "vj_predictor",
                          "cjepa_slot_masking", "cjepa_frozen"]:
            _mod = getattr(self, _mod_name, None)
            if _mod is not None:
                _mod.to(_dev)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]   # [B, L, H]
            B, _, H = last_hidden.shape
            action_tokens = last_hidden[action_indices[0], action_indices[1], :].view(B, -1, H)  # [B, action_len, H]
            embodied_action_tokens = last_hidden[embodied_action_indices[0], embodied_action_indices[1], :].view(B, -1, H)  # [B, action_len, H]
            #print(action_tokens.shape, last_hidden.shape, embodied_action_tokens.shape)
            #exit()
        
            # Step 2: JEPA Encoder (Student - Context)
            B, V, T, C, H_vid, W_vid = batch_videos.shape
            batch_videos = batch_videos.reshape(B * V, T, C, H_vid, W_vid)  # [B*V, T, C, H, W]
            input_videos = []
            for i in range(B*V):
                processed = self.vj_processor(
                    videos=batch_videos[i], return_tensors="pt"
                )["pixel_values_videos"].to(self.vj_encoder.device)
                if processed.dim() == 4:
                    processed = processed.unsqueeze(0)
                input_videos.append(processed)
            input_videos = torch.cat(input_videos, dim=0)  # [B*V, T, C, H, W]

            # Student visual features
            with torch.no_grad():
                video_embeddings = self.vj_encoder.get_vision_features(pixel_values_videos=input_videos)
                # video_embeddings is [B*V, seq_len, D] (3D, seq_len = T_latent * N_spatial)
                # Chunk across views and concat features → [B, seq_len, D*V]
                video_embeddings = torch.cat(torch.chunk(video_embeddings, chunks=V, dim=0), dim=-1)

            # Determine temporal parameters
            _, T_original, _, _, _ = input_videos.shape
            tubelet_size = self.vj_encoder.config.tubelet_size
            num_latents_temporal = T_original // tubelet_size
            num_tokens = video_embeddings.shape[1]
            tokens_per_latent = num_tokens // num_latents_temporal  # spatial tokens per timestep

            # Step 2b: Per-timestep Slot Attention
            # Reshape to [B, T_latent, N_spatial, D*V] for per-timestep processing
            video_4d = video_embeddings.view(B, num_latents_temporal, tokens_per_latent, -1)

            slots_per_t = []
            prev_slots = None
            for t_idx in range(num_latents_temporal):
                frame_feats = video_4d[:, t_idx, :, :]  # [B, N_spatial, D*V]
                slots_t = self.slot_attention(frame_feats, prev_slots)  # [B, num_slots, slot_dim]
                slots_per_t.append(slots_t)
                prev_slots = slots_t

            video_slots = torch.stack(slots_per_t, dim=1)  # [B, T_latent, num_slots, slot_dim]

            # -----------------------------------------------------------
            # Step 2b-aux: Frozen C-JEPA auxiliary prediction loss
            # The frozen predictor acts as a slot-level teacher for
            # distillation against the student's slot representations.
            # -----------------------------------------------------------
            cjepa_frozen_loss = None
            if self.cjepa_frozen is not None:
                # Clamp history length to what the frozen predictor was trained on
                _max_hist = self.cjepa_frozen.predictor.history_frames
                _T_h_cjepa = min(num_latents_temporal - 1, _max_hist)
                if _T_h_cjepa > 0:
                    # Use the last _T_h_cjepa frames so prediction target is the final slot
                    _hist_start = num_latents_temporal - 1 - _T_h_cjepa
                    history_slots = video_slots[:, _hist_start:_hist_start + _T_h_cjepa, :, :]
                    cjepa_pred_back, _ = self.cjepa_frozen(
                        history_slots, use_inference=True
                    )
                    _target_start = _hist_start + _T_h_cjepa
                    pred_frames_avail = min(
                        cjepa_pred_back.shape[1],
                        num_latents_temporal - _target_start,
                    )
                    if pred_frames_avail > 0:
                        target_slots = video_slots[
                            :, _target_start:_target_start + pred_frames_avail, :, :
                        ].detach()
                        cjepa_frozen_loss = F.mse_loss(
                            cjepa_pred_back[:, :pred_frames_avail],
                            target_slots,
                        )

            # Step 2c: Object-level masking
            # When cjepa_frozen is active, use frozen C-JEPA masking protocol
            # (mask_token + time_pos_embed + id_projector from HazelNam/CJEPA).
            # Otherwise fall back to the trainable ObjectMaskingModule.
            T_h = num_latents_temporal - 1  # history window = all but last
            if self.cjepa_slot_masking is not None:
                masked_slots, _mask_idx = self.cjepa_slot_masking.apply_masking(
                    video_slots, T_h
                )
                # masked_slots: (B, T_total, S, slot_dim)  where T_total = T_h + pred_frames
                # Trim or pad to match num_latents_temporal if needed
                if masked_slots.shape[1] != num_latents_temporal:
                    masked_slots = masked_slots[:, :num_latents_temporal, :, :]
            else:
                mask_indices = ObjectMaskingModule.sample_mask_indices(
                    self.slot_attention.num_slots
                )
                masked_slots, _mask_map = self.object_masking.apply_masking(
                    video_slots, mask_indices, T_h
                )

            # Project slots if slot_dim != predictor embed_dim
            if self.slot_proj is not None:
                masked_slots = self.slot_proj(masked_slots)  # [B, T, num_slots, embed_dim]

            # Flatten to [B, T_latent * num_slots, embed_dim] for predictor
            num_slots = self.slot_attention.num_slots
            video_embeddings = masked_slots.view(B, num_latents_temporal * num_slots, -1)

            # Student Context: first (num_latents - 1) latent steps
            input_states = video_embeddings[:, :num_slots * (num_latents_temporal - 1), :]

            # Step 2d: Get Teacher Targets using AdapterX
            with torch.no_grad():
                teacher_multimodal_features = self.teacher_encoder(
                    images=input_videos,  # [B*V, T, C, H, W]
                    action_descriptions=instructions * V  # Repeat instructions for each view
                )
                # AdapterX output: [B*V, T, N, D] (4D tensor) or [B*V, T*N, D] (3D)
                if teacher_multimodal_features.ndim == 4:
                    B_V_t, T_t, N_t, D_t = teacher_multimodal_features.shape
                else:
                    # 3D case: reshape to 4D using known teacher spatial dim
                    B_V_t = teacher_multimodal_features.shape[0]
                    D_t = teacher_multimodal_features.shape[-1]
                    total_tokens = teacher_multimodal_features.shape[1]
                    T_t = num_latents_temporal
                    N_t = total_tokens // T_t
                    teacher_multimodal_features = teacher_multimodal_features.view(B_V_t, T_t, N_t, D_t)

                # Concatenate views: [B, V, T, N, D] → [B, T, N, D*V]
                teacher_flat = teacher_multimodal_features.view(B, V, T_t, N_t, D_t)
                teacher_flat = torch.cat([teacher_flat[:, i, :, :, :] for i in range(V)], dim=-1)

                # Teacher is also dense spatial tokens (N_t). We need to convert them to slots
                # to match the student's prediction target.
                # Apply same SlotAttention to teacher features
                teacher_4d = teacher_flat.view(B, T_t, N_t, -1)
                teacher_slots_per_t = []
                teacher_prev_slots = None
                for t_idx in range(T_t):
                    t_frame_feats = teacher_4d[:, t_idx, :, :]
                    # Shared SlotAttention? Or should it be separate?
                    # For now, using shared to force alignment in same slot space.
                    t_slots = self.teacher_slot_attention(t_frame_feats, teacher_prev_slots)
                    teacher_slots_per_t.append(t_slots)
                    teacher_prev_slots = t_slots
                
                teacher_slots = torch.stack(teacher_slots_per_t, dim=1) # [B, T, num_slots, D]
                
                # Project teacher slots to match predictor output dimension (Visual + Text) * V
                teacher_slots = self.teacher_slot_proj(teacher_slots) # [B, T, 3072]

                # Flatten T and Slot
                teacher_flat = teacher_slots.view(B, T_t * num_slots, -1)

                # Slice target (last latent step)
                tokens_per_teacher_latent = num_slots  # Now it matches!
                gt_multimodal = teacher_flat[:, tokens_per_teacher_latent * (T_t - 1):, :]

            # Step 3: VJ Predictor
            predicted_states = self.vj_predictor(
                input_states,
                action_tokens
            )
            # Slice predicted states to match target (last latent step)
            predicted_states = predicted_states[:, -num_slots:, :]

            teacher_forcing_wm_loss = F.l1_loss(
                predicted_states,
                gt_multimodal,
                reduction="mean"
            )
        
        if "action" not in examples[0]:
            _cjepa_loss_scale_v = self.config.framework.get("cjepa_frozen", {}).get("loss_scale", 0.05) if self.config else 0.05
            losses = {"wm_loss": teacher_forcing_wm_loss}
            if cjepa_frozen_loss is not None:
                losses["cjepa_frozen_loss"] = cjepa_frozen_loss * _cjepa_loss_scale_v
            return losses

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            # 标签对齐：取最后 chunk_len 段
            actions = torch.tensor(
                np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype
            )  # [B, T_full, action_dim]
            actions_target = actions[:, -(self.future_action_window_size+1):, :]  # (B, chunk_len, action_dim)

            repeated_diffusion_steps = (
                self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            embodied_action_repeated = embodied_action_tokens.repeat(repeated_diffusion_steps, 1, 1)
            
            state_repeated = None
            if state is not None:
                state = torch.tensor(
                    np.array(state), device=last_hidden.device, dtype=last_hidden.dtype
                )
                #print(state.shape)
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            #print(embodied_action_repeated.shape, actions_target_repeated.shape, state_repeated.shape) if state_repeated is not None else print("No state for action model")
            #exit()
            action_loss = self.action_model(embodied_action_repeated, actions_target_repeated, state_repeated)  # (B, chunk_len, action_dim)

        _cjepa_loss_scale = self.config.framework.get("cjepa_frozen", {}).get("loss_scale", 0.05) if self.config else 0.05
        losses = {"action_loss": action_loss, "wm_loss": teacher_forcing_wm_loss * 0.1}
        if cjepa_frozen_loss is not None:
            losses["cjepa_frozen_loss"] = cjepa_frozen_loss * _cjepa_loss_scale
        return losses

    @torch.inference_mode()
    def predict_action(
        self,
        batch_images: List[List[Image.Image]],  # Batch of PIL Image list as [view1, view2]
        instructions: List[str],
        state: Optional[np.ndarray] = None,
        **kwargs: str,
    ) -> np.ndarray:
        """
        推理：单次前向直接回归未来动作（无扩散采样）。

        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory

        Args:
            batch_images: List of samples; each sample is List[PIL.Image] (multi-view).
            instructions: List[str] natural language task instructions.
            cfg_scale: >1 enables classifier-free guidance (scales conditional vs unconditional).
            use_ddim: Whether to use DDIM deterministic sampling.
            num_ddim_steps: Number of DDIM steps if enabled.
            **kwargs: Reserved.

        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)
    
        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, 
            instructions=instructions,
            prompt_replace_dict={"{actions}":self.replace_prompt, "{e_actions}":self.embodied_replace_prompt})
        
        embodied_action_indices = torch.isin(qwen_inputs['input_ids'], torch.tensor([self.embodied_action_token_id], device=qwen_inputs['input_ids'].device))
        #embodied_action_indices = ~torch.isin(qwen_inputs['input_ids'], torch.tensor(self.action_token_ids, device=qwen_inputs['input_ids'].device))
        embodied_action_indices = embodied_action_indices.nonzero(as_tuple=True)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]   # [B, L, H]
            B, _, H = last_hidden.shape
            embodied_action_tokens = last_hidden[embodied_action_indices[0], embodied_action_indices[1], :].view(B, -1, H)

        state = torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype) if state is not None else None
        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(embodied_action_tokens, state)  # (B, chunk_len, action_dim)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions, "embodied_action_tokens": embodied_action_tokens.to(dtype=torch.float32).detach().cpu().numpy()}



if __name__ == "__main__":
    from omegaconf import OmegaConf
    import debugpy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./starVLA/config/training/starvla_cotrain_oxe.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()
    
    # debugpy.listen(("localhost", 10092))
    # print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    # debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    # try get model
    cfg.framework.qwenvl.base_vlm = "Qwen/Qwen2.5-VL-3B-Instruct" 
     
    model: VLA_JEPA = VLA_JEPA(cfg)
    print(model)



    # fake sample 
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    # Create a sample
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16), # action_chunk, action_dim
        "image": [image, image], # two views
        "video": np.random.randint(0, 255, (2, 8, 224, 224, 3), dtype=np.uint8), # [V, T, H, W, 3]
        "lang": "This is a fake for testing.",
        "state" : np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16), # chunk, state_dim
    }

    batch  = [sample]  # batch size 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device) # Can't move whole model due to Qwen/Accelerate usage
    # Move scratch-training components if needed
    for module in [model.vj_predictor, model.action_model]:
        try:
            module.to(device)
        except NotImplementedError:
            print(f"Moving {type(module).__name__} from meta device using to_empty (weights uninitialized!)")
            module.to_empty(device=device)
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
            elif hasattr(module, 'apply'):
                # Try to apply reset logic
                def init_weights(m):
                    if hasattr(m, 'reset_parameters'):
                        m.reset_parameters()
                    elif hasattr(m, 'weight') and m.weight.dim() > 1:
                        nn.init.xavier_uniform_(m.weight)
                    elif hasattr(m, 'bias') and m.bias is not None:
                        nn.init.zeros_(m.bias)
                module.apply(init_weights)

    # teacher_encoder (AdapterX) uses accelerate-wrapped models (vj_encoder, gemma), 
    # so calling .to() on it directly is unsafe/redundant if they are already on device.
    # It has no trainable weights of its own (frozen fusion removed).
    
    # gemma_interface also handles its own device placement (usually).
    # ensure it's on device if it has trainable parameters not managed by accelerate.
    try:
        model.gemma_interface.to(device)
    except:
        pass # Gemma likely on device already

    # vj_encoder and qwen_vl_interface are managed by accelerate/device_map="auto"
    forward_output = model(batch)
    action_loss = forward_output['action_loss']
    print(f"Action Loss: {action_loss.item()}")

    # test predict action
    predict_output = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]], state=[batch[0]["state"]])
    normalized_actions = predict_output['normalized_actions']
    print(f"Unnormalized Action: {normalized_actions}")

    # # Advance: try forward model with dataloader
    # # can be fake sample， but here get from dataloader for simpler
    # from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn

    # vla_dataset_cfg = cfg.datasets.vla_data
    # dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

    # from torch.utils.data import DataLoader

    # train_dataloader = DataLoader(
    #     dataset,
    #     batch_size=2,
    #     num_workers=1,  # For Debug
    #     collate_fn=collate_fn,
    # )
    # # 
    # for batch in tqdm(train_dataloader, desc="Processing Batches"):
    #     batch
    #     break

    # # try get model
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    # model(batch)

    # action = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]])

    # # fake state
    # for ba in batch:
    #     ba["state"] = ba["action"][0][None]

    # model(batch)
    # action = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]], state=[batch[0]["state"]])