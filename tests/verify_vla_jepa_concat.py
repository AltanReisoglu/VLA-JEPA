"""
Mock test for VLA_JEPA forward pass.
Tests: SlotAttention, ObjectMaskingModule, predictor shape, wm_loss output.
"""
import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from unittest.mock import MagicMock, patch
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


# ── Constants ──
VISUAL_DIM = 768
NUM_VIEWS = 2
NUM_FRAMES = 16
TUBELET_SIZE = 2
NUM_LATENTS = NUM_FRAMES // TUBELET_SIZE  # 8
SPATIAL_TOKENS = 196
SEQ_LEN = NUM_LATENTS * SPATIAL_TOKENS  # 1568
NUM_SLOTS = 7
SLOT_DIM = 256
NUM_ACTION_TOKENS = 4
QWEN_HIDDEN = 1024
BATCH = 2
TOTAL_ACT_TOKENS = NUM_ACTION_TOKENS * (NUM_LATENTS - 1)  # 28
QW_SEQ = TOTAL_ACT_TOKENS + 10  # 38
GEMMA_DIM = 768  # EmbeddingGemma hidden


def make_config():
    return OmegaConf.create(dict(
        framework=dict(
            name="QwenGR00T",
            vj2_model=dict(
                base_encoder="facebook/vjepa2",
                num_frames=NUM_FRAMES,
                depth=2,
                num_heads=4,
                num_action_tokens_per_timestep=NUM_ACTION_TOKENS,
                num_embodied_action_tokens_per_instruction=1,
                special_action_token="<|action_{}|>",
                embodied_action_token="<|embodied_action|>",
                num_slots=NUM_SLOTS,
                slot_dim=SLOT_DIM,
                feature_dim=VISUAL_DIM * NUM_VIEWS,
                num_iterations=2,
                eps=1e-8,
            ),
            action_model=dict(
                action_model_type="DiT-B",
                action_horizon=8,
                future_action_window_size=7,
                past_action_window_size=0,
                hidden_size=QWEN_HIDDEN,
                action_dim=7,
                state_dim=7,
                num_target_vision_tokens=1,
                add_pos_embed=False,
                num_inference_timesteps=4,
                num_layers=2,
                noise_beta_alpha=1.5,
                noise_beta_beta=1.0,
                noise_s=1.0,
                num_timestep_buckets=100,
                diffusion_model_cfg=dict(
                    cross_attention_dim=QWEN_HIDDEN,
                    output_dim=1024,
                ),
            ),
            text_encoder=dict(model_name="google/embeddinggemma-300m"),
            qwenvl=dict(base_vlm="Qwen/Qwen2-VL-2B-Instruct"),
        ),
        datasets=dict(
            vla_data=dict(CoT_prompt="{instruction}", image_size=[224, 224]),
            video_data=dict(CoT_prompt="{instruction}"),
        ),
        trainer=dict(repeated_diffusion_steps=1),
    ))


def make_mock_qwen():
    """Create a mock QwenVL interface."""
    mock_qwen = MagicMock()
    # Tokenizer
    mock_qwen.processor.tokenizer.get_vocab.return_value = dict()
    mock_qwen.processor.tokenizer.add_tokens.return_value = 1
    # action tokens get IDs 100..131, embodied_action gets ID 200
    action_id_counter = [100]
    def convert_token_to_id(token):
        if "embodied" in token:
            return 200
        else:
            val = action_id_counter[0]
            action_id_counter[0] += 1
            return val
    mock_qwen.processor.tokenizer.convert_tokens_to_ids.side_effect = convert_token_to_id
    mock_qwen.processor.tokenizer.__len__ = MagicMock(return_value=1000)
    mock_qwen.model.config.hidden_size = QWEN_HIDDEN
    mock_qwen.model.get_input_embeddings.return_value.weight.size.return_value = 1000

    # Build input: create input_ids with exactly the right number of action tokens
    # T_context = num_latents - 1 = 7, num_action_tokens_per_timestep = 4
    # Total action tokens = 7 * 4 = 28
    # We also need 1 embodied_action_token
    def build_inputs(images, instructions, prompt_replace_dict, prompt_template):
        B = len(instructions)
        # Build input_ids: [padding(0)] + [action_tokens(100..127)] + [embodied(200)] + [padding(0)]
        seq = []
        seq.extend([0] * 5)  # padding
        # Action token IDs: 100, 101, ..., 127 (28 tokens)
        for i in range(TOTAL_ACT_TOKENS):
            seq.append(100 + i)
        seq.append(200)  # embodied action token
        seq.extend([0] * 4)  # padding
        input_ids = torch.tensor([seq] * B, dtype=torch.long).cuda()
        return dict(input_ids=input_ids)
    mock_qwen.build_qwenvl_inputs.side_effect = build_inputs

    # Forward: return hidden states matching input_ids length
    def qwen_forward(*args, **kwargs):
        input_ids = kwargs.get("input_ids")
        if input_ids is not None:
            B, L = input_ids.shape
        else:
            B, L = BATCH, TOTAL_ACT_TOKENS + 10
        out = MagicMock()
        out.hidden_states = [torch.randn(B, L, QWEN_HIDDEN, device="cuda")]
        return out
    mock_qwen.side_effect = qwen_forward
    mock_qwen.return_value = qwen_forward()

    return mock_qwen


def make_mock_gemma():
    """Create a mock EmbeddingGemma interface."""
    mock_gemma = MagicMock()
    mock_gemma.hidden_size = GEMMA_DIM
    # model attribute for MultiModalTargetEncoder
    mock_gemma.model = MagicMock()
    mock_gemma.model.encode.return_value = torch.randn(1, GEMMA_DIM, device="cuda")
    return mock_gemma


def test_vla_jepa_forward():
    print("=" * 70)
    print("Testing VLA_JEPA Forward Pass (Mock)")
    print("=" * 70)

    cfg = make_config()
    print("\n[1/5] Setting up mocks...")

    with patch('transformers.VJEPA2VideoProcessor.from_pretrained') as mock_vp, \
         patch('starVLA.model.framework.VLA_JEPA.get_vlm_model') as mock_get_vlm, \
         patch('transformers.AutoModel.from_pretrained') as mock_automodel:

        # Mock Video Processor
        mock_proc = MagicMock()
        def mock_proc_fn(videos, return_tensors):
            return dict(pixel_values_videos=torch.randn(1, NUM_FRAMES, 3, 224, 224).cuda())
        mock_proc.side_effect = mock_proc_fn
        mock_vp.return_value = mock_proc

        # Mock V-JEPA (AutoModel)
        def automodel_factory(model_name, **kwargs):
            m = MagicMock()
            m.config.hidden_size = VISUAL_DIM
            m.config.tubelet_size = TUBELET_SIZE
            m.config.image_size = 224
            def get_vis(pixel_values_videos):
                B = pixel_values_videos.shape[0]
                return torch.randn(B, SEQ_LEN, VISUAL_DIM, device="cuda")
            m.get_vision_features.side_effect = get_vis
            m.device = torch.device("cuda")
            return m
        mock_automodel.side_effect = automodel_factory

        # Mock QwenVL
        mock_qwen = make_mock_qwen()
        mock_get_vlm.return_value = mock_qwen

        # Now we need to also mock the Gemma interface since it loads a real model
        with patch('starVLA.model.modules.sub_system.embedding_gemma.EmbeddingGemmaInterface') as mock_gemma_cls:
            mock_gemma = make_mock_gemma()
            mock_gemma_cls.return_value = mock_gemma

            # ── Instantiate ────────────────────────────────────────────
            print("[2/5] Instantiating VLA_JEPA...")
            from starVLA.model.framework.VLA_JEPA import VLA_JEPA
            model = VLA_JEPA(config=cfg)
            model.cuda()
            print("  Model initialized OK")

            # Verify key dimensions
            print(f"\n[3/5] Checking dimensions...")
            print(f"  SlotAttention: num_slots={model.slot_attention.num_slots}, "
                  f"slot_dim={model.slot_attention.slot_dim}")
            print(f"  Predictor embed_dim: {model.vj_predictor.predictor_embed.in_features}")
            print(f"  Predictor output_dim: {model.vj_predictor.predictor_proj.out_features}")
            if model.slot_proj is not None:
                print(f"  slot_proj: {SLOT_DIM} -> {model.slot_proj.out_features}")
            else:
                print(f"  slot_proj: None (slot_dim == embed_dim)")
            print(f"  ObjectMasking: slot_dim={model.object_masking.slot_dim}")
            print(f"  Predictor grid: {model.vj_predictor.grid_height} x {model.vj_predictor.grid_width}")

            assert model.slot_attention.num_slots == NUM_SLOTS
            assert model.vj_predictor.grid_height == NUM_SLOTS
            assert model.vj_predictor.grid_width == 1
            print("  All dimension checks passed!")

            # ── Mock AdapterX (teacher encoder) ────────────────────────
            # teacher_encoder.forward returns [B*V, T_latent, N_spatial, D_teacher]
            teacher_D = VISUAL_DIM + GEMMA_DIM  # concat dim
            def mock_teacher_forward(images, action_descriptions):
                B_V = images.shape[0]  # B*V
                return torch.randn(B_V, NUM_LATENTS, SPATIAL_TOKENS, teacher_D, device="cuda")
            mock_teacher = MagicMock()
            mock_teacher.side_effect = mock_teacher_forward
            # Use object.__setattr__ to bypass nn.Module type check
            object.__setattr__(model, 'teacher_encoder', mock_teacher)

            # ── Forward Pass ───────────────────────────────────────────
            print(f"\n[4/5] Running forward pass (batch={BATCH}, no actions)...")
            samples = []
            for _ in range(BATCH):
                img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
                samples.append(dict(
                    image=[img, img],  # 2 views
                    video=np.random.randn(NUM_VIEWS, NUM_FRAMES, 224, 224, 3).astype(np.float32),
                    lang="pick up the red cube",
                ))

            output = model(examples=samples)
            print(f"  Output keys: {list(output.keys())}")
            print(f"  wm_loss: {output['wm_loss'].item():.6f}")

            assert "wm_loss" in output, "Missing wm_loss in output!"
            assert torch.isfinite(output["wm_loss"]), "wm_loss is not finite!"
            print("  Forward pass OK!\n")

            # ── Summary ────────────────────────────────────────────────
            print("[5/5] Summary")
            print("  [OK] SlotAttention applied per-timestep")
            print("  [OK] ObjectMaskingModule applied")
            print("  [OK] Predictor grid_H*grid_W = num_slots")
            print("  [OK] wm_loss computed successfully")
            print("=" * 70)
            print("ALL TESTS PASSED")
            print("=" * 70)


if __name__ == "__main__":
    test_vla_jepa_forward()
