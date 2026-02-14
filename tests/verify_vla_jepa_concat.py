
import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from unittest.mock import MagicMock
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from starVLA.model.framework.VLA_JEPA import VLA_JEPA

def test_vla_jepa_concat():
    print("Testing VLA_JEPA with concatenated AdapterX target...")

    # Mock Config
    cfg = OmegaConf.create({
        "framework": {
            "name": "QwenGR00T",
            "vj2_model": {
                "base_encoder": "google/vit-base-patch16-224", # Placeholder
                "num_frames": 16, # Match tubelet logic somewhat
                "depth": 2, # Small depth for speed
                "num_heads": 4,
                "num_action_tokens_per_timestep": 1,
                "num_embodied_action_tokens_per_instruction": 1,
                "special_action_token": "<|action_{}|>",
                "embodeid_action_token": "<|embodied_action|>"
            },
            "action_model": {
                "action_model_type": "DiT-B", # Added missing key
                "action_horizon": 8,
                "future_action_window_size": 7,
                "past_action_window_size": 0,
                "hidden_size": 1024, # Added missing key
                "action_dim": 7,
                "state_dim": 7,
                "num_target_vision_tokens": 1, # Added missing key
                "add_pos_embed": False, # Added missing key
                "num_inference_timesteps": 4, # Moved to top level
                "num_layers": 2, # Moved to top level
                "noise_beta_alpha": 1.5, # Added
                "noise_beta_beta": 1.0, # Added
                "noise_s": 1.0, # Added
                "num_timestep_buckets": 100, # Added
                "diffusion_model_cfg": {
                    "cross_attention_dim": 2048, # Qwen hidden size
                    "output_dim": 1024,
                    # "num_layers": 2, # Moved up
                    # "num_inference_timesteps": 4 # Moved up
                }
            },
            "text_encoder": {
                "model_name": "google/embeddinggemma-300m"
            },
            "qwenvl": {
                "base_vlm": "Qwen/Qwen2-VL-2B-Instruct", # Use valid small model if needed, or mock
                 # Actually VLA_JEPA init calls get_vlm_model which loads Qwen.
                 # Loading Qwen might be slow/heavy.
                 # Maybe we can mock get_vlm_model?
            }
        },
        "datasets": {
            "vla_data": {
                "CoT_prompt": "{instruction}",
                "image_size": [224, 224]
            },
             "video_data": {
                "CoT_prompt": "{instruction}"
            }
        },
        "trainer": {
           "repeated_diffusion_steps": 1
        }
    })
    
    # We need to mock get_vlm_model to avoid loading 2B model if possible, 
    # but VLA_JEPA uses self.qwen_vl_interface tightly.
    # Let's try to run it. If it OOMs or takes too long, we mock.
    # For now, let's assume we can load it or maybe use a tiny model?
    # Or strict mocking.
    
    # Mocking starVLA.model.modules.vlm.get_vlm_model
    # We need to inject this mock BEFORE importing VLA_JEPA fully or instantiation.
    # But we already imported.
    
    # Let's use a trick to mock the interface attribute AFTER init? No, init calls it.
    # Let's mock the keys in sys.modules or patch.
    
    print("Mocking QwenVL interface...")
    with torch.device("meta"):
         # This is risky. 
         pass

    # Actually, let's just try to instantiate. If base_vlm path is invalid it will fail.
    # Provide a placeholder Qwen path? "Qwen/Qwen2-VL-2B-Instruct" is valid but large.
    # Assuming user has decent GPU. 
    # If not, we should probably mock the Qwen part entirely.
    
    # For this verification, we care about VJEPA + AdapterX + Predictor.
    # The Action Model and Qwen are less relevant for the "concatenation" change check,
    # except that VLA_JEPA init involves them.
    
    # Let's construct VLA_JEPA but mock the heavy parts if possible.
    # But VLA_JEPA is a class.
    
    # We will assume the script runs on a machine that can handle it or we use "cpu".
    # But "google/vit-base-patch16-224" is small.
    # "google/embeddinggemma-300m" is small.
    # "Qwen/Qwen2-VL-2B-Instruct" is ~4GB.
    
    # Let's try with dummy path for Qwen and mock the loader.
    cfg.framework.qwenvl.base_vlm = "dummy/path"
    
    # Mocking get_vlm_model
    from unittest.mock import patch
    
    with patch('transformers.AutoVideoProcessor.from_pretrained') as mock_avp, \
         patch('starVLA.model.framework.VLA_JEPA.get_vlm_model') as mock_get_vlm, \
         patch('transformers.AutoModel.from_pretrained') as mock_automodel:
        
        # Setup mock Video Processor
        mock_processor = MagicMock()
        def mock_process_video(videos, return_tensors):
             return {"pixel_values_videos": torch.randn(1, 16, 3, 224, 224).cuda()} 
        mock_processor.side_effect = mock_process_video
        mock_avp.return_value = mock_processor

        # Setup mock AutoModel
        # We need to handle V-JEPA and Gemma differently if possible, or just mock all.
        # Gemma interface uses SentenceTransformer usually, but here EmbeddingGemmaInterface uses:
        # self.model = SentenceTransformer(model_name)
        # Wait, VLA_JEPA line 97 imports and instantiates EmbeddingGemmaInterface.
        # EmbeddingGemmaInterface.__init__ calls SentenceTransformer or AutoModel?
        # Let's assume it calls whatever.
        # But VLA_JEPA calls AutoModel.from_pretrained(base_encoder) line 82.
        
        def automodel_side_effect(model_name, **kwargs):
            if "vit-base" in model_name or "vjepa" in model_name:
                # Mock V-JEPA
                mock_vj = MagicMock()
                mock_vj.config.hidden_size = 768
                mock_vj.config.tubelet_size = 1
                mock_vj.config.image_size = 224
                # Mock get_vision_features
                # Mock get_vision_features
                # Input: [B, T, C, H, W] -> Output [B, T*196, 768]
                # Input: [B, C, H, W] -> Output [B, 196, 768]
                def get_vis_feat(pixel_values_videos):
                    if pixel_values_videos.ndim == 5:
                        B, T, C, H, W = pixel_values_videos.shape
                        return torch.randn(B, T*196, 768).cuda()
                    else:
                         B = pixel_values_videos.shape[0]
                         return torch.randn(B, 196, 768).cuda()
                mock_vj.get_vision_features.side_effect = get_vis_feat
                mock_vj.device = torch.device("cuda")
                return mock_vj
            else:
                # For others (e.g. Gemma used by SentenceTransformer)
                m = MagicMock()
                m.config.hidden_size = 768
                
                # Mock forward to return tensors
                def mock_forward(*args, **kwargs):
                    # SentenceTransformer might pass args or kwargs
                    # We try to guess shape or default to [1, 1, 768]
                    # input_ids usually in kwargs['input_ids']
                    input_ids = kwargs.get('input_ids')
                    if input_ids is None and len(args) > 0 and isinstance(args[0], dict):
                         input_ids = args[0].get('input_ids')
                    
                    if input_ids is not None:
                        B, L = input_ids.shape
                        last_hidden = torch.randn(B, L, 768).cuda()
                        pooler = torch.randn(B, 768).cuda()
                    else:
                        # Fallback
                        last_hidden = torch.randn(1, 4, 768).cuda()
                        pooler = torch.randn(1, 768).cuda()
                    
                    # Create tuple-like object that also access by key
                    class MockOutput(dict):
                        def __getitem__(self, key):
                            if key == 0: return self["last_hidden_state"]
                            if key == 1: return self["pooler_output"]
                            return super().__getitem__(key)
                    
                    out = MockOutput()
                    out["last_hidden_state"] = last_hidden
                    out["pooler_output"] = pooler
                    out["token_embeddings"] = last_hidden
                    return out
                
                m.forward.side_effect = mock_forward
                m.side_effect = mock_forward # call on instance
                return m
        
        mock_automodel.side_effect = automodel_side_effect

        # Check output_dim calculation
        # Visual(768) + Text(2048 or 768?)
        # EmbeddingGemma - 300m -> 768 or similar?
        # script output said "hidden_size=768" for Gemma.
        # So 768 + 768 = 1536. 
        # Predictor out = 1536 * 2 = 3072.

        # Setup mock Qwen interface
        mock_qwen = MagicMock()
        mock_qwen.processor.tokenizer.get_vocab.return_value = {"<|endoftext|>": 0}
        mock_qwen.processor.tokenizer.add_tokens.return_value = 1
        mock_qwen.processor.tokenizer.convert_tokens_to_ids.return_value = 1
        mock_qwen.processor.tokenizer.__len__.return_value = 1000
        mock_qwen.model.config.hidden_size = 1024
        mock_qwen.model.get_input_embeddings().weight.size.return_value = 1000
        
        # We need input_ids to len > 16 to contain enough action tokens (id=1)
        # T=16. Predictor likely needs 15 or 16 actions.
        # Let's provide 30 tokens, all 1s.
        mock_qwen.build_qwenvl_inputs.return_value = {'input_ids': torch.ones(1, 30, dtype=torch.long).cuda()}
        
        # Mock forward return
        mock_output = MagicMock()
        mock_output.hidden_states = [torch.randn(1, 30, 1024).cuda()]
        mock_qwen.return_value = mock_output
        
        mock_get_vlm.return_value = mock_qwen
        
        # Instantiate model
        print("Instantiating VLA_JEPA...")
        model = VLA_JEPA(config=cfg)
        model.cuda()
        
        print("Model initialized.")
        print(f"Predictor Embed Dim: {model.vj_predictor.predictor_embed.in_features}") # Should be 768*2 = 1536
        print(f"Predictor Output Dim: {model.vj_predictor.predictor_proj.out_features}") # Should be (768+gemma_dim)*2
        
        gemma_dim = model.gemma_interface.hidden_size
        visual_dim = model.vj_encoder.config.hidden_size
        expected_out_dim = (visual_dim + gemma_dim) * 2
        print(f"Expected Output Dim: {expected_out_dim}")
        
        assert model.vj_predictor.predictor_proj.out_features == expected_out_dim, "Predictor output dim mismatch!"
        
        # Run Dummy Forward
        print("Running dummy forward pass...")
        image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        sample = {
            "image": [image, image], # 2 views
            "video": np.random.randn(2, 16, 224, 224, 3).astype(np.float32), # [V, T, H, W, C] but numpy
            "lang": "fake instruction",
            # "action": ... (optional, let's skip to test wm_loss mainly)
        }
        
        # Provide config for video?
        # VLA_JEPA forward expects examples
        # batch_videos: [example["video"]] -> stack -> [B, V, T, H, W, 3]
        # Then transpose to [B, V, T, 3, H, W]
        
        batch = [sample]
        
        # To avoid detailed Qwen/Action processing, we can return early or mock more?
        # But we want to test VJEPA part which is Step 2 & 3.
        # Step 4 is Action.
        # If we don't provide "action" in sample, it returns {"wm_loss": ...} (Line 316)
        # Perfect.
        
        output = model(examples=batch)
        print("Forward pass successful.")
        print(f"Output keys: {output.keys()}")
        print(f"WM Loss: {output['wm_loss']}")
        
        assert "wm_loss" in output
        print("\nIntegration Verified!")

if __name__ == "__main__":
    test_vla_jepa_concat()
