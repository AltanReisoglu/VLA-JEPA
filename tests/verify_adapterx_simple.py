
import torch
import sys
import os

# Add project root to sys.path
sys.path.append("c:\\Users\\bahaa\\Desktop\\VLA-JEPA")

from starVLA.model.modules.sub_system.embedding_gemma import EmbeddingGemmaInterface, EmbeddingGemmaConfig
from starVLA.model.framework.AdapterX import MultiModalTargetEncoder
from transformers import AutoModel, AutoTokenizer

class TextWrapper:
    def __init__(self, model_name="google/embeddinggemma-300m"): # Use a smaller model or mock for speed if needed
        # Mocking for speed/memory if real model is big
        self.config = type('obj', (object,), {'hidden_size': 768})
        self.tokenizer = type('obj', (object,), {
            '__call__': lambda self, x, return_tensors, padding, truncation: {'input_ids': torch.tensor([[1, 2]])}
        })()
        
    def __call__(self, **kwargs):
        return type('obj', (object,), {'last_hidden_state': torch.randn(1, 2, 768)})

def test_adapterx_init():
    print("Testing AdapterX initialization...")
    
    # Mock VJEPA
    vjepa = type('obj', (object,), {})()
    vjepa.get_vision_features = lambda pixel_values_videos: torch.randn(1, 256, 2048) # B, Tokens, Dim
    
    # Text Model
    text_model = TextWrapper()
    
    try:
        adapter = MultiModalTargetEncoder(
            vjepa_path=vjepa,
            text_model=text_model,
            num_fusion_layers=1,
            freeze_vjepa=False, 
            freeze_text=False
        )
        print("AdapterX initialized successfully.")
    except Exception as e:
        print(f"AdapterX initialization failed: {e}")
        return

    # specific check for Tokenizer access
    # AdapterX calls: self.text_encoder.tokenizer
    if hasattr(adapter.text_encoder, 'tokenizer'):
         print("adapter.text_encoder has tokenizer.")
    else:
         print("adapter.text_encoder MISSING tokenizer.")

if __name__ == "__main__":
    test_adapterx_init()
