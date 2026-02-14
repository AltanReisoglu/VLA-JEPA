# models/multimodal_target.py
import torch
import torch.nn as nn
from transformers import AutoModel
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
load_dotenv()  # Load environment 
HF_TOKEN = os.getenv("HF_TOKEN")  # Hugging Face token from .env
from huggingface_hub import login
login(token=HF_TOKEN)

class MultiModalTargetEncoder(nn.Module):
    def __init__(self, 
                 vjepa_path:AutoModel,
                 text_model:AutoModel,
                 num_fusion_layers=4,
                 freeze_vjepa=True,
                 freeze_text=False):
        super().__init__()
        
        # Visual encoder
        self.v_jepa = vjepa_path
        if freeze_vjepa:
            for param in self.v_jepa.parameters():
                param.requires_grad = False
        
        # Text encoder
        self.text_encoder = text_model
        if freeze_text:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        
        # Dimension matching
        if hasattr(self.text_encoder, "config"):
            text_dim = self.text_encoder.config.hidden_size
        elif hasattr(self.text_encoder, "get_sentence_embedding_dimension"):
            text_dim = self.text_encoder.get_sentence_embedding_dimension()
        else:
             text_dim = 2048 # Fallback

        # Visual dimension - dynamic check
        if hasattr(self.v_jepa, "config"):
            visual_dim = getattr(self.v_jepa.config, "hidden_size", 2048)
        else:
            visual_dim = 2048  # Fallback to V-JEPA default if config missing
        
        if text_dim != visual_dim:
             # If dims don't match, we might need projection?
             # User said "freeze all layers". If we project, we have weights.
             # If "just concat", we don't need projection.
             pass
        self.text_dim = text_dim
        self.visual_dim = visual_dim
        self.output_dim = visual_dim + text_dim # New output dimension
        
        # Cross-attention fusion (removed for concatenation)
        # self.fusion_layers = nn.ModuleList([
        #     CrossAttentionFusionBlock(
        #         dim=visual_dim,
        #         num_heads=16,
        #         mlp_ratio=4.0
        #     )
        #     for _ in range(num_fusion_layers)
        # ])
        
    def forward(self, images, action_descriptions, return_attention=False):
        """
        Args:
            images: [B, C, H, W] or [B, T, C, H, W]
            action_descriptions: List[str] or tokenized text [B, L]
        Returns:
            fused_visual: [B, 256, visual_dim + text_dim] or [B, T, 256, visual_dim + text_dim]
        """
        # Handle temporal dimension
        has_time = images.ndim == 5
        if has_time:
            B, T, C, H, W = images.shape
            images = images.view(B * T, C, H, W)
        
        # Visual encoding and Teacher features extraction
        # Visual encoding
        with torch.no_grad(): # Ensure no grad
            if hasattr(self.v_jepa, "get_vision_features"):
                if has_time:
                    # V-JEPA expects 5D input [B, T, C, H, W]
                    visual_tokens = self.v_jepa.get_vision_features(pixel_values_videos=images.view(B, T, C, H, W))
                    # Reshape output to [B*T, N, D] for concatenation
                    visual_tokens = visual_tokens.view(B * T, -1, visual_tokens.shape[-1])
                else:
                    visual_tokens = self.v_jepa.get_vision_features(pixel_values_videos=images)
            else:
                 visual_tokens = self.v_jepa(pixel_values=images).last_hidden_state  # [B*T, 256, 2048]
        
        # Text encoding
        if isinstance(action_descriptions, list):
            # Check if text_encoder has a tokenizer (AutoModel) or if we are using SentenceTransformer directly
            if hasattr(self.text_encoder, "tokenizer"):
                 tokenizer = self.text_encoder.tokenizer
            else:
                 # Fallback/Assumption: it might be SentenceTransformer which has .tokenizer
                 tokenizer = getattr(self.text_encoder, "tokenizer", None)
            
            if tokenizer:
                text_inputs = tokenizer(
                    action_descriptions,
                    return_tensors='pt',
                    padding=True,
                    truncation=True
                ).to(images.device)
                
                # Handle SentenceTransformer explicitly
                # SentenceTransformer forward expects a dict 'features'
                if isinstance(self.text_encoder, (SentenceTransformer,)): # Using imported class or check method signature?
                    # Safer to check if it's NOT AutoModel (which usually doesn't have 'encode' method in the same way)
                    # Or check modules.
                    text_outputs = self.text_encoder(text_inputs) # Pass as dict
                else:
                    # AutoModel expects kwargs
                     text_outputs = self.text_encoder(**text_inputs)
            else:
                 # No tokenizer found, maybe it expects raw text? (e.g. some wrappers)
                 text_outputs = self.text_encoder(action_descriptions)

            # Extract headers
            # Helper to check for dict-like behavior
            is_dict_like = isinstance(text_outputs, dict) or hasattr(text_outputs, 'keys')
            
            if is_dict_like and "token_embeddings" in text_outputs:
                 text_tokens = text_outputs["token_embeddings"]
            elif hasattr(text_outputs, "last_hidden_state"):
                 text_tokens = text_outputs.last_hidden_state
            elif is_dict_like and "last_hidden_state" in text_outputs:
                 text_tokens = text_outputs["last_hidden_state"]
            else:
                 text_tokens = text_outputs
        else:
            # Maybe pre-tokenized or raw text if model handles it
            text_outputs = self.text_encoder(action_descriptions)
            
            is_dict_like = isinstance(text_outputs, dict) or hasattr(text_outputs, 'keys')

            if hasattr(text_outputs, "last_hidden_state"):
                text_tokens = text_outputs.last_hidden_state
            elif is_dict_like and "token_embeddings" in text_outputs:
                text_tokens = text_outputs["token_embeddings"]
            elif is_dict_like and "last_hidden_state" in text_outputs:
                 text_tokens = text_outputs["last_hidden_state"]
            else:
                text_tokens = text_outputs

        print(text_tokens.shape, "text_tokens.shape")
        print(visual_tokens.shape, "visual_tokens.shape")
        
        # Project text to visual dimension (removed for concatenation)
        #text_tokens = self.text_projector(text_tokens)  # [B, L, 2048]
        
        # Expand text for temporal dimension (removed for concatenation)
        """if has_time:
            text_tokens = text_tokens.unsqueeze(1).expand(-1, T, -1, -1)
            text_tokens = text_tokens.reshape(B * T, -1, self.visual_dim)"""
        
        # Prepare text for concatenation
        # We assume we want Global text info concatenated to each visual token.
        # If text_tokens is [B, L, D], take mean or use it as is?
        # User said "embeddingleri concat et".
        if text_tokens.ndim == 3: # [B, L, D]
             text_tokens = text_tokens.mean(dim=1) # [B, D]
        
        # Expand text to match visual tokens
        # visual_tokens: [B*T, N, Dv]
        BT, N, Dv = visual_tokens.shape
        # text_tokens: [B, Dt]
        
        if has_time:
             # Expand B to B*T
             text_tokens = text_tokens.unsqueeze(1).expand(-1, T, -1) # [B, T, Dt]
             text_tokens = text_tokens.reshape(B*T, -1) # [B*T, Dt]
        
        # Expand to N visual tokens
        text_tokens_expanded = text_tokens.unsqueeze(1).expand(-1, N, -1) # [B*T, N, Dt]
        
        # Concatenate
        fused_visual = torch.cat([visual_tokens, text_tokens_expanded], dim=-1) # [B*T, N, Dv+Dt]
        
        # Restore temporal dimension
        if has_time:
            fused_visual = fused_visual.view(B, T, N, -1)
        
        # No attention maps
        if return_attention:
            return fused_visual, None
        return fused_visual

# CrossAttentionFusionBlock is removed as it's no longer used for concatenation
# class CrossAttentionFusionBlock(nn.Module):
#     def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(dim)
#         self.cross_attn = nn.MultiheadAttention(
#             embed_dim=dim,
#             num_heads=num_heads,
#             dropout=dropout,
#             batch_first=True
#         )
        
#         self.norm2 = nn.LayerNorm(dim)
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, int(dim * mlp_ratio)),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(int(dim * mlp_ratio), dim),
#             nn.Dropout(dropout)
#         )
        
#     def forward(self, visual_tokens, text_tokens, return_attention=False):
#         # Cross-attention: visual queries attend to text
#         normed_visual = self.norm1(visual_tokens)
#         attn_out, attn_weights = self.cross_attn(
#             query=normed_visual,
#             key=text_tokens,
#             value=text_tokens,
#             need_weights=return_attention
#         )
#         visual_tokens = visual_tokens + attn_out
        
#         # FFN
#         visual_tokens = visual_tokens + self.mlp(self.norm2(visual_tokens))
        
#         if return_attention:
#             return visual_tokens, attn_weights
#         return visual_tokens, None


if __name__ == "__main__":
    from transformers import AutoProcessor, AutoModel
    from sentence_transformers import SentenceTransformer
    import torch

    # Load models
    # NOTE: Using a standard ViT as placeholder since specific V-JEPA model ID is tricky to find/private
    # The user should replace this with their local path or correct HF ID: e.g. "facebook/vjepa-vit-h-14"
    # vjepa_path = "facebook/vjepa-vit-h-14" 
    vjepa_path = "google/vit-base-patch16-224"
    text_model_name = "google/embeddinggemma-300m"

    print(f"Loading models... (Using {vjepa_path} as placeholder for V-JEPA)")
    try:
        vjepa_processor = AutoProcessor.from_pretrained(vjepa_path)
    except Exception as e:
        print(f"AutoProcessor failed ({e}), trying AutoImageProcessor...")
        from transformers import AutoImageProcessor
        vjepa_processor = AutoImageProcessor.from_pretrained(vjepa_path)

    vjepa_model = AutoModel.from_pretrained(vjepa_path, output_hidden_states=True)
    
    # Load SentenceTransformer with trust_remote_code=True
    text_model = SentenceTransformer(text_model_name, trust_remote_code=True)
    
    # Create AdapterX
    model = MultiModalTargetEncoder(
        vjepa_path=vjepa_model,
        text_model=text_model,
        num_fusion_layers=4,
        freeze_vjepa=True,
        freeze_text=False
    ).cuda()
    model.eval()

    # Create dummy data
    B, T, C, H, W = 2, 3, 3, 224, 224
    images = torch.randn(B, T, C, H, W).cuda()
    action_descriptions = ["push the red block", "pull the blue block"]

    print("Running forward pass...")
    with torch.no_grad():
        fused_visual, attention_maps = model(images, action_descriptions, return_attention=True)

    print(f"Output shape: {fused_visual.shape}")
    if attention_maps:
        print(f"Number of attention maps: {len(attention_maps)}")
        print(f"Attention map shape: {attention_maps[0].shape}")
    else:
        print("No attention maps returned (concatenation mode)")

    # Verify dimensions
    # assert fused_visual.shape == (B, T, 256, 2048) # Hardcoded for VJEPA
    assert fused_visual.shape[0] == B
    assert fused_visual.shape[1] == T
    # Output dim should be visual_dim + text_dim
    assert fused_visual.shape[-1] == model.visual_dim + model.text_dim
    # assert len(attention_maps) == 4
    # assert attention_maps[0].shape == (B * T, 256, 128) # Depends on num tokens and heads

    print("\nAll checks passed!")