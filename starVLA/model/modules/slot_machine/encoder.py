import torch
import torch.nn as nn
from transformers import AutoVideoProcessor, AutoModel

class SlotAttention(nn.Module):
    def __init__(self, num_slots, slot_dim, num_iterations=2):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_iterations = num_iterations
        
        # Linear attention
        self.linear_attn = nn.MultiheadAttention(
            embed_dim=slot_dim,
            num_heads=8,
            batch_first=True
        )
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim * 2),
            nn.ReLU(),
            nn.Linear(slot_dim * 2, slot_dim)
        )
        
        # LayerNorms
        self.norm1 = nn.LayerNorm(slot_dim)
        self.norm2 = nn.LayerNorm(slot_dim)
    
    def forward(self, feats, slots):
        """
        Args:
            feats: [B,T, N_feat, D] - visual features
            slots: [B,T, N_slot, D] - object slots
        Returns:
            updated_slots: [B, N_slot, D]
        """
        B, N_feat, D = feats.shape
        _, N_slot, _ = slots.shape
        
        for _ in range(self.num_iterations):
            # Cross-attention: slots attend to features
            slots, _ = self.linear_attn(
                query=slots,
                key=feats,
                value=feats
            )
            slots = self.norm1(slots + slots)
            
            # Self-attention on slots
            slots, _ = self.linear_attn(
                query=slots,
                key=slots,
                value=slots
            )
            slots = self.norm2(slots + slots)
            
            # MLP
            slots = self.mlp(slots)
        
        return slots

class VideoSAUR:
    def __init__(self,config):
        # Frozen DINOv2 backbone
        #self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        """self.dinov2.eval()
        for param in self.dinov2.parameters():
            param.requires_grad = False"""
        self.config=config
        self.vj_encoder = AutoModel.from_pretrained(self.config.framework.vj2_model.base_encoder, device_map="cuda")
        self.vj_processor = AutoVideoProcessor.from_pretrained(self.config.framework.vj2_model.base_encoder)
        
        # Projection
        self.proj = nn.Linear(384, 128)  # DINOv2 ViT-S: 384 → 128
        self.num_slots=config.framework.vj2_model.num_action_tokens_per_timestep
        
        # Slot Attention
        self.slot_attention = SlotAttention(
            num_slots=self.num_slots,
            slot_dim=128,
            num_iterations=2
        )
    
    def forward(self, frames):
        """
        Args:
            frames: [B, T, H, W, C] - batch of video sequences
        Returns:
            slots: [B, T, N, D] - object slots
                   B=batch, T=time, N=num_slots, D=slot_dim
        """
        B, T = frames.shape[:2]
        slots_all = []
        prev_slots = None
        
        for t in range(T):
            # Extract DINOv2 features
            with torch.no_grad():
                feats = self.dinov2(frames[:, t])  # [B, 196, 384]
            
            # Project
            feats = self.proj(feats)  # [B, 196, 128]
            
            # Slot Attention (conditioned on previous)
            if prev_slots is None:
                slots = torch.randn(B, self.num_slots, 128)
            else:
                slots = prev_slots
            
            slots = self.slot_attention(feats, slots)  # [B, 7, 128]
            
            slots_all.append(slots)
            prev_slots = slots
        
        return torch.stack(slots_all, dim=1)  # [B, T, 7, 128]
if __name__ == "__main__":
    from types import SimpleNamespace
    config = SimpleNamespace(
        framework=SimpleNamespace(
            vj2_model=SimpleNamespace(
                base_encoder="facebook/vit-base-patch16-224",
                num_action_tokens_per_timestep=7
            )
        )
    )
    video_saur = VideoSAUR(config)
    frames = torch.randn(2, 10, 224, 224, 3)
    slots = video_saur(frames)
    print(slots.shape)