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
from starVLA.model.framework.AdapterX import MultiModalTargetEncoder
from starVLA.training.trainer_utils.trainer_tools import resize_images
from starVLA.model.tools import FRAMEWORK_REGISTRY

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
        
        """self.vj_encoder = AutoModel.from_pretrained(self.config.framework.vj2_model.base_encoder, device_map="cuda")
        self.vj_processor = AutoVideoProcessor.from_pretrained(self.config.framework.vj2_model.base_encoder)"""

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

        self.vj_predictor = VisionTransformerPredictorAC(
            num_frames=self.config.framework.vj2_model.num_frames//tubelet_size,
            img_size=((self.vj_encoder.config.image_size, self.vj_encoder.config.image_size)),
            tubelet_size=1,
            depth=self.config.framework.vj2_model.depth,
            num_heads=self.config.framework.vj2_model.num_heads,
            embed_dim=visual_dim * 2, # multi view input (Visual only)
            action_embed_dim=hidden_size, # Use the safe hidden_size variable
            num_add_tokens=self.config.framework.vj2_model.num_action_tokens_per_timestep,
            output_dim=predictor_output_dim, # Output Visual + Text
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
             action_tokens[:self.config.framework.vj2_model.num_frames//tubelet_size - 1]]
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
            B, V, T, C, H, W = batch_videos.shape
            batch_videos = batch_videos.reshape(B*V, T, C, H, W)  # [B*V, T, C, H, W]
            input_videos = []
            for i in range(B*V):
                processed = self.vj_processor(
                    videos=batch_videos[i], return_tensors="pt"
                )["pixel_values_videos"].to(self.vj_encoder.device)
                if processed.dim() == 4:
                    processed = processed.unsqueeze(0)
                input_videos.append(processed)
            input_videos = torch.cat(input_videos, dim=0)  # [B*V, T, C, H, W]
            print(f"DEBUG: input_videos shape: {input_videos.shape}")
            
            # Student visual features
            with torch.no_grad():
                video_embeddings = self.vj_encoder.get_vision_features(pixel_values_videos=input_videos)
                # video_embeddings is [B*V, T, N, D] (4D)
                # Chunk splits batch: [ (B, T, N, D), (B, T, N, D) ]
                # Cat dim=3 (Feature) -> [B, T, N, D*2]
                video_embeddings = torch.cat(torch.chunk(video_embeddings, chunks=V, dim=0), dim=-1)
                
                # Flatten T and N -> [B, T*N, D*2]
                # B_student, T_student, N_student, D_student_2 = video_embeddings.shape
                # video_embeddings = video_embeddings.view(B_student, T_student * N_student, D_student_2)
                
                # Note: v_jepa typically returns 3D [Batch, Sequence, Dim] where Sequence = Time*Tokens
                # So we don't need to manually flatten if it's already flat.
                # However, if it returns 4D [Batch, Time, Tokens, Dim], then we need to know.
                # AdapterX suggests it returns [Batch, Time, Tokens, Dim] then reshapes to [Batch*Time, Tokens, Dim]
                # If so, video_embeddings here is [Batch*V, Time, Tokens, Dim].
                # If we chunk D=0, we get B*V -> V * [Batch, Time, Tokens, Dim].
                # If we concat D=-1 (Dim), we get [Batch, Time, Tokens, Dim*V].
                # Then we need to flatten Time+Tokens: view(Batch, -1, Dim*V).
                
                # Given error "not enough values to unpack (expected 4, got 3)", it means:
                # video_embeddings is 3D! [Batch, Sequence, Dim].
                # So chunk(0) -> V * [Batch/V, Sequence, Dim]. (Wait, chunk on 0 splits Batch*V -> Batch).
                # Cat(-1) -> [Batch, Sequence, Dim*V]. This is correct shape for predictor.
                pass
            
            # Step 3: VJ Predictor
            T_encoded = video_embeddings.shape[1] # Time * Spatial Tokens (Sequence Length)
            
            # Determine temporal parameters
            B_V, T_original, C, H, W = input_videos.shape
            
            num_tokens = video_embeddings.shape[1]
            tubelet_size = self.vj_encoder.config.tubelet_size
            num_latents_temporal = T_original // tubelet_size
            tokens_per_latent = num_tokens // num_latents_temporal
            
            # Student Context: first (num_latents - 1) latent steps
            input_states = video_embeddings[:, :tokens_per_latent * (num_latents_temporal - 1), :]
            
            # Get Teacher Targets using AdapterX
            with torch.no_grad():
                 teacher_multimodal_features = self.teacher_encoder(
                     images=input_videos, # [B*V, T, C, H, W]
                     action_descriptions=instructions * V # Repeat instructions for each view if V>1
                 )
                 # AdapterX output: [B*V, T, N, D] (4D tensor)
                 
                 B_V, T_t, N, D_t = teacher_multimodal_features.shape
                 teacher_flat = teacher_multimodal_features
                 
                 # Concatenate views [B, T*N, D*V]
                 # first reshape back to [B, V, T, N, D]
                 teacher_flat = teacher_flat.view(B, V, T_t, N, D_t)
                 # Concat features: [B, T, N, D*V]
                 teacher_flat = torch.cat([teacher_flat[:, i, :, :, :] for i in range(V)], dim=-1) 
                 
                 # Flatten T and N -> [B, T*N, D*V]
                 teacher_flat = teacher_flat.view(B, T_t * N, D_t * V)
                 
                 # Slice target (last latent step)
                 gt_multimodal = teacher_flat[:, tokens_per_latent * (num_latents_temporal - 1):, :]
            
            predicted_states = self.vj_predictor(
                input_states,
                action_tokens
            )
            # Slice predicted states to match target (last latent step)
            predicted_states = predicted_states[:, -tokens_per_latent:, :]

            teacher_forcing_wm_loss = F.l1_loss(
                predicted_states,
                gt_multimodal,
                reduction="mean"
            )
        
        if "action" not in examples[0]:
            return {"wm_loss": teacher_forcing_wm_loss}

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

        return {"action_loss": action_loss, "wm_loss": teacher_forcing_wm_loss * 0.1}

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