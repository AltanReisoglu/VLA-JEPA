"""
EmbeddingGemma vision-language wrapper

Intent: provide a Qwen-style interface for Gemma-based VLM checkpoints
that expose an AutoProcessor + causal LM model with vision support.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from accelerate.logging import get_logger
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForCausalLM, AutoProcessor, BatchFeature
from transformers.modeling_outputs import CausalLMOutputWithPast

logger = get_logger(__name__)

IGNORE_INDEX = -100


class _EmbeddingGemma_VL_Interface(nn.Module):
	"""
	Lightweight wrapper around an EmbeddingGemma-style VLM.

	Matches the Qwen interface: forward/generate/build_qwenvl_inputs and exposes
	`model` / `processor` attributes so the rest of the codebase can stay
	unchanged.
	"""

	def __init__(self, config: Optional[dict] = None, **kwargs):
		super().__init__()

		qwenvl_config = config.framework.get("qwenvl", {}) if config is not None else {}
		model_id = qwenvl_config.get("base_vlm", "google/vision-gemma")
		attn_impl = qwenvl_config.get("attn_implementation", "flash_attention_2")
		torch_dtype = qwenvl_config.get("torch_dtype", "auto")
		device_map = qwenvl_config.get("device_map", "cuda")

		self.action_token_min = qwenvl_config.get("action_token_min")
		self.action_token_max = qwenvl_config.get("action_token_max")

		model_kwargs = {
			"trust_remote_code": True,
			"device_map": device_map,
			"torch_dtype": torch_dtype,
		}
		if attn_impl is not None:
			model_kwargs["attn_implementation"] = attn_impl

		try:
			self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
		except TypeError:
			# Fallback for checkpoints that do not accept attn_implementation
			model_kwargs.pop("attn_implementation", None)
			self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

		self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
		tokenizer = getattr(self.processor, "tokenizer", None)
		if tokenizer is not None:
			tokenizer.padding_side = "left"

		self.config = config

		# Align with Qwen interface when nested configs are used
		if hasattr(self.model.config, "text_config") and hasattr(self.model.config.text_config, "hidden_size"):
			self.model.config.hidden_size = self.model.config.text_config.hidden_size

	def forward(self, **kwargs) -> CausalLMOutputWithPast:
		with torch.autocast("cuda", dtype=torch.bfloat16):
			return self.model(**kwargs)

	def generate(self, **kwargs):
		with torch.autocast("cuda", dtype=torch.float16):
			return self.model.generate(**kwargs)

	def build_qwenvl_inputs(
		self,
		images,
		instructions,
		solutions=None,
		prompt_replace_dict: Optional[Dict[str, str]] = None,
		prompt_template: Optional[str] = None,
		**kwargs,
	) -> BatchFeature:
		"""
		Build chat-style inputs: [{"role": "user", "content": [{"type": "image"...}, {"type": "text", "text": prompt}]}]
		Mirrors Qwen formatting while keeping the processor flexible.
		"""

		assert len(images) == len(instructions), "Images and instructions must have the same length"

		messages = []
		for imgs, instruction in zip(images, instructions):
			content = [{"type": "image", "image": img} for img in imgs]

			if prompt_template is None:
				if self.config is not None and "CoT_prompt" in self.config.datasets.vla_data:
					cot_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
					prompt = cot_prompt.replace("{instruction}", instruction)
				else:
					prompt = instruction
			else:
				prompt = prompt_template.replace("{instruction}", instruction)

			if prompt_replace_dict is not None:
				for key, value in prompt_replace_dict.items():
					prompt = prompt.replace(key, value)

			content.append({"type": "text", "text": prompt})
			msg = [{"role": "user", "content": content}]

			if solutions is not None:
				solution = solutions[len(messages)]
				msg.append({"role": "assistant", "content": [{"type": "text", "text": solution}]})
			messages.append(msg)

		# Text packing
		if hasattr(self.processor, "apply_chat_template"):
			texts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
		else:
			texts = [m[-1]["content"][-1]["text"] for m in messages]

		image_inputs, video_inputs = process_vision_info(messages)

		processor_kwargs: Dict[str, object] = {
			"text": texts,
			"padding": True,
			"return_tensors": "pt",
		}
		if image_inputs:
			processor_kwargs["images"] = image_inputs
		if video_inputs:
			processor_kwargs["videos"] = video_inputs

		batch_input = self.processor(**processor_kwargs)

		if solutions is not None:
			labels = batch_input["input_ids"].clone()

			if self.action_token_min is not None and self.action_token_max is not None:
				for i, seq in enumerate(labels):
					mask_seq = (seq >= self.action_token_min) & (seq <= self.action_token_max)
					nonzero_indices = torch.nonzero(mask_seq, as_tuple=False)
					if nonzero_indices.numel() > 0:
						first_action_index = nonzero_indices[0].item()
						seq[:first_action_index] = IGNORE_INDEX
					else:
						seq[:] = IGNORE_INDEX

			pad_id = getattr(getattr(self.processor, "tokenizer", None), "pad_token_id", None)
			if pad_id is not None:
				labels[labels == pad_id] = IGNORE_INDEX

			batch_input["labels"] = labels

		return batch_input.to(self.model.device)


if __name__ == "__main__":
	from omegaconf import OmegaConf
	import argparse
	import debugpy

	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--config_yaml",
		type=str,
		default="./starVLA/config/training/starvla_cotrain_oxe.yaml",
		help="Path to YAML config",
	)
	args, _ = parser.parse_known_args()

	debugpy.listen(("0.0.0.0", 10092))
	print("🔍 Rank 0 waiting for debugger attach on port 10092...")
	debugpy.wait_for_client()

	cfg = OmegaConf.load(args.config_yaml)
	cfg.framework.qwenvl.base_vlm = cfg.framework.qwenvl.get("base_vlm", "google/vision-gemma")

	_ = _EmbeddingGemma_VL_Interface(cfg)
	pass