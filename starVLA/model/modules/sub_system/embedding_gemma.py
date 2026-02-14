"""
EmbeddingGemma – text embedding sub-system.

Wraps Google's ``embedding-gemma-300m`` SentenceTransformer model to produce
dense text embeddings that can be used as conditioning signals (e.g. for
action prediction) in the starVLA pipeline.

Configuration is handled via the Pydantic ``EmbeddingGemmaConfig`` model.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

import logging
import os
logger = logging.getLogger(__name__)
from dotenv import load_dotenv
load_dotenv()  # Load environment 
HF_TOKEN = os.getenv("HF_TOKEN")  # Hugging Face token from .env
from huggingface_hub import login
login(token=HF_TOKEN)

# ──────────────────────────── Config ────────────────────────────

class EmbeddingGemmaConfig(BaseModel):
    """All knobs needed to instantiate an EmbeddingGemma wrapper."""

    model_name: str = "google/embeddinggemma-300m"
    device: str = "cuda"
    trust_remote_code: bool = True


# ──────────────────────────── Module ────────────────────────────

class EmbeddingGemmaInterface(nn.Module):
    """
    Wrapper around ``SentenceTransformer("google/embedding-gemma-300m")``.

    Provides ``forward`` (= encode) and ``encode`` methods that return
    dense embeddings ``[B, hidden_size]`` for a batch of text strings.

    Attributes:
        model:       The underlying SentenceTransformer.
        hidden_size: Embedding dimension exposed for downstream layers.
    """

    def __init__(self, config: EmbeddingGemmaConfig, **kwargs):
        super().__init__()
        self._cfg = config

        self.model = SentenceTransformer(
            config.model_name,
            device=config.device,
            trust_remote_code=config.trust_remote_code 
        )

        # Expose embedding dimension for downstream projectors
        self.hidden_size = self.model.get_sentence_embedding_dimension()
        logger.info(
            f"EmbeddingGemma loaded: {config.model_name}  "
            f"hidden_size={self.hidden_size}"
        )

    # ─────────────────── encode ───────────────────

    @torch.no_grad()
    def encode(
        self,
        sentences: List[str],
        batch_size: int = 32,
        normalize: bool = True,
        convert_to_tensor: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """
        Encode a list of sentences into dense embeddings.

        Args:
            sentences:          List of text strings.
            batch_size:         Encoding batch size.
            normalize:          L2-normalize the output embeddings.
            convert_to_tensor:  Return a ``torch.Tensor`` (default True).

        Returns:
            ``torch.Tensor`` of shape ``[len(sentences), hidden_size]``.
        """
        embeddings = self.model.encode(
            sentences,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            convert_to_tensor=convert_to_tensor,
            **kwargs,
        )
        return embeddings

    def forward(self, sentences: List[str], **kwargs) -> torch.Tensor:
        """Alias for ``encode`` – makes the module callable in nn pipelines."""
        return self.encode(sentences, **kwargs)


# ──────────────────────────── quick test ────────────────────────────

if __name__ == "__main__":
    cfg = EmbeddingGemmaConfig(model_name="google/embeddinggemma-300m", device="cpu")
    interface = EmbeddingGemmaInterface(cfg)

    test_sentences = [
        "pick up the red block",
        "move the arm to the left",
    ]
    embeddings = interface.encode(test_sentences)
    print(f"Input:  {test_sentences}")
    print(f"Output: shape={embeddings.shape}, dtype={embeddings.dtype}")
    print(f"Cosine sim: {torch.nn.functional.cosine_similarity(embeddings[0], embeddings[1], dim=0):.4f}")