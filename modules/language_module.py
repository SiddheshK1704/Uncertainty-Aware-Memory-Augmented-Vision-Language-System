"""
Language Module
===============
Encodes natural language commands into dense vector embeddings
using a lightweight sentence-transformer model (MiniLM by default).

These embeddings are later matched against object labels in the
Grounding Module using cosine similarity.
"""

from __future__ import annotations
import numpy as np
from typing import List, Union
from loguru import logger


# Default model — fast and accurate for semantic similarity
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class LanguageModule:
    """
    Wraps a sentence-transformer model to produce fixed-size embeddings
    from free-form natural language strings.

    Example usage::

        lang = LanguageModule()
        embedding = lang.encode("pick up the bottle near the table")
        # embedding.shape == (384,)

        embeddings = lang.encode_batch([
            "navigate to the chair",
            "find the tv",
        ])
        # embeddings.shape == (2, 384)
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu"):
        """
        Args:
            model_name: HuggingFace model identifier or local path
            device:     'cpu' or 'cuda'
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self._embedding_dim: int = 384  # default for MiniLM
        self._load_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, text: str) -> np.ndarray:
        """
        Encode a single text string into a unit-normalized embedding.

        Args:
            text: Natural language instruction

        Returns:
            1-D numpy array of shape (embedding_dim,)
        """
        if not text or not text.strip():
            logger.warning("encode() received empty text. Returning zero vector.")
            return np.zeros(self._embedding_dim, dtype=np.float32)

        try:
            embedding = self.model.encode(
                text.strip(),
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return np.array(embedding, dtype=np.float32)

        except Exception as exc:
            logger.error(f"Encoding failed for '{text}': {exc}")
            return np.zeros(self._embedding_dim, dtype=np.float32)

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of strings in a single batched forward pass.

        Args:
            texts: List of strings

        Returns:
            2-D numpy array of shape (N, embedding_dim)
        """
        if not texts:
            return np.zeros((0, self._embedding_dim), dtype=np.float32)

        try:
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=32,
            )
            return np.array(embeddings, dtype=np.float32)

        except Exception as exc:
            logger.error(f"Batch encoding failed: {exc}")
            return np.zeros((len(texts), self._embedding_dim), dtype=np.float32)

    def encode_labels(self, labels: List[str]) -> np.ndarray:
        """
        Convenience wrapper: encode a list of object class labels.
        Prepends "a photo of a " to improve semantic alignment.

        Args:
            labels: List of class label strings (e.g., ['chair', 'bottle'])

        Returns:
            2-D numpy array of shape (N, embedding_dim)
        """
        # Prefix helps sentence transformers understand these are visual concepts
        prompted = [f"a photo of a {lbl.strip().lower()}" for lbl in labels]
        return self.encode_batch(prompted)

    @property
    def embedding_dim(self) -> int:
        """Return the output embedding dimension."""
        return self._embedding_dim

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self):
        """Download/load the sentence-transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading language model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)

            # Determine actual embedding dim from model config
            self._embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.success(
                f"Language model loaded. Embedding dim = {self._embedding_dim}"
            )

        except ImportError:
            logger.error(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
            self._use_fallback_encoder()

        except Exception as exc:
            logger.error(f"Failed to load language model '{self.model_name}': {exc}")
            self._use_fallback_encoder()

    def _use_fallback_encoder(self):
        """
        Minimal fallback if sentence-transformers isn't available.
        Uses HuggingFace transformers directly with mean pooling.
        Only used as a last resort.
        """
        logger.warning("Using HuggingFace transformers fallback encoder.")
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch

            class _FallbackEncoder:
                def __init__(self, name):
                    self.tokenizer = AutoTokenizer.from_pretrained(name)
                    self.model = AutoModel.from_pretrained(name)
                    self.model.eval()

                def encode(self, texts, normalize_embeddings=True, **kwargs):
                    if isinstance(texts, str):
                        texts = [texts]
                    inputs = self.tokenizer(
                        texts, padding=True, truncation=True,
                        max_length=128, return_tensors="pt"
                    )
                    with torch.no_grad():
                        out = self.model(**inputs)
                    # Mean pooling
                    mask = inputs["attention_mask"].unsqueeze(-1).float()
                    emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
                    emb = emb.numpy()
                    if normalize_embeddings:
                        norms = np.linalg.norm(emb, axis=1, keepdims=True)
                        emb = emb / (norms + 1e-9)
                    return emb if len(emb) > 1 else emb[0]

                def get_sentence_embedding_dimension(self):
                    return 384

            self.model = _FallbackEncoder("microsoft/MiniLM-L12-H384-uncased")
            self._embedding_dim = 384

        except Exception as exc2:
            logger.error(f"Fallback encoder also failed: {exc2}. Embeddings will be zeros.")