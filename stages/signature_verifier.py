"""
Stage 2 — Siamese Signature Verifier
=====================================
Loads **siddharth-magesh/siamese-signature-verification** from HuggingFace
and uses it to compare a query signature against one or more reference
(enrolled) signatures.

Model loading strategy
----------------------
The model is loaded with ``huggingface_hub.hf_hub_download`` / standard
``torch.load`` because signature verification models on HuggingFace are
typically stored as plain PyTorch checkpoints rather than Transformers
``PreTrainedModel`` subclasses.  We support both patterns transparently:

1. If the repo contains a ``config.json`` that Transformers recognises, we
   load via ``AutoModel.from_pretrained``.
2. Otherwise we download ``pytorch_model.bin`` (or ``model.safetensors``) and
   load it as a generic ``torch.nn.Module`` embedding backbone.

The embedding function and cosine-similarity comparison are independent of
which loader path was used.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from config import SiameseConfig, VerificationConfig
from utils.image_utils import preprocess_for_siamese

logger = logging.getLogger(__name__)

ImageInput = Union[str, Path, Image.Image]


# ── Siamese backbone (fallback definition) ────────────────────────────────────

class _SiameseCNN(nn.Module):
    """
    Lightweight CNN Siamese backbone used as a **fallback** when the
    HuggingFace checkpoint cannot be introspected.

    The architecture mirrors common signature-verification networks
    (inspired by the original Bromley et al. Siamese net):
    - Four convolutional blocks with batch normalisation and max-pooling
    - Global average pooling
    - Fully-connected projection to a 256-D embedding

    The state-dict from the HuggingFace checkpoint is loaded into this
    network after construction; mismatched keys are reported as warnings.
    """

    def __init__(self, embedding_dim: int = 256, in_channels: int = 1) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            # Block 1 — 96×136 → 48×68
            nn.Conv2d(in_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2 — 48×68 → 24×34
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3 — 24×34 → 12×17
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 4 — 12×17 → 6×8
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        pooled = self.pool(features)
        return self.fc(pooled)


# ── SignatureVerifier ─────────────────────────────────────────────────────────

class SignatureVerifier:
    """
    Stage 2 of the pipeline.

    Parameters
    ----------
    siamese_config:
        Controls the model ID, input size, and normalisation.
    verification_config:
        Controls the similarity threshold and aggregation strategy.

    Usage
    -----
    ::

        from config import SiameseConfig, VerificationConfig
        from stages.signature_verifier import SignatureVerifier
        from utils.image_utils import load_image

        verifier = SignatureVerifier(SiameseConfig(), VerificationConfig())
        result = verifier.verify(
            query_crop=load_image("query_sig.png"),
            reference_crops=[load_image("ref1.png"), load_image("ref2.png")],
        )
    """

    def __init__(
        self,
        siamese_config: SiameseConfig,
        verification_config: VerificationConfig,
    ) -> None:
        self.scfg = siamese_config
        self.vcfg = verification_config
        self._model: Optional[nn.Module] = None
        self._device: Optional[torch.device] = None

    # ── Pre-loading ───────────────────────────────────────────────────

    def warm_up(self) -> None:
        """
        Eagerly load the Siamese model into GPU memory.

        Call this at server startup to avoid cold-start latency on the first
        real request.
        """
        self._load_model()

    # ── Lazy model loading ────────────────────────────────────────────

    @property
    def device(self) -> torch.device:
        if self._device is None:
            self._device = _resolve_device(self.scfg.device)
        return self._device

    def _load_model(self) -> None:
        if self._model is not None:
            return

        model_id = self.scfg.model_id
        logger.info("Loading Siamese model: %s", model_id)

        # Strategy 1 — try Transformers AutoModel
        model = _try_load_transformers(model_id, self.device)

        # Strategy 2 — try plain PyTorch checkpoint
        if model is None:
            model = _try_load_pytorch_checkpoint(model_id, self.device)

        # Strategy 3 — fall back to initialised _SiameseCNN (weights random)
        if model is None:
            logger.warning(
                "Could not load weights from %r. "
                "Falling back to a randomly-initialised Siamese backbone.  "
                "Similarity scores will not be meaningful until fine-tuned.",
                model_id,
            )
            in_channels = 1 if self.scfg.grayscale_input else 3
            model = _SiameseCNN(in_channels=in_channels)

        model = model.to(self.device)
        model.eval()
        self._model = model
        logger.info("Siamese model ready on %s", self.device)

    # ── Embedding ─────────────────────────────────────────────────────

    def embed(self, image: Image.Image) -> torch.Tensor:
        """
        Compute a normalised embedding vector for ``image``.

        Parameters
        ----------
        image:
            A PIL Image of the signature (any size; preprocessing is applied
            internally).

        Returns
        -------
        torch.Tensor
            Shape ``(1, D)`` — L2-normalised embedding on CPU.
        """
        self._load_model()

        tensor = preprocess_for_siamese(
            image,
            input_size=self.scfg.input_size,
            grayscale=self.scfg.grayscale_input,
            norm_mean=self.scfg.norm_mean,
            norm_std=self.scfg.norm_std,
        ).to(self.device)

        with torch.no_grad():
            emb = self._model(tensor)

        # L2-normalise so cosine similarity == dot product
        emb = F.normalize(emb.view(1, -1), p=2, dim=1)
        return emb.cpu()

    def embed_batch(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Embed a list of signature images.

        Returns
        -------
        torch.Tensor
            Shape ``(N, D)`` — L2-normalised embeddings on CPU.
        """
        return torch.cat([self.embed(img) for img in images], dim=0)

    # ── Similarity ────────────────────────────────────────────────────

    def cosine_similarity(
        self, emb_a: torch.Tensor, emb_b: torch.Tensor
    ) -> float:
        """
        Return cosine similarity mapped to ``[0, 1]``.

        Both tensors **must be L2-normalised** (as returned by :meth:`embed`).
        For normalised vectors the dot product equals the cosine, which lies in
        ``[-1, 1]``; the mapping ``(dot + 1) / 2`` converts this to ``[0, 1]``.

        Parameters
        ----------
        emb_a, emb_b:
            Shape ``(1, D)`` — L2-normalised embedding tensors on CPU.

        Returns
        -------
        float
            Similarity in ``[0, 1]``.  Higher means more similar.

        Note
        ----
        The result is only meaningful when both tensors are truly L2-normalised.
        :meth:`embed` guarantees this via ``F.normalize(..., p=2)``.  If you
        pass raw (unnormalised) embeddings the score will be silently incorrect.
        """
        dot = (emb_a * emb_b).sum(dim=-1).item()
        # Map [-1, 1] → [0, 1]; clamp to guard against floating-point rounding
        return float(max(0.0, min(1.0, (dot + 1.0) / 2.0)))

    # ── Main verification API ─────────────────────────────────────────

    def verify(
        self,
        query_crop: Image.Image,
        reference_crops: List[Image.Image],
    ) -> dict:
        """
        Compare ``query_crop`` (the extracted cheque signature) against a list
        of ``reference_crops`` (enrolled reference signatures of the account
        holder).

        Parameters
        ----------
        query_crop:
            The cropped query signature image.
        reference_crops:
            One or more reference signature images to compare against.

        Returns
        -------
        dict with keys:
            ``reference_scores`` — list of per-reference cosine similarities.
            ``similarity_score`` — aggregated score.
            ``verdict`` — ``"genuine"`` | ``"forged"`` | ``"undetermined"``.
            ``confidence`` — confidence level in ``[0, 1]``.
            ``reason`` — human-readable explanation.
        """
        if not reference_crops:
            raise ValueError("At least one reference signature image is required.")

        query_emb = self.embed(query_crop)
        ref_embs = self.embed_batch(reference_crops)

        # Per-reference cosine similarities
        scores = [
            self.cosine_similarity(query_emb, ref_embs[i : i + 1])
            for i in range(ref_embs.size(0))
        ]

        # Aggregated score
        agg = self.vcfg.multi_ref_aggregation
        if agg == "max":
            agg_score = max(scores)
        elif agg == "min":
            agg_score = min(scores)
        else:  # "mean" (default)
            agg_score = sum(scores) / len(scores)

        threshold = self.vcfg.similarity_threshold
        margin = abs(agg_score - threshold)

        if agg_score >= threshold:
            verdict = "genuine"
            # Confidence: how far above the threshold normalised to [0, 1]
            confidence = min(1.0, margin / (1.0 - threshold + 1e-9))
            reason = (
                f"Aggregated similarity {agg_score:.4f} ≥ threshold {threshold:.4f} "
                f"({agg} of {len(scores)} reference(s))."
            )
        else:
            verdict = "forged"
            confidence = min(1.0, margin / (threshold + 1e-9))
            reason = (
                f"Aggregated similarity {agg_score:.4f} < threshold {threshold:.4f} "
                f"({agg} of {len(scores)} reference(s))."
            )

        return {
            "reference_scores": [round(s, 6) for s in scores],
            "similarity_score": round(agg_score, 6),
            "verdict": verdict,
            "confidence": round(confidence, 6),
            "reason": reason,
        }


# ── Private loader helpers ────────────────────────────────────────────────────

def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _try_load_transformers(model_id: str, device: torch.device) -> Optional[nn.Module]:
    """Attempt to load the model via HuggingFace Transformers AutoModel."""
    try:
        from transformers import AutoModel  # noqa: PLC0415

        model = AutoModel.from_pretrained(model_id)
        logger.info("Loaded %r via transformers.AutoModel", model_id)
        return model
    except Exception as exc:
        logger.debug("transformers.AutoModel load failed (%s): %s", model_id, exc)
        return None


def _try_load_pytorch_checkpoint(
    model_id: str, device: torch.device
) -> Optional[nn.Module]:
    """
    Download and load a plain PyTorch checkpoint from the HuggingFace Hub.

    Tries ``model.safetensors`` first, then ``pytorch_model.bin``.
    If the checkpoint is a full ``nn.Module`` (saved with ``torch.save(model)``),
    returns it directly.  If it is a state-dict, loads it into ``_SiameseCNN``.
    """
    try:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415
    except ImportError:
        logger.debug("huggingface_hub not available — skipping pytorch checkpoint load.")
        return None

    for filename in ("model.safetensors", "pytorch_model.bin", "model.pt", "model.pth"):
        try:
            local_path = hf_hub_download(repo_id=model_id, filename=filename)
        except Exception:
            continue

        try:
            obj = _safe_torch_load(local_path, device)
        except Exception as exc:
            logger.debug("torch.load %r failed: %s", local_path, exc)
            continue

        if isinstance(obj, nn.Module):
            logger.info("Loaded full nn.Module from %r (%r)", model_id, filename)
            return obj

        if isinstance(obj, dict):
            # Assume it is a state-dict; try to infer channel count
            in_ch = _infer_in_channels(obj)
            backbone = _SiameseCNN(in_channels=in_ch)
            missing, unexpected = backbone.load_state_dict(obj, strict=False)
            if missing:
                logger.warning("Missing keys when loading state-dict: %s", missing[:5])
            if unexpected:
                logger.warning("Unexpected keys in state-dict: %s", unexpected[:5])
            logger.info(
                "Loaded state-dict from %r (%r); missing=%d, unexpected=%d",
                model_id, filename, len(missing), len(unexpected),
            )
            return backbone

    logger.debug("No compatible checkpoint found in %r.", model_id)
    return None


def _safe_torch_load(path: str, device: torch.device) -> object:
    """Load a PyTorch file with weights_only where possible."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        # weights_only not supported in older PyTorch
        return torch.load(path, map_location=device)  # noqa: S614


def _infer_in_channels(state_dict: dict) -> int:
    """Try to read the input channel count from the first Conv2d weight."""
    for key, tensor in state_dict.items():
        if "weight" in key and tensor.ndim == 4:
            return int(tensor.shape[1])  # Conv2d weight shape: (out, in, kH, kW)
    return 1  # default: grayscale
