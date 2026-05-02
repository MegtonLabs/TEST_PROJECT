"""
Signature Verification System — Configuration
==============================================
All tuneable knobs in one place. Override via environment variables or by
editing this file before instantiating the pipeline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VisualAgentConfig:
    """Configuration for the Gemma4-Visual-Agent (Stage 1)."""

    # HuggingFace model IDs — match the DGX variant defaults
    gemma_model_id: str = os.environ.get("GEMMA_HF_MODEL_ID", "google/gemma-4-E4B-it")
    falcon_model_id: str = os.environ.get("FALCON_HF_MODEL_ID", "tiiuae/Falcon-Perception")
    falcon_revision: str = os.environ.get("FALCON_HF_REVISION", "main")

    # If non-empty, load Falcon from a local directory instead of the Hub
    falcon_local_dir: Optional[str] = os.environ.get("FALCON_HF_LOCAL_DIR") or None

    # CUDA device string, e.g. "cuda:0"
    cuda_device: Optional[str] = os.environ.get("CUDA_DEVICE") or None

    # Gemma generation
    gemma_do_sample: bool = os.environ.get("GEMMA_DO_SAMPLE", "1") not in ("0", "false", "no")
    gemma_temperature: float = 0.1
    gemma_max_new_tokens: int = 1024  # extra headroom for JSON output

    # Falcon inference
    falcon_torch_compile: bool = os.environ.get("FALCON_TORCH_COMPILE", "0") not in ("0", "false", "no")
    falcon_dtype: str = os.environ.get("FALCON_TORCH_DTYPE", "bfloat16")

    # Path to the Gemma4-Visual-Agent repo root
    # (clone https://github.com/PromtEngineer/Gemma4-Visual-Agent and set this)
    visual_agent_repo_path: str = os.environ.get(
        "VISUAL_AGENT_REPO_PATH",
        os.path.join(os.path.dirname(__file__), "Gemma4-Visual-Agent", "dgx_spark_gb10"),
    )


@dataclass
class SiameseConfig:
    """Configuration for the Siamese signature-verification model (Stage 2)."""

    # HuggingFace model ID or a local directory
    model_id: str = os.environ.get(
        "SIAMESE_MODEL_ID", "siddharth-magesh/siamese-signature-verification"
    )

    # Input image dimensions expected by the model
    input_size: tuple[int, int] = (155, 220)  # (H, W) — typical for CEDAR / UTSig datasets

    # Normalisation statistics (ImageNet-style unless the model card specifies otherwise)
    norm_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    norm_std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    # When True the model expects a single-channel (grayscale) input tensor
    grayscale_input: bool = True

    # Inference device — "cuda" / "cpu" / "auto"
    device: str = os.environ.get("SIAMESE_DEVICE", "auto")


@dataclass
class VerificationConfig:
    """Thresholds and policy settings for the verification decision."""

    # Cosine-similarity threshold above which a signature is GENUINE.
    # Tune on your own labelled dataset; 0.85 is a reasonable starting point.
    similarity_threshold: float = float(os.environ.get("SIG_SIMILARITY_THRESHOLD", "0.85"))

    # When comparing against multiple reference signatures, use this aggregation.
    # Options: "max" | "mean" | "min"
    multi_ref_aggregation: str = os.environ.get("SIG_MULTI_REF_AGG", "mean")

    # Minimum pixel-variance ratio below which a cropped region is flagged as
    # noise / blank and rejected before the Siamese comparison.
    min_variance_ratio: float = float(os.environ.get("SIG_MIN_VARIANCE_RATIO", "0.001"))

    # When True, use the VLM to double-check that the cropped region is really
    # a handwritten signature (slower but more reliable).
    vlm_signature_validation: bool = (
        os.environ.get("SIG_VLM_VALIDATION", "0") not in ("0", "false", "no")
    )


@dataclass
class PipelineConfig:
    """Top-level configuration object — compose and pass to ChequeVerificationPipeline."""

    visual_agent: VisualAgentConfig = field(default_factory=VisualAgentConfig)
    siamese: SiameseConfig = field(default_factory=SiameseConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
