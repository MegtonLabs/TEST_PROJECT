"""
Cheque Signature-Verification Pipeline
=======================================
End-to-end orchestrator that wires Stage 1 (Cheque Analyser) and Stage 2
(Siamese Signature Verifier) into a single, production-ready pipeline.

Quick start
-----------
::

    from pipeline import ChequeVerificationPipeline, PipelineConfig
    from utils.image_utils import load_image

    pipeline = ChequeVerificationPipeline()

    result = pipeline.run(
        cheque_image=load_image("cheque.jpg"),
        reference_images=[
            load_image("ref_sig_1.png"),
            load_image("ref_sig_2.png"),
        ],
    )

    import json
    print(json.dumps(result.to_dict(), indent=2))

Deployment notes
----------------
* All heavy models (Gemma 4, Falcon Perception, Siamese CNN) are loaded lazily
  on the first call to ``run()``.
* For high-throughput scenarios, call ``pipeline.warm_up()`` at startup to
  preload models before the first request.
* The pipeline is *not* thread-safe by default.  Use one instance per process
  or wrap ``run()`` in a mutex if you need concurrent access.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

from PIL import Image

from config import PipelineConfig
from stages.cheque_analyzer import ChequeAnalyzer
from stages.signature_verifier import SignatureVerifier
from utils.image_utils import crop_region, load_image, validate_signature_region
from utils.result_types import ChequeData, VerificationResult

logger = logging.getLogger(__name__)

ImageInput = Union[str, Path, bytes, Image.Image]


class ChequeVerificationPipeline:
    """
    Modular two-stage pipeline for cheque signature verification.

    Parameters
    ----------
    config:
        ``PipelineConfig`` instance.  Pass ``None`` (default) to use all
        default settings.

    Stages
    ------
    Stage 1 — *ChequeAnalyzer*
        Uses the Gemma4-Visual-Agent (Gemma 4 E4B + Falcon Perception) to
        verify that the image is a cheque, extract key fields, and locate the
        handwritten signature via a bounding box.

    Stage 2 — *SignatureVerifier*
        Crops the signature using the bounding box, validates the region,
        preprocesses it, and runs it through the Siamese signature-verification
        network.  Computes cosine similarity against enrolled reference
        signatures and returns a verdict with a confidence score.
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self._analyzer = ChequeAnalyzer(self.config.visual_agent)
        self._verifier = SignatureVerifier(self.config.siamese, self.config.verification)

    # ── Pre-loading ───────────────────────────────────────────────────

    def warm_up(self) -> None:
        """
        Eagerly load all models into GPU memory.

        Call this once at server startup to avoid cold-start latency on the
        first real request.
        """
        logger.info("Warming up pipeline models …")
        self._analyzer.warm_up()
        self._verifier.warm_up()
        logger.info("Pipeline warm-up complete.")

    # ── Main entry point ──────────────────────────────────────────────

    def run(
        self,
        cheque_image: ImageInput,
        reference_images: List[ImageInput],
    ) -> VerificationResult:
        """
        Run the full cheque signature-verification pipeline.

        Parameters
        ----------
        cheque_image:
            The cheque image to verify.  Accepts file paths, URLs, raw bytes,
            numpy arrays, or PIL Images.
        reference_images:
            Enrolled reference signatures of the account holder.  At least one
            image is required.  Accepts the same types as ``cheque_image``.

        Returns
        -------
        VerificationResult
            Structured output containing:

            * ``cheque_data`` — extracted fields + signature bounding box
            * ``reference_scores`` — per-reference cosine similarities
            * ``similarity_score`` — aggregated score
            * ``verdict`` — ``"genuine"`` | ``"forged"`` | ``"undetermined"``
            * ``confidence`` — confidence level in ``[0, 1]``
            * ``reason`` — human-readable explanation
            * ``error`` — non-None if a recoverable error occurred
        """
        # ── Input loading ─────────────────────────────────────────────
        try:
            cheque_pil = load_image(cheque_image)
        except Exception as exc:
            msg = f"Failed to load cheque image: {exc}"
            logger.error(msg)
            return VerificationResult(
                cheque_data=ChequeData(is_cheque=False), error=msg
            )

        if not reference_images:
            msg = "reference_images must contain at least one image."
            logger.error(msg)
            return VerificationResult(
                cheque_data=ChequeData(is_cheque=False), error=msg
            )

        ref_pils: List[Image.Image] = []
        for idx, ref in enumerate(reference_images):
            try:
                ref_pils.append(load_image(ref))
            except Exception as exc:
                msg = f"Failed to load reference image [{idx}]: {exc}"
                logger.error(msg)
                return VerificationResult(
                    cheque_data=ChequeData(is_cheque=False), error=msg
                )

        # ── Stage 1: Cheque analysis ──────────────────────────────────
        logger.info("Stage 1 — cheque analysis")
        try:
            cheque_data = self._analyzer.analyze(cheque_pil)
        except Exception as exc:
            msg = f"Stage 1 failed: {exc}"
            logger.exception(msg)
            return VerificationResult(
                cheque_data=ChequeData(is_cheque=False), error=msg
            )

        if not cheque_data.is_cheque:
            msg = "Input image does not appear to be a bank cheque."
            logger.warning(msg)
            return VerificationResult(
                cheque_data=cheque_data,
                verdict="undetermined",
                reason=msg,
            )

        if cheque_data.signature_region is None:
            msg = "No handwritten signature was detected on the cheque."
            logger.warning(msg)
            return VerificationResult(
                cheque_data=cheque_data,
                verdict="undetermined",
                reason=msg,
            )

        # ── Signature cropping ────────────────────────────────────────
        bbox = cheque_data.signature_region.bbox
        sig_crop = crop_region(
            cheque_pil,
            bbox.x1, bbox.y1, bbox.x2, bbox.y2,
            padding=8,
        )

        # ── Signature validation ──────────────────────────────────────
        passed, note = validate_signature_region(
            sig_crop,
            min_variance_ratio=self.config.verification.min_variance_ratio,
        )
        cheque_data.signature_region.validation_passed = passed
        cheque_data.signature_region.validation_note = note

        if not passed:
            logger.warning("Signature validation failed: %s", note)
            return VerificationResult(
                cheque_data=cheque_data,
                verdict="undetermined",
                reason=f"Signature region validation failed: {note}",
            )

        # Optional VLM double-check
        if self.config.verification.vlm_signature_validation:
            is_sig, vlm_note = self._analyzer.validate_signature_with_vlm(sig_crop)
            cheque_data.signature_region.validation_note = vlm_note
            if not is_sig:
                msg = f"VLM validation rejected cropped region: {vlm_note}"
                logger.warning(msg)
                return VerificationResult(
                    cheque_data=cheque_data,
                    verdict="undetermined",
                    reason=msg,
                )

        # ── Stage 2: Signature verification ───────────────────────────
        logger.info("Stage 2 — signature verification")
        try:
            result = self._verifier.verify(
                query_crop=sig_crop,
                reference_crops=ref_pils,
            )
        except Exception as exc:
            msg = f"Stage 2 failed: {exc}"
            logger.exception(msg)
            return VerificationResult(
                cheque_data=cheque_data, error=msg
            )

        # ── Assemble final result ─────────────────────────────────────
        return VerificationResult(
            cheque_data=cheque_data,
            reference_scores=result["reference_scores"],
            similarity_score=result["similarity_score"],
            verdict=result["verdict"],
            confidence=result["confidence"],
            reason=result["reason"],
        )


# ── CLI convenience ───────────────────────────────────────────────────────────

def _cli() -> None:
    """
    Minimal command-line interface for quick testing::

        python pipeline.py cheque.jpg ref1.png [ref2.png …]
    """
    import argparse
    import json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Run the cheque signature-verification pipeline."
    )
    parser.add_argument("cheque", help="Path to the cheque image.")
    parser.add_argument(
        "references",
        nargs="+",
        help="Paths to reference signature images (at least one).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the similarity threshold (default: from config).",
    )
    parser.add_argument(
        "--vlm-validate",
        action="store_true",
        help="Enable VLM-based signature region validation.",
    )
    args = parser.parse_args()

    cfg = PipelineConfig()
    if args.threshold is not None:
        cfg.verification.similarity_threshold = args.threshold
    if args.vlm_validate:
        cfg.verification.vlm_signature_validation = True

    pipeline = ChequeVerificationPipeline(config=cfg)
    result = pipeline.run(
        cheque_image=args.cheque,
        reference_images=args.references,
    )
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    _cli()
