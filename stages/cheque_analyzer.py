"""
Stage 1 — Cheque Analyser
=========================
Wraps the Gemma4-Visual-Agent (DGX / CUDA variant) to:

1. Verify that the input image is a bank cheque.
2. Extract structured cheque fields (bank, date, payee, amount, MICR, …).
3. Return the pixel-level bounding box of the handwritten signature.

This stage **does not modify** the Visual Agent code.  It only interacts with
the agent through carefully-crafted prompts and parses the JSON responses.

Prerequisites
-------------
Clone the Visual Agent repo and point ``VisualAgentConfig.visual_agent_repo_path``
at the ``dgx_spark_gb10/`` sub-directory::

    git clone https://github.com/PromtEngineer/Gemma4-Visual-Agent.git
    export VISUAL_AGENT_REPO_PATH=/path/to/Gemma4-Visual-Agent/dgx_spark_gb10

The CUDA dependencies must be installed as described in that repo's README.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from typing import Any, Dict, Optional

from PIL import Image

from config import VisualAgentConfig
from utils.result_types import BoundingBox, ChequeData, SignatureRegion

logger = logging.getLogger(__name__)

# ── Prompt templates ──────────────────────────────────────────────────────────

_CHEQUE_EXTRACTION_PROMPT = """\
You are an expert bank document analyser.

Examine the image carefully and respond with a single JSON object — no markdown \
fences, no extra text — in the following schema:

{
  "is_cheque": <true|false>,
  "bank_name": "<string or null>",
  "date": "<string or null — as printed on the cheque>",
  "payee_name": "<string or null>",
  "amount_numeric": "<string or null — the numerical figure>",
  "amount_words": "<string or null — the written-out amount>",
  "account_number": "<string or null>",
  "cheque_number": "<string or null>",
  "micr_line": "<string or null — the full MICR strip at the bottom>",
  "signature_bbox": {
    "x1": <int — left edge in pixels>,
    "y1": <int — top edge in pixels>,
    "x2": <int — right edge in pixels>,
    "y2": <int — bottom edge in pixels>
  }
}

Rules:
- Set "is_cheque" to false if the image is not a bank cheque.
- All coordinates must be absolute pixel values relative to the original image.
- "signature_bbox" must tightly enclose only the handwritten signature; \
  set all values to null if no signature is visible.
- Return null for any field you cannot read.
- Do NOT wrap the JSON in ```json … ```.
"""

_SIGNATURE_VALIDATION_PROMPT = """\
Look at this cropped image region.
Does it contain a handwritten signature (ink strokes, cursive writing, \
initials, or a scrawl made by a person)?

Reply with a single JSON object:
{ "is_signature": <true|false>, "reason": "<one sentence>" }

Do NOT wrap the JSON in markdown code fences.
"""


# ── ChequeAnalyzer ────────────────────────────────────────────────────────────

class ChequeAnalyzer:
    """
    Stage 1 of the cheque signature-verification pipeline.

    Parameters
    ----------
    config:
        ``VisualAgentConfig`` controlling model IDs, device, etc.

    Usage
    -----
    ::

        from config import VisualAgentConfig
        from stages.cheque_analyzer import ChequeAnalyzer
        from utils.image_utils import load_image

        analyzer = ChequeAnalyzer(VisualAgentConfig())
        cheque_data = analyzer.analyze(load_image("cheque.jpg"))
    """

    def __init__(self, config: VisualAgentConfig) -> None:
        self.config = config
        self._agent_loaded = False
        self._run_gemma: Any = None  # populated by _load_agent()

    # ── Pre-loading ───────────────────────────────────────────────────

    def warm_up(self) -> None:
        """
        Eagerly load the Visual Agent models (Gemma 4 + Falcon Perception).

        Call this at server startup to avoid cold-start latency on the first
        real request.
        """
        self._load_agent()

    # ── Lazy model loading ────────────────────────────────────────────

    def _load_agent(self) -> None:
        """
        Import the Visual Agent functions from the cloned repository.

        The agent is loaded lazily on the first call to ``analyze()`` so that
        the heavy GPU models are not initialised until they are actually needed.
        """
        if self._agent_loaded:
            return

        repo_path = self.config.visual_agent_repo_path
        if not os.path.isdir(repo_path):
            raise RuntimeError(
                f"Gemma4-Visual-Agent directory not found: {repo_path!r}.\n"
                "Clone https://github.com/PromtEngineer/Gemma4-Visual-Agent and set "
                "VISUAL_AGENT_REPO_PATH to the dgx_spark_gb10 sub-directory."
            )

        # Inject the repo directory into the Python path so we can import
        # agent_studio without modifying it.
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        # Propagate config values via environment variables so agent_studio.py
        # picks them up through its own os.environ.get() calls.
        #
        # Limitation: os.environ is process-global.  When multiple
        # ChequeAnalyzer instances are created with *different* configs in the
        # same process, the first instance's values take precedence (setdefault
        # is a no-op if the key already exists).  If you need per-instance
        # model IDs, set the corresponding environment variables before
        # constructing the pipeline instead of relying on this block.
        _env_overrides = {
            "GEMMA_HF_MODEL_ID": self.config.gemma_model_id,
            "FALCON_HF_MODEL_ID": self.config.falcon_model_id,
            "FALCON_HF_REVISION": self.config.falcon_revision,
            "GEMMA_DO_SAMPLE": "1" if self.config.gemma_do_sample else "0",
            "FALCON_TORCH_COMPILE": "1" if self.config.falcon_torch_compile else "0",
            "FALCON_TORCH_DTYPE": self.config.falcon_dtype,
        }
        if self.config.falcon_local_dir:
            _env_overrides["FALCON_HF_LOCAL_DIR"] = self.config.falcon_local_dir
        if self.config.cuda_device:
            _env_overrides["CUDA_DEVICE"] = self.config.cuda_device

        for k, v in _env_overrides.items():
            os.environ.setdefault(k, v)

        import agent_studio  # noqa: PLC0415 — intentional deferred import

        # We only need the VLM reasoning function (Gemma 4).
        # For signature bbox extraction we rely on Gemma's output rather than
        # Falcon because Gemma can reason about document structure more reliably.
        self._run_gemma = agent_studio.run_gemma_reasoning
        self._run_falcon = agent_studio.run_falcon_perception
        self._agent_loaded = True
        logger.info("Gemma4-Visual-Agent loaded from %s", repo_path)

    # ── Public API ────────────────────────────────────────────────────

    def analyze(self, image: Image.Image) -> ChequeData:
        """
        Analyse a cheque image.

        Parameters
        ----------
        image:
            The input cheque image (RGB PIL Image, any resolution).

        Returns
        -------
        ChequeData
            Populated with all extractable fields and a ``signature_region``
            containing the bounding box of the handwritten signature.
        """
        self._load_agent()

        logger.info("Running cheque extraction prompt (image size: %s)", image.size)
        raw_response = self._run_gemma(image, _CHEQUE_EXTRACTION_PROMPT)
        logger.debug("Raw agent response:\n%s", raw_response)

        cheque_data = self._parse_cheque_response(raw_response, image)

        if cheque_data.signature_region is None:
            # Fallback: try to locate the signature with Falcon Perception
            logger.info("VLM did not return a signature bbox — falling back to Falcon detection.")
            cheque_data.signature_region = self._detect_signature_with_falcon(image)

        return cheque_data

    def validate_signature_with_vlm(self, crop: Image.Image) -> tuple[bool, str]:
        """
        Ask the VLM whether the cropped region is a handwritten signature.

        Returns
        -------
        (is_signature, reason)
        """
        self._load_agent()
        raw = self._run_gemma(crop, _SIGNATURE_VALIDATION_PROMPT)
        parsed = _extract_json(raw)
        if parsed and "is_signature" in parsed:
            return bool(parsed["is_signature"]), str(parsed.get("reason", ""))
        # If parsing fails, fall back to a simple keyword heuristic on the raw text
        lower = raw.lower()
        if "true" in lower or "yes" in lower or "signature" in lower:
            return True, raw[:120]
        return False, raw[:120]

    # ── Parsing helpers ───────────────────────────────────────────────

    def _parse_cheque_response(
        self, raw: str, image: Image.Image
    ) -> ChequeData:
        """Parse the VLM's JSON response into a ``ChequeData`` object."""
        parsed = _extract_json(raw)
        if parsed is None:
            logger.warning("Failed to parse JSON from agent response; treating as non-cheque.")
            return ChequeData(is_cheque=False, raw_agent_response=raw)

        is_cheque = bool(parsed.get("is_cheque", False))
        sig_region = self._parse_signature_bbox(parsed, image)

        return ChequeData(
            is_cheque=is_cheque,
            bank_name=_str_or_none(parsed.get("bank_name")),
            date=_str_or_none(parsed.get("date")),
            payee_name=_str_or_none(parsed.get("payee_name")),
            amount_numeric=_str_or_none(parsed.get("amount_numeric")),
            amount_words=_str_or_none(parsed.get("amount_words")),
            micr_line=_str_or_none(parsed.get("micr_line")),
            account_number=_str_or_none(parsed.get("account_number")),
            cheque_number=_str_or_none(parsed.get("cheque_number")),
            signature_region=sig_region,
            raw_agent_response=raw,
        )

    @staticmethod
    def _parse_signature_bbox(
        parsed: Dict[str, Any], image: Image.Image
    ) -> Optional[SignatureRegion]:
        """Extract and validate a ``SignatureRegion`` from the parsed dict."""
        bbox_raw = parsed.get("signature_bbox")
        if not bbox_raw or not isinstance(bbox_raw, dict):
            return None

        # Accept None-valued dict (model returned all-null bbox)
        if all(v is None for v in bbox_raw.values()):
            return None

        try:
            bbox = BoundingBox.from_dict({k: int(v) for k, v in bbox_raw.items() if v is not None})
        except (ValueError, TypeError, KeyError) as exc:
            logger.warning("Could not parse signature_bbox %s: %s", bbox_raw, exc)
            return None

        # Clamp to image dimensions
        W, H = image.size
        bbox = BoundingBox(
            x1=max(0, bbox.x1),
            y1=max(0, bbox.y1),
            x2=min(W, bbox.x2),
            y2=min(H, bbox.y2),
        )

        if not bbox.is_valid():
            logger.warning("Signature bbox is degenerate after clamping: %s", bbox)
            return None

        return SignatureRegion(bbox=bbox, confidence=1.0)

    def _detect_signature_with_falcon(
        self, image: Image.Image
    ) -> Optional[SignatureRegion]:
        """
        Use Falcon Perception as a fallback to locate the signature region.

        Returns the detection with the largest bounding-box area, heuristically
        assuming the signature is the largest hand-drawn element in the cheque.
        """
        try:
            dets = self._run_falcon(image, "handwritten signature")
        except Exception as exc:
            logger.warning("Falcon signature detection failed: %s", exc)
            return None

        best = None
        best_area = 0
        W, H = image.size
        for det in dets:
            bbox_list = det.get("bbox")
            if not bbox_list or len(bbox_list) < 4:
                continue
            try:
                bbox = BoundingBox.from_list([int(v) for v in bbox_list[:4]])
            except (ValueError, TypeError):
                continue
            # Clamp
            bbox = BoundingBox(
                x1=max(0, bbox.x1),
                y1=max(0, bbox.y1),
                x2=min(W, bbox.x2),
                y2=min(H, bbox.y2),
            )
            if bbox.area > best_area:
                best_area = bbox.area
                best = bbox

        if best is None:
            return None

        return SignatureRegion(bbox=best, confidence=0.8)


# ── Utility functions ─────────────────────────────────────────────────────────

def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Robustly extract the first JSON object from a (possibly decorated) string.

    Tries three strategies in order:
    1. Direct ``json.loads`` on the full string.
    2. Strip markdown code fences then retry.
    3. Regex search for the first ``{ … }`` block.
    """
    text = text.strip()

    # Strategy 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: strip markdown fences
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 3: find the outermost { … } block
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def _str_or_none(value: Any) -> Optional[str]:
    if value is None or (isinstance(value, str) and value.strip().lower() in ("null", "none", "")):
        return None
    return str(value).strip()
