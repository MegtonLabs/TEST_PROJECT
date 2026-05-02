"""
Structured output types for the cheque signature-verification pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BoundingBox:
    """Pixel-space bounding box returned by the Visual Agent."""

    x1: int
    y1: int
    x2: int
    y2: int

    # ── conveniences ──────────────────────────────────────────────────

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    def is_valid(self) -> bool:
        return self.width > 0 and self.height > 0

    def to_list(self) -> List[int]:
        return [self.x1, self.y1, self.x2, self.y2]

    def to_dict(self) -> Dict[str, int]:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}

    @classmethod
    def from_list(cls, coords: List[int], fmt: str = "xyxy") -> "BoundingBox":
        """
        Create a ``BoundingBox`` from a four-element coordinate list.

        Parameters
        ----------
        coords:
            Four integers ``[a, b, c, d]``.
        fmt:
            ``"xyxy"`` (default) — ``[x1, y1, x2, y2]`` absolute pixel coords.
            ``"xywh"`` — ``[x, y, width, height]`` origin + size.

        Raises
        ------
        ValueError
            If the resulting box is degenerate (zero or negative size).
        """
        a, b, c, d = coords
        if fmt == "xywh":
            x1, y1, x2, y2 = int(a), int(b), int(a) + int(c), int(b) + int(d)
        else:
            x1, y1, x2, y2 = int(a), int(b), int(c), int(d)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"Degenerate bounding box [{x1}, {y1}, {x2}, {y2}]: "
                "x2 must be > x1 and y2 must be > y1.  "
                "Pass fmt='xywh' if the coordinates are in origin+size format."
            )
        return cls(x1, y1, x2, y2)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BoundingBox":
        if all(k in d for k in ("x1", "y1", "x2", "y2")):
            return cls(int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"]))
        if all(k in d for k in ("x", "y", "width", "height")):
            x, y = int(d["x"]), int(d["y"])
            return cls(x, y, x + int(d["width"]), y + int(d["height"]))
        if all(k in d for k in ("left", "top", "right", "bottom")):
            return cls(int(d["left"]), int(d["top"]), int(d["right"]), int(d["bottom"]))
        raise ValueError(f"Cannot parse BoundingBox from: {d}")


@dataclass
class SignatureRegion:
    """Signature location together with optional confidence from the Visual Agent."""

    bbox: BoundingBox
    confidence: float = 1.0
    validation_passed: bool = True
    validation_note: str = ""


@dataclass
class ChequeData:
    """Structured cheque fields extracted by the Visual Agent (Stage 1)."""

    is_cheque: bool
    bank_name: Optional[str] = None
    date: Optional[str] = None
    payee_name: Optional[str] = None
    amount_numeric: Optional[str] = None
    amount_words: Optional[str] = None
    micr_line: Optional[str] = None
    account_number: Optional[str] = None
    cheque_number: Optional[str] = None
    signature_region: Optional[SignatureRegion] = None

    # Raw JSON string returned by the Visual Agent (useful for debugging)
    raw_agent_response: str = ""

    def to_dict(self) -> Dict[str, Any]:
        sig = None
        if self.signature_region is not None:
            sig = {
                "bbox": self.signature_region.bbox.to_dict(),
                "confidence": self.signature_region.confidence,
                "validation_passed": self.signature_region.validation_passed,
                "validation_note": self.signature_region.validation_note,
            }
        return {
            "is_cheque": self.is_cheque,
            "bank_name": self.bank_name,
            "date": self.date,
            "payee_name": self.payee_name,
            "amount_numeric": self.amount_numeric,
            "amount_words": self.amount_words,
            "micr_line": self.micr_line,
            "account_number": self.account_number,
            "cheque_number": self.cheque_number,
            "signature_region": sig,
        }


@dataclass
class VerificationResult:
    """
    Final output of the complete cheque signature-verification pipeline.

    All fields are populated by Stage 2 (Siamese verifier).  Stage 1 results
    are embedded in ``cheque_data``.
    """

    # ── Stage 1 output ────────────────────────────────────────────────
    cheque_data: ChequeData

    # ── Stage 2 output ────────────────────────────────────────────────
    # Per-reference similarity scores (cosine similarity in [0, 1])
    reference_scores: List[float] = field(default_factory=list)

    # Aggregated score used for the final decision
    similarity_score: float = 0.0

    # "genuine" | "forged" | "undetermined"
    verdict: str = "undetermined"

    # Normalised confidence in the verdict (0–1)
    confidence: float = 0.0

    # Human-readable reason for the decision
    reason: str = ""

    # ── Error / skip information ──────────────────────────────────────
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cheque_data": self.cheque_data.to_dict(),
            "reference_scores": self.reference_scores,
            "similarity_score": round(self.similarity_score, 6),
            "verdict": self.verdict,
            "confidence": round(self.confidence, 6),
            "reason": self.reason,
            "error": self.error,
        }
