"""Load and validate read-only evidence emitted by the tracking/evidence pipeline."""

from .mask_evidence import load_evidence_csv

__all__ = ["load_evidence_csv"]
