#!/usr/bin/env python3
"""Generate the three current Object Permanence curves from existing evidence."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from modules.curves import compute_motion_smoothness, compute_object_existence, compute_shape_normality
from modules.evidence import load_evidence_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Object Existence, Shape Normality, and Motion Smoothness curves")
    parser.add_argument("--evidence-csv", help="Current per_instance_frame_evidence.csv")
    parser.add_argument("--tracking-dir", help="Directory containing evidence/per_instance_frame_evidence.csv")
    parser.add_argument("--video", help="Optional provenance only; no model inference is run")
    parser.add_argument("--task", help="Optional provenance only")
    parser.add_argument("--target", help="Optional provenance only")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if bool(args.evidence_csv) == bool(args.tracking_dir):
        parser.error("provide exactly one of --evidence-csv or --tracking-dir")
    source = Path(args.evidence_csv) if args.evidence_csv else Path(args.tracking_dir) / "evidence/per_instance_frame_evidence.csv"
    evidence = load_evidence_csv(source)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    per_instance = []
    for instance_id, rows in evidence.groupby("instance_id", sort=True):
        rows = rows.sort_values("frame_idx", kind="stable").reset_index(drop=True)
        curves = compute_object_existence(rows).merge(compute_shape_normality(rows), on=["frame", "time_sec"]).merge(compute_motion_smoothness(rows), on=["frame", "time_sec"])
        curves.insert(2, "instance_id", instance_id); per_instance.append(curves)
    all_curves = pd.concat(per_instance, ignore_index=True)
    # Current group aggregation is the per-frame mean of available instance evidence.
    three = all_curves.groupby(["frame", "time_sec"], as_index=False)[["object_existence", "shape_normality", "motion_smoothness"]].mean()
    three.to_csv(out / "three_curves.csv", index=False)
    three[["frame", "time_sec", "object_existence"]].to_csv(out / "object_existence_curve.csv", index=False)
    three[["frame", "time_sec", "shape_normality"]].to_csv(out / "shape_normality_curve.csv", index=False)
    three[["frame", "time_sec", "motion_smoothness"]].to_csv(out / "motion_smoothness_curve.csv", index=False)
    all_curves.to_csv(out / "per_instance_three_curves.csv", index=False)
    fig, axis = plt.subplots(figsize=(12, 4.5))
    for key, label in [("object_existence", "Object Existence"), ("shape_normality", "Shape Normality"), ("motion_smoothness", "Motion Smoothness")]:
        axis.plot(three["frame"], three[key], label=label)
    axis.set(xlabel="Frame", ylabel="Score", title="Object Permanence Curves", ylim=(0, 1)); axis.grid(alpha=.25); axis.legend(); fig.tight_layout(); fig.savefig(out / "three_curves.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
