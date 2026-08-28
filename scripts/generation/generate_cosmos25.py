#!/usr/bin/env python3
"""Small wrapper around NVIDIA Cosmos-Predict2.5's official inference entry."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a locally installed NVIDIA Cosmos-Predict2.5 checkout")
    parser.add_argument("--input", required=True, help="Conditioning media (ignored only by text2world)")
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-file")
    parser.add_argument("--output", required=True)
    parser.add_argument("--inference-type", choices=("text2world", "image2world", "video2world"), required=True)
    parser.add_argument("--model", default="2B/post-trained", help="Official examples/inference.py model value")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cosmos-repo", required=True, help="Local clone of github.com/nvidia-cosmos/cosmos-predict2.5")
    parser.add_argument("--dry-run", action="store_true", help="Print the official command without executing it")
    args = parser.parse_args()
    if bool(args.prompt) == bool(args.prompt_file): parser.error("provide exactly one of --prompt or --prompt-file")
    repo, entry = Path(args.cosmos_repo), Path(args.cosmos_repo) / "examples/inference.py"
    if not repo.is_dir() or not entry.is_file(): parser.error("--cosmos-repo must contain examples/inference.py from Cosmos-Predict2.5")
    prompt = Path(args.prompt_file).read_text().strip() if args.prompt_file else args.prompt
    payload = {"inference_type": args.inference_type, "name": Path(args.input).stem, "prompt": prompt}
    if args.inference_type != "text2world": payload["input_path"] = str(Path(args.input).resolve())
    Path(args.output).mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", prefix="cosmos25_", delete=False) as handle:
        json.dump(payload, handle); bundle = Path(handle.name)
    command = [sys.executable, str(entry), "-i", str(bundle), "-o", str(Path(args.output).resolve()),
               f"--inference-type={args.inference_type}", f"--model={args.model}", f"--seed={args.seed}"]
    try:
        if args.dry_run: print(" ".join(command))
        else: subprocess.run(command, cwd=repo, check=True)
    finally:
        bundle.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
