# NVIDIA Cosmos-Predict2.5 setup

Cosmos-Predict2.5 is an optional data-generation module. Install it in a **separate environment** from RoboEngine → XMem so its CUDA, PyTorch, and Cosmos dependencies cannot disturb the validated tracking stack.

- Official repository: <https://github.com/nvidia-cosmos/cosmos-predict2.5>
- Recommended default model: `nvidia/Cosmos-Predict2.5-2B`
- Hugging Face: <https://huggingface.co/nvidia/Cosmos-Predict2.5-2B>
- Default official model value: `2B/post-trained`

The 14B model is an optional high-quality choice, not the default.

Follow the official setup guide. Its stated requirements include Linux x86-64, Python 3.10.x, an Ampere-or-newer NVIDIA GPU, NVIDIA driver `>= 570.124.06`, and CUDA 12.8.1. RTX 30-series cards, including the RTX 3090, are Ampere generation. Do not substitute the tracking environment's CUDA 12.4 for the official Cosmos recommendation.

Model access requires a Hugging Face account, read token, and acceptance of the NVIDIA Open Model License. Authenticate interactively (`hf auth login`) or supply credentials through your environment; never place a token in code or configuration.

```bash
export HF_HOME=/path/to/large_disk/huggingface
git clone https://github.com/nvidia-cosmos/cosmos-predict2.5.git /path/to/cosmos-predict2.5
```

The official 2B Video2World configuration at 720p / 16 FPS requires about 32.54 GB GPU VRAM. A 24 GB RTX 3090 is therefore not guaranteed to run that standard configuration. Try memory offloading, reduced resolution/workload, or an alternative official inference configuration, and validate with a smoke test.

This packaging step intentionally does not download Cosmos checkpoints. `generate_cosmos25.py --help` is a wrapper-only smoke test; model inference is skipped until a user installs the upstream project and weights.
