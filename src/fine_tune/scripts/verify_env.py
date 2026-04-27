"""Sanity-check the GPU + Python environment on a fine-tune pod.

Run this:
  - immediately after setup_pod.sh
  - after any pip install / version bump
  - before kicking off a long training run

Exits non-zero if any check fails. Intended to catch the common "looks installed
but actually broken" failure mode where torch was inadvertently upgraded past the
driver's CUDA capability or kernels got ABI-mismatched.
"""
from __future__ import annotations

import importlib
import os
import sys
from typing import Callable


def check(name: str, fn: Callable[[], str]) -> bool:
    try:
        msg = fn()
        print(f"  OK  {name}: {msg}")
        return True
    except Exception as e:
        print(f"  ✗   {name}: {e}")
        return False


def main() -> int:
    checks: list[tuple[str, Callable[[], str]]] = []

    def torch_version() -> str:
        import torch
        return f"{torch.__version__} (cuda={torch.version.cuda})"

    def torch_cuda() -> str:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() == False — driver/CUDA mismatch")
        n = torch.cuda.device_count()
        names = [torch.cuda.get_device_name(i) for i in range(n)]
        return f"{n} GPU(s): {names}"

    def transformers_loadable() -> str:
        from transformers import AutoConfig  # noqa: F401
        import transformers
        return transformers.__version__

    def peft_loadable() -> str:
        from peft import LoraConfig, get_peft_model  # noqa: F401
        import peft
        return peft.__version__

    def accelerate_loadable() -> str:
        import accelerate
        return accelerate.__version__

    def torchvision_loadable() -> str:
        # torchvision import-time crashes on ABI mismatch; force the heavy import path
        from torchvision.io import ImageReadMode  # noqa: F401
        import torchvision
        return torchvision.__version__

    def flash_attn_loadable() -> str:
        import flash_attn
        # Force the CUDA module to import — this is what fails on ABI mismatch
        from flash_attn import flash_attn_interface  # noqa: F401
        return flash_attn.__version__

    def fla_loadable() -> str:
        # flash-linear-attention exposes its kernels via fla.*
        import fla
        return getattr(fla, "__version__", "installed")

    def causal_conv1d_loadable() -> str:
        import causal_conv1d
        # Force the CUDA backing module — this is the ABI-mismatch tripwire
        from causal_conv1d import causal_conv1d_fn  # noqa: F401
        return getattr(causal_conv1d, "__version__", "installed")

    def bitsandbytes_loadable() -> str:
        import bitsandbytes as bnb
        return bnb.__version__

    def hf_token_present() -> str:
        tok = os.environ.get("HF_TOKEN") or _read_token_file()
        if not tok:
            raise RuntimeError("no HF_TOKEN env var and no /workspace/.hf_token")
        return f"token len={len(tok)} (source={'env' if os.environ.get('HF_TOKEN') else 'file'})"

    def hf_home_persistent() -> str:
        hf_home = os.environ.get("HF_HOME", "")
        if not hf_home.startswith("/workspace"):
            raise RuntimeError(f"HF_HOME={hf_home!r} not under /workspace — caches will be wiped on pod restart")
        return hf_home

    def cuda_alloc_conf() -> str:
        v = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        if "expandable_segments" not in v:
            raise RuntimeError(f"PYTORCH_CUDA_ALLOC_CONF={v!r} should include expandable_segments:True")
        return v

    def small_forward_pass() -> str:
        import torch
        a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
        c = a @ b
        torch.cuda.synchronize()
        return f"bf16 matmul OK on {torch.cuda.get_device_name(0)}, c.norm()={c.float().norm().item():.2e}"

    checks = [
        ("torch", torch_version),
        ("torch.cuda", torch_cuda),
        ("torchvision", torchvision_loadable),
        ("transformers", transformers_loadable),
        ("peft", peft_loadable),
        ("accelerate", accelerate_loadable),
        ("flash_attn", flash_attn_loadable),
        ("flash-linear-attention", fla_loadable),
        ("causal_conv1d", causal_conv1d_loadable),
        ("bitsandbytes", bitsandbytes_loadable),
        ("HF_HOME", hf_home_persistent),
        ("PYTORCH_CUDA_ALLOC_CONF", cuda_alloc_conf),
        ("HF_TOKEN", hf_token_present),
        ("bf16 matmul", small_forward_pass),
    ]

    print("Environment checks:")
    failed = sum(1 for name, fn in checks if not check(name, fn))
    print()
    if failed:
        print(f"FAILED ({failed}/{len(checks)} checks). Do NOT start training until these are green.")
        return 1
    print(f"All {len(checks)} checks passed. Env is ready.")
    return 0


def _read_token_file() -> str | None:
    p = "/workspace/.hf_token"
    if os.path.exists(p):
        return open(p).read().strip()
    return None


if __name__ == "__main__":
    sys.exit(main())
