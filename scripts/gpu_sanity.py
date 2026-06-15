"""Phase 0 GPU sanity: print device, compute capability, and a bf16 matmul check."""

from __future__ import annotations

import torch

from _common import REPO_ROOT  # noqa: F401  (sets sys.path)


def main() -> None:
    print(f"torch {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("No CUDA device — training/eval will run on CPU (smoke only).")
        return
    cap = torch.cuda.get_device_capability()
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"compute capability: sm_{cap[0]}{cap[1]}")
    free, total = torch.cuda.mem_get_info()
    print(f"memory: {free / 1e9:.1f} GB free / {total / 1e9:.1f} GB total")
    a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
    c = (a @ b).float()
    print(f"bf16 matmul ok: {torch.isfinite(c).all().item()}")
    if cap[0] >= 12:
        print("sm_120 (Blackwell) detected — confirm torch is a cu128 build (see pyproject).")


if __name__ == "__main__":
    main()
