"""AdSwapAI R&D, 2026-09-02: environment check for the SAM3 experiments.

Run this first. It verifies the venv, the GPU, the sam3 package, Hugging Face
authentication and (optionally) downloads the SAM3 checkpoint.
"""
import sys


def main() -> int:
    print(f"python  : {sys.version.split()[0]}  ({sys.executable})")

    try:
        import torch
        print(f"torch   : {torch.__version__}  cuda={torch.cuda.is_available()}", end="")
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            print(f"  {torch.cuda.get_device_name(0)}  free {free / 2**30:.1f} / {total / 2**30:.1f} GB")
        else:
            print()
    except Exception as exc:  # noqa: BLE001
        print(f"torch   : NOT OK ({exc})")
        return 1

    try:
        import sam3  # noqa: F401
        print(f"sam3    : ok ({sam3.__file__})")
    except Exception as exc:  # noqa: BLE001
        print(f"sam3    : NOT OK ({exc})")
        return 1

    try:
        from huggingface_hub import whoami
        info = whoami()
        print(f"hf auth : logged in as {info.get('name')}")
    except Exception as exc:  # noqa: BLE001
        print("hf auth : NOT logged in. Accept the terms at https://huggingface.co/facebook/sam3 "
              "and run  hf auth login  (or huggingface-cli login) in your terminal.")
        print(f"          ({exc.__class__.__name__})")
        return 1

    if "--download" in sys.argv:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id="facebook/sam3", filename="sam3.pt")
        print(f"checkpoint: {path}")

    print("environment OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
