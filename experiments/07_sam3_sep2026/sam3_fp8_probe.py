"""AdSwapAI R&D, 2026-09-03: fp8 for the SAM3 ViT linear layers (torchao float8 dynamic quantisation).

The backbone is at the bf16 GEMM ceiling of the RTX 5070 Ti (~85 TFLOPS);
torch._scaled_mm in fp8 measures 181 TFLOPS on the same shapes. This script
quantises the 128 Linear layers of the ViT trunk (qkv, proj, fc1, fc2 in 32
blocks) with torchao's Float8DynamicActivationFloat8Weight (weights fp8 once,
activations fp8 per call with a dynamic scale) and measures:
  * backbone ms per frame, eager and under torch.compile,
  * detection parity against the bf16 model: max |score diff| over all
    queries, number of detections above the threshold, mean mask IoU on the
    queries the bf16 model keeps.
Variants: bf16, fp8 per-tensor scaling, fp8 per-row scaling.
Output: output/fp8/summary.json and a printed table.
"""
import json
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

# ----------------------------------------------------------------------------- settings
VIDEO = "data/2.mp4"
PROMPT = "sponsor banner"
CONFIDENCE = 0.35
RESOLUTION = 1008
FRAMES = 30                  # frames 0..29 (wide shot, 13-16 boards); parity is checked on all of them
VARIANTS = ["bf16", "fp8_tensor"]   # fp8_row: torch._scaled_mm row-wise scaling is not supported on sm_120 (torch 2.11)
COMPILE_MODE = "default"     # torch.compile mode for the vision backbone
OUTPUT_DIR = "output/fp8"
# --------------------------------------------------------------------------------------

SDPA = [SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


def load_frames(n):
    cap = cv2.VideoCapture(VIDEO)
    frames = []
    while len(frames) < n:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(torch.from_numpy(f).cuda())
    cap.release()
    return frames


def preprocess(frame_bgr):
    x = frame_bgr.flip(-1).permute(2, 0, 1).float().unsqueeze(0)
    x = F.interpolate(x, (RESOLUTION, RESOLUTION), mode="bilinear", align_corners=False, antialias=True)
    return (x / 255.0 - 0.5) / 0.5


def to_bf16(module):
    n = 0
    for p in module.parameters():
        if p.is_floating_point() and p.dtype != torch.bfloat16:
            p.data = p.data.to(torch.bfloat16)
            n += 1
    for m in module.modules():
        for name, b in list(m._buffers.items()):
            if b is not None and b.is_floating_point() and b.dtype != torch.bfloat16:
                m._buffers[name] = b.to(torch.bfloat16)
                n += 1
    return n


def patch_fused_mlp():
    """SAM3's ViT MLP calls aten._addmm_activation on fc1's raw weight (fused linear + GELU), which bypasses
    the module and is not implemented for torchao's Float8Tensor. For fp8 weights use linear + GELU instead."""
    import sam3.model.vitdet as vitdet
    original = vitdet.addmm_act

    def addmm_act_fp8_aware(activation, linear, mat1):
        if "Float8" in type(linear.weight).__name__:
            act = activation() if isinstance(activation, type) else activation
            return act(linear(mat1.to(torch.bfloat16)))
        return original(activation, linear, mat1)

    vitdet.addmm_act = addmm_act_fp8_aware


class Model:
    """SAM3 image model with cached text prompt; returns all-query scores and low-res masks."""

    def __init__(self, quant=None):
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        self.model = build_sam3_image_model()
        to_bf16(self.model.backbone)
        self.p = Sam3Processor(self.model, resolution=RESOLUTION, confidence_threshold=CONFIDENCE)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            self.text = self.model.backbone.forward_text([PROMPT], device="cuda")
        self.prompt = self.model._get_dummy_prompt()
        self.vb = self.model.backbone.vision_backbone           # uncompiled reference
        if quant is not None:
            from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerRow, PerTensor, quantize_
            patch_fused_mlp()
            gran = PerRow() if quant == "row" else PerTensor()
            cfg = Float8DynamicActivationFloat8WeightConfig(granularity=gran, set_inductor_config=False)
            quantize_(self.vb, cfg, filter_fn=lambda m, fqn: isinstance(m, torch.nn.Linear) and "trunk.blocks" in fqn)
            n = 0
            for _, m in self.vb.named_modules():
                if isinstance(m, torch.nn.Linear) and "Float8" in type(m.weight).__name__:
                    # autocast keeps LayerNorm output in fp32; the fp8 linear wants bf16 in (and gives bf16 out)
                    m.register_forward_pre_hook(lambda mod, args: (args[0].to(torch.bfloat16),) + tuple(args[1:]))
                    n += 1
            print(f"  quantised {n} Linear layers to fp8 ({quant})", flush=True)

    def set_compiled(self, on):
        self.model.backbone.vision_backbone = torch.compile(self.vb, mode=COMPILE_MODE) if on else self.vb

    @torch.no_grad()
    def backbone(self, x):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16), sdpa_kernel(SDPA, set_priority=True):
            return self.model.backbone.forward_image(x)

    @torch.no_grad()
    def decode(self, bb):
        bb.update(self.text)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = self.model.forward_grounding(backbone_out=bb, find_input=self.p.find_stage,
                                               geometric_prompt=self.prompt, find_target=None)
        probs = (out["pred_logits"].sigmoid() * out["presence_logit_dec"].sigmoid().unsqueeze(1)).squeeze(-1)[0].float()
        masks = out["pred_masks"][0].float() > 0                # (Q, h, w) binary, decoder resolution
        return probs, masks


def sync_ms(fn, *args):
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn(*args)
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t) * 1000


def run(model, frames, reference=None):
    """Times the backbone over the frames and compares every frame's queries to the reference."""
    for f in frames[:3]:
        model.decode(model.backbone(preprocess(f)))
    torch.cuda.synchronize()
    bb_ms, outputs = [], []
    for f in frames:
        bb, ms = sync_ms(model.backbone, preprocess(f))
        bb_ms.append(ms)
        outputs.append(model.decode(bb))
    result = {"backbone_ms": round(float(np.mean(bb_ms)), 1), "backbone_ms_min": round(float(np.min(bb_ms)), 1),
              "kept_per_frame": round(float(np.mean([int((p > CONFIDENCE).sum()) for p, _ in outputs])), 1)}
    if reference is not None:
        dprob, ious, agree = [], [], []
        for (p, m), (pr, mr) in zip(outputs, reference):
            dprob.append(float((p - pr).abs().max()))
            keep = pr > CONFIDENCE
            agree.append(float(((p > CONFIDENCE) == keep).float().mean()))
            if keep.any():
                a, b = m[keep].flatten(1).float(), mr[keep].flatten(1).float()
                inter = (a * b).sum(1)
                union = a.sum(1) + b.sum(1) - inter
                ious.append(float((inter / union.clamp(min=1)).mean()))
        result.update({"max_score_diff": round(max(dprob), 4), "mean_mask_iou_vs_bf16": round(float(np.mean(ious)), 4),
                       "kept_set_agreement": round(float(np.mean(agree)), 4)})
    return result, outputs


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"GPU {torch.cuda.get_device_name(0)}  torch {torch.__version__}", flush=True)
    import torchao
    print(f"torchao {torchao.__version__}", flush=True)
    frames = load_frames(FRAMES)
    results, reference = [], None
    for name in VARIANTS:
        quant = None if name == "bf16" else name.split("_")[1]
        print(f"\n=== {name} ===", flush=True)
        try:
            model = Model(quant)
            for compiled in (False, True):
                model.set_compiled(compiled)
                t = time.time()
                res, outputs = run(model, frames, reference)
                res.update({"variant": name, "compiled": compiled, "run_s": round(time.time() - t, 1),
                            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)})
                if reference is None:
                    reference = outputs
                print(json.dumps(res), flush=True)
                results.append(res)
            del model
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            print(f"  {name} failed: {exc!r}", flush=True)
            results.append({"variant": name, "error": repr(exc)})

    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\n=== summary ===")
    print(f"{'variant':<12}{'compiled':>9}{'backbone ms':>13}{'kept':>6}{'max dscore':>12}{'mask IoU':>10}{'kept agree':>12}")
    for r in results:
        if "error" in r:
            print(f"{r['variant']:<12} ERROR {r['error'][:110]}")
            continue
        print(f"{r['variant']:<12}{str(r['compiled']):>9}{r['backbone_ms']:>13}{r['kept_per_frame']:>6}"
              f"{str(r.get('max_score_diff', '-')):>12}{str(r.get('mean_mask_iou_vs_bf16', '-')):>10}{str(r.get('kept_set_agreement', '-')):>12}")


if __name__ == "__main__":
    main()
