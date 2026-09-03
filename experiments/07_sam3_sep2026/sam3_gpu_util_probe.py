"""AdSwapAI R&D, 2026-09-03: how busy is the GPU inside the SAM3 model, and how do we fill it?

After the post-processing moved to the GPU, 94 % of a frame is the SAM3
model itself: backbone ~99 ms, decoder ~50 ms. A ViT-L at 1008 px is about
4 TFLOP per frame, which a 5070 Ti can do in well under 50 ms, so the model
is not compute-bound: it is losing time to kernel launches, dtype casts
(fp32 weights cast to bf16 by autocast on every frame) and small kernels.

Part 1 profiles a few frames with torch.profiler: GPU-busy fraction, number
of kernels per frame, the biggest kernels.
Part 2 times the backbone and the decoder under variants that remove that
overhead:
  autocast        as in sam3_hybrid_track.py (fp32 weights + bf16 autocast)
  bf16            weights converted to bf16 once, no autocast
  bf16_graph_bb   bf16 + the backbone captured as one CUDA graph (fixed 1008 input)
  bf16_graph_all  bf16 + backbone and decoder captured as CUDA graphs
  batch2/batch4   bf16 backbone on 2 / 4 frames at once (per-frame cost)
Each variant also reports detections on frame 0 so a speed-up that changes
the result is visible.

Output: output/gpu_util/summary.json, trace.json (open in chrome://tracing
or https://ui.perfetto.dev), printed table.
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
FRAMES = 30                  # frames 0..29 of the clip (wide shot, 13-16 boards)
PROFILE_FRAMES = 5
OUTPUT_DIR = "output/gpu_util"
VARIANTS = ["autocast", "bf16", "bf16_graph_bb", "bf16_graph_all", "batch2", "batch4"]
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


def preprocess(frames_bgr):
    """(B,H,W,3) uint8 BGR on the GPU -> (B,3,R,R) float normalised like Sam3Processor."""
    x = frames_bgr.flip(-1).permute(0, 3, 1, 2).float()
    x = F.interpolate(x, (RESOLUTION, RESOLUTION), mode="bilinear", align_corners=False, antialias=True)
    return (x / 255.0 - 0.5) / 0.5


class Runner:
    """Backbone + grounding decoder with a cached text prompt; optional CUDA graphs."""

    def __init__(self, model, processor, prompt, dtype_ctx):
        from sam3.model import box_ops
        self.model, self.p, self.box_ops, self.ctx = model, processor, box_ops, dtype_ctx
        with torch.inference_mode(), self.ctx():
            self.text = model.backbone.forward_text([prompt], device="cuda")
        self.prompt = model._get_dummy_prompt()
        self.graph_bb = self.graph_dec = None

    @torch.inference_mode()
    def backbone(self, x):
        if self.graph_bb is not None:
            self.static_x.copy_(x)
            self.graph_bb.replay()
            return self.static_bb
        with self.ctx(), sdpa_kernel(SDPA, set_priority=True):
            return self.model.backbone.forward_image(x)

    @torch.inference_mode()
    def decoder(self, backbone_out):
        if self.graph_dec is not None:
            self.graph_dec.replay()
            return self.static_dec
        backbone_out.update(self.text)
        with self.ctx():
            return self.model.forward_grounding(backbone_out=backbone_out, find_input=self.p.find_stage,
                                                geometric_prompt=self.prompt, find_target=None)

    def postprocess(self, out):
        probs = (out["pred_logits"].sigmoid() * out["presence_logit_dec"].sigmoid().unsqueeze(1)).squeeze(-1)
        keep = probs > CONFIDENCE
        return probs[keep].float(), out["pred_masks"][keep].float(), out["pred_boxes"][keep].float()

    @torch.inference_mode()
    def capture(self, x, with_decoder):
        """Capture backbone (and decoder) as CUDA graphs on static buffers."""
        self.static_x = x.clone()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), self.ctx(), sdpa_kernel(SDPA, set_priority=True):
            for _ in range(3):
                bb = self.model.backbone.forward_image(self.static_x)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g), self.ctx(), sdpa_kernel(SDPA, set_priority=True):
            self.static_bb = self.model.backbone.forward_image(self.static_x)
        self.graph_bb = g
        if not with_decoder:
            return
        self.static_bb.update(self.text)
        self.tensor_cache = CachedTorchTensor()
        with self.tensor_cache:
            with torch.cuda.stream(s), self.ctx():
                for _ in range(3):
                    self.model.forward_grounding(backbone_out=self.static_bb, find_input=self.p.find_stage,
                                                 geometric_prompt=self.prompt, find_target=None)
            torch.cuda.current_stream().wait_stream(s)
            g2 = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g2), self.ctx():
                self.static_dec = self.model.forward_grounding(backbone_out=self.static_bb, find_input=self.p.find_stage,
                                                               geometric_prompt=self.prompt, find_target=None)
        self.graph_dec = g2


class CachedTorchTensor:
    """Makes ``torch.tensor(<python list>, device="cuda")`` return a cached GPU tensor for repeated constant inputs.

    SAM3's encoder builds ``spatial_shapes`` from a Python list on every call; that is a
    CPU->GPU copy, which CUDA graph capture rejects. With a fixed input size the values never
    change, so the tensor created during warm-up is reused (and the copy disappears).
    """

    def __init__(self):
        self.cache, self.orig = {}, torch.tensor

    def __enter__(self):
        def cached(data, *args, **kwargs):
            dev = kwargs.get("device")
            if isinstance(data, (list, tuple)) and dev is not None and "cuda" in str(dev):
                key = (repr(data), str(kwargs.get("dtype")))
                if key not in self.cache:
                    self.cache[key] = self.orig(data, *args, **kwargs)
                return self.cache[key]
            return self.orig(data, *args, **kwargs)
        torch.tensor = cached
        return self

    def __exit__(self, *exc):
        torch.tensor = self.orig


def to_bf16(model):
    """Convert float parameters and buffers to bf16 in place; complex buffers (RoPE) and integer buffers stay."""
    n = 0
    for p in model.parameters():
        if p.is_floating_point() and p.dtype != torch.bfloat16:
            p.data = p.data.to(torch.bfloat16)
            n += 1
    for m in model.modules():
        for name, b in list(m._buffers.items()):
            if b is not None and b.is_floating_point() and b.dtype != torch.bfloat16:
                m._buffers[name] = b.to(torch.bfloat16)
                n += 1
    return n


def sync_ms(fn, *args):
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn(*args)
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t) * 1000


def profile_frames(runner, frames):
    """torch.profiler on a few frames: GPU busy fraction, kernels per frame, top kernels."""
    from torch.profiler import ProfilerActivity, profile, record_function
    for f in frames[:3]:                                                # warm-up
        runner.decoder(runner.backbone(preprocess(f.unsqueeze(0))))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for f in frames[:PROFILE_FRAMES]:
            with record_function("backbone"):
                bb = runner.backbone(preprocess(f.unsqueeze(0)))
            with record_function("decoder"):
                out = runner.decoder(bb)
            with record_function("postprocess"):
                runner.postprocess(out)
        torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - t0) * 1000 / PROFILE_FRAMES
    prof.export_chrome_trace(os.path.join(OUTPUT_DIR, "trace.json"))
    from torch.autograd import DeviceType
    events = prof.key_averages()
    # only real GPU kernels (device_type CUDA); aten ops and annotations would double-count their kernels
    kernel_rows = [(e.key, e.self_device_time_total / 1000 / PROFILE_FRAMES, e.count / PROFILE_FRAMES)
                   for e in events if e.device_type == DeviceType.CUDA and e.self_device_time_total > 0
                   and not getattr(e, "is_user_annotation", False) and "SAM3Image." not in e.key
                   and e.key not in ("backbone", "decoder", "postprocess")]
    gpu_ms = sum(r[1] for r in kernel_rows)
    n_kernels = sum(r[2] for r in kernel_rows)
    kernel_rows.sort(key=lambda r: -r[1])
    phase = {}
    for e in events:
        if e.key in ("backbone", "decoder", "postprocess"):
            phase[e.key] = {"cpu_ms": round(e.cpu_time_total / 1000 / PROFILE_FRAMES, 1),
                            "gpu_ms": round(e.device_time_total / 1000 / PROFILE_FRAMES, 1)}
    result = {
        "wall_ms_per_frame": round(wall_ms, 1), "gpu_kernel_ms_per_frame": round(gpu_ms, 1),
        "gpu_busy_pct": round(100 * gpu_ms / wall_ms, 1), "kernel_launches_per_frame": int(n_kernels),
        "phases": phase,
        "top_kernels": [{"name": k[:90], "ms_per_frame": round(ms, 2), "calls_per_frame": round(n, 1)} for k, ms, n in kernel_rows[:15]],
    }
    print(f"\nprofile ({PROFILE_FRAMES} frames): wall {wall_ms:.1f} ms/frame, GPU kernels {gpu_ms:.1f} ms/frame "
          f"= {result['gpu_busy_pct']}% busy, {int(n_kernels)} kernel launches/frame")
    for k, v in phase.items():
        print(f"  {k:12s} cpu {v['cpu_ms']:7.1f} ms   gpu {v['gpu_ms']:7.1f} ms")
    print("  top kernels (ms/frame, calls/frame):")
    for r in result["top_kernels"][:12]:
        print(f"    {r['ms_per_frame']:6.2f}  {r['calls_per_frame']:6.1f}  {r['name']}")
    return result


def bench(name, runner, frames):
    batch = 2 if name == "batch2" else 4 if name == "batch4" else 1
    if "graph" in name:
        runner.capture(preprocess(frames[0].unsqueeze(0)), with_decoder=name.endswith("all"))
    for f in frames[:3]:                                                # warm-up
        if batch == 1:
            runner.decoder(runner.backbone(preprocess(f.unsqueeze(0))))
        else:
            runner.backbone(preprocess(torch.stack(frames[:batch])))
    torch.cuda.synchronize()
    bb_ms, dec_ms, n = 0.0, 0.0, 0
    t_loop = time.perf_counter()
    if batch == 1:
        for f in frames:
            bb, ms1 = sync_ms(lambda: runner.backbone(preprocess(f.unsqueeze(0))))
            out, ms2 = sync_ms(runner.decoder, bb)
            bb_ms, dec_ms, n = bb_ms + ms1, dec_ms + ms2, n + 1
        # unsynchronised loop: what the pipeline actually sees
        torch.cuda.synchronize()
        t_loop = time.perf_counter()
        for f in frames:
            probs, masks, boxes = runner.postprocess(runner.decoder(runner.backbone(preprocess(f.unsqueeze(0)))))
            probs.cpu()
        torch.cuda.synchronize()
    else:
        for i in range(0, len(frames) - batch + 1, batch):
            _, ms1 = sync_ms(lambda: runner.backbone(preprocess(torch.stack(frames[i:i + batch]))))
            bb_ms, n = bb_ms + ms1, n + batch
        dec_ms = float("nan")
    loop_ms = (time.perf_counter() - t_loop) * 1000 / max(n, 1)
    # sanity: detections on frame 0
    if batch == 1:
        probs, masks, boxes = runner.postprocess(runner.decoder(runner.backbone(preprocess(frames[0].unsqueeze(0)))))
        det = {"count": int(probs.numel()), "mean_score": round(float(probs.mean()), 3) if probs.numel() else None}
    else:
        det = None
    row = {"variant": name, "batch": batch, "backbone_ms": round(bb_ms / n, 1), "decoder_ms": round(dec_ms / n, 1) if batch == 1 else None,
           "loop_ms_per_frame": round(loop_ms, 1) if batch == 1 else None,
           "fps_model_only": round(1000 / loop_ms, 1) if batch == 1 else round(1000 / (bb_ms / n), 1),
           "frame0": det, "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    print(json.dumps(row), flush=True)
    return row


def main():
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"GPU {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    frames = load_frames(FRAMES)
    model = build_sam3_image_model()
    processor = Sam3Processor(model, resolution=RESOLUTION, confidence_threshold=CONFIDENCE)
    # bf16 variants keep autocast on: with bf16 weights its casts are no-ops, and the few fp32 tensors the model
    # creates internally (position encodings, arange) are still matched to the bf16 weights.
    autocast = lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)  # noqa: E731

    results = {"profile_autocast": None, "variants": []}
    runner = Runner(model, processor, PROMPT, autocast)
    results["profile_autocast"] = profile_frames(runner, frames)

    for name in VARIANTS:
        print(f"\n=== {name} ===", flush=True)
        try:
            if name == "autocast":
                r = Runner(model, processor, PROMPT, autocast)
            else:
                if not getattr(model, "_is_bf16", False):
                    # only the vision-language backbone (ViT + neck + text encoder): the grounding decoder and
                    # the heads run parts in fp32 on purpose and reject bf16 weights
                    print(f"  converted {to_bf16(model.backbone)} backbone tensors to bf16", flush=True)
                    model._is_bf16 = True
                r = Runner(model, processor, PROMPT, autocast)
            results["variants"].append(bench(name, r, frames))
            if name == "bf16_graph_all":
                results["profile_bf16_graph_all"] = profile_frames(r, frames)
            del r
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            print(f"  {name} failed: {exc!r}", flush=True)
            results["variants"].append({"variant": name, "error": repr(exc)})

    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\n=== summary (model only, ms per frame) ===")
    print(f"{'variant':<16}{'batch':>6}{'backbone':>10}{'decoder':>9}{'loop':>8}{'fps':>7}   frame0")
    for r in results["variants"]:
        if "error" in r:
            print(f"{r['variant']:<16} ERROR {r['error'][:110]}")
            continue
        print(f"{r['variant']:<16}{r['batch']:>6}{r['backbone_ms']:>10}{str(r['decoder_ms']):>9}{str(r['loop_ms_per_frame']):>8}{r['fps_model_only']:>7}   {r['frame0']}")


if __name__ == "__main__":
    main()
