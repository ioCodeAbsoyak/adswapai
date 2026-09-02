#!/usr/bin/env python3
"""
AdSwapAI backend service.

Flask API in front of a single GPU worker. Jobs are queued and processed one
at a time (two Mask R-CNN models per frame do not benefit from being run
concurrently on one GPU, and sequential processing keeps memory predictable).

Endpoints (all proxied by the nginx frontend):
  POST /process                 submit a job (multipart form)
  GET  /process-status          all jobs, or ?job_id=... / ?browser_session=...
  GET  /jobs/<job_id>           one job
  GET  /all-jobs                all jobs
  GET  /health                  service / GPU / queue status
  GET  /videos/<filename>       serve a processed video (range requests OK)
  GET  /processed_videos        list processed videos
  GET  /sample-videos           list bundled sample videos
"""
from __future__ import annotations

import logging
import os
import queue
import re
import shutil
import tempfile
import threading
import time
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

import torch
from flask import Flask, abort, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

from pipeline import JobParams, Models, decode_image, get_device, process_video

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("adswap.app")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.environ.get("PROCESSED_DIR", os.path.join(BASE_DIR, "processed_videos"))
SAMPLE_VIDEO_DIR = os.environ.get("SAMPLE_VIDEO_DIR", os.path.join(BASE_DIR, "sample_videos"))
BILLBOARD_WEIGHTS = os.environ.get("BILLBOARD_WEIGHTS", os.path.join(BASE_DIR, "model_final.pth"))
COCO_WEIGHTS = os.environ.get("COCO_WEIGHTS", os.path.join(BASE_DIR, "models", "mask_rcnn_R_50_FPN_3x.pkl"))
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "500"))
PORT = int(os.environ.get("PORT", "5000"))
JOB_RETENTION_DONE = 2 * 3600     # seconds to keep finished jobs in memory
JOB_RETENTION_MAX = 5 * 3600      # seconds to keep any job in memory
ALLOWED_VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

models: Optional[Models] = None
jobs: Dict[str, Dict[str, Any]] = {}          # JSON-safe public state
job_inputs: Dict[str, Dict[str, Any]] = {}    # private inputs (paths, arrays, params)
jobs_lock = threading.Lock()
job_queue: "queue.Queue[str]" = queue.Queue()
active_job_id: Optional[str] = None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _form_float(name: str, default: float, lo: Optional[float] = None, hi: Optional[float] = None) -> float:
    raw = request.form.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    if lo is not None:
        value = max(lo, value)
    if hi is not None:
        value = min(hi, value)
    return value


def _form_int(name: str, default: int, lo: int, hi: int) -> int:
    return int(round(_form_float(name, float(default), float(lo), float(hi))))


def _form_bool(name: str, default: bool) -> bool:
    raw = request.form.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _update(job_id: str, **fields: Any) -> None:
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id].update(fields)


def _video_ready(path: str, min_size: int = 1024) -> bool:
    try:
        return os.path.isfile(path) and os.path.getsize(path) > min_size
    except OSError:
        return False


def _snapshot() -> Dict[str, Dict[str, Any]]:
    """JSON-safe copy of all jobs with live queue positions."""
    with jobs_lock:
        copy = {jid: dict(job) for jid, job in jobs.items()}
    queued = sorted((j for j in copy.values() if j["status"] == "queued"), key=lambda j: j["submitted_time"])
    for pos, job in enumerate(queued, start=1):
        job["queue_position"] = pos
    for job in copy.values():
        if job["status"] == "done":
            job["video_ready"] = _video_ready(os.path.join(PROCESSED_DIR, job["final_output_filename"]))
    return copy


def _cleanup_old_jobs() -> None:
    now = time.time()
    with jobs_lock:
        stale = [
            jid for jid, job in jobs.items()
            if (job["status"] in {"done", "error"} and now - (job.get("end_time") or now) > JOB_RETENTION_DONE)
            or (now - job["submitted_time"] > JOB_RETENTION_MAX)
        ]
        for jid in stale:
            jobs.pop(jid, None)
            job_inputs.pop(jid, None)
    for jid in stale:
        logger.info("Dropped old job %s from memory", jid)


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #
def _run_job(job_id: str) -> None:
    global active_job_id
    inputs = job_inputs.get(job_id)
    if inputs is None:
        return
    active_job_id = job_id
    _update(job_id, status="processing", start_time=time.time())
    logger.info("Job %s started: %s", job_id, inputs["input_path"])

    def on_progress(current: int, total: int) -> None:
        _update(job_id, current_frame=current, total_frames=max(total, current))

    out_path = inputs["output_path"]
    try:
        stats = process_video(inputs["input_path"], out_path, inputs["params"], models,
                              replacement=inputs.get("replacement"), progress_cb=on_progress)
        if not _video_ready(out_path):
            raise RuntimeError("Output video is missing or empty")
        name = os.path.basename(out_path)
        _update(job_id, status="done", is_processing=False, end_time=time.time(),
                current_frame=stats["frames"], total_frames=stats["frames"],
                final_output_filename=name, video_url=f"/videos/{name}", video_ready=True,
                processing_fps=stats["processing_fps"], elapsed_seconds=stats["elapsed_seconds"])
        logger.info("Job %s done in %.1fs (%.1f fps)", job_id, stats["elapsed_seconds"], stats["processing_fps"] or 0)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Job %s failed", job_id)
        _update(job_id, status="error", is_processing=False, end_time=time.time(), error=str(exc))
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
            except OSError:
                pass
    finally:
        active_job_id = None
        tmp = inputs.get("temp_dir")
        if tmp and os.path.isdir(tmp):
            shutil.rmtree(tmp, ignore_errors=True)
        inputs.pop("replacement", None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _cleanup_old_jobs()


def _worker() -> None:
    while True:
        job_id = job_queue.get()
        try:
            _run_job(job_id)
        finally:
            job_queue.task_done()


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.route("/health", methods=["GET"])
def health():
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return jsonify({
        "status": "ok" if models is not None else "loading",
        "device": get_device(),
        "gpu": gpu,
        "queue_length": job_queue.qsize(),
        "active_job": active_job_id,
        "jobs_in_memory": len(jobs),
    }), (200 if models is not None else 503)


@app.route("/process", methods=["POST"])
def process():
    if models is None:
        abort(503, "Models are still loading")

    # Client-supplied ids are used in file names: keep only safe characters
    job_id = re.sub(r"[^A-Za-z0-9_.-]", "", request.form.get("client_job_id", ""))[:128] or str(uuid.uuid4())
    browser_session = request.form.get("browser_session", "unknown-browser")[:128]
    mode = request.form.get("mode", "image").strip().lower()
    if mode not in {"image", "mask"}:
        abort(400, "mode must be image or mask")

    # --- input video: uploaded file or bundled sample -----------------------
    temp_dir: Optional[str] = None
    sample_name = (request.form.get("sample_video") or "").strip()
    upload = request.files.get("video")
    if sample_name:
        safe = secure_filename(sample_name)
        in_path = os.path.join(SAMPLE_VIDEO_DIR, safe)
        if not safe or not os.path.isfile(in_path):
            abort(400, f"Unknown sample video: {sample_name}")
        display_name = safe
    elif upload is not None and upload.filename:
        safe = secure_filename(upload.filename) or "upload.mp4"
        if os.path.splitext(safe)[1].lower() not in ALLOWED_VIDEO_EXT:
            abort(400, "Unsupported video format")
        temp_dir = tempfile.mkdtemp(prefix="adswap_")
        in_path = os.path.join(temp_dir, safe)
        upload.save(in_path)
        display_name = safe
    else:
        abort(400, "Missing video file (field video) or sample_video name")

    # --- parameters ----------------------------------------------------------
    params = JobParams(
        mode=mode,
        conf_threshold=_form_float("conf_threshold", 0.5, 0.05, 0.95),
        human_conf_threshold=_form_float("human_conf_threshold", 0.5, 0.05, 0.95),
        min_mask_size=_form_float("min_mask_size", 0.0, 0.0, 1.0),
        enable_human_filter=_form_bool("enable_human_filter", True),
        hold_frames=_form_int("hold_frames", 3, 0, 15),
        feather=_form_float("feather", 2.0, 0.0, 10.0),
    )
    if mode == "mask":
        r = _form_int("mask_color_r", 0, 0, 255)
        g = _form_int("mask_color_g", 255, 0, 255)
        b = _form_int("mask_color_b", 0, 0, 255)
        params.mask_color_bgr = (b, g, r)
        params.mask_alpha = _form_float("mask_alpha", 0.5, 0.05, 1.0)

    replacement = None
    if mode == "image":
        rep = request.files.get("replacement")
        if rep is None:
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)
            abort(400, "Missing replacement image (field replacement)")
        replacement = decode_image(rep.read())
        if replacement is None:
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)
            abort(400, "Failed to decode replacement image")

    # --- register and enqueue -------------------------------------------------
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_name = f"processed_{datetime.now().strftime('%Y%m%d%H%M%S')}_{job_id[:8]}.mp4"
    out_path = os.path.join(PROCESSED_DIR, out_name)

    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "browser_session": browser_session,
            "file_name": display_name,
            "mode": mode,
            "status": "queued",
            "is_processing": True,          # kept for older frontends: queued or running
            "current_frame": 0,
            "total_frames": 0,
            "submitted_time": time.time(),
            "start_time": None,
            "end_time": None,
            "output_filename": out_name,
            "video_ready": False,
        }
        job_inputs[job_id] = {
            "input_path": in_path, "output_path": out_path, "temp_dir": temp_dir,
            "params": params, "replacement": replacement,
        }
    job_queue.put(job_id)
    position = job_queue.qsize()
    logger.info("Job %s queued (position %d): %s mode=%s", job_id, position, display_name, mode)
    return jsonify({"success": True, "message": "Job queued", "job_id": job_id, "queue_position": position})


@app.route("/process-status", methods=["GET"])
def process_status():
    snapshot = _snapshot()
    job_id = request.args.get("job_id")
    browser_session = request.args.get("browser_session")
    if job_id:
        if job_id not in snapshot:
            return jsonify({}), 404
        return jsonify({job_id: snapshot[job_id]})
    if browser_session:
        return jsonify({jid: j for jid, j in snapshot.items() if j.get("browser_session") == browser_session})
    return jsonify(snapshot)


@app.route("/jobs/<job_id>", methods=["GET"])
def job_detail(job_id: str):
    snapshot = _snapshot()
    if job_id not in snapshot:
        abort(404, "Job not found")
    return jsonify(snapshot[job_id])


@app.route("/all-jobs", methods=["GET"])
def all_jobs():
    return jsonify(_snapshot())


@app.route("/videos/<path:filename>", methods=["GET"])
@app.route("/processed_videos/<path:filename>", methods=["GET"])
def serve_video(filename: str):
    safe = secure_filename(filename)
    if not safe or not _video_ready(os.path.join(PROCESSED_DIR, safe)):
        abort(404, "Video not found or not ready")
    return send_from_directory(PROCESSED_DIR, safe, mimetype="video/mp4", conditional=True)


@app.route("/processed_videos", methods=["GET"])
def list_processed_videos():
    try:
        names = sorted(f for f in os.listdir(PROCESSED_DIR) if f.lower().endswith(".mp4"))
    except FileNotFoundError:
        names = []
    return jsonify(names)


@app.route("/sample-videos", methods=["GET"])
def list_sample_videos():
    try:
        names = sorted(f for f in os.listdir(SAMPLE_VIDEO_DIR)
                       if os.path.splitext(f)[1].lower() in ALLOWED_VIDEO_EXT and "_processed" not in f)
    except FileNotFoundError:
        names = []
    return jsonify(names)


@app.errorhandler(413)
def too_large(_):
    return jsonify({"success": False, "error": f"Upload exceeds {MAX_UPLOAD_MB} MB"}), 413


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    global models
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    logger.info("Loading models (device=%s)...", get_device())
    models = Models(BILLBOARD_WEIGHTS, COCO_WEIGHTS)
    try:
        models.warm_up()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Warm-up failed (continuing): %s", exc)
    threading.Thread(target=_worker, name="gpu-worker", daemon=True).start()
    logger.info("Serving on 0.0.0.0:%d (processed dir: %s, samples: %s)", PORT, PROCESSED_DIR, SAMPLE_VIDEO_DIR)
    app.run(host="0.0.0.0", port=PORT, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
