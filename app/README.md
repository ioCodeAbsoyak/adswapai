# AdSwapAI web application

The current, working product: a Flask + Detectron2 backend behind an nginx
static frontend, packaged with Docker Compose. Upload a sports clip (or pick a
bundled sample), choose an ad image or a colour mask, and get back a video in
which every detected pitch-side board carries the new ad.

```
app/
├── docker-compose.yml        backend (GPU) + frontend (nginx)
├── backend/
│   ├── app.py                Flask API + single GPU worker queue
│   ├── pipeline.py           detection / tracking / replacement / encoding
│   ├── cli.py                run the pipeline on a file without the web app
│   ├── dockerfile            CUDA 12.8 · torch 2.7 · Detectron2 (2-stage build)
│   ├── requirements.txt
│   ├── model_final.pth       <- put the custom billboard model here (not in git)
│   └── models/               COCO Mask R-CNN weights are baked in at build time
├── frontend/
│   ├── dockerfile            nginx:stable-alpine
│   ├── nginx.conf            static site + API reverse proxy
│   └── static/               index / demo / about / technology / admin pages
│       └── sampleVideos/     <- put 1.mp4, 2.mp4, 3.mp4 here (not in git)
└── processed_videos/         output videos (bind-mounted into the backend)
```

## Prerequisites

* Docker Desktop (or Docker Engine) with the NVIDIA container runtime
* An NVIDIA GPU; tested on an RTX 5070 Ti (driver 610.88)
* ~20 GB free disk for the backend image
* The large assets that are kept out of git, see [`../docs/assets.md`](../docs/assets.md):
  `backend/model_final.pth` (custom Mask R-CNN, 351 MB) and the three sample
  clips in `frontend/static/sampleVideos/`

## Run

```bash
cd app
docker compose up -d --build        # first build: 15-40 min (torch + Detectron2)
docker compose logs -f backend      # wait for "Serving on 0.0.0.0:5000"
```

* Web demo: http://localhost  (`FRONTEND_PORT=8088 docker compose up -d` for another port)
* Admin / job list: http://localhost/admin.html
* Backend API directly: http://localhost:5000/health

Stop and remove everything: `docker compose down --rmi all`

### Command line (no web UI)

```bash
docker compose exec backend python3 cli.py sample_videos/2.mp4 processed_videos/out.mp4 \
    --replacement ad_images/bilboards6.jpg --conf 0.5 --hold-frames 3
docker compose exec backend python3 cli.py sample_videos/2.mp4 processed_videos/mask.mp4 --mode mask
```

## How a frame is processed

1. **Billboard detection** — custom Mask R-CNN (Detectron2, R50-FPN, one class
   `billboard`) returns instance masks; score threshold from the request.
2. **Protected regions** — the stock COCO Mask R-CNN finds people and sports
   balls; their pixels are subtracted from every board mask so players stay in
   front of the ad.
3. **Temporal smoothing** — an IoU tracker keeps a board alive for
   `hold_frames` frames when detection drops out (removes flicker).
4. **Classification** — masks wider than 60 % of the frame are "big"
   (perimeter LED strip), the rest "small".
5. **Replacement** — big boards get the ad repeated horizontally with its
   aspect ratio preserved; small boards get one perspective-warped ad. Corners
   come from the minimum-area rectangle snapped to the mask's convex hull;
   edges are feathered. In mask mode every board is tinted instead.
6. **Encoding** — frames are streamed to ffmpeg (libx264, `faststart`), the
   source audio track is copied.

Throughput on an RTX 5070 Ti at 1080p: about 6 fps (two Mask R-CNN passes per frame).

## API

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/process` | multipart form: `video` file **or** `sample_video` name, `mode` (`image`/`mask`), `replacement` image (image mode), `conf_threshold`, `human_conf_threshold`, `min_mask_size`, `enable_human_filter`, `hold_frames`, `feather`, `mask_color_r/g/b`, `mask_alpha`, `client_job_id`, `browser_session` |
| GET | `/process-status?job_id=…` | job state: `queued` / `processing` / `done` / `error`, frame progress, `video_url` |
| GET | `/jobs/<id>`, `/all-jobs` | job details |
| GET | `/videos/<file>` | processed video (range requests supported) |
| GET | `/processed_videos`, `/sample-videos` | listings |
| GET | `/health` | device, GPU name, queue length, active job |

Jobs run one at a time on a single GPU worker; the UI shows the queue position.

## Known limitations

* Per-frame detection with a small custom dataset (about 150 labelled frames):
  the model is tuned to the sample clips, new footage needs more training data.
* Wide boards use a single homography per frame; on strongly curved LED strips
  the ad still slides a little.
* No live/stream mode. The Flask development server is used on purpose for the
  demo; put a WSGI server in front for anything public.
