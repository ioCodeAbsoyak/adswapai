# 05 · The web application (8 May – 5 Jun 2025)

Once the Detectron2 billboard model existed, the pipeline was wrapped in a
Flask API behind an nginx site so that anybody could try it in a browser. In
four weeks the app went from one blocking endpoint to a job-tracked, threaded
backend with an admin page, perspective-correct pasting and "smart tiling" for
full-width boards, then was rebranded from Altervision to AdSwap AI.

| Snapshot | Date | Backend | Frontend |
|----------|------|---------|----------|
| `01_sync_flask_first_version/` | 8 May | Single synchronous `/process`: custom Mask R-CNN for boards, COCO Mask R-CNN for people/ball, colour mask or axis-aligned image paste, ffmpeg transcode. | One inline HTML page. |
| (14 May, not kept) | 14 May | Job dictionary, `client_job_id`, `/process-status`, `/all-jobs`, two-stage Dockerfile building Detectron2 as a wheel. | Admin page listing jobs. |
| (16 May, not kept) | 16 May | Processing moved to a daemon thread, `replace_using_mask` rewritten with `minAreaRect` + `getPerspectiveTransform`, boards wider than 60 % of the frame split off as "big". | Marketing site (hero, about, technology), JS split into `demo.js`. |
| `02_final_may2025/` | 24 May / 5 Jun | `smart_tile_replacement` for big boards (contour → `approxPolyDP` → tiled ad → `warpPerspective`), pip cache layers in the Dockerfile. | AdSwap AI rebrand, landing page with a before/after slider, nginx 301 from `altervision.tv`. |

`02_final_may2025/` holds the backend, Docker and nginx files plus the
original `demo.js`; the full static site of that version lives on in
[`app/frontend/static`](../../app/frontend/static) (the current app kept the
pages and only reworked the JavaScript).

## Known weaknesses of this version (all fixed in `app/`)

* Wide boards in far shots were silently skipped: the corner heuristic
  (`x+y` min/max) collapsed on thin slanted strips.
* Human/ball protection did not apply to small boards in image mode.
* A different threshold slider value rebuilt a whole Mask R-CNN per request.
* Temp upload folders were never deleted, ffmpeg failures left the job in
  "processing" forever, only two API routes were proxied by nginx.
* DeepSORT was installed in the image but never used by the code.

## Running

Do not run this snapshot; use [`app/`](../../app). It is kept to show the
state the project was in when the investor search started (June 2025).
