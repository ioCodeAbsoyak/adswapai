"""AdSwapAI R&D, 2025-04-20: self-contained working snapshot - CSRT tracking with
re-detection every 30 frames, seam-blended horizontal tiling for wide boards, and
homography paste with a blurred alpha mask (Detectron2 Mask R-CNN billboard model)."""

# replacement.py

import os
import cv2
import numpy as np
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

VIDEO_PATH = "data/adVideo1.mp4"        # sample clip, see repo docs/assets.md
REPLACEMENT_DIR = "data/replace"         # folder of ad images where several are used
MODEL_PATH = "model_final.pth"           # custom Detectron2 billboard model, see docs/assets.md
OUTPUT_DIR = "output"

def tile_image(img, target_w, target_h, overlap=30):
    """
    1) resize img proportionally to target_h,
    2) tile it horizontally to build the canvas,
    3) blend the seams using overlap.
    """
    h_i, w_i = img.shape[:2]
    scale    = target_h / h_i
    new_w    = max(1, int(w_i * scale))
    resized  = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)

    n_tiles = int(np.ceil(target_w / new_w))
    canvas  = np.zeros((target_h, n_tiles * new_w, 3), dtype=img.dtype)

    for i in range(n_tiles):
        x0   = i * new_w
        tile = resized.copy()
        if 0 < i < n_tiles:
            alpha = np.linspace(0, 1, overlap)[None, :, None]
            left  = canvas[:, x0:x0+overlap]
            right = resized[:, :overlap]
            tile[:, :overlap] = (left * (1-alpha) + right * alpha).astype(img.dtype)
        canvas[:, x0:x0+new_w] = tile

    return canvas[:, :target_w]


def replace_on_polygon(frame, poly_pts, repl_img, alpha=0.9):
    """
    Apply a homography to the actual mask polygon and alpha-blend the replacement onto it.
    """
    h_f, w_f = frame.shape[:2]
    dst      = np.array(poly_pts, dtype=np.float32)
    h_r, w_r = repl_img.shape[:2]
    src      = np.array([[0,0],[w_r,0],[w_r,h_r],[0,h_r]], dtype=np.float32)

    # billboard aspect ratio
    bill_w      = np.linalg.norm(dst[1]-dst[0])
    bill_h      = np.linalg.norm(dst[3]-dst[0])
    aspect_bill = bill_w / bill_h
    aspect_rep  = w_r   / h_r

    if aspect_bill > 1.5 * aspect_rep:
        # very wide billboard -> tile
        bg   = tile_image(repl_img, int(bill_w), int(bill_h), overlap=30)
        src2 = np.array(
            [[0,0],[bg.shape[1],0],[bg.shape[1],bg.shape[0]],[0,bg.shape[0]]],
            dtype=np.float32
        )
        H    = cv2.getPerspectiveTransform(src2, dst)
        warp = cv2.warpPerspective(bg, H, (w_f, h_f), flags=cv2.INTER_LINEAR)
        mask_src = np.ones((bg.shape[0], bg.shape[1]), dtype=np.uint8)*255
        mask     = cv2.warpPerspective(mask_src, H, (w_f, h_f), flags=cv2.INTER_LINEAR)
    else:
        # normal stretch
        H    = cv2.getPerspectiveTransform(src, dst)
        warp = cv2.warpPerspective(repl_img, H, (w_f, h_f), flags=cv2.INTER_LINEAR)
        mask = cv2.warpPerspective(
            np.ones((h_r, w_r), np.uint8)*255,
            H, (w_f, h_f), flags=cv2.INTER_LINEAR
        )

    # antialias mask
    mask = cv2.GaussianBlur(mask, (15,15), 0) / 255.0
    mask = mask[:, :, None]

    # alpha blend
    out = frame * (1 - mask) + warp * (mask * alpha)
    return out.astype(np.uint8)


def main():
    # 1) Detectron2 predictor
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS                  = MODEL_PATH
    cfg.MODEL.ROI_HEADS.NUM_CLASSES    = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE                   = "cuda"
    predictor                          = DefaultPredictor(cfg)

    # 2) Replacement images
    repl_dir   = REPLACEMENT_DIR
    repl_files = sorted(f for f in os.listdir(repl_dir)
                        if f.lower().endswith((".jpg",".png")))
    repl_imgs  = [cv2.imread(os.path.join(repl_dir,f)) for f in repl_files]
    n_repl     = len(repl_imgs)

    # 3) Video I/O
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cap    = cv2.VideoCapture(VIDEO_PATH)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(os.path.join(OUTPUT_DIR, "output.mp4"), fourcc, fps, (w,h))

    detect_interval = 30
    trackers        = []
    frame_idx       = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        if frame_idx % detect_interval == 0:
            # 1) clear previous trackers
            trackers.clear()

            # 2) re-detect & replace (mask-based)
            outputs = predictor(frame)
            inst    = outputs["instances"].to("cpu")
            masks   = inst.pred_masks.numpy().astype(np.uint8) * 255

            for i, mask in enumerate(masks):
                # get the actual contour
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
                cnt      = max(cnts, key=cv2.contourArea)
                # polygon reduced to 4 points
                eps    = 0.02 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, eps, True)
                if len(approx)==4:
                    quad = approx.reshape(4,2).astype(np.float32)
                else:
                    quad = cv2.boxPoints(cv2.minAreaRect(cnt)).astype(np.float32)

                # start tracker
                x,y,wb,hb = cv2.boundingRect(cnt)
                tr        = cv2.TrackerCSRT_create()
                tr.init(frame, (x,y,wb,hb))
                trackers.append({'tracker':tr,'img_id':i % n_repl})

                # mask-based replace
                frame = replace_on_polygon(frame, quad, repl_imgs[i % n_repl])

        else:
            # 3) track & replace using the tracker only (use the same mask homography!)
            for t in trackers:
                ok, box = t['tracker'].update(frame)
                if not ok: continue
                x,y,wb,hb = map(int, box)
                quad = np.array([[x,y],[x+wb,y],[x+wb,y+hb],[x,y+hb]],
                                dtype=np.float32)
                frame = replace_on_polygon(frame, quad,
                                           repl_imgs[t['img_id']])

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print("Done - output.mp4 created.")


if __name__ == "__main__":
    main()
