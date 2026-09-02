"""AdSwapAI R&D, 2025-04-26: compact class-based pipeline - RAFT optical flow
(torchvision) warps masks between detections, IoU-matched tracks, exponential
mask smoothing, and people/ball exclusion (Detectron2 Mask R-CNN billboard model)."""

import cv2
import argparse
import numpy as np
import os
import sys
import torch
import random
import logging
import time
from typing import List, Tuple, Optional
from dataclasses import dataclass
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog, DatasetCatalog
from torchvision.models.optical_flow import raft_large, raft_small, Raft_Large_Weights, Raft_Small_Weights
from torchvision.transforms.functional import to_tensor
from torchvision.utils import flow_to_image

# Logging setup
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BillboardReplacer")

@dataclass
class BillboardDetection:
    mask: np.ndarray
    box: Tuple[int, int, int, int]
    score: float

@dataclass
class BillboardTrack:
    id: int
    mask: np.ndarray
    box: Tuple[int, int, int, int]
    score: float
    color: Tuple[int, int, int]
    age: int
    time_since_detection: int

class BillboardTracker:
    def __init__(self, raft_model: torch.nn.Module, device: torch.device, max_age: int = 10):
        self.raft_model = raft_model
        self.device = device
        self.max_age = max_age
        self.prev_frame = None
        self.tracks: List[BillboardTrack] = []
        self.next_id = 1
        self.mask_history = {}
        self.history_length = 3

    def preprocess(self, frame: np.ndarray) -> torch.Tensor:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t = to_tensor(frame_rgb).to(self.device).unsqueeze(0)
        return t

    def calc_flow(self, frame: np.ndarray) -> Optional[torch.Tensor]:
        cur = self.preprocess(frame)
        if self.prev_frame is None:
            self.prev_frame = cur
            return None
        with torch.no_grad():
            flow_preds = self.raft_model(self.prev_frame, cur)
        flow = flow_preds[-1][0]
        self.prev_frame = cur
        return flow

    @staticmethod
    def warp_mask(mask: np.ndarray, flow: torch.Tensor) -> np.ndarray:
        h, w = mask.shape
        flow_np = flow.cpu().numpy()
        dx, dy = flow_np[0], flow_np[1]
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        warped = cv2.remap(mask.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR)
        return warped > 0.5

    def update_history(self, tid: int, mask: np.ndarray):
        self.mask_history.setdefault(tid, []).append(mask)
        if len(self.mask_history[tid]) > self.history_length:
            self.mask_history[tid].pop(0)

    def smooth_mask(self, tid: int) -> np.ndarray:
        masks = self.mask_history.get(tid, [])
        if not masks:
            return None
        weights = np.array([0.5**i for i in range(len(masks))][::-1])
        weights /= weights.sum()
        sm = np.zeros_like(masks[0], dtype=np.float32)
        for m, w in zip(masks, weights): sm += m.astype(np.float32)*w
        return sm > 0.5

    def track(self, frame: np.ndarray, detections: List[BillboardDetection]) -> List[BillboardTrack]:
        flow = self.calc_flow(frame)
        if flow is None:
            self.tracks = [BillboardTrack(self.next_id+i, d.mask, d.box, d.score,
                                           (random.randint(0,255),)*3, 0, 0)
                           for i,d in enumerate(detections)]
            for t in self.tracks: self.update_history(t.id, t.mask)
            self.next_id += len(detections)
            return self.tracks
        # Warp existing
        preds = []
        for tr in self.tracks:
            wmask = self.warp_mask(tr.mask, flow)
            preds.append(BillboardTrack(tr.id, wmask, tr.box, tr.score, tr.color,
                                        tr.age+1, tr.time_since_detection+1))
        # IoU matching
        iou = lambda a,b: (a&b).sum()/float((a|b).sum()+1e-6)
        matches, used_d, used_t = [], set(), set()
        for di,d in enumerate(detections):
            best = -1; bi=None
            for ti,t in enumerate(preds):
                if ti in used_t: continue
                val = iou(d.mask, t.mask)
                if val>best: best, bi = val, ti
            if best>0.3:
                matches.append((di,bi)); used_d.add(di); used_t.add(bi)
        new_tracks=[]
        # matched
        for di,ti in matches:
            d, t = detections[di], preds[ti]
            self.update_history(t.id, d.mask)
            sm = self.smooth_mask(t.id)
            new_tracks.append(BillboardTrack(t.id, sm if sm is not None else d.mask, d.box,
                                            d.score, t.color, t.age, 0))
        # unmatched old
        for i,t in enumerate(preds):
            if i not in used_t and t.time_since_detection<self.max_age:
                new_tracks.append(t)
        # unmatched new
        for i,d in enumerate(detections):
            if i not in used_d and d.score>0.4:
                nt = BillboardTrack(self.next_id, d.mask, d.box, d.score,
                                    (random.randint(0,255),)*3, 0,0)
                new_tracks.append(nt)
                self.update_history(nt.id, d.mask)
                self.next_id+=1
        self.tracks=new_tracks
        return new_tracks

class BillboardReplacer:
    def __init__(self, replacement_img: np.ndarray):
        self.variants = []
        for i in range(3):
            var = cv2.convertScaleAbs(replacement_img, alpha=1.0 + (i-1)*0.1, beta=5)
            self.variants.append(var)
        self.last_pts = {}

    def choose(self, tid: int) -> np.ndarray:
        return self.variants[tid % len(self.variants)]

    def order_pts(self, pts):
        s = pts.sum(axis=1); d = np.diff(pts,axis=1).ravel()
        return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],dtype=np.float32)

    def replace(self, frame, track: BillboardTrack, fg_mask=None):
        mask = track.mask.astype(np.uint8)*255
        contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return frame
        cnt = max(contours, key=cv2.contourArea)
        if cnt.shape[0]<4: return frame
        repl = self.choose(track.id)
        approx = cv2.approxPolyDP(cnt,0.02*cv2.arcLength(cnt,True),True)
        hull = cv2.convexHull(approx)
        if hull.shape[0]==4:
            dst = np.array([[0,0],[repl.shape[1]-1,0],[repl.shape[1]-1,repl.shape[0]-1],[0,repl.shape[0]-1]],dtype=np.float32)
            src = self.order_pts(hull.reshape(-1,2))
            M = cv2.getPerspectiveTransform(dst, src)
            warp = cv2.warpPerspective(repl, M, (frame.shape[1],frame.shape[0]),borderMode=cv2.BORDER_TRANSPARENT)
            msk = np.zeros((frame.shape[0],frame.shape[1]),dtype=np.uint8)
            cv2.fillPoly(msk, [hull],255)
            valid = (msk==255)
            if fg_mask is not None: valid &= (fg_mask==0)
            out = frame.copy()
            for c in range(3): out[:,:,c][valid] = warp[:,:,c][valid]
            return out
        # fallback resize
        x,y,w,h = cv2.boundingRect(cnt)
        small = cv2.resize(repl,(w,h))
        mask2 = np.zeros((h,w),dtype=np.uint8)
        cnt2 = cnt - [x,y]
        cv2.drawContours(mask2,[cnt2],0,255,-1)
        out=frame.copy()
        for c in range(3):
            region = out[y:y+h,x:x+w,c]
            valid = mask2==255
            if fg_mask is not None:
                valid &= (fg_mask[y:y+h,x:x+w]==0)
            region[valid] = small[:,:,c][valid]
            out[y:y+h,x:x+w,c]=region
        return out

class VideoProcessor:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def setup_billboard(self):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.MODEL.WEIGHTS = self.args.billboard_model
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.args.conf
        cfg.MODEL.DEVICE = str(self.device)
        if "billboard" not in DatasetCatalog.list():
            DatasetCatalog.register("billboard", lambda: [])
            MetadataCatalog.get("billboard").set(thing_classes=["billboard"])
        pred = DefaultPredictor(cfg)
        return pred

    def setup_coco(self):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.args.coco_conf
        cfg.MODEL.DEVICE = str(self.device)
        pred = DefaultPredictor(cfg)
        return pred

    def setup_raft(self):
        weights = Raft_Small_Weights.DEFAULT if self.args.fast else Raft_Large_Weights.DEFAULT
        model = raft_small(weights) if self.args.fast else raft_large(weights)
        model.to(self.device).eval()
        return model

    def run(self):
        # prepare
        b_pred = self.setup_billboard()
        c_pred = self.setup_coco()
        raft = self.setup_raft()
        tracker = BillboardTracker(raft, self.device, max_age=self.args.max_age)
        repl = BillboardReplacer(cv2.imread(self.args.repl_image))

        cap = cv2.VideoCapture(self.args.input)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w,h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.args.output, fourcc, fps, (w,h))
        frame_num=0
        while True:
            ret,frm = cap.read()
            if not ret: break
            frame_num+=1
            # detect billboards
            inst = b_pred(frm)["instances"].to("cpu")
            bboxes, masks, scores = inst.pred_boxes.tensor.numpy(), inst.pred_masks.numpy(), inst.scores.numpy()
            dets = [BillboardDetection(masks[i], tuple(map(int, bboxes[i][:4])), float(scores[i]))
                    for i in range(len(scores))]
            # detect foreground
            inst2 = c_pred(frm)["instances"].to("cpu")
            cls = inst2.pred_classes.numpy()
            keep = [i for i,c in enumerate(cls) if c in (0,32)]
            fg = np.zeros((h,w),np.uint8)
            for i in keep:
                fg |= (inst2.pred_masks.numpy()[i]*255).astype(np.uint8)
            # track
            tracks = tracker.track(frm, dets)
            # replace
            out_frame = frm.copy()
            for t in tracks:
                if t.score>0.4:
                    out_frame = repl.replace(out_frame, t, fg)
            out.write(out_frame)
        cap.release(); out.release()

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--billboard-model', default='model_final.pth', help="Custom Detectron2 billboard model, see docs/assets.md")
    p.add_argument('--input', default='data/adVideo1.mp4', help="Sample clip, see repo docs/assets.md")
    p.add_argument('--output', default='output/result.mp4')
    p.add_argument('--repl-image', default='data/replace.jpg')
    p.add_argument('--conf', type=float, default=0.5)
    p.add_argument('--coco-conf', type=float, default=0.5)
    p.add_argument('--max-age', type=int, default=5)
    p.add_argument('--fast', action='store_true')
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    VideoProcessor(args).run()
