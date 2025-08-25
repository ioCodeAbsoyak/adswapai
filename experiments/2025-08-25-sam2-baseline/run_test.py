import os
import cv2
import torch
import yaml
import time
from pathlib import Path

# SAM2 importları (submodule içinden)
import sys
ROOT = Path(__file__).resolve().parents[2]  # adswapai/
sys.path.insert(0, str(ROOT / "external" / "sam2"))
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

def load_cfg(fp):
    with open(fp, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def check_device_info():
    """GPU/CPU durumunu kontrol et"""
    print("=" * 50)
    print("DEVICE INFORMATION")
    print("=" * 50)
    
    # CUDA kullanılabilirlik
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        print(f"Current CUDA Device: {torch.cuda.current_device()}")
        print(f"CUDA Device Name: {torch.cuda.get_device_name()}")
        print(f"CUDA Memory Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"CUDA Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
        print(f"CUDA Memory Reserved: {torch.cuda.memory_reserved() / 1024**3:.1f} GB")
    else:
        print("CUDA not available - will use CPU")
    
    print("=" * 50)

def main():
    # Device bilgilerini göster
    check_device_info()
    
    cfg = load_cfg("config.yaml")
    img = cv2.imread(cfg["image_path"])
    assert img is not None, f"Image not found: {cfg['image_path']}"
    out_p = Path(cfg["output_dir"])
    out_p.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = cfg["checkpoint"]
    model_cfg = cfg["model_cfg"]
    device = cfg.get("device", "cuda")
    dtype = torch.bfloat16 if cfg.get("dtype", "bfloat16") == "bfloat16" else torch.float16

    # Device kontrolü ve ayarlama
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA requested but not available, falling back to CPU")
        device = "cpu"
    
    print(f"\n🎯 Using device: {device}")
    print(f"🎯 Using dtype: {dtype}")

    # Model yükleme - timing ile
    print("\n📥 Loading SAM2 model...")
    start_time = time.time()
    sam2_model = build_sam2(model_cfg, checkpoint, device=device)
    
    # Prompt tipine göre predictor seç
    prompt = cfg["prompt"]
    if prompt["type"] == "auto":
        # Automatic mask generator for detecting all objects
        mask_generator = SAM2AutomaticMaskGenerator(
            sam2_model,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,
        )
        predictor = None
        print("🤖 Using SAM2AutomaticMaskGenerator for detecting all objects")
    else:
        # Regular predictor for point/box prompts
        predictor = SAM2ImagePredictor(sam2_model)
        mask_generator = None
        print("🎯 Using SAM2ImagePredictor for targeted segmentation")
        
    load_time = time.time() - start_time
    print(f"✅ Model loaded in {load_time:.2f} seconds")
    
    # Model device'ını kontrol et
    model_device = next(sam2_model.parameters()).device
    print(f"🔍 Model is on device: {model_device}")

    
    # Inference - timing ile
    print(f"\n🚀 Starting inference on {model_device}...")
    start_time = time.time()
    
    # BGR->RGB conversion with copy to avoid negative strides
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"📷 Image size: {rgb_img.shape}")
    
    prompt = cfg["prompt"]
    print(f"🎯 Using prompt type: {prompt['type']}")
    
    if prompt["type"] == "auto":
        # Auto mask generation with SAM2AutomaticMaskGenerator
        print("   Generating masks for all objects...")
        with torch.inference_mode():
            masks_info = mask_generator.generate(rgb_img)
        
        print(f"🔍 Found {len(masks_info)} objects")
        # Convert to format compatible with visualization
        masks = [info['segmentation'] for info in masks_info]
        scores = [info['stability_score'] for info in masks_info]
        
    else:
        # Targeted segmentation with prompts
        with torch.inference_mode(), torch.autocast(device_type=device, dtype=dtype):
            predictor.set_image(rgb_img)
            
            if prompt["type"] == "point":
                # Tek pozitif nokta
                import numpy as np
                pts = np.array([prompt["point"]])[None, ...]  # (1,1,2)
                labels = np.array([1])[None, ...]             # (1,1) 1=foreground
                print(f"   Point: {prompt['point']}")
                masks, scores, _ = predictor.predict(point_coords=pts, point_labels=labels)
            elif prompt["type"] == "box":
                import numpy as np
                # örnek: [x1,y1,x2,y2]
                box = np.array(prompt["box"])[None, ...]
                print(f"   Box: {prompt['box']}")
                masks, scores, _ = predictor.predict(box=box)
    
    inference_time = time.time() - start_time
    print(f"✅ Inference completed in {inference_time:.2f} seconds")
    print(f"🔍 Found {len(masks)} masks")
    
    if len(scores) > 0:
        print(f"📊 Scores range: {min(scores):.3f} - {max(scores):.3f}")
        print(f"📊 Top 5 scores: {sorted(scores, reverse=True)[:5]}")
    
    # GPU memory durumu (eğer CUDA kullanılıyorsa)
    if device == "cuda" and torch.cuda.is_available():
        print(f"🔍 GPU Memory after inference:")
        print(f"   Allocated: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
        print(f"   Reserved: {torch.cuda.memory_reserved() / 1024**3:.1f} GB")

    # Masks görselleştirme
    import numpy as np
    
    if len(masks) == 0:
        print("❌ No masks found!")
        return
    
    if len(masks) > 1:
        # Multiple masks - show top scoring ones
        print(f"🎨 Creating visualization with multiple masks...")
        
        # Sort by score and take top masks
        sorted_indices = np.argsort(scores)[::-1]
        top_count = min(5, len(masks))  # Show top 5 masks
        top_masks = [masks[sorted_indices[i]] for i in range(top_count)]
        top_scores = [scores[sorted_indices[i]] for i in range(top_count)]
        
        # Create colored overlay for multiple objects
        overlay = img.copy()
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]  # Different colors
        
        for i, (mask, score) in enumerate(zip(top_masks, top_scores)):
            if isinstance(mask, np.ndarray):
                m = (mask > 0).astype("uint8") * 255
            else:
                m = (mask > 0.5).astype("uint8") * 255
            color = colors[i % len(colors)]
            overlay[m == 255] = color
            print(f"   Mask {i+1}: score={score:.3f}, pixels={np.sum(m==255)}, color={color}")
        
        vis = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
    else:
        # Single mask
        print(f"🎨 Single mask with score: {scores[0]:.3f}")
        
        mask = masks[0]
        if isinstance(mask, np.ndarray):
            m = (mask > 0).astype("uint8") * 255
        else:
            m = (mask > 0.5).astype("uint8") * 255
        overlay = img.copy()
        overlay[m == 255] = (0, 255, 0)  # yeşil maske
        vis = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

    cv2.imwrite(str(out_p), vis)
    print(f"💾 Saved result to: {out_p}")
    
    # Özet bilgi
    print(f"\n📊 SUMMARY:")
    print(f"   Device used: {model_device}")
    print(f"   Model load time: {load_time:.2f}s")
    print(f"   Inference time: {inference_time:.2f}s")
    print(f"   Total time: {load_time + inference_time:.2f}s")

if __name__ == "__main__":
    main()
