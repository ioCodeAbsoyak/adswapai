import os
import cv2
import torch
import yaml
import time
import numpy as np
from pathlib import Path
from datetime import datetime            # Billboard detection criteria - larger than players but not entire background
            if 0.5 <= area_percentage <= 15.0:  # Between 0.5% and 15% of frame (billboards are larger)
                detected_object_ids.append(i)
                detected_masks.append(mask)
                print(f"   ✅ Billboard {i} at {point}, area: {area_percentage:.2f}%")
            else:
                print(f"   ❌ Billboard {i} rejected, area: {area_percentage:.2f}%")M2 imports (from submodule)
import sys
ROOT = Path(__file__).resolve().parents[2]  # adswapai/
sys.path.insert(0, str(ROOT / "external" / "sam2"))
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_video_predictor import SAM2VideoPredictor

def load_cfg(fp):
    with open(fp, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def check_device_info():
    """Check GPU/CPU status"""
    print("=" * 50)
    print("VIDEO TRACKING - DEVICE INFORMATION")
    print("=" * 50)
    
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

def extract_frames_from_video(video_path, output_dir):
    """Extract frames from video"""
    print(f"📹 Extracting frames from: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📊 Video info: {total_frames} frames, {fps:.2f} FPS, {width}x{height}")
    
    # Create frames directory
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    frame_paths = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # SAM2 format: file names starting with numbers only
        frame_path = frames_dir / f"{frame_idx:06d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        frame_paths.append(str(frame_path))
        frame_idx += 1
        
        if frame_idx % 100 == 0:
            print(f"   Extracted {frame_idx}/{total_frames} frames...")
    
    cap.release()
    print(f"✅ Extracted {len(frame_paths)} frames to {frames_dir}")
    
    return frame_paths, fps, (width, height)

def detect_players_in_first_frame(predictor, frame_paths, device):
    """Detect players in first frame using manual point selection"""
    print(f"🎯 Detecting players in first frame...")
    
    # Load first frame
    first_frame = cv2.imread(frame_paths[0])
    rgb_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    
    # Initialize video predictor
    inference_state = predictor.init_state(video_path=str(Path(frame_paths[0]).parent))
    
    height, width = rgb_frame.shape[:2]
    print(f"   Frame size: {width}x{height}")
    
    # Manually selected points where billboards/advertising boards are likely to be
    # Focus on stadium perimeter areas where advertising is typically placed
    manual_billboard_points = [
        # Top stadium billboards (background)
        [width//4, height//8],          # Top left billboard
        [width//2, height//8],          # Top center billboard
        [3*width//4, height//8],        # Top right billboard
        
        # Side advertising boards
        [width//10, height//3],         # Left side upper
        [9*width//10, height//3],       # Right side upper
        [width//8, height//2],          # Left side center
        [7*width//8, height//2],        # Right side center
        
        # Bottom/sideline advertising
        [width//4, 7*height//8],        # Bottom left
        [width//2, 7*height//8],        # Bottom center
        [3*width//4, 7*height//8],      # Bottom right
    ]
    
    print(f"   Testing {len(manual_billboard_points)} strategic billboard positions")
    
    detected_object_ids = []
    detected_masks = []
    
    for i, point in enumerate(manual_billboard_points):
        try:
            points = np.array([point], dtype=np.float32)
            labels = np.array([1], np.int32)  # 1 = foreground
            
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=i,
                points=points,
                labels=labels,
            )
            
            # Get mask
            mask = (out_mask_logits[0] > 0.0).cpu().numpy()
            mask_area = np.sum(mask)
            total_area = height * width
            area_percentage = (mask_area / total_area) * 100
            
            # Very relaxed criteria - just avoid full background
            if 0.1 <= area_percentage <= 8.0:  # Between 0.1% and 8% of frame
                detected_object_ids.append(i)
                detected_masks.append(mask)
                print(f"   ✅ Player {i} at {point}, area: {area_percentage:.2f}%")
            else:
                print(f"   ❌ Point {i} at {point}, area: {area_percentage:.2f}% (outside 0.1%-8%)")
                
        except Exception as e:
            print(f"   ❌ Failed at point {i} {point}: {e}")
    
    # If still no detection, force some basic detections
    if len(detected_object_ids) == 0:
        print("� Forcing basic detections...")
        basic_points = [
            [width//2, height//2],      # Center
            [width//3, height//2],      # Left
            [2*width//3, height//2],    # Right
        ]
        
        for i, point in enumerate(basic_points):
            try:
                points = np.array([point], dtype=np.float32)
                labels = np.array([1], np.int32)
                
                obj_id = 100 + i  # Use different IDs
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=obj_id,
                    points=points,
                    labels=labels,
                )
                
                mask = (out_mask_logits[0] > 0.0).cpu().numpy()
                mask_area = np.sum(mask)
                area_percentage = (mask_area / total_area) * 100
                
                # Very relaxed - any reasonable size
                if 0.05 <= area_percentage <= 20:
                    detected_object_ids.append(obj_id)
                    detected_masks.append(mask)
                    print(f"   🔧 Forced detection {i} at {point}, area: {area_percentage:.2f}%")
                    
            except Exception as e:
                print(f"   ❌ Forced detection failed: {e}")
    
    print(f"🎯 Successfully detected {len(detected_object_ids)} objects for tracking")
    return inference_state, detected_object_ids, detected_masks

def propagate_masks_through_video(predictor, inference_state, frame_paths):
    """Propagate masks through entire video"""
    print(f"🚀 Propagating masks through {len(frame_paths)} frames...")
    
    start_time = time.time()
    
    # Run propagation for all frames
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        
        if out_frame_idx % 50 == 0:
            print(f"   Processed frame {out_frame_idx}/{len(frame_paths)-1}")
    
    propagation_time = time.time() - start_time
    print(f"✅ Propagation completed in {propagation_time:.2f} seconds")
    
    return video_segments

def create_output_video(frame_paths, video_segments, fps, frame_size, output_path):
    """Create output video with tracked masks"""
    print(f"🎬 Creating output video: {output_path}")
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)
    
    if not out.isOpened():
        print(f"❌ Failed to open video writer for {output_path}")
        return
    
    # Colors (different for each player)
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Red  
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 128, 0),  # Orange
        (128, 255, 0),  # Lime
    ]
    
    total_frames = len(frame_paths)
    
    for frame_idx, frame_path in enumerate(frame_paths):
        frame = cv2.imread(frame_path)
        
        if frame is None:
            print(f"⚠️ Could not read frame {frame_idx}: {frame_path}")
            continue
            
        original_frame = frame.copy()
        
        if frame_idx in video_segments:
            # This frame has masks
            overlay = frame.copy()
            
            for obj_id, mask in video_segments[frame_idx].items():
                if obj_id < len(colors):
                    color = colors[obj_id]
                    
                    # Ensure mask is correct shape and type
                    if isinstance(mask, np.ndarray):
                        # Handle 3D masks by removing batch dimension
                        if len(mask.shape) == 3 and mask.shape[0] == 1:
                            mask = mask.squeeze(0)
                        
                        # Make sure mask has correct dimensions
                        if mask.shape[:2] == frame.shape[:2]:  # H, W should match
                            # Convert to boolean mask
                            bool_mask = mask.astype(bool)
                            # Apply color to mask regions
                            overlay[bool_mask] = color
                        else:
                            print(f"⚠️ Mask shape {mask.shape} doesn't match frame shape {frame.shape}")
                    else:
                        print(f"⚠️ Invalid mask type for object {obj_id}")
            
            # Blend original and overlay
            frame = cv2.addWeighted(original_frame, 0.7, overlay, 0.3, 0)
        
        out.write(frame)
        
        if frame_idx % 50 == 0 or frame_idx == total_frames - 1:
            print(f"   Wrote frame {frame_idx}/{total_frames}")
    
    out.release()
    print(f"✅ Output video saved: {output_path}")
    
    # Verify file was created
    if Path(output_path).exists():
        file_size = Path(output_path).stat().st_size
        print(f"   Video file size: {file_size / (1024*1024):.1f} MB")
    else:
        print(f"❌ Video file was not created successfully")

def main():
    # Show device information
    check_device_info()
    
    cfg = load_cfg("config_video.yaml")
    
    video_path = cfg["video_path"]
    checkpoint = cfg["checkpoint"]
    model_cfg = cfg["model_cfg"]
    device = cfg.get("device", "cuda")
    
    # Device check
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA requested but not available, falling back to CPU")
        device = "cpu"
    
    print(f"\n🎯 Using device: {device}")
    
    # Output path with timestamp (YYYYMMDD_HHMMSS format)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("./outputs")
    output_video = output_dir / f"output_{timestamp}.mp4"
    output_dir.mkdir(exist_ok=True)
    
    print(f"📁 Output video will be: {output_video}")
    
    # Load video predictor
    print("\n📥 Loading SAM2 video predictor...")
    start_time = time.time()
    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)
    load_time = time.time() - start_time
    print(f"✅ Model loaded in {load_time:.2f} seconds")
    
    # Extract frames
    frame_paths, fps, frame_size = extract_frames_from_video(video_path, output_dir)
    
    # Detect players in first frame
    inference_state, detected_object_ids, detected_masks = detect_players_in_first_frame(predictor, frame_paths, device)
    
    if len(detected_object_ids) == 0:
        print("⚠️ No objects detected, but proceeding with video creation...")
        # Create video without masks
        video_segments = {}
    else:
        # Track masks through video
        video_segments = propagate_masks_through_video(predictor, inference_state, frame_paths)
    
    # Always create output video
    print(f"\n🎬 Creating output video...")
    create_output_video(frame_paths, video_segments, fps, frame_size, output_video)
    
    # Clean GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Clean up frames directory to save space
    frames_dir = output_dir / "frames"
    if frames_dir.exists():
        print(f"🧹 Cleaning up temporary frames...")
        import shutil
        shutil.rmtree(frames_dir)
    
    # Summary
    print(f"\n📊 SUMMARY:")
    print(f"   Input video: {video_path}")
    print(f"   Output video: {output_video}")
    print(f"   Objects tracked: {len(detected_object_ids)}")
    print(f"   Total frames: {len(frame_paths)}")
    print(f"   FPS: {fps:.2f}")
    print(f"   Device used: {device}")
    print(f"   Video size: {frame_size}")
    
    if output_video.exists():
        file_size_mb = output_video.stat().st_size / (1024 * 1024)
        print(f"   Output file size: {file_size_mb:.1f} MB")
        print(f"✅ SUCCESS: Video saved to {output_video}")
    else:
        print(f"❌ ERROR: Video file was not created!")
    
    return str(output_video)

if __name__ == "__main__":
    main()
