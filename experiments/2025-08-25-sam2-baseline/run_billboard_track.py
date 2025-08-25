import os
import cv2
import torch
import yaml
import time
import numpy as np
from pathlib import Path
from datetime import datetime

# SAM2 imports (from submodule)
import sys
ROOT = Path(__file__).resolve().parents[2]  # adswapai/
sys.path.insert(0, str(ROOT / "external" / "sam2"))
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_video_predictor import SAM2VideoPredictor

def load_cfg(fp):
    with open(fp, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def check_device_info():
    """Check device information and display GPU specs if available"""
    print("🔍 Device Information:")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Set memory allocation strategy
        torch.cuda.empty_cache()
        print(f"   Using device: cuda")
        return "cuda"
    else:
        print(f"   Using device: cpu")
        return "cpu"

def extract_video_frames(video_path, frame_dir):
    """Extract frames from video"""
    print(f"📹 Extracting frames from {video_path}")
    
    # Create frame directory
    frame_dir = Path(frame_dir)
    frame_dir.mkdir(exist_ok=True)
    
    # Clear existing frames
    for f in frame_dir.glob("*.jpg"):
        f.unlink()
    
    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"   Video FPS: {fps}")
    print(f"   Total frames: {frame_count}")
    
    frame_paths = []
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_path = frame_dir / f"{i:06d}.jpg"  # SAM2 expects just numbers
        cv2.imwrite(str(frame_path), frame)
        frame_paths.append(str(frame_path))
        
        if i % 100 == 0:
            print(f"   Extracted frame {i}/{frame_count}")
    
    cap.release()
    print(f"✅ Extracted {len(frame_paths)} frames")
    return frame_paths, fps

def detect_billboards(predictor, frame_paths, inference_state):
    """Detect billboards/advertising boards in the video"""
    print(f"🎯 Detecting billboards in first frame...")
    
    # Load first frame
    first_frame = cv2.imread(frame_paths[0])
    rgb_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    
    height, width = rgb_frame.shape[:2]
    print(f"   Frame size: {width}x{height}")
    
    # Strategic points for billboard/advertising board detection
    # Focus on stadium perimeter areas where advertising is typically placed
    billboard_detection_points = [
        # Top stadium billboards (background)
        [width//4, height//6],          # Top left billboard area
        [width//2, height//6],          # Top center billboard area
        [3*width//4, height//6],        # Top right billboard area
        
        # Side advertising boards
        [width//8, height//3],          # Left side upper
        [7*width//8, height//3],        # Right side upper
        [width//10, height//2],         # Far left center
        [9*width//10, height//2],       # Far right center
        
        # Bottom/sideline advertising
        [width//4, 4*height//5],        # Bottom left
        [width//2, 4*height//5],        # Bottom center
        [3*width//4, 4*height//5],      # Bottom right
        
        # Corner billboard areas
        [width//6, height//4],          # Top left corner
        [5*width//6, height//4],        # Top right corner
    ]
    
    print(f"   Testing {len(billboard_detection_points)} billboard positions")
    
    detected_object_ids = []
    detected_masks = []
    
    for i, point in enumerate(billboard_detection_points):
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
            
            # Billboard detection criteria - much more relaxed for any visible objects
            if 0.01 <= area_percentage <= 50.0:  # Between 0.01% and 50% of frame (very broad range)
                detected_object_ids.append(i)
                detected_masks.append(mask)
                print(f"   ✅ Billboard {i} at {point}, area: {area_percentage:.2f}%")
            else:
                print(f"   ❌ Billboard {i} rejected, area: {area_percentage:.2f}%")
                
        except Exception as e:
            print(f"   ❌ Failed at billboard position {i} {point}: {e}")
    
    # If no billboards detected, try alternative approach
    if len(detected_object_ids) == 0:
        print("🔧 No billboards detected, trying alternative positions...")
        alternative_points = [
            [width//8, height//8],      # Top left
            [7*width//8, height//8],    # Top right
            [width//2, height//8],      # Top center
            [width//8, 7*height//8],    # Bottom left
            [7*width//8, 7*height//8],  # Bottom right
        ]
        
        for i, point in enumerate(alternative_points):
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
                
                # Very relaxed criteria for alternative positions
                if 0.01 <= area_percentage <= 50:
                    detected_object_ids.append(obj_id)
                    detected_masks.append(mask)
                    print(f"   🔧 Alternative billboard {i} at {point}, area: {area_percentage:.2f}%")
                    
            except Exception as e:
                print(f"   ❌ Alternative billboard detection failed: {e}")
    
    print(f"🎯 Successfully detected {len(detected_object_ids)} billboards for tracking")
    return detected_object_ids

def propagate_masks(predictor, inference_state, total_frames, detected_object_ids):
    """Propagate masks through video frames"""
    print(f"🔄 Propagating masks through {total_frames} frames...")
    
    start_time = time.time()
    
    # Run propagation
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    
    end_time = time.time()
    print(f"✅ Mask propagation completed in {end_time - start_time:.2f}s")
    
    return video_segments

def create_output_video(frame_paths, video_segments, output_path, fps, detected_object_ids):
    """Create output video with billboard masks"""
    print(f"🎬 Creating output video: {output_path}")
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(frame_paths[0])
    height, width, channels = first_frame.shape
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Color palette for different billboards
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 128, 0),  # Orange
        (128, 0, 255),  # Purple
        (0, 255, 128),  # Light green
        (255, 128, 128),# Light red
        (128, 255, 128),# Light green
        (128, 128, 255),# Light blue
    ]
    
    total_frames = len(frame_paths)
    
    for frame_idx in range(total_frames):
        # Read frame
        frame = cv2.imread(frame_paths[frame_idx])
        original_frame = frame.copy()
        
        # Create overlay
        overlay = frame.copy()
        
        # Apply masks for this frame
        if frame_idx in video_segments:
            for obj_id in detected_object_ids:
                if obj_id in video_segments[frame_idx]:
                    mask = video_segments[frame_idx][obj_id]
                    
                    # Handle 3D masks by removing batch dimension
                    if len(mask.shape) == 3 and mask.shape[0] == 1:
                        mask = mask.squeeze(0)
                    
                    # Ensure mask is correct shape and type
                    if isinstance(mask, np.ndarray):
                        # Make sure mask has correct dimensions
                        if mask.shape[:2] == frame.shape[:2]:  # H, W should match
                            # Convert to boolean mask
                            bool_mask = mask.astype(bool)
                            # Apply color to mask regions
                            if obj_id < len(colors):
                                color = colors[obj_id]
                                overlay[bool_mask] = color
                        else:
                            print(f"⚠️ Mask shape {mask.shape} doesn't match frame shape {frame.shape}")
                    else:
                        print(f"⚠️ Invalid mask type for billboard {obj_id}")
            
            # Blend original and overlay
            frame = cv2.addWeighted(original_frame, 0.7, overlay, 0.3, 0)
        
        out.write(frame)
        
        if frame_idx % 50 == 0 or frame_idx == total_frames - 1:
            print(f"   Wrote frame {frame_idx}/{total_frames}")
    
    out.release()
    print(f"✅ Output video saved: {output_path}")
    
    # Get file size
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    print(f"   Video file size: {file_size:.1f} MB")

def main():
    # Check device
    device = check_device_info()
    
    # Load config
    config_path = "config_video.yaml"
    config = load_cfg(config_path)
    print(f"📋 Loaded config from {config_path}")
    
    # Paths
    video_path = Path(config["video_path"])
    sam2_checkpoint = Path(config["checkpoint"])  # config uses "checkpoint" not "sam2_checkpoint"
    model_cfg = config["model_cfg"]
    frame_dir = Path("temp_frames")
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    
    print(f"🎥 Input video: {video_path}")
    
    # Extract frames
    frame_paths, fps = extract_video_frames(video_path, frame_dir)
    
    # Initialize SAM2 predictor
    print(f"🤖 Loading SAM2 model...")
    print(f"   Checkpoint: {sam2_checkpoint}")
    print(f"   Config: {model_cfg}")
    
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    
    # Initialize video predictor
    inference_state = predictor.init_state(video_path=str(frame_dir))
    
    # Detect billboards
    detected_object_ids = detect_billboards(predictor, frame_paths, inference_state)
    
    if len(detected_object_ids) == 0:
        print("❌ No billboards detected! Exiting...")
        return
    
    # Propagate masks
    video_segments = propagate_masks(predictor, inference_state, len(frame_paths), detected_object_ids)
    
    # Create output video
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = outputs_dir / f"billboard_output_{timestamp}.mp4"
    
    create_output_video(frame_paths, video_segments, output_path, fps, detected_object_ids)
    
    # Cleanup
    print(f"🧹 Cleaning up temporary frames...")
    for frame_path in frame_paths:
        os.remove(frame_path)
    frame_dir.rmdir()
    
    # Summary
    print(f"\n📊 SUMMARY:")
    print(f"   Input video: {video_path}")
    print(f"   Output video: {output_path}")
    print(f"   Billboards tracked: {len(detected_object_ids)}")
    print(f"   Total frames: {len(frame_paths)}")
    print(f"   FPS: {fps:.2f}")
    print(f"   Device used: {device}")
    print(f"   Video size: ({frame_paths and cv2.imread(frame_paths[0]).shape[1] or 0}, {frame_paths and cv2.imread(frame_paths[0]).shape[0] or 0})")
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    print(f"   Output file size: {file_size:.1f} MB")
    print(f"✅ SUCCESS: Billboard tracking video saved to {output_path}")

if __name__ == "__main__":
    main()
