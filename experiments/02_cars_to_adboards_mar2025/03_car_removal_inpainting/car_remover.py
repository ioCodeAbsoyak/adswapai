"""AdSwapAI R&D, 2025-03-13: batched Mask R-CNN car removal with temporal smoothing (class CarRemover)."""

import torch
import torchvision
import cv2
import numpy as np
import time
import os
from tqdm import tqdm
import threading

class CarRemover:
    """
    Class to detect and remove vehicles from video using deep learning and inpainting.
    Uses the MaskRCNN model for detection and OpenCV for inpainting.
    """
    
    def __init__(self, device=None):
        """
        Initialize the car remover with necessary models and configurations.
        
        Args:
            device: PyTorch device to use (defaults to CUDA if available)
        """
        # Set up device
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Configure CUDA performance optimizations
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Vehicle classes in COCO dataset
        self.VEHICLE_CLASSES = [3, 8, 6, 4]  # car, truck, bus, motorcycle
        self.CONFIDENCE_THRESHOLD = 0.70
        
        # Load model
        print("Loading MaskRCNN model...")
        weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights).to(self.device)
        self.model.eval()
        print("Model loaded successfully")
    
    def process_batch(self, frames, prev_masks=None, prev_inpainted=None):
        """
        Process a batch of frames to remove vehicles.
        
        Args:
            frames: List of frames to process
            prev_masks: Previous frame masks for temporal consistency
            prev_inpainted: Previous inpainted frames for temporal smoothing
            
        Returns:
            Tuple of (inpainted_frames, new_masks, new_inpainted)
        """
        if not frames:
            return [], [], []
            
        batch_size = len(frames)
        frame_height, frame_width = frames[0].shape[:2]
        
        # Prepare tensors
        input_tensors = []
        for frame in frames:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = torchvision.transforms.functional.to_tensor(frame_rgb)
            input_tensors.append(frame_tensor)
        
        input_batch = torch.stack(input_tensors).to(self.device)
        
        # Model inference
        with torch.no_grad():
            outputs = self.model(input_batch)
        
        # Initialize results
        inpainted_frames = []
        new_masks = []
        new_inpainted = []
        
        # Process each frame
        for i, frame in enumerate(frames):
            # Extract mask
            vehicle_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
            
            # Process vehicle detections
            detections = outputs[i]
            
            for j in range(len(detections["boxes"])):
                score = detections["scores"][j].item()
                label = detections["labels"][j].item()
                
                if score > self.CONFIDENCE_THRESHOLD and label in self.VEHICLE_CLASSES:
                    mask = detections["masks"][j, 0].cpu().numpy()
                    mask_binary = (mask > 0.5).astype(np.uint8)
                    
                    box = detections["boxes"][j].cpu().numpy().astype(np.int32)
                    x1, y1, x2, y2 = box
                    
                    # Add extra space for shadow areas
                    extra_bottom = int((y2 - y1) * 0.2)
                    y2_extended = min(y2 + extra_bottom, frame_height)
                    
                    # Add to vehicle mask
                    vehicle_mask = cv2.bitwise_or(vehicle_mask, mask_binary)
                    
                    # Add shadow region
                    shadow_region = np.zeros_like(vehicle_mask)
                    shadow_region[y2:y2_extended, max(0, x1-10):min(x2+10, frame_width)] = 1
                    vehicle_mask = cv2.bitwise_or(vehicle_mask, shadow_region)
            
            # Clean up mask
            kernel = np.ones((7, 7), np.uint8)
            vehicle_mask = cv2.dilate(vehicle_mask, kernel, iterations=1)
            vehicle_mask = cv2.morphologyEx(vehicle_mask, cv2.MORPH_CLOSE, kernel)
            
            # Use previous mask for consistency if available
            if prev_masks is not None and i < len(prev_masks) and prev_masks[i] is not None:
                vehicle_mask = cv2.bitwise_or(vehicle_mask, prev_masks[i])
            
            # Prepare for inpainting
            inpaint_mask = (vehicle_mask * 255).astype(np.uint8)
            
            # Two-stage inpainting
            inpainted_ns = cv2.inpaint(frame, inpaint_mask, 7, cv2.INPAINT_NS)
            inpainted_result = cv2.inpaint(inpainted_ns, inpaint_mask, 3, cv2.INPAINT_TELEA)
            
            # Temporal smoothing
            if prev_inpainted is not None and i < len(prev_inpainted) and prev_inpainted[i] is not None:
                # Blend with previous frame in masked regions only
                # (outside the mask the blend amount is 0, so those pixels keep the current frame)
                mask_3ch = cv2.merge([inpaint_mask, inpaint_mask, inpaint_mask]) / 255.0
                blend_factor = 0.7  # Weight for current frame inside the mask
                blend_amount = mask_3ch * (1.0 - blend_factor)

                inpainted_result = inpainted_result.astype(float) * (1.0 - blend_amount) + \
                                   prev_inpainted[i].astype(float) * blend_amount
                inpainted_result = inpainted_result.astype(np.uint8)
            
            # Save results
            inpainted_frames.append(inpainted_result)
            new_masks.append(vehicle_mask)
            new_inpainted.append(inpainted_result)
        
        return inpainted_frames, new_masks, new_inpainted
    
    def process_video(self, video_path, output_path=None, batch_size=2, gpu_monitor=None):
        """
        Process an entire video to remove vehicles.
        
        Args:
            video_path: Path to input video file
            output_path: Path to output video file (defaults to "CarsRemoved_" + input filename)
            batch_size: Initial batch size
            gpu_monitor: Optional GPUMonitor instance for dynamic batch sizing
            
        Returns:
            Path to the output video file
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {frame_width}x{frame_height}, {fps} FPS, {total_frames} frames")
        
        # Set output path
        if output_path is None:
            output_dir = os.path.dirname(video_path)
            output_name = f"CarsRemoved_{os.path.basename(video_path)}"
            output_path = os.path.join(output_dir, output_name)
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        # Performance tracking variables
        start_time = time.time()
        frames_processed = 0
        batch_history = []
        fps_history = []
        
        # Processing variables
        prev_masks = []
        prev_inpainted = []
        current_batch_size = batch_size
        
        # Progress bar
        pbar = tqdm(total=total_frames, desc="Processing")
        last_batch_update_time = time.time()
        
        try:
            while frames_processed < total_frames:
                batch_start_time = time.time()
                
                # Update batch size if we have a GPU monitor
                if gpu_monitor is not None and time.time() - last_batch_update_time > 3.0:
                    new_batch_size = gpu_monitor.get_optimal_batch_size(current_batch_size)
                    if new_batch_size != current_batch_size:
                        print(f"\nBatch size updated: {current_batch_size} -> {new_batch_size}")
                        if gpu_monitor is not None:
                            metrics = gpu_monitor.get_metrics()
                            print(f"GPU Memory: {metrics['memory_usage']*100:.1f}%, " +
                                  f"GPU Utilization: {metrics['gpu_utilization']*100:.1f}%")
                        
                        current_batch_size = new_batch_size
                        
                        # Update history variables
                        if len(prev_masks) != current_batch_size:
                            prev_masks = [None] * current_batch_size
                        if len(prev_inpainted) != current_batch_size:
                            prev_inpainted = [None] * current_batch_size
                    
                    last_batch_update_time = time.time()
                
                # Collect frames for the batch
                frames = []
                for _ in range(current_batch_size):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)
                
                if not frames:
                    break
                
                # Process batch
                inpainted_frames, new_masks, new_inpainted = self.process_batch(
                    frames, 
                    prev_masks=prev_masks[:len(frames)], 
                    prev_inpainted=prev_inpainted[:len(frames)]
                )
                
                # Update previous data for next batch
                prev_masks = new_masks
                prev_inpainted = new_inpainted
                
                # Write to output video
                for frame in inpainted_frames:
                    out.write(frame)
                
                # Update progress
                frames_processed += len(frames)
                pbar.update(len(frames))
                
                # Calculate batch FPS
                batch_time = time.time() - batch_start_time
                batch_fps = len(frames) / batch_time
                fps_history.append(batch_fps)
                if len(fps_history) > 20:
                    fps_history.pop(0)
                
                # Record batch history
                batch_history.append(current_batch_size)
                if len(batch_history) > 20:
                    batch_history.pop(0)
                
                # Average FPS
                avg_fps = sum(fps_history) / len(fps_history)
                
                # Update progress bar
                postfix = {
                    'Batch': current_batch_size,
                    'FPS': f"{avg_fps:.1f}",
                    'Processed': f"{frames_processed}/{total_frames}"
                }
                
                # Add GPU metrics if available
                if gpu_monitor is not None:
                    metrics = gpu_monitor.get_metrics()
                    postfix['GPU Mem'] = f"{metrics['memory_usage']*100:.1f}%"
                    postfix['GPU Util'] = f"{metrics['gpu_utilization']*100:.1f}%"
                    
                pbar.set_postfix(postfix)
        
        finally:
            # Clean up
            pbar.close()
            cap.release()
            out.release()
        
        # Final statistics
        total_time = time.time() - start_time
        overall_fps = frames_processed / total_time
        print(f"\nProcessed {frames_processed} frames in {total_time:.2f} seconds ({overall_fps:.2f} FPS)")
        
        if batch_history:
            print(f"Average batch size: {sum(batch_history)/len(batch_history):.1f}")
        
        print(f"Output saved to: {output_path}")
        return output_path