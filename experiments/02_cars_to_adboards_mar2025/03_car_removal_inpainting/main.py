"""AdSwapAI R&D, 2025-03-13: argparse CLI wiring GPUMonitor + CarRemover."""

import os
import argparse
from gpu_monitor import GPUMonitor
from car_remover import CarRemover
import torch

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Remove vehicles from video using AI")
    
    parser.add_argument("input_video", type=str, help="Path to input video file")
    parser.add_argument("--output", "-o", type=str, help="Path to output video file")
    parser.add_argument("--batch-size", "-b", type=int, default=2, 
                        help="Initial batch size (will be adjusted dynamically if possible)")
    parser.add_argument("--disable-dynamic-batch", action="store_true", 
                        help="Disable dynamic batch size adjustment")
    parser.add_argument("--no-gpu", action="store_true", 
                        help="Force CPU usage even if GPU is available")
    
    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_args()
    
    # Check if input video exists
    if not os.path.exists(args.input_video):
        print(f"Error: Input video not found: {args.input_video}")
        return
    
    # Set device
    device = torch.device('cpu') if args.no_gpu else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize GPU monitor if needed
    gpu_monitor = None
    if not args.disable_dynamic_batch and device.type == 'cuda':
        gpu_monitor = GPUMonitor()
        gpu_monitor.start_monitoring()
    
    try:
        # Initialize car remover
        car_remover = CarRemover(device=device)
        
        # Process video
        output_path = car_remover.process_video(
            video_path=args.input_video,
            output_path=args.output,
            batch_size=args.batch_size,
            gpu_monitor=gpu_monitor
        )
        
        print(f"Vehicle removal complete! Output saved to: {output_path}")
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        # Clean up GPU monitor
        if gpu_monitor is not None:
            gpu_monitor.stop_monitoring()

if __name__ == "__main__":
    main()