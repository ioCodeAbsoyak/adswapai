"""AdSwapAI R&D, 2025-03-13: NVML/torch GPU monitor with dynamic batch size."""

import torch
import time
import threading
import psutil
import numpy as np

# Try to import NVML for advanced GPU monitoring
try:
    import pynvml
    HAS_NVML = True
    print("NVML available - Advanced GPU monitoring enabled")
except ImportError:
    HAS_NVML = False
    print("NVML not found - Basic GPU monitoring will be used")

class GPUMonitor:
    """
    A class to monitor GPU usage and provide optimal batch size recommendations.
    This class can be used in any deep learning project to dynamically adjust batch size.
    """
    
    def __init__(self, device=0):
        """
        Initialize the GPU monitor.
        
        Args:
            device (int): GPU device index to monitor (default: 0)
        """
        self.device = device
        self.running = False
        self.current_memory_usage = 0
        self.current_gpu_utilization = 0
        self.memory_threshold = 0.75  # 75% memory usage threshold
        self.util_threshold_high = 0.8  # 80% is considered high GPU utilization
        self.util_threshold_low = 0.4   # 40% is considered low GPU utilization
        
        # Initialize NVML if available
        self.nvml_initialized = False
        
        if HAS_NVML and torch.cuda.is_available():
            try:
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(self.device)
                self.gpu_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                self.total_memory = self.gpu_info.total
                self.nvml_initialized = True
                print(f"GPU monitoring initialized for device {device}")
            except Exception as e:
                print(f"NVML initialization failed: {e}")
                self.nvml_initialized = False
        
        # Fallback to torch metrics
        if not self.nvml_initialized and torch.cuda.is_available():
            self.total_memory = torch.cuda.get_device_properties(self.device).total_memory
            print(f"Using basic torch metrics for GPU {device}")
    
    def start_monitoring(self):
        """Start the GPU monitoring thread."""
        if not torch.cuda.is_available():
            print("CUDA not available. Monitoring disabled.")
            return
            
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("GPU monitoring started")
    
    def stop_monitoring(self):
        """Stop the GPU monitoring thread."""
        self.running = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1.0)
        
        # Shutdown NVML if it was initialized
        if self.nvml_initialized:
            try:
                pynvml.nvmlShutdown()
                print("NVML shutdown complete")
            except:
                pass
    
    def _monitor_loop(self):
        """Internal monitoring loop that updates GPU metrics."""
        while self.running:
            if self.nvml_initialized:
                try:
                    # Get detailed GPU metrics with NVML
                    self.gpu_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                    self.current_memory_usage = self.gpu_info.used / self.gpu_info.total
                    
                    util_rates = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                    self.current_gpu_utilization = util_rates.gpu / 100.0
                except Exception:
                    # Fallback to basic torch metrics if error occurs
                    self.current_memory_usage = torch.cuda.memory_allocated(self.device) / self.total_memory
                    self.current_gpu_utilization = 0.5  # Make an educated guess
            else:
                # Basic torch metrics
                self.current_memory_usage = torch.cuda.memory_allocated(self.device) / self.total_memory
                self.current_gpu_utilization = 0.5  # Make an educated guess
            
            time.sleep(0.5)  # Update every half second
    
    def get_optimal_batch_size(self, current_batch_size, min_batch=1, max_batch=16):
        """
        Calculate the optimal batch size based on current GPU usage.
        
        Args:
            current_batch_size (int): The current batch size being used
            min_batch (int): Minimum allowable batch size (default: 1)
            max_batch (int): Maximum allowable batch size (default: 16)
            
        Returns:
            int: Recommended batch size
        """
        if not torch.cuda.is_available():
            return current_batch_size
            
        # Decrease batch size if GPU memory usage is too high
        if self.current_memory_usage > self.memory_threshold:
            return max(min_batch, current_batch_size - 1)
        
        # Increase batch size if GPU utilization is low and memory permits
        if self.current_gpu_utilization < self.util_threshold_low and self.current_memory_usage < 0.7:
            return min(max_batch, current_batch_size + 1)
        
        # If GPU utilization is very high, check other factors too
        if self.current_gpu_utilization > self.util_threshold_high:
            # Check CPU usage as well
            cpu_usage = psutil.cpu_percent() / 100.0
            
            # If both GPU and CPU are under high load, decrease batch size for balance
            if cpu_usage > 0.85:
                return max(min_batch, current_batch_size - 1)
        
        # Keep current batch size if no change needed
        return current_batch_size
    
    def get_metrics(self):
        """
        Get current GPU metrics.
        
        Returns:
            dict: Dictionary containing current GPU metrics
        """
        if self.nvml_initialized:
            memory_used = self.gpu_info.used / 1024**2
            memory_total = self.total_memory / 1024**2
        else:
            memory_used = torch.cuda.memory_allocated(self.device) / 1024**2
            memory_total = self.total_memory / 1024**2
            
        return {
            "memory_usage": self.current_memory_usage,
            "gpu_utilization": self.current_gpu_utilization,
            "memory_used_mb": memory_used,
            "memory_total_mb": memory_total,
        }
    
    @staticmethod
    def is_available():
        """Check if GPU monitoring is available on this system."""
        return torch.cuda.is_available()

# Basic test if run directly
if __name__ == "__main__":
    monitor = GPUMonitor()
    monitor.start_monitoring()
    
    try:
        # Example monitoring loop
        for i in range(10):
            metrics = monitor.get_metrics()
            print(f"GPU Memory: {metrics['memory_usage']*100:.1f}%, " +
                  f"Utilization: {metrics['gpu_utilization']*100:.1f}%")
            time.sleep(1)
    finally:
        monitor.stop_monitoring()