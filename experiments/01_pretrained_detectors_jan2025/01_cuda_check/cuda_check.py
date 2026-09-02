"""AdSwapAI R&D, 2025-01-17: CUDA smoke test."""
import torch
from torch.cuda import memory_allocated, memory_reserved

# Tensor operation
device = torch.device('cuda:0')  # Explicitly select the GPU device
x = torch.rand(1024, 1024, device=device)  # Allocate a large tensor

# Check GPU memory usage
print("Allocated memory:", memory_allocated())
print("Reserved memory:", memory_reserved())
