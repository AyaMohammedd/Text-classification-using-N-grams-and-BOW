import torch
import time

# Check MPS availability
print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built with CUDA: {torch.backends.mps.is_built()}")

# Create a test tensor on MPS
try:
    device = torch.device('mps')
    x = torch.randn(10000, 10000, device=device)
    y = torch.randn(10000, 10000, device=device)
    
    # Perform a GPU operation
    start_time = time.time()
    z = torch.matmul(x, y)
    end_time = time.time()
    
    print(f"\nGPU test successful!")
    print(f"Matrix multiplication time: {end_time - start_time:.4f} seconds")
    print(f"Result tensor on: {z.device}")
    
except Exception as e:
    print(f"\nError using MPS: {e}")
    print("Falling back to CPU...")
    
    device = torch.device('cpu')
    x = torch.randn(10000, 10000, device=device)
    y = torch.randn(10000, 10000, device=device)
    
    start_time = time.time()
    z = torch.matmul(x, y)
    end_time = time.time()
    
    print(f"\nCPU test successful!")
    print(f"Matrix multiplication time: {end_time - start_time:.4f} seconds")
