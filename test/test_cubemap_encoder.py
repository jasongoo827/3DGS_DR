import torch
import torch.nn as nn
import sys
import os

# Add local submodule path for testing before installation if possible, 
# or rely on installation.
try:
    from submodules.cubemap_encoder.cubemap_encoder import CubeMapEncoder
except ImportError:
    print("Could not import CubeMapEncoder. Attempting to add path...")
    sys.path.append(os.path.join(os.getcwd(), "submodules/cubemap_encoder"))
    from cubemap_encoder import CubeMapEncoder

def test_cubemap_encoder():
    print("Initializing CubeMapEncoder...")
    device = torch.device("cuda")
    encoder = CubeMapEncoder(resolution=128).to(device)
    
    # 1. Forward Test
    print("Running Forward Pass...")
    N = 1000
    dirs = torch.nn.functional.normalize(torch.randn(N, 3, device=device), dim=1)
    colors = encoder(dirs)
    
    print(f"Output color shape: {colors.shape}")
    assert colors.shape == (N, 3)
    
    # 2. Gradient Test
    print("Running Backward Pass...")
    loss = colors.sum()
    loss.backward()
    
    print("Gradient check:", encoder.cubemap.grad is not None)
    if encoder.cubemap.grad is not None:
        print(f"Gradient shape: {encoder.cubemap.grad.shape}")
        print(f"Gradient stats: Min {encoder.cubemap.grad.min()}, Max {encoder.cubemap.grad.max()}")
        if encoder.cubemap.grad.abs().sum() == 0:
             print("WARNING: Gradient is all zeros! This might be correct if sampled texels were 0 and input was 0, but unlikely with random init.")
        else:
             print("Gradient flow confirmed.")

if __name__ == "__main__":
    test_cubemap_encoder()
