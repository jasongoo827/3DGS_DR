import torch
import sys
import os
sys.path.append(os.getcwd())
try: 
    from scene.gaussian_model import GaussianModel
except ImportError:
    sys.path.append(".")
    from scene.gaussian_model import GaussianModel

def test_gaussian_model():
    print("Initializing GaussianModel...")
    gm = GaussianModel(sh_degree=3)
    
    # Mock data initialization
    num_points = 100
    gm._xyz = torch.nn.Parameter(torch.randn(num_points, 3).cuda())
    gm._features_dc = torch.nn.Parameter(torch.randn(num_points, 1, 3).cuda())
    gm._features_rest = torch.nn.Parameter(torch.randn(num_points, 15, 3).cuda())
    gm._scaling = torch.nn.Parameter(torch.rand(num_points, 3).cuda())
    gm._rotation = torch.nn.Parameter(torch.rand(num_points, 4).cuda())
    gm._opacity = torch.nn.Parameter(torch.rand(num_points, 1).cuda())
    
    # Init reflection strength manually for test since we didn't call create_from_pcd
    gm._reflection_strength = torch.nn.Parameter(torch.zeros(num_points, 1).cuda())
    
    # 1. Check Normal Calculation
    print("Testing get_normal()...")
    normals = gm.get_normal()
    print(f"Normals shape: {normals.shape}")
    assert normals.shape == (num_points, 3)
    
    norms = torch.norm(normals, dim=1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-5):
        print("Normals not normalized! Norms:", norms)
    else:
        print("Normals are normalized.")
    
    # 2. Check Reflection Strength
    print("Testing get_reflection_strength...")
    ref = gm.get_reflection_strength
    print(f"Refl strength shape: {ref.shape}")
    assert ref.shape == (num_points, 1)
    # sigmoid(0) = 0.5
    assert torch.all(ref >= 0) and torch.all(ref <= 1)
    
    # 3. Check View Dependent Flipping
    print("Testing Normal Flipping...")
    n0 = normals[0]
    v_same = n0 # dot > 0
    
    view_dirs = v_same.unsqueeze(0).repeat(num_points, 1) # (100, 3)
    
    normals_flipped = gm.get_normal(view_dirs=view_dirs)
    n0_flipped = normals_flipped[0]
    
    dot_after = torch.dot(n0_flipped, v_same)
    print(f"Dot before: {torch.dot(n0, v_same).item():.3f}, Dot after: {dot_after.item():.3f}")
    
    if dot_after < 0:
        print("Flipping logic works.")
    else:
        print("Flipping logic FAILED.")
        exit(1)

    print("\nVerification Successful!")

if __name__ == "__main__":
    test_gaussian_model()
