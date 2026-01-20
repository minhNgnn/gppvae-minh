"""
Test SM kernel with use_angle_input=True vs False.
Diagnose why gradients may not flow when use_angle_input=True.
"""
import torch
import numpy as np
import sys
sys.path.insert(0, 'GPPVAE/pysrc/coil100')

from kernels import SMCircleKernel, wrapped_lag_distance, get_angles_degrees

Q = 18
angles = get_angles_degrees(Q)

print("=" * 70)
print("SM Kernel: use_angle_input=False vs True")
print("=" * 70)

# Test 1: Compare distance matrices
print("\n=== Distance Matrix Comparison ===")
D_wrapped = wrapped_lag_distance(angles)
diff_raw = torch.abs(angles.unsqueeze(1) - angles.unsqueeze(0))
D_angle = diff_raw / 360.0

print(f"Wrapped lag distance (use_angle_input=False):")
print(f"  D[0,1] (20°): {D_wrapped[0,1].item():.2f}")
print(f"  D[0,9] (180°): {D_wrapped[0,9].item():.2f}")
print(f"  D[0,17] (340°→20° wrapped): {D_wrapped[0,17].item():.2f}")
print(f"  Range: [{D_wrapped.min().item():.1f}, {D_wrapped.max().item():.1f}]")

print(f"\nNormalized angle diff (use_angle_input=True):")
print(f"  D[0,1] (20°): {D_angle[0,1].item():.4f}")
print(f"  D[0,9] (180°): {D_angle[0,9].item():.4f}")
print(f"  D[0,17] (340°): {D_angle[0,17].item():.4f}")
print(f"  Range: [{D_angle.min().item():.4f}, {D_angle.max().item():.4f}]")

# Test 2: Exp term with default variance
print("\n=== Exp Term with Default Variance (0.0001) ===")
var = 0.0001

print("use_angle_input=False (D in degrees 0-180):")
for i, d in [(1, D_wrapped[0,1].item()), (9, D_wrapped[0,9].item())]:
    exp_val = np.exp(-2 * np.pi**2 * var * d**2)
    print(f"  D={d:.1f}° → exp = {exp_val:.6f}")

print("\nuse_angle_input=True (D normalized 0-1):")
for i, d in [(1, D_angle[0,1].item()), (9, D_angle[0,9].item())]:
    exp_val = np.exp(-2 * np.pi**2 * var * d**2)
    print(f"  D={d:.4f} → exp = {exp_val:.6f}")

print("\n⚠️ PROBLEM: With use_angle_input=True, exp ≈ 1.0 for ALL distances!")
print("   This means kernel is nearly constant → no gradient signal!")

# Test 3: Find correct variance for angle mode
print("\n=== Finding Correct Variance for use_angle_input=True ===")
print("We want exp(D=0.5) ≈ 0.3 for reasonable correlation at 180°")
print("exp(-2π²·var·0.5²) = 0.3  →  var ≈ 0.24")

for var in [0.0001, 0.01, 0.1, 0.24, 1.0, 10.0]:
    exp_near = np.exp(-2 * np.pi**2 * var * 0.0556**2)  # 20°
    exp_mid = np.exp(-2 * np.pi**2 * var * 0.5**2)      # 180°
    print(f"  var={var:6.4f}: exp(20°)={exp_near:.4f}, exp(180°)={exp_mid:.4f}")

# Test 4: Gradient flow with both modes
print("\n=== Gradient Flow Test ===")

for use_angle in [False, True]:
    mode_name = "angle" if use_angle else "wrapped"
    
    # Use appropriate default variance for each mode
    if use_angle:
        # Need larger variance for normalized D
        kernel = SMCircleKernel(num_mixtures=2, use_angle_input=True)
        # Override with correct variance
        with torch.no_grad():
            kernel.log_variances.fill_(np.log(0.24))  # Good for normalized D
    else:
        kernel = SMCircleKernel(num_mixtures=2, use_angle_input=False)
    
    K = kernel(angles)
    loss = K.sum()
    loss.backward()
    
    print(f"\nuse_angle_input={use_angle} ({mode_name} mode):")
    print(f"  K[0,1]={K[0,1].item():.4f}, K[0,9]={K[0,9].item():.4f}, K[0,17]={K[0,17].item():.4f}")
    print(f"  means.grad: {kernel.means.grad}")
    print(f"  log_variances.grad: {kernel.log_variances.grad}")
    
    # Check if gradients are meaningful
    if kernel.means.grad is not None:
        grad_norm = kernel.means.grad.norm().item()
        print(f"  means.grad_norm: {grad_norm:.4f} {'✓' if grad_norm > 0.01 else '✗ (too small!)'}")

# Test 5: What default variance should be for angle mode?
print("\n" + "=" * 70)
print("SOLUTION: Need different default variance for use_angle_input=True")
print("=" * 70)
print("""
When use_angle_input=False:
  D is in degrees [0, 180], so variance ~0.0001 works
  exp(-2π²·0.0001·20²) ≈ 0.45

When use_angle_input=True:
  D is normalized [0, 1], so need variance ~1.0
  exp(-2π²·1.0·0.056²) ≈ 0.94 (adjacent)
  exp(-2π²·1.0·0.5²) ≈ 0.08 (opposite)

FIX: Set default variance based on use_angle_input flag!
""")
