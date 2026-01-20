"""
Test if SM kernel parameters actually update during training simulation.
"""
import sys
sys.path.insert(0, 'GPPVAE/pysrc/coil100')

import torch
from torch import optim
import numpy as np
from vmod import Vmodel
from kernels import get_angles_degrees

print("=" * 70)
print("Testing SM Kernel Parameter Updates During Training")
print("=" * 70)

P = 10  # 10 objects
Q = 18  # 18 views
p = 8   # embedding dim

# Test both modes
for use_angle_input in [False, True]:
    mode_name = "angle" if use_angle_input else "wrapped"
    print(f"\n{'='*70}")
    print(f"use_angle_input={use_angle_input} ({mode_name} mode)")
    print(f"{'='*70}")
    
    # Create Vmodel with SM kernel
    vm = Vmodel(P, Q, p, view_kernel='sm_circle', 
                num_mixtures=2, use_angle_input=use_angle_input)
    
    # Show initial kernel parameters
    print(f"\nInitial kernel parameters:")
    print(f"  variances: {vm.kernel.variances.detach().numpy()}")
    print(f"  means (freqs): {vm.kernel.means.detach().numpy()}")
    print(f"  weights: {vm.kernel.weights.detach().numpy()}")
    
    # Check kernel matrix
    K_init = vm.get_kernel_matrix()
    print(f"\nInitial kernel matrix:")
    print(f"  K[0,0]={K_init[0,0].item():.4f}, K[0,1]={K_init[0,1].item():.4f}, K[0,9]={K_init[0,9].item():.4f}")
    
    # Setup optimizer (include all vm parameters)
    optimizer = optim.Adam(vm.parameters(), lr=0.01)
    
    # Simulate training steps
    print(f"\nSimulating 10 training steps...")
    param_history = []
    
    for step in range(10):
        optimizer.zero_grad()
        
        # Random batch
        d = torch.randint(0, P, (32,))
        w = torch.randint(0, Q, (32,))
        
        # Forward pass
        V = vm(d, w)
        
        # Dummy loss (just sum of V to create gradients)
        loss = V.sum()
        loss.backward()
        
        # Check gradients before step
        if step == 0:
            print(f"\nGradients at step 0:")
            print(f"  x0.grad_norm: {vm.x0.grad.norm().item():.4f}")
            print(f"  kernel.means.grad: {vm.kernel.means.grad}")
            print(f"  kernel.log_variances.grad: {vm.kernel.log_variances.grad}")
            print(f"  kernel.log_weights.grad: {vm.kernel.log_weights.grad}")
        
        # Store parameters before update
        param_history.append({
            'variances': vm.kernel.variances.detach().clone().numpy(),
            'means': vm.kernel.means.detach().clone().numpy(),
            'weights': vm.kernel.weights.detach().clone().numpy()
        })
        
        optimizer.step()
    
    # Compare first and last parameters
    print(f"\nParameter changes (step 0 → step 9):")
    print(f"  variances: {param_history[0]['variances']} → {param_history[-1]['variances']}")
    print(f"  means: {param_history[0]['means']} → {param_history[-1]['means']}")
    print(f"  weights: {param_history[0]['weights']} → {param_history[-1]['weights']}")
    
    # Check if parameters actually changed
    var_changed = not np.allclose(param_history[0]['variances'], param_history[-1]['variances'], rtol=1e-4)
    mean_changed = not np.allclose(param_history[0]['means'], param_history[-1]['means'], rtol=1e-4)
    weight_changed = not np.allclose(param_history[0]['weights'], param_history[-1]['weights'], rtol=1e-4)
    
    status = "✅ OK" if (var_changed and mean_changed and weight_changed) else "❌ NOT UPDATING"
    print(f"\nStatus: {status}")
    print(f"  Variances changed: {var_changed}")
    print(f"  Means changed: {mean_changed}")
    print(f"  Weights changed: {weight_changed}")

print(f"\n{'='*70}")
print("Test complete!")
print(f"{'='*70}")
