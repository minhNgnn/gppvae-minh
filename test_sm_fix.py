"""Verify SM kernel fix for use_angle_input=True"""
import sys
sys.path.insert(0, 'GPPVAE/pysrc/coil100')
import torch
import numpy as np
from kernels import SMCircleKernel, get_angles_degrees

Q = 18
angles = get_angles_degrees(Q)

print('=== Testing Fixed SM Kernel ===')
print()

# Test use_angle_input=False (should use var=0.0001)
k1 = SMCircleKernel(num_mixtures=2, use_angle_input=False)
print('use_angle_input=False:')
print(f'  Default variance: {k1.variances.detach().numpy()}')
K1 = k1(angles)
print(f'  K[0,1]={K1[0,1].item():.4f}, K[0,9]={K1[0,9].item():.4f}, K[0,17]={K1[0,17].item():.4f}')

# Test use_angle_input=True (should use var=1.0)
k2 = SMCircleKernel(num_mixtures=2, use_angle_input=True)
print()
print('use_angle_input=True:')
print(f'  Default variance: {k2.variances.detach().numpy()}')
K2 = k2(angles)
print(f'  K[0,1]={K2[0,1].item():.4f}, K[0,9]={K2[0,9].item():.4f}, K[0,17]={K2[0,17].item():.4f}')

# Gradient test
print()
print('=== Gradient Flow Test ===')
for use_angle, k in [(False, k1), (True, k2)]:
    for p in k.parameters():
        p.grad = None
    K = k(angles)
    loss = K.sum()
    loss.backward()
    grad_norm = k.means.grad.norm().item()
    status = 'OK' if grad_norm > 0.1 else 'FAIL'
    print(f'  use_angle_input={use_angle}: means.grad_norm={grad_norm:.4f} [{status}]')

print()
print('Fix verified!')
