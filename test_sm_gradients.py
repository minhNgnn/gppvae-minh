#!/usr/bin/env python
"""Test if SM kernel parameters receive gradients."""

import sys
sys.path.insert(0, 'GPPVAE/pysrc/coil100')

import torch
from kernels import SMCircleKernel, create_kernel
from vmod import Vmodel

# Test 1: Check if SMCircleKernel parameters are nn.Parameters
print('=== Test 1: SMCircleKernel Parameter Registration ===')
sm = SMCircleKernel(num_mixtures=2)
print('Parameters in SMCircleKernel:')
for name, param in sm.named_parameters():
    print(f'  {name}: shape={param.shape}, requires_grad={param.requires_grad}')

# Test 2: Check if parameters flow through Vmodel
print()
print('=== Test 2: Vmodel Parameter Registration ===')
vm = Vmodel(P=10, Q=18, p=8, view_kernel='sm_circle', num_mixtures=2)
print(f'Total parameters in Vmodel: {sum(p.numel() for p in vm.parameters())}')
print('Parameters breakdown:')
for name, param in vm.named_parameters():
    print(f'  {name}: shape={param.shape}, requires_grad={param.requires_grad}')

# Test 3: Simulate gradient flow
print()
print('=== Test 3: Gradient Flow Test ===')
vm = Vmodel(P=10, Q=18, p=8, view_kernel='sm_circle', num_mixtures=2)
d = torch.tensor([0, 1, 2, 3], dtype=torch.long)
w = torch.tensor([0, 1, 2, 3], dtype=torch.long)

V = vm(d, w)
loss = V.sum()
loss.backward()

print('Gradients after backward:')
for name, param in vm.named_parameters():
    if param.grad is not None:
        print(f'  {name}: grad_norm={param.grad.norm().item():.6f}')
    else:
        print(f'  {name}: NO GRADIENT!')

# Test 4: Check the v() method - this is where kernel matrix is computed
print()
print('=== Test 4: Check v() method (Cholesky of kernel) ===')
vm2 = Vmodel(P=10, Q=18, p=8, view_kernel='sm_circle', num_mixtures=2)

# Get initial parameters
means_before = vm2.kernel.means.clone().detach()
print(f'Initial means: {means_before.numpy()}')

# Compute L = v() and check if it requires grad
L = vm2.v()
print(f'L = vm.v() requires_grad: {L.requires_grad}')

# The problem: v() uses cholesky which might break gradient flow
K = vm2.kernel(vm2.angles)
print(f'K = kernel(angles) requires_grad: {K.requires_grad}')

# Check if cholesky preserves gradients
try:
    L_chol = torch.linalg.cholesky(K + 1e-6 * torch.eye(18))
    print(f'L_chol requires_grad: {L_chol.requires_grad}')
except Exception as e:
    print(f'Cholesky failed: {e}')
