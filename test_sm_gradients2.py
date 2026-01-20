#!/usr/bin/env python
"""Diagnose why SM kernel parameters have zero gradients."""

import sys
sys.path.insert(0, 'GPPVAE/pysrc/coil100')

import torch
import numpy as np
from kernels import SMCircleKernel, get_angles_degrees

print('=== Diagnosing SM Kernel Gradient Issue ===\n')

# Create kernel and angles
sm = SMCircleKernel(num_mixtures=2)
angles = get_angles_degrees(18)

print('Initial parameters:')
print(f'  means: {sm.means.data.numpy()}')
print(f'  weights: {sm.weights.data.numpy()}')
print(f'  variances: {sm.variances.data.numpy()}')

# Compute kernel matrix
K = sm(angles)
print(f'\nK shape: {K.shape}')
print(f'K requires_grad: {K.requires_grad}')
print(f'K[0,:5]: {K[0,:5].detach().numpy()}')

# Try to backprop through K directly
loss = K.sum()
loss.backward()

print('\nGradients after K.sum().backward():')
print(f'  means.grad: {sm.means.grad}')
print(f'  log_weights.grad: {sm.log_weights.grad}')
print(f'  log_variances.grad: {sm.log_variances.grad}')

# Check if wrapped_lag_distance is the problem
print('\n=== Checking wrapped_lag_distance ===')
from kernels import wrapped_lag_distance
D = wrapped_lag_distance(angles)
print(f'D requires_grad: {D.requires_grad}')
print(f'D[0,:5]: {D[0,:5].numpy()}')

# The issue: angles is a buffer (not parameter), so D has no gradient
# But the kernel computation should still have gradient through weights/means/variances

print('\n=== Manual SM kernel forward ===')
sm2 = SMCircleKernel(num_mixtures=2)
angles2 = get_angles_degrees(18)
D2 = wrapped_lag_distance(angles2)

K2 = torch.zeros_like(D2)
for q in range(sm2.num_mixtures):
    w = sm2.weights[q]
    mu = sm2.means[q]
    var = sm2.variances[q]
    
    print(f'\nMixture {q}:')
    print(f'  w={w.item():.4f}, mu={mu.item():.4f}, var={var.item():.4f}')
    
    exp_term = torch.exp(-2 * (np.pi ** 2) * var * (D2 ** 2))
    cos_term = torch.cos(2 * np.pi * mu * D2)
    
    print(f'  exp_term[0,1]={exp_term[0,1].item():.6f}')
    print(f'  cos_term[0,1]={cos_term[0,1].item():.6f}')
    print(f'  contribution[0,1]={(w * exp_term * cos_term)[0,1].item():.6f}')
    
    K2 = K2 + w * exp_term * cos_term

print(f'\nK2[0,1] = {K2[0,1].item():.6f}')
print(f'K2 requires_grad: {K2.requires_grad}')

# Check if the issue is with torch.zeros_like
print('\n=== Issue: torch.zeros_like(D) ===')
print(f'D2.requires_grad: {D2.requires_grad}')
# torch.zeros_like copies the requires_grad from input!
# Since D doesn't require grad (angles is a buffer), zeros_like also doesn't!
K3 = torch.zeros(18, 18)  # This should work
for q in range(sm2.num_mixtures):
    w = sm2.weights[q]
    mu = sm2.means[q]
    var = sm2.variances[q]
    exp_term = torch.exp(-2 * (np.pi ** 2) * var * (D2 ** 2))
    cos_term = torch.cos(2 * np.pi * mu * D2)
    K3 = K3 + w * exp_term * cos_term

print(f'K3 requires_grad: {K3.requires_grad}')
loss3 = K3.sum()
loss3.backward()
print(f'After K3.sum().backward():')
print(f'  means.grad: {sm2.means.grad}')
