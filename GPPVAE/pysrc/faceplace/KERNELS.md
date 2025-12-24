# GP-VAE Kernel Library

This document describes the different kernel functions available for modeling view correlations in GP-VAE.

## Overview

The `kernels.py` module provides 7 different kernel implementations for modeling correlations between view angles. Each kernel has different properties and is suitable for different scenarios.

## Available Kernels

### 1. **Full Rank Kernel** (`'full'`, `'fullrank'`)
```python
kernel = FullRankKernel(n_views=9)
```
- **Description**: Learns an arbitrary Q×Q covariance matrix
- **Parameters**: Q²/2 (45 for 9 views)
- **Pros**: Most expressive, no assumptions
- **Cons**: Most parameters, can overfit
- **Use when**: You have lots of data and want maximum flexibility

### 2. **Linear Kernel** (`'linear'`)
```python
kernel = LinearKernel(n_views=9, rank=3)
```
- **Description**: K = V @ V^T where V are learned view embeddings
- **Parameters**: Q × rank (27 for rank=3, 9 views)
- **Pros**: Low-rank structure, original GP-VAE kernel
- **Cons**: Doesn't explicitly use angle information
- **Use when**: Default choice, works well in practice
- **Note**: This is the kernel used in the original Casale et al. (2018) paper

### 3. **RBF Angle Kernel** (`'rbf'`, `'gaussian'`)
```python
kernel = RBFAngleKernel(n_views=9, lengthscale=1.0)
```
- **Formula**: k(θ, θ') = exp(-(θ - θ')² / (2ℓ²))
- **Parameters**: 1 (lengthscale ℓ)
- **Pros**: Very smooth, simple
- **Cons**: Not periodic (doesn't know 0° = 360°)
- **Use when**: Views are not full rotations (e.g., [-90°, 90°])

### 4. **Periodic Kernel** (`'periodic'`) ⭐ **RECOMMENDED FOR ROTATIONS**
```python
kernel = PeriodicKernel(n_views=9, lengthscale=1.0)
```
- **Formula**: k(θ, θ') = exp(-2·sin²((θ - θ')/2) / ℓ²)
- **Parameters**: 1 (lengthscale ℓ)
- **Pros**: **Correctly handles periodicity** (0° = 360°), smooth
- **Cons**: None for periodic data
- **Use when**: Modeling full rotations (face views 0-360°)
- **Why it's best**: Uses sin²(Δθ/2) which is naturally periodic

### 5. **Rational Quadratic Kernel** (`'rational_quadratic'`, `'rq'`, `'cauchy'`)
```python
kernel = RationalQuadraticKernel(n_views=9, lengthscale=1.0, alpha=1.0)
```
- **Formula**: k(θ, θ') = (1 + (θ - θ')² / ℓ²)^(-α)
- **Parameters**: 2 (lengthscale ℓ, shape α)
- **Pros**: Heavier tails than RBF, mixture of scales
- **Cons**: Not periodic
- **Use when**: You want smooth transitions with occasional long-range correlations

### 6. **Matérn Kernel** (`'matern'`)
```python
kernel = MaternKernel(n_views=9, lengthscale=1.0, nu=1.5)
```
- **Formula** (ν=1.5): k(r) = (1 + √3·r/ℓ) · exp(-√3·r/ℓ)
- **Parameters**: 1 (lengthscale ℓ)
- **Pros**: More realistic than RBF, less smooth
- **Cons**: Not periodic
- **Use when**: RBF is too smooth, you want ν=1.5 (once differentiable) or ν=2.5
- **Options**: `nu=1.5` (default) or `nu=2.5`

### 7. **Von Mises Kernel** (`'vonmises'`, `'von_mises'`) ⭐ **RECOMMENDED FOR ANGLES**
```python
kernel = VonMisesKernel(n_views=9, kappa=1.0)
```
- **Formula**: k(θ, θ') = exp(κ · cos(θ - θ'))
- **Parameters**: 1 (concentration κ)
- **Pros**: **Designed specifically for circular data**, naturally periodic
- **Cons**: None for angular data
- **Use when**: Modeling angles/rotations (alternative to Periodic)
- **Why it's good**: This is the circular analog of the Gaussian distribution

## Quick Start

### Creating a Kernel

```python
from kernels import get_kernel

# Use factory function
kernel = get_kernel('periodic', n_views=9, lengthscale=0.5)

# Or instantiate directly
from kernels import PeriodicKernel
kernel = PeriodicKernel(n_views=9, lengthscale=0.5)
```

### Getting the Covariance Matrix

```python
# Full Q×Q matrix
K_full = kernel.get_full_matrix()  # [9, 9]

# Subset for specific views
view_indices = torch.tensor([0, 1, 4, 7])
K_subset = kernel(view_indices)  # [4, 4]
```

### Visualizing Kernels

```python
import matplotlib.pyplot as plt
import torch

kernel = get_kernel('periodic', n_views=9)
K = kernel.get_full_matrix().detach().numpy()

plt.imshow(K, cmap='viridis')
plt.colorbar()
plt.title('Periodic Kernel')
plt.xlabel('View 1')
plt.ylabel('View 2')
plt.show()
```

## Recommendations by Use Case

### For Face View Angles (0°, 15°, 30°, ..., 90°)
**Recommended**: `'periodic'` or `'vonmises'`
- These correctly handle the fact that rotations wrap around
- Periodic kernel is smoother, Von Mises is specifically designed for circular data

### For Limited View Range (e.g., only front views -45° to +45°)
**Recommended**: `'rbf'` or `'matern'`
- No periodicity needed
- RBF for smooth, Matérn for more realistic

### For Maximum Flexibility (with lots of data)
**Recommended**: `'full'` or `'linear'` with high rank
- Full rank: learns everything from data
- Linear: good compromise between flexibility and regularization

### For Comparing with Original GP-VAE Paper
**Recommended**: `'linear'` with `rank=Q` (number of views)
- This replicates the original Casale et al. (2018) kernel

## Parameters

### Lengthscale (ℓ)
- Controls how quickly correlations decay with distance
- **Small ℓ**: Nearby views are correlated, distant views are independent
- **Large ℓ**: Even distant views are correlated
- Log-parameterized internally: always positive

### Kappa (κ) - Von Mises only
- Controls concentration around the mean direction
- **Small κ** (→0): Nearly uniform, all views correlated equally
- **Large κ**: Sharp peak, only nearby views correlated
- Similar role to inverse lengthscale

### Alpha (α) - Rational Quadratic only  
- Controls tail heaviness
- **α → ∞**: Becomes RBF kernel
- **Small α**: Heavier tails, long-range correlations

### Nu (ν) - Matérn only
- Controls smoothness
- **ν = 1.5**: Once differentiable (less smooth)
- **ν = 2.5**: Twice differentiable (smoother)
- **ν → ∞**: Becomes RBF (infinitely differentiable)

## Integration with GP-VAE

To use these kernels in your GP-VAE training, you'll need to modify `vmod.py` to use a kernel instead of raw embeddings. See the next section for integration code.

## Example: Comparing All Kernels

```python
from kernels import get_kernel
import matplotlib.pyplot as plt
import numpy as np

n_views = 9
angles = np.linspace(0, 2*np.pi, n_views, endpoint=False)

kernels = {
    'Linear': get_kernel('linear', n_views, rank=3),
    'RBF': get_kernel('rbf', n_views, angles=angles),
    'Periodic': get_kernel('periodic', n_views, angles=angles),
    'Von Mises': get_kernel('vonmises', n_views, angles=angles),
    'Matérn': get_kernel('matern', n_views, angles=angles, nu=1.5),
}

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for ax, (name, kernel) in zip(axes, kernels.items()):
    K = kernel.get_full_matrix().detach().numpy()
    im = ax.imshow(K, cmap='viridis', vmin=0, vmax=1)
    ax.set_title(name)
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('kernel_comparison.png')
```

## Performance Considerations

| Kernel | Parameters | Computation | Memory |
|--------|------------|-------------|--------|
| Full Rank | Q²/2 | O(Q²) | O(Q²) |
| Linear | Q×r | O(Nr²) | O(Nr) |
| RBF | 1 | O(N²) | O(N²) |
| Periodic | 1 | O(N²) | O(N²) |
| Rational Quadratic | 2 | O(N²) | O(N²) |
| Matérn | 1 | O(N²) | O(N²) |
| Von Mises | 1 | O(N²) | O(N²) |

Where:
- Q = number of views (e.g., 9)
- r = rank (for linear kernel)
- N = batch size

## References

1. **Original GP-VAE**: Casale et al. (2018) "Gaussian Process Prior Variational Autoencoders"
2. **Periodic Kernel**: MacKay (1998) "Introduction to Gaussian Processes"
3. **Von Mises**: Mardia & Jupp (2000) "Directional Statistics"
4. **Matérn**: Rasmussen & Williams (2006) "Gaussian Processes for Machine Learning"

## Testing

Run the module directly to generate visualizations:
```bash
cd GPPVAE/pysrc/faceplace
python kernels.py
```

This will create `kernel_comparison.png` showing all kernel matrices.
