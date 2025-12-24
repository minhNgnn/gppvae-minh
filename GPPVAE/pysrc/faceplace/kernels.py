"""
Kernel functions for Gaussian Process priors in GP-VAE.

This module provides various kernel implementations for modeling correlations
in the latent space, particularly for view angles in the face dataset.

Each kernel can be used to construct covariance matrices for the GP prior.
"""

import torch
import numpy as np
from torch import nn
from torch.autograd import Variable


class ViewKernel(nn.Module):
    """Base class for view angle kernels."""
    
    def __init__(self, n_views):
        """
        Args:
            n_views: Number of discrete view angles (e.g., 9 for FacePlaces)
        """
        super(ViewKernel, self).__init__()
        self.n_views = n_views
    
    def forward(self, view_indices):
        """
        Compute kernel matrix for given view indices.
        
        Args:
            view_indices: Tensor of view indices [N]
        
        Returns:
            Kernel matrix [N, N]
        """
        raise NotImplementedError


class FullRankKernel(ViewKernel):
    """
    Full-rank kernel: learns arbitrary Q×Q covariance matrix.
    Most expressive but requires Q²/2 parameters.
    """
    
    def __init__(self, n_views):
        super(FullRankKernel, self).__init__(n_views)
        # Learn Cholesky factor of covariance (lower triangular)
        self.L = nn.Parameter(torch.eye(n_views))
    
    def forward(self, view_indices):
        # K = L @ L^T (positive definite)
        K_full = torch.mm(self.L, self.L.t())
        # Index into full kernel matrix
        return K_full[view_indices][:, view_indices]
    
    def get_full_matrix(self):
        """Get the full Q×Q kernel matrix."""
        return torch.mm(self.L, self.L.t())


class LinearKernel(ViewKernel):
    """
    Linear kernel: K = V @ V^T where V are learned view embeddings.
    This is the original kernel from Casale et al. (2018).
    Requires Q×r parameters where r is the rank.
    """
    
    def __init__(self, n_views, rank):
        """
        Args:
            n_views: Number of views
            rank: Rank of the factorization (embedding dimension)
        """
        super(LinearKernel, self).__init__(n_views)
        self.rank = rank
        # Learn view embeddings
        self.v0 = nn.Parameter(torch.randn(n_views, rank) * 0.1)
    
    def forward(self, view_indices):
        # Get embeddings for the views
        V = self.v0[view_indices]  # [N, rank]
        # K = V @ V^T
        return torch.mm(V, V.t())  # [N, N]
    
    def get_full_matrix(self):
        """Get the full Q×Q kernel matrix."""
        return torch.mm(self.v0, self.v0.t())


class RBFAngleKernel(ViewKernel):
    """
    RBF (Gaussian) kernel on angles:
    k(θ, θ') = exp(-(θ - θ')² / (2ℓ²))
    
    Assumes angles are evenly spaced in [0, 2π).
    Smooth but not periodic (doesn't know 0° = 360°).
    """
    
    def __init__(self, n_views, angles=None, lengthscale=1.0):
        """
        Args:
            n_views: Number of views
            angles: Optional array of angles in radians. If None, assumes evenly spaced.
            lengthscale: Initial lengthscale parameter ℓ
        """
        super(RBFAngleKernel, self).__init__(n_views)
        
        # Set up angles (default: evenly spaced)
        if angles is None:
            angles = np.linspace(0, 2*np.pi, n_views, endpoint=False)
        self.register_buffer('angles', torch.tensor(angles, dtype=torch.float32))
        
        # Learnable lengthscale (log-parameterized for positivity)
        self.log_lengthscale = nn.Parameter(torch.log(torch.tensor(lengthscale)))
    
    def forward(self, view_indices):
        # Get angles for the views
        theta = self.angles[view_indices]  # [N]
        
        # Compute pairwise squared differences
        theta_i = theta.unsqueeze(1)  # [N, 1]
        theta_j = theta.unsqueeze(0)  # [1, N]
        sq_dist = (theta_i - theta_j) ** 2  # [N, N]
        
        # RBF kernel
        lengthscale = torch.exp(self.log_lengthscale)
        K = torch.exp(-sq_dist / (2 * lengthscale ** 2))
        
        return K
    
    def get_full_matrix(self):
        """Get the full Q×Q kernel matrix."""
        theta = self.angles
        theta_i = theta.unsqueeze(1)
        theta_j = theta.unsqueeze(0)
        sq_dist = (theta_i - theta_j) ** 2
        lengthscale = torch.exp(self.log_lengthscale)
        return torch.exp(-sq_dist / (2 * lengthscale ** 2))


class PeriodicKernel(ViewKernel):
    """
    Periodic kernel for angles:
    k(θ, θ') = exp(-2·sin²((θ - θ')/2) / ℓ²)
    
    This kernel correctly handles the periodicity: k(0°, 360°) = 1.
    Best choice for modeling rotations/angles.
    """
    
    def __init__(self, n_views, angles=None, lengthscale=1.0):
        """
        Args:
            n_views: Number of views
            angles: Optional array of angles in radians. If None, assumes evenly spaced.
            lengthscale: Initial lengthscale parameter ℓ
        """
        super(PeriodicKernel, self).__init__(n_views)
        
        # Set up angles
        if angles is None:
            angles = np.linspace(0, 2*np.pi, n_views, endpoint=False)
        self.register_buffer('angles', torch.tensor(angles, dtype=torch.float32))
        
        # Learnable lengthscale
        self.log_lengthscale = nn.Parameter(torch.log(torch.tensor(lengthscale)))
    
    def forward(self, view_indices):
        theta = self.angles[view_indices]
        
        # Compute angular differences
        theta_i = theta.unsqueeze(1)
        theta_j = theta.unsqueeze(0)
        diff = theta_i - theta_j
        
        # Periodic kernel: uses sin²(Δθ/2) which is periodic
        lengthscale = torch.exp(self.log_lengthscale)
        K = torch.exp(-2 * torch.sin(diff / 2) ** 2 / lengthscale ** 2)
        
        return K
    
    def get_full_matrix(self):
        """Get the full Q×Q kernel matrix."""
        theta = self.angles
        theta_i = theta.unsqueeze(1)
        theta_j = theta.unsqueeze(0)
        diff = theta_i - theta_j
        lengthscale = torch.exp(self.log_lengthscale)
        return torch.exp(-2 * torch.sin(diff / 2) ** 2 / lengthscale ** 2)


class RationalQuadraticKernel(ViewKernel):
    """
    Rational Quadratic (Cauchy) kernel:
    k(θ, θ') = (1 + (θ - θ')² / ℓ²)^(-α)
    
    Can be seen as a mixture of RBF kernels with different lengthscales.
    When α → ∞, becomes RBF. Heavier tails than RBF.
    """
    
    def __init__(self, n_views, angles=None, lengthscale=1.0, alpha=1.0):
        """
        Args:
            n_views: Number of views
            angles: Optional array of angles in radians
            lengthscale: Initial lengthscale parameter ℓ
            alpha: Shape parameter (controls tail heaviness)
        """
        super(RationalQuadraticKernel, self).__init__(n_views)
        
        if angles is None:
            angles = np.linspace(0, 2*np.pi, n_views, endpoint=False)
        self.register_buffer('angles', torch.tensor(angles, dtype=torch.float32))
        
        self.log_lengthscale = nn.Parameter(torch.log(torch.tensor(lengthscale)))
        self.log_alpha = nn.Parameter(torch.log(torch.tensor(alpha)))
    
    def forward(self, view_indices):
        theta = self.angles[view_indices]
        
        theta_i = theta.unsqueeze(1)
        theta_j = theta.unsqueeze(0)
        sq_dist = (theta_i - theta_j) ** 2
        
        lengthscale = torch.exp(self.log_lengthscale)
        alpha = torch.exp(self.log_alpha)
        
        K = (1 + sq_dist / (lengthscale ** 2)) ** (-alpha)
        
        return K
    
    def get_full_matrix(self):
        """Get the full Q×Q kernel matrix."""
        theta = self.angles
        theta_i = theta.unsqueeze(1)
        theta_j = theta.unsqueeze(0)
        sq_dist = (theta_i - theta_j) ** 2
        lengthscale = torch.exp(self.log_lengthscale)
        alpha = torch.exp(self.log_alpha)
        return (1 + sq_dist / (lengthscale ** 2)) ** (-alpha)


class MaternKernel(ViewKernel):
    """
    Matérn kernel (ν=3/2):
    k(r) = (1 + √3·r/ℓ) · exp(-√3·r/ℓ)
    where r = |θ - θ'|
    
    Less smooth than RBF but more realistic for many real-world processes.
    ν=3/2 means once differentiable (vs RBF which is infinitely differentiable).
    """
    
    def __init__(self, n_views, angles=None, lengthscale=1.0, nu=1.5):
        """
        Args:
            n_views: Number of views
            angles: Optional array of angles in radians
            lengthscale: Initial lengthscale parameter ℓ
            nu: Smoothness parameter (1.5 or 2.5 are common)
        """
        super(MaternKernel, self).__init__(n_views)
        
        if angles is None:
            angles = np.linspace(0, 2*np.pi, n_views, endpoint=False)
        self.register_buffer('angles', torch.tensor(angles, dtype=torch.float32))
        
        self.log_lengthscale = nn.Parameter(torch.log(torch.tensor(lengthscale)))
        self.nu = nu  # Fixed smoothness parameter
    
    def forward(self, view_indices):
        theta = self.angles[view_indices]
        
        theta_i = theta.unsqueeze(1)
        theta_j = theta.unsqueeze(0)
        dist = torch.abs(theta_i - theta_j)
        
        lengthscale = torch.exp(self.log_lengthscale)
        
        if self.nu == 1.5:
            # ν = 3/2
            sqrt3_r = np.sqrt(3.0) * dist / lengthscale
            K = (1 + sqrt3_r) * torch.exp(-sqrt3_r)
        elif self.nu == 2.5:
            # ν = 5/2
            sqrt5_r = np.sqrt(5.0) * dist / lengthscale
            K = (1 + sqrt5_r + 5.0/3.0 * (dist/lengthscale)**2) * torch.exp(-sqrt5_r)
        else:
            raise NotImplementedError(f"Matérn with ν={self.nu} not implemented")
        
        return K
    
    def get_full_matrix(self):
        """Get the full Q×Q kernel matrix."""
        theta = self.angles
        theta_i = theta.unsqueeze(1)
        theta_j = theta.unsqueeze(0)
        dist = torch.abs(theta_i - theta_j)
        lengthscale = torch.exp(self.log_lengthscale)
        
        if self.nu == 1.5:
            sqrt3_r = np.sqrt(3.0) * dist / lengthscale
            return (1 + sqrt3_r) * torch.exp(-sqrt3_r)
        elif self.nu == 2.5:
            sqrt5_r = np.sqrt(5.0) * dist / lengthscale
            return (1 + sqrt5_r + 5.0/3.0 * (dist/lengthscale)**2) * torch.exp(-sqrt5_r)


class VonMisesKernel(ViewKernel):
    """
    Von Mises kernel (also called wrapped Gaussian):
    k(θ, θ') = exp(κ · cos(θ - θ'))
    
    This is the circular analog of the Gaussian distribution.
    κ (kappa) controls concentration: larger κ = more concentrated.
    Naturally periodic and designed specifically for circular data.
    """
    
    def __init__(self, n_views, angles=None, kappa=1.0):
        """
        Args:
            n_views: Number of views
            angles: Optional array of angles in radians
            kappa: Concentration parameter (κ > 0)
        """
        super(VonMisesKernel, self).__init__(n_views)
        
        if angles is None:
            angles = np.linspace(0, 2*np.pi, n_views, endpoint=False)
        self.register_buffer('angles', torch.tensor(angles, dtype=torch.float32))
        
        # Log-parameterized kappa for positivity
        self.log_kappa = nn.Parameter(torch.log(torch.tensor(kappa)))
    
    def forward(self, view_indices):
        theta = self.angles[view_indices]
        
        theta_i = theta.unsqueeze(1)
        theta_j = theta.unsqueeze(0)
        diff = theta_i - theta_j
        
        kappa = torch.exp(self.log_kappa)
        K = torch.exp(kappa * torch.cos(diff))
        
        # Normalize so diagonal is 1
        K_diag = torch.exp(kappa * torch.ones_like(K[0, 0]))
        K = K / K_diag
        
        return K
    
    def get_full_matrix(self):
        """Get the full Q×Q kernel matrix."""
        theta = self.angles
        theta_i = theta.unsqueeze(1)
        theta_j = theta.unsqueeze(0)
        diff = theta_i - theta_j
        kappa = torch.exp(self.log_kappa)
        K = torch.exp(kappa * torch.cos(diff))
        K_diag = torch.exp(kappa * torch.ones_like(K[0, 0]))
        return K / K_diag


def get_kernel(kernel_name, n_views, **kwargs):
    """
    Factory function to create kernel instances.
    
    Args:
        kernel_name: Name of kernel ('full', 'linear', 'rbf', 'periodic', 
                     'rational_quadratic', 'matern', 'vonmises')
        n_views: Number of discrete views
        **kwargs: Additional arguments passed to kernel constructor
    
    Returns:
        ViewKernel instance
    
    
    Example:
        >>> kernel = get_kernel('periodic', n_views=9, lengthscale=0.5)
        >>> K = kernel.get_full_matrix()
    """
    kernel_map = {
        'full': FullRankKernel,
        'fullrank': FullRankKernel,
        'linear': LinearKernel,
        'rbf': RBFAngleKernel,
        'gaussian': RBFAngleKernel,
        'periodic': PeriodicKernel,
        'rational_quadratic': RationalQuadraticKernel,
        'rq': RationalQuadraticKernel,
        'cauchy': RationalQuadraticKernel,
        'matern': MaternKernel,
        'vonmises': VonMisesKernel,
        'von_mises': VonMisesKernel,
    }
    
    kernel_name_lower = kernel_name.lower()
    if kernel_name_lower not in kernel_map:
        available = ', '.join(sorted(kernel_map.keys()))
        raise ValueError(f"Unknown kernel '{kernel_name}'. Available: {available}")
    
    kernel_class = kernel_map[kernel_name_lower]
    return kernel_class(n_views, **kwargs)


# Example usage and testing
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # Setup
    n_views = 9
    angles = np.linspace(0, 2*np.pi, n_views, endpoint=False)
    
    # Create different kernels
    kernels = {
        'Full Rank': FullRankKernel(n_views),
        'Linear (rank=3)': LinearKernel(n_views, rank=3),
        'RBF': RBFAngleKernel(n_views, angles),
        'Periodic': PeriodicKernel(n_views, angles),
        'Rational Quadratic': RationalQuadraticKernel(n_views, angles),
        'Matérn (ν=1.5)': MaternKernel(n_views, angles, nu=1.5),
        'Von Mises': VonMisesKernel(n_views, angles),
    }
    
    # Visualize kernel matrices
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, (name, kernel) in enumerate(kernels.items()):
        K = kernel.get_full_matrix().detach().numpy()
        
        ax = axes[idx]
        im = ax.imshow(K, cmap='viridis', vmin=0, vmax=1)
        ax.set_title(name)
        ax.set_xlabel('View angle')
        ax.set_ylabel('View angle')
        plt.colorbar(im, ax=ax)
    
    # Hide the last subplot if we have fewer kernels than subplots
    if len(kernels) < len(axes):
        axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig('kernel_comparison.png', dpi=150)
    print("Saved kernel comparison to kernel_comparison.png")
    
    # Print kernel properties
    print("\nKernel Properties:")
    print("=" * 60)
    for name, kernel in kernels.items():
        K = kernel.get_full_matrix().detach().numpy()
        n_params = sum(p.numel() for p in kernel.parameters())
        print(f"{name:25s} | Params: {n_params:3d} | K[0,0]={K[0,0]:.3f} | K[0,1]={K[0,1]:.3f}")
