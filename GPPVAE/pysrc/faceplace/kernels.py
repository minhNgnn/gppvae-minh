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
    
    def __init__(self, n_views, jitter=1e-6):
        super(FullRankKernel, self).__init__(n_views)
        # Learn Cholesky factor of covariance (lower triangular)
        self.L = nn.Parameter(torch.eye(n_views))
        self.jitter = jitter
    
    def forward(self, view_indices):
        # K = L @ L^T (positive definite)
        K_full = torch.mm(self.L, self.L.t())
        # Add jitter for numerical stability
        K_full = K_full + self.jitter * torch.eye(self.n_views, device=K_full.device)
        # Index into full kernel matrix
        return K_full[view_indices][:, view_indices]
    
    def get_full_matrix(self):
        """Get the full Q×Q kernel matrix."""
        K = torch.mm(self.L, self.L.t())
        return K + self.jitter * torch.eye(self.n_views, device=K.device)


class LinearKernel(ViewKernel):
    """
    Linear kernel: K = V @ V^T where V are learned view embeddings.
    This is the original kernel from Casale et al. (2018).
    Requires Q×r parameters where r is the rank.
    """
    
    def __init__(self, n_views, rank, jitter=1e-6):
        """
        Args:
            n_views: Number of views
            rank: Rank of the factorization (embedding dimension)
            jitter: Small value added to diagonal for numerical stability
        """
        super(LinearKernel, self).__init__(n_views)
        self.rank = rank
        self.jitter = jitter
        # Learn view embeddings
        self.v0 = nn.Parameter(torch.randn(n_views, rank) * 0.1)
    
    def forward(self, view_indices):
        # Get embeddings for the views
        V = self.v0[view_indices]  # [N, rank]
        # K = V @ V^T
        K = torch.mm(V, V.t())  # [N, N]
        # Add jitter for numerical stability
        K = K + self.jitter * torch.eye(K.shape[0], device=K.device)
        return K
    
    def get_full_matrix(self):
        """Get the full Q×Q kernel matrix."""
        K = torch.mm(self.v0, self.v0.t())
        return K + self.jitter * torch.eye(self.n_views, device=K.device)


class RBFAngleKernel(ViewKernel):
    """
    RBF (Gaussian) kernel on angles:
    k(θ, θ') = exp(-(θ - θ')² / (2ℓ²))
    
    Can work with:
    1. Discrete view indices (0, 1, 2, ..., n_views-1) - uses angles buffer
    2. Continuous angle values (e.g., normalized angles in [-1, 1])
    
    Smooth but not periodic (doesn't know 0° = 360°).
    """
    
    def __init__(self, n_views, angles=None, lengthscale=1.0, angle_scale='normalized'):
        """
        Args:
            n_views: Number of views
            angles: Optional array of angles. If None, uses normalized angles [-1, 1]
            lengthscale: Initial lengthscale parameter ℓ
            angle_scale: Scale of input angles ('normalized' for [-1,1], 'radians' for [0,2π])
        """
        super(RBFAngleKernel, self).__init__(n_views)
        
        # Set up reference angles for discrete indices
        if angles is None:
            if angle_scale == 'normalized':
                # Map to [-1, 1] range (matching data_parser_interpolation.py)
                angles = np.linspace(-1.0, 1.0, n_views)
            elif angle_scale == 'radians':
                angles = np.linspace(0, 2*np.pi, n_views, endpoint=False)
            else:
                raise ValueError(f"Unknown angle_scale: {angle_scale}")
        
        self.register_buffer('angles', torch.tensor(angles, dtype=torch.float32))
        self.angle_scale = angle_scale
        
        # Learnable lengthscale (log-parameterized for positivity)
        self.log_lengthscale = nn.Parameter(torch.log(torch.tensor(lengthscale)))
    
    def forward(self, view_indices):
        """
        Compute kernel matrix for discrete view indices.
        
        Args:
            view_indices: Integer indices [N] into the angles buffer
        
        Returns:
            K: Kernel matrix [N, N]
        """
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
    
    def forward_continuous(self, angles_continuous):
        """
        Compute kernel matrix between continuous angles and reference angles.
        
        This enables interpolation: given continuous angle values,
        compute their similarity to the discrete reference angles.
        
        Args:
            angles_continuous: Continuous angle values [N] (e.g., [-1, 1] or [0, 2π])
        
        Returns:
            K_cross: Cross-kernel matrix [N, n_views]
                     K_cross[i, j] = k(angles_continuous[i], angles_reference[j])
        """
        # Ensure continuous angles are 1D
        if len(angles_continuous.shape) > 1:
            angles_continuous = angles_continuous.squeeze(-1)
        
        # Compute cross-kernel: k(w_i, θ_j) for all i, j
        theta_cont = angles_continuous.unsqueeze(1)  # [N, 1]
        theta_ref = self.angles.unsqueeze(0)  # [1, Q]
        sq_dist = (theta_cont - theta_ref) ** 2  # [N, Q]
        
        # RBF kernel
        lengthscale = torch.exp(self.log_lengthscale)
        K_cross = torch.exp(-sq_dist / (2 * lengthscale ** 2))
        
        return K_cross
    
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
    Matérn kernel (ν=3/2 or ν=5/2):
    k(r) = (1 + √3·r/ℓ) · exp(-√3·r/ℓ)  for ν=3/2
    where r = |θ - θ'|
    
    Can work with:
    1. Discrete view indices (uses angles buffer)
    2. Continuous angle values
    
    Less smooth than RBF but more realistic for many real-world processes.
    ν=3/2 means once differentiable (vs RBF which is infinitely differentiable).
    """
    
    def __init__(self, n_views, angles=None, lengthscale=1.0, nu=1.5, angle_scale='normalized'):
        """
        Args:
            n_views: Number of views
            angles: Optional array of angles
            lengthscale: Initial lengthscale parameter ℓ
            nu: Smoothness parameter (1.5 or 2.5 are common)
            angle_scale: Scale of input angles ('normalized' for [-1,1], 'radians' for [0,2π])
        """
        super(MaternKernel, self).__init__(n_views)
        
        if angles is None:
            if angle_scale == 'normalized':
                angles = np.linspace(-1.0, 1.0, n_views)
            elif angle_scale == 'radians':
                angles = np.linspace(0, 2*np.pi, n_views, endpoint=False)
            else:
                raise ValueError(f"Unknown angle_scale: {angle_scale}")
        
        self.register_buffer('angles', torch.tensor(angles, dtype=torch.float32))
        self.angle_scale = angle_scale
        
        self.log_lengthscale = nn.Parameter(torch.log(torch.tensor(lengthscale)))
        self.nu = nu  # Fixed smoothness parameter
    
    def _compute_kernel(self, dist, lengthscale):
        """Helper to compute Matérn kernel from distances."""
        if self.nu == 1.5:
            # ν = 3/2
            sqrt3_r = np.sqrt(3.0) * dist / lengthscale
            return (1 + sqrt3_r) * torch.exp(-sqrt3_r)
        elif self.nu == 2.5:
            # ν = 5/2
            sqrt5_r = np.sqrt(5.0) * dist / lengthscale
            return (1 + sqrt5_r + 5.0/3.0 * (dist/lengthscale)**2) * torch.exp(-sqrt5_r)
        else:
            raise NotImplementedError(f"Matérn with ν={self.nu} not implemented")
    
    def forward(self, view_indices):
        theta = self.angles[view_indices]
        
        theta_i = theta.unsqueeze(1)
        theta_j = theta.unsqueeze(0)
        dist = torch.abs(theta_i - theta_j)
        
        lengthscale = torch.exp(self.log_lengthscale)
        return self._compute_kernel(dist, lengthscale)
    
    def forward_continuous(self, angles_continuous):
        """
        Compute kernel matrix between continuous angles and reference angles.
        
        Args:
            angles_continuous: Continuous angle values [N]
        
        Returns:
            K_cross: Cross-kernel matrix [N, n_views]
        """
        if len(angles_continuous.shape) > 1:
            angles_continuous = angles_continuous.squeeze(-1)
        
        theta_cont = angles_continuous.unsqueeze(1)  # [N, 1]
        theta_ref = self.angles.unsqueeze(0)  # [1, Q]
        dist = torch.abs(theta_cont - theta_ref)  # [N, Q]
        
        lengthscale = torch.exp(self.log_lengthscale)
        return self._compute_kernel(dist, lengthscale)
    
    def get_full_matrix(self):
        """Get the full Q×Q kernel matrix."""
        theta = self.angles
        theta_i = theta.unsqueeze(1)
        theta_j = theta.unsqueeze(0)
        dist = torch.abs(theta_i - theta_j)
        lengthscale = torch.exp(self.log_lengthscale)
        return self._compute_kernel(dist, lengthscale)


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


class SpectralMixtureKernel(ViewKernel):
    """
    Spectral Mixture (SM) kernel from Wilson & Adams (2013):
    k(τ) = Σᵢ wᵢ · exp(-2π²τ²vᵢ) · cos(2πτμᵢ)
    
    where τ = θ - θ' and for each mixture component i:
    - wᵢ: mixture weight (positive, sum to 1)
    - μᵢ: mixture mean (frequency)
    - vᵢ: mixture variance (inverse lengthscale, positive)
    
    This is extremely flexible and can approximate many stationary kernels
    including RBF, Periodic, Matérn, etc. The kernel learns the spectral
    density (Fourier transform of covariance) as a mixture of Gaussians.
    
    Key advantages:
    - Can discover and model multiple periodicities
    - Can learn complex, non-standard covariance structures
    - Can approximate other kernels automatically
    
    Parameters: 3 × n_components (weights, means, variances)
    
    Implementation notes:
    - Weights use softplus (not softmax) for independent scaling
    - Frequencies initialized based on data characteristics
    - Variances initialized for smooth interpolation
    """
    
    def __init__(self, n_views, angles=None, n_components=3, angle_scale='normalized'):
        """
        Args:
            n_views: Number of views
            angles: Optional array of angles
            n_components: Number of mixture components (Q in SM paper)
            angle_scale: Scale of input angles ('normalized' for [-1,1], 'radians' for [0,2π])
        """
        super(SpectralMixtureKernel, self).__init__(n_views)
        
        self.n_components = n_components
        
        # Set up reference angles
        if angles is None:
            if angle_scale == 'normalized':
                angles = np.linspace(-1.0, 1.0, n_views)
            elif angle_scale == 'radians':
                angles = np.linspace(0, 2*np.pi, n_views, endpoint=False)
            else:
                raise ValueError(f"Unknown angle_scale: {angle_scale}")
        
        self.register_buffer('angles', torch.tensor(angles, dtype=torch.float32))
        self.angle_scale = angle_scale
        
        # Compute data statistics for smart initialization
        if angle_scale == 'normalized':
            angle_range = 2.0  # [-1, 1]
            nyquist_freq = n_views / (2 * angle_range)  # Maximum meaningful frequency
        else:
            angle_range = 2 * np.pi
            nyquist_freq = n_views / angle_range
        
        # Initialize mixture parameters
        # Mixture weights: use log-space parameterization with softplus
        # Initialize uniformly (will learn relative importance)
        self.log_mixture_weights = nn.Parameter(torch.zeros(n_components))
        
        # Mixture means (frequencies): initialize to capture different scales
        # Space frequencies logarithmically to cover low to high frequencies
        if n_components == 1:
            init_means = torch.tensor([nyquist_freq / 4])
        else:
            # Logarithmic spacing from low to Nyquist frequency
            log_freqs = torch.linspace(
                np.log(nyquist_freq / 10), 
                np.log(nyquist_freq), 
                n_components
            )
            init_means = torch.exp(log_freqs)
        
        self.mixture_means = nn.Parameter(init_means)
        
        # Mixture variances (inverse lengthscales): 
        # Initialize for smooth interpolation (not too peaked)
        # Variance controls frequency bandwidth
        init_vars = torch.ones(n_components) * (nyquist_freq / (4 * n_components))
        self.log_mixture_vars = nn.Parameter(torch.log(init_vars))
    
    def _compute_sm_kernel(self, tau):
        """
        Compute SM kernel for given distance matrix.
        
        Args:
            tau: Distance matrix [N, M] where tau[i,j] = θ_i - θ_j
        
        Returns:
            K: Kernel matrix [N, M]
        """
        # Get parameters (using softplus for positive weights that can scale independently)
        # Softplus allows weights to grow beyond 1, unlike softmax
        mixture_weights = torch.nn.functional.softplus(self.log_mixture_weights)  # [Q]
        mixture_means = self.mixture_means  # [Q]
        mixture_vars = torch.exp(self.log_mixture_vars)  # [Q]
        
        # Expand dimensions for broadcasting
        tau_expanded = tau.unsqueeze(-1)  # [N, M, 1]
        weights = mixture_weights.unsqueeze(0).unsqueeze(0)  # [1, 1, Q]
        means = mixture_means.unsqueeze(0).unsqueeze(0)  # [1, 1, Q]
        vars = mixture_vars.unsqueeze(0).unsqueeze(0)  # [1, 1, Q]
        
        # Compute each component: wᵢ · exp(-2π²τ²vᵢ) · cos(2πτμᵢ)
        exponential_term = torch.exp(-2 * np.pi**2 * tau_expanded**2 * vars)
        cosine_term = torch.cos(2 * np.pi * tau_expanded * means)
        
        # Sum over mixture components
        K = torch.sum(weights * exponential_term * cosine_term, dim=-1)
        
        # Add small jitter for numerical stability (only on diagonal)
        if tau.shape[0] == tau.shape[1]:  # Square matrix (not cross-covariance)
            jitter = 1e-6
            K = K + jitter * torch.eye(tau.shape[0], device=K.device)
        
        return K
    
    def forward(self, view_indices):
        """
        Compute kernel matrix for discrete view indices.
        
        Args:
            view_indices: Integer indices [N] into the angles buffer
        
        Returns:
            K: Kernel matrix [N, N]
        """
        theta = self.angles[view_indices]  # [N]
        
        # Compute pairwise differences
        theta_i = theta.unsqueeze(1)  # [N, 1]
        theta_j = theta.unsqueeze(0)  # [1, N]
        tau = theta_i - theta_j  # [N, N]
        
        return self._compute_sm_kernel(tau)
    
    def forward_continuous(self, angles_continuous):
        """
        Compute kernel matrix between continuous angles and reference angles.
        
        Args:
            angles_continuous: Continuous angle values [N]
        
        Returns:
            K_cross: Cross-kernel matrix [N, n_views]
        """
        if len(angles_continuous.shape) > 1:
            angles_continuous = angles_continuous.squeeze(-1)
        
        # Compute cross-differences
        theta_cont = angles_continuous.unsqueeze(1)  # [N, 1]
        theta_ref = self.angles.unsqueeze(0)  # [1, Q]
        tau = theta_cont - theta_ref  # [N, Q]
        
        return self._compute_sm_kernel(tau)
    
    def get_full_matrix(self):
        """Get the full Q×Q kernel matrix."""
        theta = self.angles
        theta_i = theta.unsqueeze(1)
        theta_j = theta.unsqueeze(0)
        tau = theta_i - theta_j
        return self._compute_sm_kernel(tau)
    
    def get_spectral_density(self, frequencies):
        """
        Get the learned spectral density at given frequencies.
        This is useful for visualization and interpretation.
        
        Args:
            frequencies: Array of frequency values [F]
        
        Returns:
            spectral_density: Values of spectral density at each frequency [F]
        """
        mixture_weights = torch.nn.functional.softplus(self.log_mixture_weights)
        mixture_means = self.mixture_means
        mixture_vars = torch.exp(self.log_mixture_vars)
        
        freqs = torch.tensor(frequencies, dtype=torch.float32, device=self.angles.device)
        freqs_expanded = freqs.unsqueeze(-1)  # [F, 1]
        weights = mixture_weights.unsqueeze(0)  # [1, Q]
        means = mixture_means.unsqueeze(0)  # [1, Q]
        vars = mixture_vars.unsqueeze(0)  # [1, Q]
        
        # Spectral density: Σᵢ wᵢ · N(ω; μᵢ, vᵢ)
        # Each component is a Gaussian in frequency space
        density = torch.sum(
            weights * torch.exp(-(freqs_expanded - means)**2 / (2 * vars)) / torch.sqrt(2 * np.pi * vars),
            dim=-1
        )
        
        return density.detach().cpu().numpy()


def get_kernel(kernel_name, n_views, **kwargs):
    """
    Factory function to create kernel instances.
    
    Args:
        kernel_name: Name of kernel ('full', 'linear', 'rbf', 'periodic', 
                     'rational_quadratic', 'matern', 'vonmises', 'spectral_mixture')
        n_views: Number of discrete views
        **kwargs: Additional arguments passed to kernel constructor
    
    Returns:
        ViewKernel instance
    
    
    Example:
        >>> kernel = get_kernel('periodic', n_views=9, lengthscale=0.5)
        >>> K = kernel.get_full_matrix()
        >>> # Spectral mixture with 5 components
        >>> kernel = get_kernel('spectral_mixture', n_views=9, n_components=5)
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
        'spectral_mixture': SpectralMixtureKernel,
        'sm': SpectralMixtureKernel,
        'spectral': SpectralMixtureKernel,
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
    angles_norm = np.linspace(-1.0, 1.0, n_views)
    
    # Create different kernels
    kernels = {
        'Full Rank': FullRankKernel(n_views),
        'Linear (rank=3)': LinearKernel(n_views, rank=3),
        'RBF': RBFAngleKernel(n_views, angles),
        'Periodic': PeriodicKernel(n_views, angles),
        'Rational Quadratic': RationalQuadraticKernel(n_views, angles),
        'Matérn (ν=1.5)': MaternKernel(n_views, angles, nu=1.5),
        'Von Mises': VonMisesKernel(n_views, angles),
        'SM (Q=3)': SpectralMixtureKernel(n_views, angles_norm, n_components=3, angle_scale='normalized'),
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
    
    # Visualize spectral density for SM kernel
    sm_kernel = kernels['SM (Q=3)']
    freqs = np.linspace(-3, 3, 200)
    spectral_density = sm_kernel.get_spectral_density(freqs)
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(sm_kernel.get_full_matrix().detach().numpy(), cmap='viridis')
    plt.colorbar()
    plt.title('SM Kernel Matrix')
    
    plt.subplot(1, 2, 2)
    plt.plot(freqs, spectral_density)
    plt.xlabel('Frequency')
    plt.ylabel('Spectral Density')
    plt.title('Learned Spectral Density (SM Kernel)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sm_kernel_analysis.png', dpi=150)
    print("Saved SM kernel analysis to sm_kernel_analysis.png")
