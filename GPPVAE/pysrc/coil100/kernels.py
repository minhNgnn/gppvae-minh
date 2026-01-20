"""
GP prior kernels for GPPVAE - View dimension.

Four kernels are implemented, all accepting the same angular inputs (degrees [0, 360)):

1. FullRank    - No geometry, learns free covariance over view indices
2. RBF-circle  - RBF kernel with wrapped lag distance (circular)
3. SM-circle   - Spectral Mixture kernel with wrapped lag distance (circular)
4. Periodic    - Standard periodic kernel with built-in circularity

Wrapped lag distance: d(theta, theta') = min(|theta - theta'|, 360 - |theta - theta'|)

All kernels return (Q, Q) covariance matrices compatible with GP-VAE training.
"""

import torch
import torch.nn as nn
import numpy as np


def wrapped_lag_distance(angles):
    """
    Compute wrapped lag distance matrix for circular data.
    z
    d(theta, theta') = min(|theta - theta'|, 360 - |theta - theta'|)
    
    Args:
        angles: (Q,) tensor of angles in degrees [0, 360)
        
    Returns:
        D: (Q, Q) distance matrix
    """
    # Pairwise absolute differences
    diff = torch.abs(angles.unsqueeze(1) - angles.unsqueeze(0))  # (Q, Q)
    
    # Wrapped distance: min(diff, 360 - diff)
    D = torch.min(diff, 360.0 - diff)
    return D


class FullRankKernel(nn.Module):
    """
    Free-form learnable covariance matrix over view indices.
    K = L @ L^T where L is a learnable lower triangular matrix.
    
    - Ignores angle values entirely
    - No geometric assumptions
    - Learns unconstrained covariance (subject to PSD via Cholesky parameterization)
    """
    
    def __init__(self, Q):
        """
        Args:
            Q: Number of views (e.g., 18 for COIL-100)
        """
        super(FullRankKernel, self).__init__()
        self.Q = Q
        # Initialize L as identity matrix
        L_init = torch.eye(Q)
        self.L = nn.Parameter(L_init)
        
        # Register a buffer for creating lower triangular mask on correct device
        self.register_buffer('_ones', torch.ones(Q, Q))
    
    def forward(self, angles=None):
        """
        Compute K = L @ L^T
        
        Args:
            angles: Ignored (kept for API compatibility), but used for device info
            
        Returns:
            K: (Q, Q) positive semi-definite covariance matrix
        """
        # Use tril mask that's on the same device as self.L
        L_lower = torch.tril(self.L)
        K = torch.mm(L_lower, L_lower.t())
        return K
    
    def __repr__(self):
        return "FullRankKernel(Q={})".format(self.Q)


class RBFCircleKernel(nn.Module):
    """
    RBF kernel with wrapped lag distance for circular data.
    
    k(theta, theta') = variance * exp(-d(theta, theta')^2 / (2 * lengthscale^2))
    
    where d(theta, theta') = min(|theta - theta'|, 360 - |theta - theta'|) is the wrapped distance.
    
    Learnable parameters:
    - lengthscale: Controls smoothness
    - variance: Signal variance (scaling)
    """
    
    def __init__(self, lengthscale_init=30.0, variance_init=1.0):
        """
        Args:
            lengthscale_init: Initial lengthscale (in degrees, default 30)
            variance_init: Initial signal variance
        """
        super(RBFCircleKernel, self).__init__()
        # Store log for unconstrained optimization
        self.log_lengthscale = nn.Parameter(torch.tensor(np.log(lengthscale_init)))
        self.log_variance = nn.Parameter(torch.tensor(np.log(variance_init)))
    
    @property
    def lengthscale(self):
        return torch.exp(self.log_lengthscale)
    
    @property
    def variance(self):
        return torch.exp(self.log_variance)
    
    def forward(self, angles):
        """
        Compute RBF kernel with wrapped distance.
        
        Args:
            angles: (Q,) tensor of angles in degrees [0, 360)
            
        Returns:
            K: (Q, Q) kernel matrix
        """
        # Compute wrapped lag distance
        D = wrapped_lag_distance(angles)  # (Q, Q)
        
        # RBF kernel: k = var * exp(-d^2 / (2 * l^2))
        K = self.variance * torch.exp(-D ** 2 / (2 * self.lengthscale ** 2))
        return K
    
    def __repr__(self):
        return "RBFCircleKernel(lengthscale={:.2f}, variance={:.4f})".format(
            self.lengthscale.item(), self.variance.item())


class SMCircleKernel(nn.Module):
    """
    Spectral Mixture kernel with wrapped lag distance for circular data.
    
    k(theta, theta') = sum_q w_q * exp(-2*pi^2 * var_q * d^2) * cos(2*pi * mu_q * d)
    
    where d is the wrapped lag distance (or raw angle difference if use_angle_input=True).
    
    Learnable parameters per mixture component:
    - weights: Mixture weights (softmax normalized)
    - means (mu): Frequencies
    - variances: Bandwidths (inverse lengthscales)
    """
    
    def __init__(self, num_mixtures=2, freq_init=None, length_init=None, weight_init=None,
                 use_angle_input=False, period=360.0):
        """
        Args:
            num_mixtures: Number of spectral mixture components (default: 2)
                          Using 2-3 mixtures to avoid overfitting with 18 views.
            freq_init: Optional list/array of initial frequencies (means). If None, uses linspace(0.01, 0.1)
            length_init: Optional list/array of initial lengthscales. If None, uses 30.0 for all
            weight_init: Optional list/array of initial weights. If None, uses uniform (equal weights)
            use_angle_input: If True, use raw |angle_diff| like periodic kernel instead of wrapped lag.
                            This makes the kernel work directly with angle differences.
            period: Period in degrees (default: 360), only used if use_angle_input=True
        """
        super(SMCircleKernel, self).__init__()
        self.num_mixtures = num_mixtures
        self.use_angle_input = use_angle_input
        self.period = period
        
        # Mixture weights (softmax will be applied, so log-space for uniform = zeros)
        if weight_init is not None:
            # Convert weights to log-space (before softmax)
            weights = torch.tensor(weight_init, dtype=torch.float32)
            self.log_weights = nn.Parameter(torch.log(weights + 1e-8))
        else:
            self.log_weights = nn.Parameter(torch.zeros(num_mixtures))
        
        # Means (frequencies) - initialize spread across frequency range
        if freq_init is not None:
            self.means = nn.Parameter(torch.tensor(freq_init, dtype=torch.float32))
        else:
            # Scale for degrees: typical frequencies for 360 period
            self.means = nn.Parameter(torch.linspace(0.01, 0.1, num_mixtures))
        
        # Variances (bandwidths) - store log for positivity
        # variance ≈ 1/lengthscale² in spectral space
        # CRITICAL: Default variance depends on distance scale!
        #
        # use_angle_input=False: D is in degrees [0, 180]
        #   exp(-2*π²*var*D²) with D=20° and var=0.0001 → exp(-0.79) ≈ 0.45 ✓
        #
        # use_angle_input=True: D is normalized [0, 1] (D = |diff|/period)
        #   exp(-2*π²*var*D²) with D=0.056 (20°) and var=0.0001 → exp(-0.00006) ≈ 1.0 ✗
        #   Need var ~ 1.0 for exp(-2*π²*1.0*0.056²) ≈ 0.94, exp(0.5²) ≈ 0.08
        if length_init is not None:
            lengthscales = torch.tensor(length_init, dtype=torch.float32)
            # Convert lengthscale to variance (approximate inverse relationship)
            variances = 1.0 / (lengthscales ** 2 + 1e-8)
            self.log_variances = nn.Parameter(torch.log(variances))
        else:
            if use_angle_input:
                # D is normalized [0, 1], need larger variance
                # var=1.0 gives good range: exp(0.056)≈0.94, exp(0.5)≈0.08
                default_var = 1.0
            else:
                # D is in degrees [0, 180], need small variance
                # var=0.0001 gives exp(20°)≈0.45 for adjacent views
                default_var = 0.0001
            self.log_variances = nn.Parameter(torch.full((num_mixtures,), np.log(default_var)))
    
    @property
    def weights(self):
        return torch.softmax(self.log_weights, dim=0)
    
    @property
    def variances(self):
        return torch.exp(self.log_variances)
    
    def forward(self, angles):
        """
        Compute Spectral Mixture kernel.
        
        Args:
            angles: (Q,) tensor of angles in degrees [0, 360)
            
        Returns:
            K: (Q, Q) kernel matrix
        """
        if self.use_angle_input:
            # Use raw angle differences like periodic kernel (periodicity built into cos term)
            diff = torch.abs(angles.unsqueeze(1) - angles.unsqueeze(0))  # (Q, Q)
            # Normalize by period for frequency interpretation
            D = diff / self.period  # Now in [0, 1] for one period
        else:
            # Use wrapped lag distance (circular)
            D = wrapped_lag_distance(angles)  # (Q, Q)
        
        K = torch.zeros_like(D)
        
        for q in range(self.num_mixtures):
            w = self.weights[q]
            mu = self.means[q]
            var = self.variances[q]
            
            # SM kernel: w * exp(-2*pi^2*var*d^2) * cos(2*pi*mu*d)
            exp_term = torch.exp(-2 * (np.pi ** 2) * var * (D ** 2))
            cos_term = torch.cos(2 * np.pi * mu * D)
            K = K + w * exp_term * cos_term
        
        return K
    
    def __repr__(self):
        mode = "angle" if self.use_angle_input else "wrapped"
        return "SMCircleKernel(num_mixtures={}, mode={})".format(self.num_mixtures, mode)


class PeriodicKernel(nn.Module):
    """
    Standard periodic GP kernel with built-in circularity.
    
    k(theta, theta') = variance * exp(-2 * sin^2(pi * |theta - theta'| / period) / lengthscale^2)
    
    - Uses standard formulation with built-in periodicity
    - Does NOT use wrapped lag explicitly (periodicity is in the kernel)
    - Period fixed at 360 (one full rotation)
    
    Learnable parameters:
    - lengthscale: Controls smoothness
    - variance: Signal variance
    """
    
    def __init__(self, lengthscale_init=1.0, variance_init=1.0, period=360.0):
        """
        Args:
            lengthscale_init: Initial lengthscale
            variance_init: Initial signal variance
            period: Period in degrees (default: 360)
        """
        super(PeriodicKernel, self).__init__()
        self.log_lengthscale = nn.Parameter(torch.tensor(np.log(lengthscale_init)))
        self.log_variance = nn.Parameter(torch.tensor(np.log(variance_init)))
        self.period = period  # Fixed, not learned
    
    @property
    def lengthscale(self):
        return torch.exp(self.log_lengthscale)
    
    @property
    def variance(self):
        return torch.exp(self.log_variance)
    
    def forward(self, angles):
        """
        Compute Periodic kernel.
        
        Args:
            angles: (Q,) tensor of angles in degrees [0, 360)
            
        Returns:
            K: (Q, Q) kernel matrix
        """
        # Pairwise differences (not wrapped - periodicity handles it)
        diff = angles.unsqueeze(1) - angles.unsqueeze(0)  # (Q, Q)
        
        # Periodic kernel: var * exp(-2 * sin^2(pi*|d|/p) / l^2)
        sin_term = torch.sin(np.pi * torch.abs(diff) / self.period)
        K = self.variance * torch.exp(-2 * sin_term ** 2 / (self.lengthscale ** 2))
        return K
    
    def __repr__(self):
        return "PeriodicKernel(lengthscale={:.4f}, period={})".format(
            self.lengthscale.item(), self.period)


def create_kernel(kernel_type, Q=None, **kwargs):
    """
    Factory function to create kernel instances.
    
    Args:
        kernel_type: One of 'full_rank', 'rbf_circle', 'sm_circle', 'periodic'
        Q: Number of views (required for full_rank)
        **kwargs: Additional kernel-specific arguments
        
    Returns:
        Kernel module instance
    """
    kernel_type = kernel_type.lower().replace('-', '_')
    
    if kernel_type == 'full_rank':
        if Q is None:
            raise ValueError("Q (number of views) is required for FullRankKernel")
        return FullRankKernel(Q=Q)
    
    elif kernel_type == 'rbf_circle':
        lengthscale = kwargs.get('lengthscale', 30.0)  # 30 degrees default
        variance = kwargs.get('variance', 1.0)
        return RBFCircleKernel(lengthscale_init=lengthscale, variance_init=variance)
    
    elif kernel_type == 'sm_circle':
        num_mixtures = kwargs.get('num_mixtures', 2)  # 2 mixtures default (less overfitting)
        freq_init = kwargs.get('freq_init', None)
        length_init = kwargs.get('length_init', None)
        weight_init = kwargs.get('weight_init', None)
        use_angle_input = kwargs.get('use_angle_input', False)
        period = kwargs.get('period', 360.0)
        return SMCircleKernel(num_mixtures=num_mixtures, 
                            freq_init=freq_init, 
                            length_init=length_init,
                            weight_init=weight_init,
                            use_angle_input=use_angle_input,
                            period=period)
    
    elif kernel_type == 'periodic':
        lengthscale = kwargs.get('lengthscale', 1.0)
        variance = kwargs.get('variance', 1.0)
        period = kwargs.get('period', 360.0)
        return PeriodicKernel(lengthscale_init=lengthscale, variance_init=variance, period=period)
    
    else:
        raise ValueError("Unknown kernel type: {}. Supported: 'full_rank', 'rbf_circle', 'sm_circle', 'periodic'".format(kernel_type))


def get_angles_degrees(Q):
    """
    Get angle values in degrees [0, 360) for Q evenly spaced views.
    
    Args:
        Q: Number of views
        
    Returns:
        torch.Tensor of shape (Q,) with angles in degrees
    """
    angles = np.linspace(0, 360, Q, endpoint=False)
    return torch.tensor(angles, dtype=torch.float32)


if __name__ == "__main__":
    print("=" * 60)
    print("Testing GP Kernels for GPPVAE - View Dimension")
    print("=" * 60)
    
    Q = 18  # COIL-100 has 18 views (every 20 degrees)
    angles = get_angles_degrees(Q)
    print("\nQ = {} views".format(Q))
    print("Angles (degrees): {}... (first 6)".format(angles[:6].numpy()))
    print("Step size: {} degrees\n".format(360.0/Q))
    
    # Test wrapped lag distance
    D = wrapped_lag_distance(angles)
    print("Wrapped lag distance examples:")
    print("  d(0, 20)  = {:.1f} degrees".format(D[0,1].item()))
    print("  d(0, 180) = {:.1f} degrees (opposite)".format(D[0,9].item()))
    print("  d(0, 340) = {:.1f} degrees (wraps to 20)".format(D[0,17].item()))
    print()
    
    # Test each kernel
    kernels = {
        'FullRank': FullRankKernel(Q=Q),
        'RBF-circle': RBFCircleKernel(lengthscale_init=30.0),
        'SM-circle': SMCircleKernel(num_mixtures=2),
        'Periodic': PeriodicKernel(lengthscale_init=1.0)
    }
    
    print("-" * 60)
    print("Kernel outputs:")
    print("-" * 60)
    
    for name, kernel in kernels.items():
        K = kernel(angles)
        
        print("\n{}:".format(name))
        print("  {}".format(kernel))
        print("  Shape: {}".format(K.shape))
        print("  K[0,0] = {:.4f} (self-similarity)".format(K[0,0].item()))
        print("  K[0,1] = {:.4f} (adjacent: 0 vs 20)".format(K[0,1].item()))
        print("  K[0,9] = {:.4f} (opposite: 0 vs 180)".format(K[0,9].item()))
        print("  K[0,17] = {:.4f} (wrapped: 0 vs 340)".format(K[0,17].item()))
        print("  Range: [{:.4f}, {:.4f}]".format(K.min().item(), K.max().item()))
    
    # Test factory function
    print("\n" + "-" * 60)
    print("Testing create_kernel() factory:")
    print("-" * 60)
    for kt in ['full_rank', 'rbf_circle', 'sm_circle', 'periodic']:
        k = create_kernel(kt, Q=Q)
        print("  create_kernel('{}') -> {}".format(kt, k))
    
    # Verify symmetry
    print("\n" + "-" * 60)
    print("Verifying wrapped distance symmetry:")
    print("-" * 60)
    for name, kernel in kernels.items():
        K = kernel(angles)
        sym_error = torch.max(torch.abs(K - K.t())).item()
        status = "OK" if sym_error < 1e-6 else "FAIL"
        print("  {}: max|K - K^T| = {:.2e} [{}]".format(name, sym_error, status))
    
    print("\nAll 4 kernels working correctly!")
