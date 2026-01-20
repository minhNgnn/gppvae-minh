"""
Enhanced SM Kernel with Custom Initialization Option

This shows how to add optional initialization parameters to SMCircleKernel
while maintaining backward compatibility.
"""

import torch
import torch.nn as nn
import numpy as np


class SMCircleKernelEnhanced(nn.Module):
    """
    Spectral Mixture kernel with OPTIONAL custom initialization.
    
    All parameters (weights, frequencies, variances) are LEARNED during training,
    but you can provide better starting values for faster convergence.
    """
    
    def __init__(self, 
                 num_mixtures=2,
                 frequencies_init=None,
                 variances_init=None, 
                 weights_init=None):
        """
        Args:
            num_mixtures: Number of spectral mixture components
            frequencies_init: Optional list/array of initial frequencies (means)
                             Default: linspace(0.01, 0.1, num_mixtures)
            variances_init: Optional list/array of initial variances (bandwidths)
                           Default: all 1.0
            weights_init: Optional list/array of initial weights (before softmax)
                         Default: all equal (log_weights = 0)
        
        Example for interpolation task (train views 40° apart):
            SMCircleKernelEnhanced(
                num_mixtures=2,
                frequencies_init=[1/360, 1/40],  # Full rotation + train spacing
                variances_init=[0.01, 0.01],     # Tight bandwidths
                weights_init=[0.5, 0.5]          # Equal importance
            )
        """
        super(SMCircleKernelEnhanced, self).__init__()
        self.num_mixtures = num_mixtures
        
        # Initialize frequencies (means)
        if frequencies_init is not None:
            frequencies_init = torch.tensor(frequencies_init, dtype=torch.float32)
            assert len(frequencies_init) == num_mixtures, \
                f"frequencies_init must have {num_mixtures} elements"
            self.means = nn.Parameter(frequencies_init)
        else:
            # Default: spread across frequency range
            self.means = nn.Parameter(torch.linspace(0.01, 0.1, num_mixtures))
        
        # Initialize variances (bandwidths)
        if variances_init is not None:
            variances_init = torch.tensor(variances_init, dtype=torch.float32)
            assert len(variances_init) == num_mixtures, \
                f"variances_init must have {num_mixtures} elements"
            # Store as log for positivity constraint
            self.log_variances = nn.Parameter(torch.log(variances_init))
        else:
            # Default: unit variance
            self.log_variances = nn.Parameter(torch.zeros(num_mixtures))
        
        # Initialize weights
        if weights_init is not None:
            weights_init = torch.tensor(weights_init, dtype=torch.float32)
            assert len(weights_init) == num_mixtures, \
                f"weights_init must have {num_mixtures} elements"
            # Convert to log space (before softmax)
            # log_weights such that softmax(log_weights) ≈ weights_init
            self.log_weights = nn.Parameter(torch.log(weights_init))
        else:
            # Default: equal weights (log_weights=0 → softmax → 1/num_mixtures each)
            self.log_weights = nn.Parameter(torch.zeros(num_mixtures))
    
    @property
    def weights(self):
        return torch.softmax(self.log_weights, dim=0)
    
    @property
    def variances(self):
        return torch.exp(self.log_variances)
    
    def forward(self, angles):
        """Compute Spectral Mixture kernel with wrapped distance."""
        from kernels import wrapped_lag_distance  # Import helper
        
        D = wrapped_lag_distance(angles)
        K = torch.zeros_like(D)
        
        for q in range(self.num_mixtures):
            w = self.weights[q]
            mu = self.means[q]
            var = self.variances[q]
            
            exp_term = torch.exp(-2 * (np.pi ** 2) * var * (D ** 2))
            cos_term = torch.cos(2 * np.pi * mu * D)
            K = K + w * exp_term * cos_term
        
        return K
    
    def __repr__(self):
        return (f"SMCircleKernelEnhanced(num_mixtures={self.num_mixtures}, "
                f"freqs={self.means.data.numpy()}, "
                f"vars={self.variances.data.numpy()})")


def create_kernel_enhanced(kernel_type, Q=None, **kwargs):
    """
    Enhanced factory with custom SM initialization support.
    
    Example:
        create_kernel_enhanced('sm_circle', Q=18,
                              num_mixtures=2,
                              frequencies_init=[1/360, 1/40],
                              variances_init=[0.01, 0.01])
    """
    from kernels import FullRankKernel, RBFCircleKernel, PeriodicKernel
    
    kernel_type = kernel_type.lower().replace('-', '_')
    
    if kernel_type == 'full_rank':
        if Q is None:
            raise ValueError("Q required for FullRankKernel")
        return FullRankKernel(Q=Q)
    
    elif kernel_type == 'rbf_circle':
        lengthscale = kwargs.get('lengthscale', 30.0)
        variance = kwargs.get('variance', 1.0)
        return RBFCircleKernel(lengthscale_init=lengthscale, variance_init=variance)
    
    elif kernel_type == 'sm_circle':
        # Extract all possible SM parameters
        num_mixtures = kwargs.get('num_mixtures', 2)
        frequencies_init = kwargs.get('frequencies_init', None)
        variances_init = kwargs.get('variances_init', None)
        weights_init = kwargs.get('weights_init', None)
        
        return SMCircleKernelEnhanced(
            num_mixtures=num_mixtures,
            frequencies_init=frequencies_init,
            variances_init=variances_init,
            weights_init=weights_init
        )
    
    elif kernel_type == 'periodic':
        lengthscale = kwargs.get('lengthscale', 1.0)
        variance = kwargs.get('variance', 1.0)
        period = kwargs.get('period', 360.0)
        return PeriodicKernel(lengthscale_init=lengthscale, variance_init=variance, period=period)
    
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")


# ============================================================================
# DEMO: Compare generic vs custom initialization
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SM Kernel: Generic vs Custom Initialization")
    print("=" * 70)
    
    from kernels import get_angles_degrees
    angles = get_angles_degrees(18)
    
    print("\n1️⃣  GENERIC INITIALIZATION:")
    print("-" * 70)
    kernel_generic = SMCircleKernelEnhanced(num_mixtures=2)
    print(kernel_generic)
    K_gen = kernel_generic(angles)
    print(f"Initial K[0,1] = {K_gen[0,1].item():.4f} (adjacent views)")
    print(f"Initial K[0,2] = {K_gen[0,2].item():.4f} (40° apart)")
    
    print("\n2️⃣  CUSTOM INITIALIZATION (Interpolation-specific):")
    print("-" * 70)
    kernel_custom = SMCircleKernelEnhanced(
        num_mixtures=2,
        frequencies_init=[1/360, 1/40],   # 360° period + 40° train spacing
        variances_init=[0.01, 0.01],      # Tight bandwidths
        weights_init=[0.5, 0.5]           # Equal importance
    )
    print(kernel_custom)
    K_custom = kernel_custom(angles)
    print(f"Initial K[0,1] = {K_custom[0,1].item():.4f} (adjacent views)")
    print(f"Initial K[0,2] = {K_custom[0,2].item():.4f} (40° apart)")
    
    print("\n3️⃣  NOTEBOOK USAGE:")
    print("-" * 70)
    print("""
    # In notebook CONFIG:
    'kernel_kwargs': {
        'num_mixtures': 2,
        'frequencies_init': [1/360, 1/40],    # Optional: better init
        'variances_init': [0.01, 0.01],       # Optional: better init
        'weights_init': [0.5, 0.5],           # Optional: better init
    }
    
    # Or keep it simple (model will learn):
    'kernel_kwargs': {
        'num_mixtures': 2,  # Use defaults for the rest
    }
    """)
    
    print("\n✅ Both approaches work - custom init may converge faster!")
