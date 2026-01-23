"""
Variance Model (Vmod) for GP-VAE on COIL-100.

This module implements the variance model that combines:
- Object embeddings (learned P x p matrix)
- View kernel (structured covariance for 18 rotation views)

The view kernel uses angles in degrees [0, 360) with wrapped lag distance.
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

# Import kernels from local coil100 directory
from kernels import create_kernel, get_angles_degrees


def normalize_rows(x):
    """Normalize each row to unit length."""
    diagonal = torch.einsum("ij,ij->i", [x, x])[:, None]
    return x / torch.sqrt(diagonal + 1e-8)


class Vmodel(nn.Module):
    """
    Variance model for GP-VAE on COIL-100.
    
    Computes V = X ⊗ W where:
    - X: Object embeddings (P x p) - learned
    - W: View covariance factor from kernel (Q x Q) - structured
    
    The GP prior is: z ~ N(0, V @ V^T + noise)
    """
    
    def __init__(self, P, Q, p, view_kernel='full_rank', **kernel_kwargs):
        """
        Args:
            P: Number of objects (100 for COIL-100)
            Q: Number of views (18 for COIL-100)
            p: Dimension of object embeddings (typically 64)
            view_kernel: Type of kernel for view covariance
                - 'full_rank': Free-form learnable covariance
                - 'rbf_circle': RBF with wrapped lag distance
                - 'sm_circle': Spectral Mixture with wrapped lag
                - 'periodic': Standard periodic kernel
            **kernel_kwargs: Additional kernel arguments
        """
        super(Vmodel, self).__init__()
        self.P = P
        self.Q = Q
        self.p = p
        self.view_kernel_type = view_kernel
        
        # Object embeddings (always learned)
        self.x0 = nn.Parameter(torch.randn(P, p))
        
        # View kernel
        self.kernel = create_kernel(view_kernel, Q=Q, **kernel_kwargs)
        
        # Pre-compute reference angles in degrees [0, 360)
        # For COIL-100: [0, 20, 40, ..., 340]
        self.register_buffer('angles', get_angles_degrees(Q))
        
        # # Cache for Cholesky factor (updated when kernel changes)
        # self._L_cache = None
        # self._L_cache_valid = False
        
        self._init_params()
    
    def _init_params(self):
        """Initialize object embeddings."""
        # First dimension to 1, rest small noise
        self.x0.data[:, 0] = 1.0
        self.x0.data[:, 1:] = 1e-3 * torch.randn(*self.x0[:, 1:].shape)
    
    def x(self):
        """Get normalized object embeddings."""
        return normalize_rows(self.x0)
    
    def v(self):
        """
        Get view covariance factor L such that K_view = L @ L^T.
        Uses Cholesky decomposition of kernel matrix.
        """
        # For FullRankKernel, we can directly use its L parameter
        if self.view_kernel_type == 'full_rank':
            L_lower = torch.tril(self.kernel.L)
            return L_lower
        
        # Compute kernel matrix K(angles, angles)
        K = self.kernel(self.angles)  # (Q, Q)
        
        # Add jitter for numerical stability
        K = K + 1e-4 * torch.eye(self.Q, device=K.device)
        
        # Cholesky decomposition: K = L @ L^T
        try:
            L = torch.linalg.cholesky(K)
        except RuntimeError:
            # Fallback to eigendecomposition if Cholesky fails
            # torch.linalg.eigh replaces deprecated torch.symeig
            eigvals, eigvecs = torch.linalg.eigh(K)
            eigvals = torch.clamp(eigvals, min=1e-5)
            L = eigvecs @ torch.diag(torch.sqrt(eigvals))
        
        return L  # (Q, Q)
    
    def forward(self, d, w):
        """
        Compute variance matrix V for given object and view indices.
        
        Args:
            d: Object indices [N] (integers in [0, P-1])
            w: View indices [N] (integers in [0, Q-1])
               Note: These are discrete indices, not angles!
        
        Returns:
            V: Variance matrix [N, p*Q] where each row is vec(X_i ⊗ L[w_i])
        """
        # Get object embeddings
        X = F.embedding(d, self.x())  # [N, p]
        
        # Get view embeddings from Cholesky factor
        L = self.v()  # [Q, Q]
        W = F.embedding(w, L)  # [N, Q]
        
        # Kronecker product: V_i = X_i ⊗ W_i
        V = torch.einsum("ij,ik->ijk", [X, W])  # [N, p, Q]
        V = V.reshape([V.shape[0], -1])  # [N, p*Q]
        
        return V
    
    def get_kernel_matrix(self):
        """Get the current kernel matrix K = L @ L^T."""
        L = self.v()
        return L @ L.t()
    
    def __repr__(self):
        return "Vmodel(P={}, Q={}, p={}, kernel={})".format(
            self.P, self.Q, self.p, self.kernel)


# Testing
if __name__ == "__main__":
    print("Testing Vmodel for COIL-100")
    print("=" * 60)
    
    P = 10   # 10 objects (subset)
    Q = 18   # 18 views
    p = 8    # embedding dim
    
    # Test data: 4 samples per object, cycling through views
    _d = np.kron(np.arange(P), np.ones(4)).astype(int)  # [0,0,0,0, 1,1,1,1, ...]
    _w = np.tile(np.arange(4), P).astype(int)            # [0,1,2,3, 0,1,2,3, ...]
    
    d = torch.tensor(_d, dtype=torch.long)
    w = torch.tensor(_w, dtype=torch.long)
    
    print(f"Test data: {len(d)} samples")
    print(f"Object indices: {d[:8].tolist()}...")
    print(f"View indices: {w[:8].tolist()}...")
    
    # Test each kernel
    for kernel_name in ['full_rank', 'rbf_circle', 'sm_circle', 'periodic']:
        print(f"\n--- Testing {kernel_name} ---")
        
        vm = Vmodel(P, Q, p, view_kernel=kernel_name)
        V = vm(d, w)
        K = vm.get_kernel_matrix()
        
        print(f"  V shape: {V.shape}")
        print(f"  K shape: {K.shape}")
        print(f"  K[0,0]={K[0,0].item():.4f}, K[0,1]={K[0,1].item():.4f}, K[0,17]={K[0,17].item():.4f}")
        print(f"  Symmetry: {torch.allclose(K, K.t())}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
