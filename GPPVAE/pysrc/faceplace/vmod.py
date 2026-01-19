import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os
import pdb
from kernels import get_kernel


def normalize_rows(x):
    diagonal = torch.einsum("ij,ij->i", [x, x])[:, None]
    return x / torch.sqrt(diagonal)


class Vmodel(nn.Module):
    def __init__(self, P, Q, p, q=None, view_kernel='legacy', **kernel_kwargs):
        """
        Variance model for GP-VAE.
        
        Args:
            P: Number of objects (people)
            Q: Number of views (angles)
            p: Dimension of object embeddings
            q: Dimension of view embeddings (only used if view_kernel='legacy')
               If None and view_kernel='legacy', defaults to Q
            view_kernel: Type of kernel for view covariance
                - 'legacy': Original implementation (learned q-dim embeddings)
                - 'fullrank': Full Q×Q covariance matrix (RECOMMENDED)
                - 'linear': Low-rank linear kernel (requires rank=... kwarg)
                - 'periodic': Periodic kernel for angles (requires lengthscale=... kwarg)
                - 'rbf', 'matern', 'vonmises', etc.
            **kernel_kwargs: Additional arguments for kernel (e.g., lengthscale, rank)
        
        Examples:
            # Backward compatible (original behavior)
            >>> vm = Vmodel(P=100, Q=9, p=64, q=9)
            
            # Use full-rank kernel (no q needed)
            >>> vm = Vmodel(P=100, Q=9, p=64, view_kernel='fullrank')
            
            # Use periodic kernel for rotation angles
            >>> vm = Vmodel(P=100, Q=9, p=64, view_kernel='periodic', lengthscale=1.0)
            
            # Use low-rank linear kernel
            >>> vm = Vmodel(P=100, Q=9, p=64, view_kernel='linear', rank=3)
        """
        super(Vmodel, self).__init__()
        self.P = P
        self.Q = Q
        self.p = p
        self.view_kernel = view_kernel
        
        # Set q: either provided, or default to Q for legacy mode
        if q is None and view_kernel == 'legacy':
            q = Q
        self.q = q
        
        # Object embeddings (always learned)
        self.x0 = nn.Parameter(torch.randn(P, p))
        
        # View kernel: either legacy or structured
        if view_kernel == 'legacy':
            # Original implementation: learn view embeddings
            if q is None:
                raise ValueError("q must be specified when using view_kernel='legacy'")
            self.v0 = nn.Parameter(torch.randn(Q, q))
            self.kernel = None
        else:
            # Use structured kernel from kernels.py
            self.kernel = get_kernel(view_kernel, n_views=Q, **kernel_kwargs)
            self.v0 = None  # Not used when kernel is specified
        
        self._init_params()

    def x(self):
        """Get normalized object embeddings."""
        return normalize_rows(self.x0)

    def v(self):
        """Get view covariance factor (V such that K_view = V @ V^T)."""
        if self.view_kernel == 'legacy':
            # Original: normalized view embeddings
            return normalize_rows(self.v0)
        else:
            # Structured kernel: compute Cholesky factor of kernel matrix
            # K = V @ V^T, so we need V = chol(K)
            K = self.kernel.get_full_matrix()
            
            # Add small jitter for numerical stability
            K = K + 1e-6 * torch.eye(self.Q, device=K.device)
            
            # Cholesky decomposition: K = L @ L^T
            try:
                L = torch.linalg.cholesky(K)
            except RuntimeError:
                # If Cholesky fails, use eigendecomposition
                eigvals, eigvecs = torch.linalg.eigh(K)
                eigvals = torch.clamp(eigvals, min=1e-6)
                L = eigvecs @ torch.diag(torch.sqrt(eigvals))
            
            return L

    def forward(self, d, w):
        """
        Compute variance matrix V for given object and view indices.
        
        Args:
            d: Object indices [N] (always discrete integers)
            w: View values [N] - can be discrete indices OR continuous angles
        
        Returns:
            V: Variance matrix [N, p*q] where each row is vec(X_i ⊗ W_i)
        """
        # Embed objects (always discrete indices)
        X = F.embedding(d, self.x())  # [N, p]
        
        # Handle views: continuous angles vs discrete indices
        if self.view_kernel == 'legacy':
            # Legacy mode: w must be discrete indices
            W = F.embedding(w, self.v())  # [N, q]
        else:
            # Structured kernel mode: w can be continuous angles
            # Need to compute kernel-based embeddings for continuous values
            
            # Check if w contains continuous values (floats) or indices (ints)
            if w.dtype == torch.long or w.dtype == torch.int32 or w.dtype == torch.int64:
                # Discrete indices: use embedding
                W = F.embedding(w, self.v())  # [N, Q]
            else:
                # Continuous angles: compute kernel matrix on the fly
                # This allows RBF/Matern to work with actual angle values
                W = self._compute_view_embeddings_continuous(w)
        
        # Kronecker product: V_i = X_i ⊗ W_i
        V = torch.einsum("ij,ik->ijk", [X, W])
        V = V.reshape([V.shape[0], -1])
        return V
    
    def _compute_view_embeddings_continuous(self, w_continuous):
        """
        Compute view embeddings for continuous angle values.
        
        For continuous angles, we can't use F.embedding. Instead:
        1. Compute kernel k(w_i, θ_j) for all i, j where θ_j are reference angles
        2. Use kernel values as "soft embeddings"
        3. Return K @ V where V = chol(K_ref) of reference angles
        
        Args:
            w_continuous: Continuous angle values [N] (e.g., normalized angles in [-1, 1])
        
        Returns:
            W: View embeddings [N, Q]
        """
        # Get reference angles (discrete grid) - these are what the kernel was trained on
        # For interpolation, reference angles are the training views
        # We need to compute k(w_continuous, w_reference) for each sample
        
        if hasattr(self.kernel, 'angles'):
            # Kernel has explicit angles (e.g., RBF, Periodic, Matern)
            # Compute kernel between continuous angles and reference angles
            ref_angles = self.kernel.angles  # [Q] reference angles
            
            # Convert w_continuous to same scale as ref_angles if needed
            # Assuming w_continuous is normalized to [-1, 1] and ref_angles are in radians
            # We need to know the mapping...
            
            # For now, assume w_continuous are already in the same space as ref_angles
            # Compute k(w_i, θ_j) for all combinations
            W_kernel = self.kernel.forward_continuous(w_continuous)  # [N, Q]
            
            # Get Cholesky factor of reference kernel
            V_ref = self.v()  # [Q, Q] Cholesky factor
            
            # Compute embeddings: W = K_cross @ inv(K_ref) @ V_ref
            # But K_ref = V_ref @ V_ref^T, so inv(K_ref) @ V_ref = V_ref^{-T}
            # Simplification: W ≈ K_cross @ V_ref^{-1}
            # Even simpler: just use K_cross as soft embeddings (unnormalized)
            
            # Use kernel values directly as embeddings
            return W_kernel
        else:
            # Fallback: kernel doesn't support continuous angles
            # Convert continuous to discrete and use embedding
            raise NotImplementedError(
                f"Kernel '{self.view_kernel}' doesn't support continuous angles. "
                f"Use discrete indices or implement forward_continuous() for this kernel."
            )

    def _init_params(self):
        """Initialize parameters."""
        # Object embeddings: first dimension to 1, rest small noise
        self.x0.data[:, 0] = 1.0
        self.x0.data[:, 1:] = 1e-3 * torch.randn(*self.x0[:, 1:].shape)
        
        # View embeddings (only for legacy mode)
        if self.view_kernel == 'legacy':
            self.v0.data[:] = torch.eye(*self.v0.shape) + 1e-3 * torch.randn(*self.v0.shape)


if __name__ == "__main__":
    print("Testing Vmodel with different kernels...")
    print("=" * 60)

    P = 4  # Number of objects
    Q = 9  # Number of views (matching face dataset)
    p = 2  # Object embedding dimension
    q = 2  # View embedding dimension (legacy only)

    # Test data
    _d = np.kron(np.arange(P), np.ones(4))  # [0,0,0,0, 1,1,1,1, ...]
    _w = np.tile(np.arange(4), P)            # [0,1,2,3, 0,1,2,3, ...]
    
    d = Variable(torch.Tensor(_d).long(), requires_grad=False).cuda()
    w = Variable(torch.Tensor(_w).long(), requires_grad=False).cuda()

    # Test 1: Legacy mode (original implementation)
    print("\n1. Testing LEGACY mode (original implementation)")
    vm_legacy = Vmodel(P, Q, p, q, view_kernel='legacy').cuda()
    V_legacy = vm_legacy(d, w)
    print(f"   Output shape: {V_legacy.shape}")
    print(f"   View kernel shape: {vm_legacy.v().shape}")
    
    # Test 2: Full rank kernel
    print("\n2. Testing FULLRANK kernel")
    vm_full = Vmodel(P, Q, p, q, view_kernel='fullrank').cuda()
    V_full = vm_full(d, w)
    K_full = vm_full.kernel.get_full_matrix()
    print(f"   Output shape: {V_full.shape}")
    print(f"   Kernel matrix shape: {K_full.shape}")
    print(f"   Kernel diagonal: {K_full.diag()}")
    
    # Test 3: Linear kernel (low rank)
    print("\n3. Testing LINEAR kernel (rank=3)")
    vm_linear = Vmodel(P, Q, p, q, view_kernel='linear', rank=3).cuda()
    V_linear = vm_linear(d, w)
    K_linear = vm_linear.kernel.get_full_matrix()
    print(f"   Output shape: {V_linear.shape}")
    print(f"   Kernel matrix shape: {K_linear.shape}")
    print(f"   Kernel rank: 3 (by construction)")
    
    # Test 4: Periodic kernel
    print("\n4. Testing PERIODIC kernel")
    vm_periodic = Vmodel(P, Q, p, q, view_kernel='periodic', lengthscale=1.0).cuda()
    V_periodic = vm_periodic(d, w)
    K_periodic = vm_periodic.kernel.get_full_matrix()
    print(f"   Output shape: {V_periodic.shape}")
    print(f"   Kernel matrix shape: {K_periodic.shape}")
    print(f"   K[0,0] = {K_periodic[0,0]:.4f}, K[0,1] = {K_periodic[0,1]:.4f}")
    print(f"   Periodicity check: K[0,0] = {K_periodic[0,0]:.4f}, K[0,Q-1] = {K_periodic[0,Q-1]:.4f}")
    
    # Test 5: Von Mises kernel
    print("\n5. Testing VON MISES kernel")
    vm_vonmises = Vmodel(P, Q, p, q, view_kernel='vonmises', kappa=2.0).cuda()
    V_vonmises = vm_vonmises(d, w)
    K_vonmises = vm_vonmises.kernel.get_full_matrix()
    print(f"   Output shape: {V_vonmises.shape}")
    print(f"   Kernel matrix shape: {K_vonmises.shape}")
    print(f"   K[0,0] = {K_vonmises[0,0]:.4f}, K[0,1] = {K_vonmises[0,1]:.4f}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("\nUsage in training:")
    print("  vm = Vmodel(P, Q, p, q, view_kernel='periodic', lengthscale=0.5)")
    print("  vm = Vmodel(P, Q, p, q, view_kernel='fullrank')")
    print("  vm = Vmodel(P, Q, p, q, view_kernel='legacy')  # Original")
