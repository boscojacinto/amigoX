"""
Compute the column-order function g on the mesh.

Goal: g : M_C → R such that ⟨J∇f, ∇g⟩ ≈ 1 everywhere,
i.e. g increases at unit arc-length speed along each isoline of f.

We minimise:
    E(g) = ∫_M |⟨T, ∇g⟩ - 1|² dA  +  ε ∫_M |∇g|² dA

where T = J∇f  (90° rotation of ∇f in the tangent plane).

The Euler-Lagrange equation gives a sparse linear system
    (K_aniso + ε L) g = b
subject to g = 0 on the seam vertices.

For regions of negative Gaussian curvature we use a curvature-adapted
sampling rate via h(k) = tanh(-k/α)/2 + 1  (Section 7.1.2 of the paper).
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .mesh_ops import gradient_basis, scalar_gradient, cotangent_laplacian


def compute_column_order(V, F, f, seam_verts, eps: float = 1e-3, alpha: float = 10.0):
    """
    Solve for g given geodesic distance f and seam vertex indices.

    Parameters
    ----------
    V, F        : mesh geometry
    f           : geodesic distance at each vertex, shape (n_verts,)
    seam_verts  : vertex indices along the cut (Dirichlet BC: g = 0)
    eps         : Laplacian regularisation weight
    alpha       : curvature-adaptation parameter (paper uses 10)

    Returns
    -------
    g : ndarray, shape (n_verts,)
    """
    n = len(V)
    grad_phi, areas, normals = gradient_basis(V, F)

    # ∇f per triangle
    grad_f = scalar_gradient(V, F, f, grad_phi)

    # T = J∇f = n × ∇f  (90° rotation in the tangent plane)
    T = np.cross(normals, grad_f)          # (n_faces, 3)

    # Curvature-adapted speed: h(k) where k = directional curvature along isoline
    # We approximate k as |∇f| variation — use T magnitude as proxy.
    T_norm = np.linalg.norm(T, axis=1)    # (n_faces,)
    h = np.tanh(-T_norm / alpha) / 2 + 1  # adaptation factor per face
    # Target directional derivative of g along T = h (instead of 1)
    target = h                             # shape (n_faces,)

    # a_k[f] = T_f · ∇φ_k  for k = 0, 1, 2
    a = np.einsum("fi,fki->fk", T, grad_phi)   # (n_faces, 3) — a[:,k] = T·∇φ_k

    # -----------------------------------------------------------------------
    # Assemble anisotropic stiffness K_aniso
    # K[i,j] += area_f * a[f,k_i] * a[f,k_j]  for each face f and local (k_i,k_j)
    # -----------------------------------------------------------------------
    weighted_a = a * areas[:, None]             # (n_faces, 3)

    rows_list, cols_list, vals_list = [], [], []
    for ki in range(3):
        for kj in range(3):
            rows_list.append(F[:, ki])
            cols_list.append(F[:, kj])
            vals_list.append(weighted_a[:, ki] * a[:, kj])

    all_rows = np.concatenate(rows_list)
    all_cols = np.concatenate(cols_list)
    all_vals = np.concatenate(vals_list)
    K_aniso = sp.csr_matrix((all_vals, (all_rows, all_cols)), shape=(n, n))

    # -----------------------------------------------------------------------
    # Assemble RHS: b[i] += area_f * target_f * a[f, k]
    # -----------------------------------------------------------------------
    b = np.zeros(n)
    for k in range(3):
        np.add.at(b, F[:, k], areas * target * a[:, k])

    # -----------------------------------------------------------------------
    # Laplacian regularisation
    # -----------------------------------------------------------------------
    L, _ = cotangent_laplacian(V, F)
    # L is negative semi-definite; we add -eps*L so that the full system
    # K_aniso - eps*L is positive (semi-)definite.
    A = K_aniso - eps * L

    # -----------------------------------------------------------------------
    # Apply Dirichlet BC: g = 0 on seam_verts
    # -----------------------------------------------------------------------
    seam_set = np.zeros(n, dtype=bool)
    seam_set[seam_verts] = True
    free = np.where(~seam_set)[0]

    A_free = A[free][:, free]
    b_free = b[free]
    # subtract contribution of constrained (zero) dofs — they're already zero

    g = np.zeros(n)
    if len(free) > 0:
        g_free = spla.spsolve(A_free.tocsr(), b_free)
        g[free] = g_free

    # Ensure g is non-negative (arc length is always ≥ 0)
    g = np.maximum(g, 0.0)
    return g
