"""
Core mesh operations: loading, per-triangle gradient basis, cotangent Laplacian.
"""

import numpy as np
import scipy.sparse as sp
import trimesh


def load_mesh(path: str):
    """Load a triangle mesh and return (V, F) as float64/int32 arrays."""
    mesh = trimesh.load(path, force="mesh", process=False)
    V = np.array(mesh.vertices, dtype=np.float64)
    F = np.array(mesh.faces, dtype=np.int32)
    return V, F


def weld_vertices(V, F, tol=1e-6):
    """
    Merge coincident (duplicated) vertices into one.

    Many ``.obj`` exports split the mesh along seams, leaving several vertices
    at the exact same position. Topologically this disconnects the surface, so
    the heat-method geodesic can't propagate across the seam and clamps whole
    regions to a constant value (which then can't be crocheted — they have no
    isolines). Welding restitches those seams.

    Returns
    -------
    V_w     : (m, 3) welded vertex positions (m <= n)
    F_w     : (k, 3) faces re-indexed onto V_w, with degenerate faces removed
    old2new : (n,) map from each original vertex index to its welded index
    """
    V = np.asarray(V, dtype=np.float64)
    diag = float(np.linalg.norm(V.max(axis=0) - V.min(axis=0))) or 1.0
    key = np.round(V / (diag * tol)).astype(np.int64)
    _, old2new = np.unique(key, axis=0, return_inverse=True)
    old2new = np.asarray(old2new).reshape(-1)

    m = int(old2new.max()) + 1
    V_w = np.zeros((m, 3), dtype=np.float64)
    counts = np.zeros(m)
    np.add.at(V_w, old2new, V)
    np.add.at(counts, old2new, 1.0)
    V_w /= counts[:, None]

    F_w = old2new[np.asarray(F)]
    nondegen = ((F_w[:, 0] != F_w[:, 1])
                & (F_w[:, 1] != F_w[:, 2])
                & (F_w[:, 0] != F_w[:, 2]))
    F_w = F_w[nondegen].astype(np.int32)
    return V_w, F_w, old2new


def normalize_to_unit_area(V, F):
    """Scale V so that total surface area equals 1. Returns (V_norm, scale)."""
    areas = face_areas(V, F)
    total = areas.sum()
    scale = np.sqrt(total)
    return V / scale, scale


def face_areas(V, F):
    """Area of each triangle, shape (n_faces,)."""
    p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cross = np.cross(p1 - p0, p2 - p0)
    return 0.5 * np.linalg.norm(cross, axis=1)


def face_normals(V, F):
    """Unit outward normal of each triangle, shape (n_faces, 3)."""
    p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cross = np.cross(p1 - p0, p2 - p0)
    norm = np.linalg.norm(cross, axis=1, keepdims=True)
    return cross / (norm + 1e-300)


def gradient_basis(V, F):
    """
    Per-triangle gradient of each barycentric function.

    Returns
    -------
    grad_phi : ndarray, shape (n_faces, 3, 3)
        grad_phi[f, k, :] = gradient of φ_k (k=0,1,2) within face f.
    areas : ndarray, shape (n_faces,)
    normals : ndarray, shape (n_faces, 3)
    """
    p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cross = np.cross(p1 - p0, p2 - p0)          # (n_faces, 3)
    double_area = np.linalg.norm(cross, axis=1)  # (n_faces,)
    n = cross / (double_area[:, None] + 1e-300)  # unit normals

    # Edge opposite to vertex k
    e_opp = np.stack([p2 - p1, p0 - p2, p1 - p0], axis=1)  # (n_faces, 3, 3)

    # ∇φ_k = (n × e_opp_k) / (2 * area) = (n × e_opp_k) / double_area
    n_exp = n[:, None, :]                         # (n_faces, 1, 3)
    grad_phi = np.cross(n_exp, e_opp) / double_area[:, None, None]

    return grad_phi, 0.5 * double_area, n


def scalar_gradient(V, F, scalar, grad_phi=None):
    """
    Gradient of a scalar function (defined at vertices) within each face.

    Parameters
    ----------
    scalar : ndarray, shape (n_verts,)
    grad_phi : precomputed from gradient_basis (optional)

    Returns
    -------
    grad : ndarray, shape (n_faces, 3)
    """
    if grad_phi is None:
        grad_phi, _, _ = gradient_basis(V, F)

    f0 = scalar[F[:, 0]]
    f1 = scalar[F[:, 1]]
    f2 = scalar[F[:, 2]]

    # ∇f = Σ_k f_k ∇φ_k
    return (f0[:, None] * grad_phi[:, 0, :]
            + f1[:, None] * grad_phi[:, 1, :]
            + f2[:, None] * grad_phi[:, 2, :])


def cotangent_laplacian(V, F):
    """
    Negative-semidefinite cotangent Laplacian L and lumped mass matrix M.

    The convention here is the geometric Laplacian:
        L[i,j] = (cot α + cot β) / 2  for edge (i,j)
        L[i,i] = -Σ_{j~i} L[i,j]

    Returns
    -------
    L : scipy.sparse.csr_matrix, shape (n_verts, n_verts)
    M : scipy.sparse.diags, shape (n_verts, n_verts)  (lumped mass)
    """
    n = len(V)
    p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]

    def _cot(a, b, c):
        ab, ac = b - a, c - a
        dot = np.einsum("fi,fi->f", ab, ac)
        cross_norm = np.linalg.norm(np.cross(ab, ac), axis=1)
        return dot / (cross_norm + 1e-300)

    cot0 = _cot(p0, p1, p2)  # angle at v0, weight for edge (v1, v2)
    cot1 = _cot(p1, p0, p2)  # angle at v1, weight for edge (v0, v2)
    cot2 = _cot(p2, p0, p1)  # angle at v2, weight for edge (v0, v1)

    ii, jj, vv = [], [], []
    for i_local, j_local, cot in [
        (1, 2, cot0), (0, 2, cot1), (0, 1, cot2)
    ]:
        vi = F[:, i_local]
        vj = F[:, j_local]
        w = 0.5 * cot
        ii += [vi, vj, vi, vj]
        jj += [vj, vi, vi, vj]
        vv += [w, w, -w, -w]

    rows = np.concatenate(ii)
    cols = np.concatenate(jj)
    vals = np.concatenate(vv)
    L = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))

    # Lumped mass: one-third of surrounding face areas per vertex
    areas = face_areas(V, F)
    mass = np.zeros(n)
    for k in range(3):
        np.add.at(mass, F[:, k], areas / 3.0)
    M = sp.diags(mass)

    return L, M


def vertex_adjacency(F, n_verts):
    """Return list-of-lists: adjacency[v] = list of neighbouring vertex indices."""
    adj = [[] for _ in range(n_verts)]
    for f in F:
        for a, b in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])]:
            adj[a].append(b)
            adj[b].append(a)
    return [list(set(nb)) for nb in adj]


def vertex_faces(F, n_verts):
    """Return list-of-lists: vf[v] = list of face indices containing v."""
    vf = [[] for _ in range(n_verts)]
    for fi, f in enumerate(F):
        for v in f:
            vf[v].append(fi)
    return vf
