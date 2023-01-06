# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import torch
import nvdiffrast.torch as dr

from . import util
from . import mesh

######################################################################################
# Computes the image gradient, useful for kd/ks smoothness losses
######################################################################################
def image_grad(buf, std=0.01):
    t, s = torch.meshgrid(torch.linspace(-1.0 + 1.0 / buf.shape[1], 1.0 - 1.0 / buf.shape[1], buf.shape[1], device="cuda"), 
                          torch.linspace(-1.0 + 1.0 / buf.shape[2], 1.0 - 1.0 / buf.shape[2], buf.shape[2], device="cuda"),
                          indexing='ij')
    tc   = torch.normal(mean=0, std=std, size=(buf.shape[0], buf.shape[1], buf.shape[2], 2), device="cuda") + torch.stack((s, t), dim=-1)[None, ...]
    tap  = dr.texture(buf, tc, filter_mode='linear', boundary_mode='clamp')
    return torch.abs(tap[..., :-1] - buf[..., :-1]) * tap[..., -1:] * buf[..., -1:]

######################################################################################
# Computes the avergage edge length of a mesh. 
# Rough estimate of the tessellation of a mesh. Can be used e.g. to clamp gradients
######################################################################################
def avg_edge_length(v_pos, t_pos_idx):
    e_pos_idx = mesh.compute_edges(t_pos_idx)
    edge_len  = util.length(v_pos[e_pos_idx[:, 0]] - v_pos[e_pos_idx[:, 1]])
    return torch.mean(edge_len)

######################################################################################
# Laplacian regularization using umbrella operator (Fujiwara / Desbrun).
# https://mgarland.org/class/geom04/material/smoothing.pdf
######################################################################################
def laplace_regularizer_const(v_pos, t_pos_idx):
    term = torch.zeros_like(v_pos)
    norm = torch.zeros_like(v_pos[..., 0:1])

    v0 = v_pos[t_pos_idx[:, 0], :]
    v1 = v_pos[t_pos_idx[:, 1], :]
    v2 = v_pos[t_pos_idx[:, 2], :]

    term.scatter_add_(0, t_pos_idx[:, 0:1].repeat(1,3), (v1 - v0) + (v2 - v0))
    term.scatter_add_(0, t_pos_idx[:, 1:2].repeat(1,3), (v0 - v1) + (v2 - v1))
    term.scatter_add_(0, t_pos_idx[:, 2:3].repeat(1,3), (v0 - v2) + (v1 - v2))

    two = torch.ones_like(v0) * 2.0
    norm.scatter_add_(0, t_pos_idx[:, 0:1], two)
    norm.scatter_add_(0, t_pos_idx[:, 1:2], two)
    norm.scatter_add_(0, t_pos_idx[:, 2:3], two)

    term = term / torch.clamp(norm, min=1.0)

    return torch.mean(term**2)

######################################################################################
# Smooth vertex normals
######################################################################################
def normal_consistency(v_pos, t_pos_idx):
    # Compute face normals
    v0 = v_pos[t_pos_idx[:, 0], :]
    v1 = v_pos[t_pos_idx[:, 1], :]
    v2 = v_pos[t_pos_idx[:, 2], :]

    face_normals = util.safe_normalize(torch.cross(v1 - v0, v2 - v0))

    tris_per_edge = mesh.compute_edge_to_face_mapping(t_pos_idx)

    # Fetch normals for both faces sharind an edge
    n0 = face_normals[tris_per_edge[:, 0], :]
    n1 = face_normals[tris_per_edge[:, 1], :]

    # Compute error metric based on normal difference
    term = torch.clamp(util.dot(n0, n1), min=-1.0, max=1.0)
    term = (1.0 - term) * 0.5

    return torch.mean(torch.abs(term))

def laplacian_cot(verts, faces):
    """
    Compute the cotangent laplacian
    Inspired by https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/loss/mesh_laplacian_smoothing.html
    Parameters
    ----------
    verts : torch.Tensor
        Vertex positions.
    faces : torch.Tensor
        array of triangle faces.
    """

    # V = sum(V_n), F = sum(F_n)
    V, F = verts.shape[0], faces.shape[0]

    face_verts = verts[faces]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    # Side lengths of each triangle, of shape (sum(F_n),)
    # A is the side opposite v1, B is opposite v2, and C is opposite v3
    A = (v1 - v2).norm(dim=1)
    B = (v0 - v2).norm(dim=1)
    C = (v0 - v1).norm(dim=1)

    # Area of each triangle (with Heron's formula); shape is (sum(F_n),)
    s = 0.5 * (A + B + C)
    # note that the area can be negative (close to 0) causing nans after sqrt()
    # we clip it to a small positive value
    area = (s * (s - A) * (s - B) * (s - C)).clamp_(min=1e-12).sqrt()

    # Compute cotangents of angles, of shape (sum(F_n), 3)
    A2, B2, C2 = A * A, B * B, C * C
    cota = (B2 + C2 - A2) / area
    cotb = (A2 + C2 - B2) / area
    cotc = (A2 + B2 - C2) / area
    cot = torch.stack([cota, cotb, cotc], dim=1)
    cot /= 4.0

    # Construct a sparse matrix by basically doing:
    # L[v1, v2] = cota
    # L[v2, v0] = cotb
    # L[v0, v1] = cotc
    ii = faces[:, [1, 2, 0]]
    jj = faces[:, [2, 0, 1]]
    idx = torch.stack([ii, jj], dim=0).view(2, F * 3)
    L = torch.sparse.FloatTensor(idx, cot.view(-1), (V, V))

    # Make it symmetric; this means we are also setting
    # L[v2, v1] = cota
    # L[v0, v2] = cotb
    # L[v1, v0] = cotc
    L += L.t()

    # Add the diagonal indices
    vals = torch.sparse.sum(L, dim=0).to_dense()
    indices = torch.arange(V, device='cuda')
    idx = torch.stack([indices, indices], dim=0)
    L = torch.sparse.FloatTensor(idx, vals, (V, V)) - L
    return L

def laplacian_uniform(verts, faces):
    """
    Compute the uniform laplacian
    Parameters
    ----------
    verts : torch.Tensor
        Vertex positions.
    faces : torch.Tensor
        array of triangle faces.
    """
    V = verts.shape[0]
    F = faces.shape[0]

    # Neighbor indices
    ii = faces[:, [1, 2, 0]].flatten()
    jj = faces[:, [2, 0, 1]].flatten()
    adj = torch.stack([torch.cat([ii, jj]), torch.cat([jj, ii])], dim=0).unique(dim=1)
    adj_values = torch.ones(adj.shape[1], device='cuda', dtype=torch.float)

    # Diagonal indices
    diag_idx = adj[0]

    # Build the sparse matrix
    idx = torch.cat((adj, torch.stack((diag_idx, diag_idx), dim=0)), dim=1)
    values = torch.cat((-adj_values, adj_values))

    # The coalesce operation sums the duplicate indices, resulting in the
    # correct diagonal
    return torch.sparse_coo_tensor(idx, values, (V,V)).coalesce()
