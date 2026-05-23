# gaussian_shell.py
# Differentiable spherical-shell Gaussian surface for 3D reconstruction.
#
# Idea: instead of a dense O(N^3) voxel grid, represent the object as a set of
# K shells, each a sphere of N Gaussians at fixed directions d_i with a
# *learnable* radial position r_i (plus learnable opacity and color).
# Optimizing r_i sculpts the surface in/out — "differentiable 3D sculpting".
#
# Layers:
#   - Multi-layer (concentric): K shells at staggered radii — captures
#     overhangs and concavities because rays can hit multiple shells.
#   - Multi-anchor: each shell can have its own anchor point (defaults to
#     origin); useful for branching / disjoint-lobed shapes.

import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

# Allow TF32 on Ampere+ (A100/H100). Free perf for the projection matmul.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from video_orbit_voxel_recon import (
    make_intrinsics,
    build_orbit_poses_from_period,
    load_frames_as_tensors,
    estimate_orbit_period,
    estimate_orbit_direction,
)


# ----------------- Sphere sampling & neighbors -----------------

def fibonacci_sphere(n):
    """N nearly-uniform unit vectors on a sphere via the Fibonacci lattice."""
    k = torch.arange(n, dtype=torch.float64)
    phi = math.pi * (3.0 - math.sqrt(5.0))
    z = 1.0 - 2.0 * (k + 0.5) / n
    r_xy = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
    theta = phi * k
    x = r_xy * torch.cos(theta)
    y = r_xy * torch.sin(theta)
    return torch.stack([x, y, z], dim=-1).float()


def knn_indices(points, k):
    """For each point, indices of its k nearest neighbors (excluding itself).

    points: [N, 3]   Returns: [N, k] long indices.
    """
    d2 = torch.cdist(points, points)
    d2.fill_diagonal_(float("inf"))
    return torch.topk(d2, k=k, largest=False).indices


# ----------------- Shell module -----------------

class GaussianShell(nn.Module):
    """Stack of K spherical shells of Gaussians.

    Each shell s has its own anchor a_s (default origin), N directions d_i
    (Fibonacci-sampled, fixed), and per-Gaussian learnable radius r_i,
    opacity, and color. Position of Gaussian (s, i) in world space:
        p = a_s + r_{s,i} * d_i

    Args:
        n_per_shell:   number of Gaussians per shell
        n_shells:      number of concentric / offset shells
        init_radii:    list/tensor of initial radii for each shell (len = n_shells).
                       Defaults to evenly spaced radii in [0.4, 1.6].
        anchors:       [n_shells, 3] tensor of anchor points (default zeros).
                       Pass non-zero anchors for the multi-anchor variant.
        sigma_world:   world-space std of each Gaussian. Defaults to the
                       inter-Gaussian spacing on the outermost shell.
        share_directions:   if True (default), all shells share one direction
                            grid — makes per-shell Laplacian neighbors trivial.
    """

    def __init__(
        self,
        n_per_shell=8000,
        n_shells=1,
        init_radii=None,
        anchors=None,
        sigma_world=None,
        init_alpha=0.5,
        alpha_cap=1.0,
        n_neighbors=8,
        share_directions=True,
    ):
        super().__init__()
        self.n_per_shell = int(n_per_shell)
        self.n_shells = int(n_shells)
        self.share_directions = bool(share_directions)
        # Per-Gaussian alpha is multiplied by this. <1.0 prevents any single
        # splat from fully saturating opacity, so transmittance never goes to
        # zero and gradient leaks to occluded Gaussians (back of object).
        self.alpha_cap = float(alpha_cap)

        if init_radii is None:
            if n_shells == 1:
                init_radii = [1.2]
            else:
                init_radii = list(np.linspace(0.4, 1.6, n_shells))
        init_radii = torch.as_tensor(init_radii, dtype=torch.float32)
        assert init_radii.numel() == n_shells

        if anchors is None:
            anchors = torch.zeros(n_shells, 3, dtype=torch.float32)
        else:
            anchors = torch.as_tensor(anchors, dtype=torch.float32).reshape(n_shells, 3)
        self.register_buffer("anchors", anchors)

        # Directions: shared across shells, or independent per shell
        if share_directions:
            d = fibonacci_sphere(n_per_shell)              # [N, 3]
            self.register_buffer("directions", d)          # [N, 3]
        else:
            d_list = [fibonacci_sphere(n_per_shell) for _ in range(n_shells)]
            self.register_buffer("directions", torch.stack(d_list, dim=0))  # [K, N, 3]

        # Sigma_world default: spacing on outermost shell
        if sigma_world is None:
            spacing = math.sqrt(4.0 * math.pi / n_per_shell)
            outer_r = float(init_radii.max().item())
            sigma_world = spacing * outer_r * 0.6
        self.sigma_world = float(sigma_world)

        # Learnable params: per-shell, per-Gaussian
        # radius_param: [K, N]
        radius_init = init_radii[:, None].expand(n_shells, n_per_shell).contiguous()
        self.radius_param = nn.Parameter(radius_init.clone())

        # opacity logits: [K, N]; init alpha low to let optimization grow it
        a0 = math.log(init_alpha / (1.0 - init_alpha))
        self.alpha_logit = nn.Parameter(
            torch.full((n_shells, n_per_shell), a0, dtype=torch.float32)
        )

        # color logits: [K, N, 3]; ~0 → sigmoid ≈ 0.5
        self.color_logit = nn.Parameter(torch.zeros(n_shells, n_per_shell, 3))

        # Neighbor graph for Laplacian smoothness on each shell
        if share_directions:
            nbr = knn_indices(self.directions, n_neighbors)         # [N, k]
            self.register_buffer("neighbors", nbr)                   # [N, k]
        else:
            nbrs = [knn_indices(self.directions[s], n_neighbors)
                    for s in range(n_shells)]
            self.register_buffer("neighbors", torch.stack(nbrs, 0))  # [K, N, k]

        # Depth-based floater suppression mask (True = suppress alpha to 0)
        self.register_buffer("suppress_mask",
                             torch.zeros(n_shells, n_per_shell, dtype=torch.bool))

    # ----- accessors -----

    def positions(self):
        """[K*N, 3] — world-space centers of every Gaussian."""
        if self.share_directions:
            d = self.directions[None, :, :]                          # [1, N, 3]
        else:
            d = self.directions                                      # [K, N, 3]
        p = self.radius_param[..., None] * d + self.anchors[:, None, :]
        return p.reshape(-1, 3)

    def alphas(self):
        a = torch.sigmoid(self.alpha_logit).reshape(-1)              # [K*N]
        if self.alpha_cap < 1.0:
            a = a * self.alpha_cap
        if self.suppress_mask.any():
            a = a * (1.0 - self.suppress_mask.reshape(-1).float())
        return a

    def colors(self):
        return torch.sigmoid(self.color_logit).reshape(-1, 3)         # [K*N, 3]

    # ----- regularizers -----

    def laplacian_radius_loss(self):
        """Mean squared difference between r_i and its on-sphere neighbors,
        per shell, averaged over all shells."""
        r = self.radius_param                                         # [K, N]
        if self.share_directions:
            r_nb = r[:, self.neighbors]                               # [K, N, k]
        else:
            # gather along N independently per shell
            K, N, k = self.neighbors.shape
            r_nb = torch.gather(
                r.unsqueeze(-1).expand(K, N, k),
                1,
                self.neighbors,
            )
        return ((r.unsqueeze(-1) - r_nb) ** 2).mean()

    def alpha_sparsity_loss(self):
        return self.alphas().mean()

    # ----- ops -----

    @torch.no_grad()
    def clamp_radii(self, r_min=0.0, r_max=2.0):
        self.radius_param.clamp_(min=r_min, max=r_max)

    @classmethod
    def from_npz(cls, path):
        """Reconstruct a GaussianShell from a save_shell_npz file."""
        data = np.load(path, allow_pickle=False)
        n_shells = int(data["n_shells"])
        n_per_shell = int(data["n_per_shell"])
        share = bool(data["share_directions"])
        sigma_world = float(data["sigma_world"])
        anchors = torch.from_numpy(data["anchors"]).float()

        alpha_cap = float(data["alpha_cap"]) if "alpha_cap" in data else 1.0

        # Build with placeholder radii; we overwrite below.
        obj = cls(
            n_per_shell=n_per_shell, n_shells=n_shells,
            init_radii=[1.0] * n_shells, anchors=anchors,
            sigma_world=sigma_world, share_directions=share,
            alpha_cap=alpha_cap,
        )

        # Overwrite directions buffer (Fibonacci is deterministic, but be safe)
        d = torch.from_numpy(data["directions"]).float()
        obj.directions.copy_(d)

        # Overwrite radii
        obj.radius_param.data.copy_(torch.from_numpy(data["radii"]).float())

        # Saved alphas/colors are post-sigmoid; invert to logits.
        alphas = torch.from_numpy(data["alphas"]).float().clamp(1e-6, 1 - 1e-6)
        obj.alpha_logit.data.copy_(torch.logit(alphas))
        colors = torch.from_numpy(data["colors"]).float().clamp(1e-6, 1 - 1e-6)
        obj.color_logit.data.copy_(torch.logit(colors))

        # Load suppress_mask if present
        if "suppress_mask" in data:
            obj.suppress_mask.copy_(
                torch.from_numpy(data["suppress_mask"].astype(bool)))
        return obj


# ----------------- Differentiable rendering -----------------

def _composite_chunk(
    u, v, sigma_screen, alpha, color, xs, ys, T, img, acc, eps
):
    """Front-to-back alpha-composite a single chunk of (already depth-sorted)
    Gaussians into the running (T, img, acc) accumulators.

    All inputs are 1D over Gaussians (length k); xs, ys are [H, W].
    """
    du = xs[None, :, :] - u[:, None, None]                            # [k, H, W]
    dv = ys[None, :, :] - v[:, None, None]
    inv_2s2 = 0.5 / (sigma_screen[:, None, None] ** 2)
    g = torch.exp(-(du * du + dv * dv) * inv_2s2)
    w = (alpha[:, None, None] * g).clamp(max=0.999)                   # [k, H, W]

    # Within-chunk front-to-back transmittance
    log_om = torch.log1p(-w + eps)                                    # log(1 - w)
    log_T_chunk = torch.cumsum(log_om, dim=0)
    T_after = torch.exp(log_T_chunk)                                  # [k, H, W]
    T_before = torch.cat(
        [torch.ones_like(T_after[:1]), T_after[:-1]], dim=0
    )                                                                  # [k, H, W]

    # Combine with running transmittance from prior chunks
    contrib = T[None] * T_before * w                                  # [k, H, W]
    img = img + (contrib[..., None] * color[:, None, None, :]).sum(dim=0).permute(2, 0, 1)
    acc = acc + contrib.sum(dim=0)
    T = T * T_after[-1]
    return T, img, acc


def render_shell(shell, K, R, t, H, W, near=0.05, chunk_size=2048, eps=1e-6,
                 use_checkpoint=True):
    """Render the Gaussian shell stack for one camera view.

    use_checkpoint=True wraps each chunk in torch.utils.checkpoint so per-chunk
    activations are recomputed during backward — bounds peak memory to roughly
    one chunk's worth of intermediates rather than all chunks combined.

    Returns:
        img:  [3, H, W] in [0, 1]
        acc:  [H, W]    accumulated alpha
    """
    device = shell.radius_param.device
    p_world = shell.positions()                                       # [G, 3]
    p_cam = p_world @ R.T + t.view(1, 3)                              # [G, 3]
    z = p_cam[:, 2]

    fx = K[0, 0]; fy = K[1, 1]
    cx = K[0, 2]; cy = K[1, 2]
    safe_z = torch.clamp(z, min=near)
    u = fx * p_cam[:, 0] / safe_z + cx
    v = fy * p_cam[:, 1] / safe_z + cy

    sigma_screen = (shell.sigma_world * fx) / safe_z
    sigma_screen = sigma_screen.clamp(min=0.5)                        # avoid sub-pixel collapse

    front = (z > near).float()
    alpha = shell.alphas() * front                                    # [G]
    color = shell.colors()                                            # [G, 3]

    # Depth sort (sort indices are non-differentiable — that's fine, gradients
    # still flow through the values)
    order = torch.argsort(z, dim=0)
    u_s = u[order]; v_s = v[order]
    s_s = sigma_screen[order]
    a_s = alpha[order]
    c_s = color[order]

    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )

    T = torch.ones(H, W, device=device)
    img = torch.zeros(3, H, W, device=device)
    acc = torch.zeros(H, W, device=device)

    G = u_s.shape[0]
    chunk = G if (chunk_size is None or chunk_size >= G) else int(chunk_size)
    eps_t = torch.tensor(eps, device=device, dtype=torch.float32)
    do_ckpt = use_checkpoint and torch.is_grad_enabled()
    for i in range(0, G, chunk):
        j = min(i + chunk, G)
        if do_ckpt:
            T, img, acc = torch.utils.checkpoint.checkpoint(
                _composite_chunk,
                u_s[i:j], v_s[i:j], s_s[i:j], a_s[i:j], c_s[i:j],
                xs, ys, T, img, acc, eps_t,
                use_reentrant=False,
            )
        else:
            T, img, acc = _composite_chunk(
                u_s[i:j], v_s[i:j], s_s[i:j], a_s[i:j], c_s[i:j],
                xs, ys, T, img, acc, eps_t,
            )
    return img, acc


def render_shell_views(shell, K, poses, img_size, chunk_size=2048):
    H, W = img_size
    imgs, accs = [], []
    for R, t in poses:
        img, acc = render_shell(shell, K, R, t, H, W, chunk_size=chunk_size)
        imgs.append(img)
        accs.append(acc)
    return imgs, accs


# ----------------- Tile-based renderer -----------------
#
# Per-Gaussian frustum/bbox cull: each splat only contributes to pixels within
# ±k·σ_screen of its center. We tile the image into T×T blocks, build a per-
# tile list of overlapping Gaussians, and run the compositing only over each
# tile's pixels with its own slice of Gaussians. Same math, far less waste.

def render_shell_tiled(
    shell, K, R, t, H, W,
    near=0.05, tile_size=16, sigma_extent=3.0,
    use_checkpoint=False, chunk_size=4096, eps=1e-6,
):
    device = shell.radius_param.device
    p_world = shell.positions()                                       # [G, 3]
    p_cam = p_world @ R.T + t.view(1, 3)
    z = p_cam[:, 2]

    fx = K[0, 0]; fy = K[1, 1]; cx = K[0, 2]; cy = K[1, 2]
    safe_z = torch.clamp(z, min=near)
    u = fx * p_cam[:, 0] / safe_z + cx
    v = fy * p_cam[:, 1] / safe_z + cy

    sigma_screen = (shell.sigma_world * fx) / safe_z
    sigma_screen = sigma_screen.clamp(min=0.5)

    front = (z > near).float()
    alpha = shell.alphas() * front
    color = shell.colors()

    # Sort by depth — depth ordering is global, identical across all tiles
    order = torch.argsort(z, dim=0)
    u_s = u[order]; v_s = v[order]
    s_s = sigma_screen[order]; a_s = alpha[order]; c_s = color[order]
    front_s = front[order]

    # Screen-space bbox per Gaussian (±k·σ), clamped to image
    ext = sigma_extent * s_s
    u_lo = u_s - ext; u_hi = u_s + ext
    v_lo = v_s - ext; v_hi = v_s + ext
    # Quick reject: any Gaussian whose bbox is entirely off-screen
    on_screen = (u_hi >= 0) & (u_lo < W) & (v_hi >= 0) & (v_lo < H) & (front_s > 0.5)
    if on_screen.any():
        idx_kept = on_screen.nonzero(as_tuple=False).squeeze(-1)
        u_s = u_s[idx_kept]; v_s = v_s[idx_kept]
        s_s = s_s[idx_kept]; a_s = a_s[idx_kept]; c_s = c_s[idx_kept]
        u_lo = u_lo[idx_kept]; u_hi = u_hi[idx_kept]
        v_lo = v_lo[idx_kept]; v_hi = v_hi[idx_kept]
    else:
        return (torch.zeros(3, H, W, device=device),
                torch.zeros(H, W, device=device))

    # Convert each Gaussian's bbox into a tile index range (inclusive)
    tx_lo = torch.clamp((u_lo / tile_size).floor().long(), 0)
    tx_hi = torch.clamp((u_hi / tile_size).floor().long(), max=(W - 1) // tile_size)
    ty_lo = torch.clamp((v_lo / tile_size).floor().long(), 0)
    ty_hi = torch.clamp((v_hi / tile_size).floor().long(), max=(H - 1) // tile_size)

    n_tx = (W + tile_size - 1) // tile_size
    n_ty = (H + tile_size - 1) // tile_size

    out_img = torch.zeros(3, H, W, device=device)
    out_acc = torch.zeros(H, W, device=device)
    eps_t = torch.tensor(eps, device=device, dtype=torch.float32)
    do_ckpt = use_checkpoint and torch.is_grad_enabled()

    def _tile_chunk(u_c, v_c, s_c, a_c, c_c, xs, ys, T, tile_img, tile_acc, eps_t):
        du = xs[None] - u_c[:, None, None]
        dv = ys[None] - v_c[:, None, None]
        inv_2s2 = 0.5 / (s_c[:, None, None] ** 2)
        g = torch.exp(-(du * du + dv * dv) * inv_2s2)
        w = (a_c[:, None, None] * g).clamp(max=0.999)
        log_om = torch.log1p(-w + eps_t)
        log_T_chunk = torch.cumsum(log_om, dim=0)
        T_after = torch.exp(log_T_chunk)
        T_before = torch.cat(
            [torch.ones_like(T_after[:1]), T_after[:-1]], dim=0
        )
        contrib = T[None] * T_before * w
        tile_img = tile_img + (contrib[..., None]
                               * c_c[:, None, None, :]).sum(dim=0).permute(2, 0, 1)
        tile_acc = tile_acc + contrib.sum(dim=0)
        T = T * T_after[-1]
        return T, tile_img, tile_acc

    for ty in range(n_ty):
        in_row = (ty_lo <= ty) & (ty_hi >= ty)
        if not in_row.any():
            continue
        row_idx = in_row.nonzero(as_tuple=False).squeeze(-1)
        # Subset for the row
        u_r = u_s[row_idx]; v_r = v_s[row_idx]; s_r = s_s[row_idx]
        a_r = a_s[row_idx]; c_r = c_s[row_idx]
        tx_lo_r = tx_lo[row_idx]; tx_hi_r = tx_hi[row_idx]

        y0 = ty * tile_size
        y1 = min(y0 + tile_size, H)

        for tx in range(n_tx):
            in_tile = (tx_lo_r <= tx) & (tx_hi_r >= tx)
            if not in_tile.any():
                continue
            t_idx = in_tile.nonzero(as_tuple=False).squeeze(-1)
            u_t = u_r[t_idx]; v_t = v_r[t_idx]; s_t = s_r[t_idx]
            a_t = a_r[t_idx]; c_t = c_r[t_idx]
            G_tile = t_idx.numel()

            x0 = tx * tile_size
            x1 = min(x0 + tile_size, W)
            tH, tW = y1 - y0, x1 - x0

            ys, xs = torch.meshgrid(
                torch.arange(y0, y1, device=device, dtype=torch.float32),
                torch.arange(x0, x1, device=device, dtype=torch.float32),
                indexing="ij",
            )

            # Chunked compositing with running transmittance
            T = torch.ones(tH, tW, device=device)
            tile_img = torch.zeros(3, tH, tW, device=device)
            tile_acc = torch.zeros(tH, tW, device=device)
            chunk = min(chunk_size, G_tile)

            for i in range(0, G_tile, chunk):
                j = min(i + chunk, G_tile)
                args = (u_t[i:j], v_t[i:j], s_t[i:j], a_t[i:j], c_t[i:j],
                        xs, ys, T, tile_img, tile_acc, eps_t)
                if do_ckpt:
                    T, tile_img, tile_acc = torch.utils.checkpoint.checkpoint(
                        _tile_chunk, *args, use_reentrant=False,
                    )
                else:
                    T, tile_img, tile_acc = _tile_chunk(*args)

            out_img[:, y0:y1, x0:x1] = tile_img
            out_acc[y0:y1, x0:x1] = tile_acc

    return out_img, out_acc


def render_shell_tiled_row(
    shell, K, R, t, H, W,
    near=0.05, tile_size=32, sigma_extent=3.0,
    chunk_size=2048, eps=1e-6,
):
    """Row-batched tile renderer.

    Eliminates the inner per-tile loop: for each tile *row* (n_ty iterations
    total, not n_tx*n_ty), gathers Gaussians overlapping that row and
    composites them against the entire row-strip [tile_size × W] in one go,
    chunked along Gaussians for memory. Trades smaller kernels for ~n_tx
    fewer python iterations / kernel launches.
    """
    device = shell.radius_param.device
    p_world = shell.positions()
    p_cam = p_world @ R.T + t.view(1, 3)
    z = p_cam[:, 2]

    fx = K[0, 0]; fy = K[1, 1]; cx = K[0, 2]; cy = K[1, 2]
    safe_z = torch.clamp(z, min=near)
    u = fx * p_cam[:, 0] / safe_z + cx
    v = fy * p_cam[:, 1] / safe_z + cy

    sigma_screen = (shell.sigma_world * fx) / safe_z
    sigma_screen = sigma_screen.clamp(min=0.5)

    front = (z > near).float()
    alpha = shell.alphas() * front
    color = shell.colors()

    order = torch.argsort(z, dim=0)
    u_s = u[order]; v_s = v[order]
    s_s = sigma_screen[order]; a_s = alpha[order]; c_s = color[order]
    front_s = front[order]

    ext = sigma_extent * s_s
    on_screen = ((u_s + ext >= 0) & (u_s - ext < W)
                 & (v_s + ext >= 0) & (v_s - ext < H)
                 & (front_s > 0.5))

    # Per-Gaussian tile-row range
    ty_lo = ((v_s - ext) / tile_size).floor().long().clamp(0, (H - 1) // tile_size)
    ty_hi = ((v_s + ext) / tile_size).floor().long().clamp(0, (H - 1) // tile_size)

    n_ty = (H + tile_size - 1) // tile_size
    out_img = torch.zeros(3, H, W, device=device)
    out_acc = torch.zeros(H, W, device=device)

    for ty in range(n_ty):
        in_row = on_screen & (ty_lo <= ty) & (ty_hi >= ty)
        row_idx = in_row.nonzero(as_tuple=False).squeeze(-1)
        G_row = row_idx.numel()
        if G_row == 0:
            continue

        u_r = u_s[row_idx]; v_r = v_s[row_idx]; s_r = s_s[row_idx]
        a_r = a_s[row_idx]; c_r = c_s[row_idx]

        y0 = ty * tile_size
        y1 = min(y0 + tile_size, H)
        rH = y1 - y0

        ys, xs = torch.meshgrid(
            torch.arange(y0, y1, device=device, dtype=torch.float32),
            torch.arange(0, W, device=device, dtype=torch.float32),
            indexing="ij",
        )

        # Chunked compositing along Gaussians to bound memory; each chunk is
        # gradient-checkpointed so its activations are recomputed in backward
        # instead of retained — critical since row strips are large.
        T = torch.ones(rH, W, device=device)
        row_img = torch.zeros(3, rH, W, device=device)
        row_acc = torch.zeros(rH, W, device=device)
        chunk = min(chunk_size, G_row)
        eps_t = torch.tensor(eps, device=device, dtype=torch.float32)
        do_ckpt = torch.is_grad_enabled()

        def _row_chunk(u_c, v_c, s_c, a_c, c_c, xs, ys, T, row_img, row_acc, eps_t):
            du = xs[None] - u_c[:, None, None]
            dv = ys[None] - v_c[:, None, None]
            inv_2s2 = 0.5 / (s_c[:, None, None] ** 2)
            g = torch.exp(-(du * du + dv * dv) * inv_2s2)
            w = (a_c[:, None, None] * g).clamp(max=0.999)
            log_om = torch.log1p(-w + eps_t)
            log_T_chunk = torch.cumsum(log_om, dim=0)
            T_after = torch.exp(log_T_chunk)
            T_before = torch.cat(
                [torch.ones_like(T_after[:1]), T_after[:-1]], dim=0
            )
            contrib = T[None] * T_before * w
            row_img = row_img + (contrib[..., None]
                                 * c_c[:, None, None, :]).sum(dim=0).permute(2, 0, 1)
            row_acc = row_acc + contrib.sum(dim=0)
            T = T * T_after[-1]
            return T, row_img, row_acc

        for i in range(0, G_row, chunk):
            j = min(i + chunk, G_row)
            args = (u_r[i:j], v_r[i:j], s_r[i:j], a_r[i:j], c_r[i:j],
                    xs, ys, T, row_img, row_acc, eps_t)
            if do_ckpt:
                T, row_img, row_acc = torch.utils.checkpoint.checkpoint(
                    _row_chunk, *args, use_reentrant=False,
                )
            else:
                T, row_img, row_acc = _row_chunk(*args)

        out_img[:, y0:y1, :] = row_img
        out_acc[y0:y1, :] = row_acc

    return out_img, out_acc


def render_shell_dispatch(shell, K, R, t, H, W,
                          chunk_size=2048, tile_size=0, tile_mode="grid",
                          use_checkpoint=True):
    """Pick tiled or dense renderer based on tile_size and tile_mode.

    tile_mode:
      'grid' — original per-(row, col) tile loop (smallest tiles, most python overhead)
      'row'  — row-batched: one composite per tile-row, fewer kernel launches
    """
    if tile_size and tile_size > 0:
        if tile_mode == "row":
            return render_shell_tiled_row(
                shell, K, R, t, H, W,
                tile_size=tile_size, chunk_size=chunk_size,
            )
        return render_shell_tiled(
            shell, K, R, t, H, W, tile_size=tile_size,
            use_checkpoint=use_checkpoint, chunk_size=chunk_size,
        )
    return render_shell(
        shell, K, R, t, H, W,
        chunk_size=chunk_size, use_checkpoint=use_checkpoint,
    )


# ----------------- Export -----------------

def export_shell_ply(shell, out_path, alpha_threshold=0.05):
    """Write the surviving Gaussians as a vertex-only PLY with RGBA."""
    pos = shell.positions().detach().cpu().numpy()                    # [G, 3]
    rgb = (shell.colors().detach().cpu().numpy() * 255.0).astype(np.uint8)
    a = shell.alphas().detach().cpu().numpy()
    keep = a > alpha_threshold
    pos = pos[keep]; rgb = rgb[keep]; a = a[keep]
    n = len(pos)
    with open(out_path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        for prop in ("x", "y", "z"):
            f.write(f"property float {prop}\n")
        for prop in ("red", "green", "blue", "alpha"):
            f.write(f"property uchar {prop}\n")
        f.write("end_header\n")
        for (x, y, z), c, ai in zip(pos, rgb, a):
            f.write(f"{x} {y} {z} {int(c[0])} {int(c[1])} {int(c[2])} {int(ai*255)}\n")


def save_shell_npz(shell, out_path):
    np.savez(
        out_path,
        directions=shell.directions.detach().cpu().numpy(),
        anchors=shell.anchors.detach().cpu().numpy(),
        radii=shell.radius_param.detach().cpu().numpy(),
        alphas=torch.sigmoid(shell.alpha_logit).detach().cpu().numpy(),
        colors=torch.sigmoid(shell.color_logit).detach().cpu().numpy(),
        sigma_world=np.float32(shell.sigma_world),
        alpha_cap=np.float32(shell.alpha_cap),
        n_shells=np.int32(shell.n_shells),
        n_per_shell=np.int32(shell.n_per_shell),
        share_directions=np.bool_(shell.share_directions),
        suppress_mask=shell.suppress_mask.detach().cpu().numpy(),
    )


# ----------------- DA-based foreground mask -----------------

def da_mask_from_inv(da_inv_maps, thresh=0.3, morph_kernel=5):
    """Foreground masks from DA inverse-depth: pixels with inv_depth > thresh
    are kept (closer than the threshold). Light morph cleanup closes pinholes
    and removes specks.

    Args:
        da_inv_maps: [V, H, W] float32 array of raw DA inv-depth (larger=closer)
        thresh: absolute threshold on inv-depth; pixels at or below are masked
        morph_kernel: size of morphological structuring element

    Returns:
        [V, H, W] float32 in {0.0, 1.0}
    """
    V, H, W = da_inv_maps.shape
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
    out = np.zeros((V, H, W), dtype=np.float32)
    for v in range(V):
        m = (da_inv_maps[v] > thresh).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        out[v] = m.astype(np.float32) / 255.0
    return out


# ----------------- Depth-based floater suppression -----------------

def compute_suppress_mask(shell, da_inv_maps, poses, intrinsics, img_size,
                          scale=2.36, min_views_frac=0.3,
                          near_plane=0.05, far_plane=4.0):
    """Compute a [K, N] bool mask of floater Gaussians using Depth Anything.

    Per-view checks for each Gaussian:
      - in_frustum = on-screen AND near_plane < cam_z < far_plane
      - product = cam_z * da_inv_at_pixel
      - is_floater = in_frustum AND product < scale
    Suppress if (#floater views) / (#in_frustum views) > min_views_frac.

    The denominator uses a *pure geometric frustum* (no DA): a view counts
    only if the Gaussian falls within its near-/far-plane-clamped pyramid.
    This stops orbit-opposite views — where a near-camera floater sits way
    behind the scene and would otherwise vote "not a floater" — from diluting
    the floater fraction.

    Args:
        shell: GaussianShell (positions are read with no_grad)
        da_inv_maps: [V, H, W] numpy array of DA inverse-depth maps
        poses: list of (R, t) tuples (torch tensors, any device)
        intrinsics: [3, 3] torch tensor
        img_size: (H, W)
        scale: threshold for "looks like a floater" (product < scale)
        min_views_frac: fraction of in-frustum views required to suppress
        near_plane: minimum cam_z for a view to count
        far_plane: maximum cam_z for a view to count. Tune to orbit_radius
                   + scene_radius (≈ 4.0 for our orbit_radius=2.5,
                   scene_radius=1.5).

    Returns:
        [K, N] bool tensor (on CPU)
    """
    H, W = img_size
    V = len(poses)
    n_shells = shell.n_shells
    n_per_shell = shell.n_per_shell

    # Get positions as numpy (no autograd)
    with torch.no_grad():
        positions = shell.positions().detach().cpu().numpy()  # [G, 3]
    G = len(positions)

    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])

    cam_z_all = np.zeros((V, G), dtype=np.float32)
    da_at_pixel = np.zeros((V, G), dtype=np.float32)
    visible = np.zeros((V, G), dtype=bool)

    for vi, (R, t) in enumerate(poses):
        R_np = R.detach().cpu().numpy()
        t_np = t.detach().cpu().numpy().flatten()
        p_cam = positions @ R_np.T + t_np[None, :]
        z = p_cam[:, 2]
        safe_z = np.clip(z, near_plane, None)
        u = fx * p_cam[:, 0] / safe_z + cx
        v = fy * p_cam[:, 1] / safe_z + cy

        # Pure geometric frustum: on-screen AND inside [near_plane, far_plane]
        in_frustum = ((z > near_plane) & (z < far_plane)
                      & (u >= 0) & (u < W) & (v >= 0) & (v < H))
        cam_z_all[vi] = z
        visible[vi] = in_frustum

        u_int = np.clip(u.astype(int), 0, W - 1)
        v_int = np.clip(v.astype(int), 0, H - 1)
        da_at_pixel[vi] = da_inv_maps[vi][v_int, u_int]

    # product < scale → floater (only counts in views where Gaussian is in-frustum)
    product = cam_z_all * da_at_pixel
    is_floater = visible & (product < scale)
    n_vis = visible.sum(axis=0).clip(1)
    floater_frac = is_floater.sum(axis=0) / n_vis
    # Require at least one in-frustum view; otherwise the fraction is vacuous.
    seen_any = visible.any(axis=0)
    suppress = seen_any & (floater_frac > min_views_frac)

    return torch.from_numpy(suppress.reshape(n_shells, n_per_shell))


# ----------------- Multi-GPU helpers -----------------

def _make_shell_replicas(master, devices):
    """Build a GaussianShell replica on each device, mirroring the master's
    architecture and copying its parameters/buffers once."""
    replicas = []
    for dev in devices:
        replica = GaussianShell(
            n_per_shell=master.n_per_shell,
            n_shells=master.n_shells,
            init_radii=[1.0] * master.n_shells,
            anchors=master.anchors.detach().cpu(),
            sigma_world=master.sigma_world,
            alpha_cap=master.alpha_cap,
            share_directions=master.share_directions,
        ).to(dev)
        with torch.no_grad():
            replica.directions.copy_(master.directions.to(dev, non_blocking=True))
            replica.neighbors.copy_(master.neighbors.to(dev, non_blocking=True))
            replica.anchors.copy_(master.anchors.to(dev, non_blocking=True))
            replica.suppress_mask.copy_(master.suppress_mask.to(dev, non_blocking=True))
            for mp, rp in zip(master.parameters(), replica.parameters()):
                rp.copy_(mp.detach().to(dev, non_blocking=True))
        replicas.append(replica)
    return replicas


def _broadcast_params(master, replicas):
    """Copy master parameters → all replicas (no autograd)."""
    with torch.no_grad():
        for replica in replicas:
            dev = replica.radius_param.device
            for mp, rp in zip(master.parameters(), replica.parameters()):
                rp.copy_(mp.detach().to(dev, non_blocking=True))


def _gpu_worker(gpu_id, replica, Kc, poses_dev, gt_dev, mask_dev,
                batch_idx, batch_mask_sum, tile_size, chunk_size, H, W,
                tile_mode="grid"):
    """Per-GPU thread: forward+backward on its slice of views, return MSE
    accumulator and a snapshot of parameter grads (on its own device)."""
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(gpu_id)

    for p in replica.parameters():
        if p.grad is not None:
            p.grad.zero_()

    mse_acc = 0.0
    for vi in batch_idx:
        R_v, t_v = poses_dev[vi]
        img_v, _ = render_shell_dispatch(
            replica, Kc, R_v, t_v, H, W,
            chunk_size=chunk_size, tile_size=tile_size,
            tile_mode=tile_mode, use_checkpoint=True,
        )
        diff2 = (img_v - gt_dev[vi]) ** 2 * mask_dev[vi]
        l_mse_v = diff2.sum() / batch_mask_sum
        l_mse_v.backward()
        mse_acc += l_mse_v.item()

    torch.cuda.synchronize(device)
    grads = [
        (p.grad.detach().clone() if p.grad is not None
         else torch.zeros_like(p))
        for p in replica.parameters()
    ]
    return mse_acc, grads


# ----------------- Training -----------------

def train_shell_from_video(
    video_path,
    orbit_period_frames,
    direction,
    start_frame=0,
    frame_step=1,
    n_per_shell=8000,
    n_shells=1,
    init_radii=None,
    anchors=None,
    init_npz=None,
    img_res=(128, 128),
    n_iters=2000,
    fov_y_deg=45.0,
    out_dir="shell_out",
    lr_radius=2e-2,
    lr_alpha=5e-2,
    lr_color=5e-2,
    lambda_lap=0.5,
    lambda_alpha=1e-3,
    chunk_size=2048,
    tile_size=0,
    tile_mode="grid",
    n_gpus=1,
    debug_sync=False,
    views_per_iter=None,
    snapshot_every=0,
    snapshot_views=4,
    use_openai_bg_removal=False,
    openai_api_key=None,
    log_every=10,
    save_views=True,
    cull_every=0,
    cull_scale=2.36,
    cull_min_views_frac=0.3,
    cull_da_device=None,
    cull_far_plane=4.0,
    alpha_cap=1.0,
    mask_from_da=False,
    da_mask_thresh=0.3,
    da_mask_morph=5,
):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Shells: {n_shells}  N/shell: {n_per_shell}  total: {n_shells*n_per_shell}")
    print(f"Resolution: {img_res[0]}x{img_res[1]}  iters: {n_iters}")
    print("=" * 60)

    # Frames + masks
    frame_indices = list(range(start_frame,
                               start_frame + orbit_period_frames,
                               frame_step))
    gt_stack, mask_stack, used_indices = load_frames_as_tensors(
        video_path, frame_indices,
        img_res=img_res, device=device,
        use_openai_bg_removal=use_openai_bg_removal,
        openai_api_key=openai_api_key,
    )
    V, C, H, W = gt_stack.shape
    print(f"Loaded {V} frames at {H}x{W}")

    # Camera + poses
    Kc = make_intrinsics(H, W, fov_y_deg=fov_y_deg, device=device)
    poses = build_orbit_poses_from_period(
        used_indices,
        orbit_period_frames=orbit_period_frames,
        direction=direction,
        radius=2.5,
        device=device,
        theta0=0.0,
    )

    # Shell
    if init_npz is not None:
        print(f"[INIT] Loading shell from {init_npz}")
        shell = GaussianShell.from_npz(init_npz).to(device)
        n_per_shell = shell.n_per_shell
        n_shells = shell.n_shells
    else:
        shell = GaussianShell(
            n_per_shell=n_per_shell,
            n_shells=n_shells,
            init_radii=init_radii,
            anchors=anchors,
            alpha_cap=alpha_cap,
        ).to(device)

    opt = optim.Adam([
        {"params": [shell.radius_param], "lr": lr_radius},
        {"params": [shell.alpha_logit],  "lr": lr_alpha},
        {"params": [shell.color_logit],  "lr": lr_color},
    ])

    r_max_clamp = max(2.5, float(shell.radius_param.max().item()) * 1.5)

    # Snapshot subdirs for intermediate renders
    snap_dir = out_dir / "snapshots"
    if snapshot_every and snapshot_every > 0:
        snap_dir.mkdir(parents=True, exist_ok=True)
        # Pick evenly-spaced views (across the orbit) for snapshots
        snap_idx = np.linspace(0, V - 1, num=min(snapshot_views, V),
                               dtype=int).tolist()
        print(f"[SNAPSHOT] every {snapshot_every} iters → views {snap_idx} "
              f"into {snap_dir}")

    do_subsample = views_per_iter is not None and views_per_iter < V
    if do_subsample:
        print(f"[SUBSAMPLE] {views_per_iter}/{V} views per iter")

    # Multi-GPU setup: replicate model+data on each GPU once
    n_gpus_avail = torch.cuda.device_count() if torch.cuda.is_available() else 0
    n_gpus_use = max(1, min(int(n_gpus), n_gpus_avail or 1))
    if n_gpus_use > 1:
        devices = [f"cuda:{i}" for i in range(n_gpus_use)]
        print(f"[MULTI-GPU] data-parallel across {n_gpus_use} GPUs: {devices}")
        replicas = _make_shell_replicas(shell, devices)
        K_per_gpu = [Kc.to(d, non_blocking=True) for d in devices]
        poses_per_gpu = [
            [(R.to(d, non_blocking=True), t.to(d, non_blocking=True))
             for (R, t) in poses]
            for d in devices
        ]
        gt_per_gpu = [gt_stack.to(d, non_blocking=True) for d in devices]
        mask_per_gpu = [mask_stack.to(d, non_blocking=True) for d in devices]
    else:
        replicas = K_per_gpu = poses_per_gpu = gt_per_gpu = mask_per_gpu = None

    # DA computation: shared between the floater cull and (optional) DA-based
    # foreground mask. Run once if either feature needs it.
    da_inv_maps = None
    need_da = (cull_every > 0) or mask_from_da
    if need_da:
        da_device = cull_da_device or device
        reasons = []
        if cull_every > 0:
            reasons.append(f"cull_every={cull_every}, scale={cull_scale}, "
                           f"min_views_frac={cull_min_views_frac}")
        if mask_from_da:
            reasons.append(f"mask_from_da thresh={da_mask_thresh}")
        print(f"[DA] Running Depth Anything once on {da_device} "
              f"({'; '.join(reasons)})...")
        from depth_filter_shell import get_da_inv_depths
        da_inv_maps = get_da_inv_depths(
            video_path, used_indices, (H, W), device=da_device)
        print(f"[DA] inv-depth maps ready: {da_inv_maps.shape}")

        if mask_from_da:
            da_mask = da_mask_from_inv(
                da_inv_maps, thresh=da_mask_thresh,
                morph_kernel=da_mask_morph)
            old_cov = float(mask_stack.mean().item())
            mask_stack = (torch.from_numpy(da_mask)
                          .unsqueeze(1)            # [V, 1, H, W]
                          .to(device))
            new_cov = float(mask_stack.mean().item())
            print(f"[MASK] Replaced FG mask with DA-derived "
                  f"(coverage: {old_cov*100:.1f}% → {new_cov*100:.1f}%)")
            # Re-replicate per-GPU mask if multi-GPU
            if n_gpus_use > 1:
                mask_per_gpu = [mask_stack.to(d, non_blocking=True)
                                for d in devices]

        # Floor for cull stability (avoid 0 in cull formula)
        if cull_every > 0:
            da_inv_maps = np.clip(da_inv_maps, 0.01, None)

    ema_loss = None
    ema_alpha = 0.9                                                   # EMA decay

    for it in range(n_iters):
        t0 = time.time()
        opt.zero_grad()
        mse_acc = 0.0

        if do_subsample:
            batch_idx = torch.randperm(V)[:views_per_iter].tolist()
        else:
            batch_idx = list(range(V))

        # Per-iter normalizer over only the sampled views, so loss magnitude
        # is comparable across iters and across batch sizes.
        batch_mask_sum_master = mask_stack[batch_idx].sum() * 3.0 + 1e-6

        if n_gpus_use > 1:
            # Sync master params to replicas
            _broadcast_params(shell, replicas)

            # Split batch round-robin across GPUs
            batches = [batch_idx[i::n_gpus_use] for i in range(n_gpus_use)]
            bms_per_gpu = [
                torch.tensor(float(batch_mask_sum_master.item()), device=d)
                for d in devices
            ]

            if debug_sync:
                # Sequential per-GPU loop — same correctness, no threading.
                results = [
                    _gpu_worker(
                        gpu_id, replicas[gpu_id],
                        K_per_gpu[gpu_id], poses_per_gpu[gpu_id],
                        gt_per_gpu[gpu_id], mask_per_gpu[gpu_id],
                        batches[gpu_id], bms_per_gpu[gpu_id],
                        tile_size, chunk_size, H, W,
                        tile_mode=tile_mode,
                    )
                    for gpu_id in range(n_gpus_use)
                ]
            else:
                with ThreadPoolExecutor(max_workers=n_gpus_use) as ex:
                    futures = [
                        ex.submit(
                            _gpu_worker, gpu_id, replicas[gpu_id],
                            K_per_gpu[gpu_id], poses_per_gpu[gpu_id],
                            gt_per_gpu[gpu_id], mask_per_gpu[gpu_id],
                            batches[gpu_id], bms_per_gpu[gpu_id],
                            tile_size, chunk_size, H, W,
                            tile_mode,
                        )
                        for gpu_id in range(n_gpus_use)
                    ]
                    results = [f.result() for f in futures]

            mse_acc = sum(r[0] for r in results)
            # Sum replica grads onto master
            for gpu_id, (_, grads) in enumerate(results):
                for mp, g in zip(shell.parameters(), grads):
                    g_master = g.to(device, non_blocking=True)
                    if mp.grad is None:
                        mp.grad = g_master.clone()
                    else:
                        mp.grad.add_(g_master)
        else:
            # Single-GPU per-view forward+backward
            for vi in batch_idx:
                R_v, t_v = poses[vi]
                img_v, _ = render_shell_dispatch(
                    shell, Kc, R_v, t_v, H, W,
                    chunk_size=chunk_size, tile_size=tile_size,
                    tile_mode=tile_mode, use_checkpoint=True,
                )
                mask_v = mask_stack[vi]                                # [1, H, W]
                gt_v = gt_stack[vi]                                     # [3, H, W]
                diff2 = (img_v - gt_v) ** 2 * mask_v
                l_mse_v = diff2.sum() / batch_mask_sum_master           # contribution
                l_mse_v.backward()
                mse_acc += l_mse_v.item()

        # Regularizers — cheap, single backward over them on master only.
        loss_lap = shell.laplacian_radius_loss()
        loss_alpha = shell.alpha_sparsity_loss()
        loss_reg = lambda_lap * loss_lap + lambda_alpha * loss_alpha
        loss_reg.backward()

        opt.step()
        shell.clamp_radii(r_min=0.0, r_max=r_max_clamp)

        # Periodic floater suppression mask update
        if cull_every > 0 and (it + 1) % cull_every == 0:
            mask = compute_suppress_mask(
                shell, da_inv_maps, poses, Kc, (H, W),
                scale=cull_scale, min_views_frac=cull_min_views_frac,
                far_plane=cull_far_plane,
            )
            shell.suppress_mask.copy_(mask.to(device))
            n_suppressed = int(mask.sum())
            print(f"[CULL@{it+1}] suppressed {n_suppressed}/"
                  f"{shell.n_shells * shell.n_per_shell} Gaussians")
            # Sync to replicas
            if n_gpus_use > 1:
                with torch.no_grad():
                    for replica in replicas:
                        replica.suppress_mask.copy_(
                            shell.suppress_mask.to(replica.radius_param.device,
                                                   non_blocking=True))

        total_iter = mse_acc + loss_reg.item()
        ema_loss = total_iter if ema_loss is None else (
            ema_alpha * ema_loss + (1.0 - ema_alpha) * total_iter
        )

        if it % log_every == 0 or it == 0:
            r = shell.radius_param.detach()
            a = shell.alphas().detach()
            print(
                f"[{it}/{n_iters}] loss={total_iter:.4e} ema={ema_loss:.4e} "
                f"(mse={mse_acc:.4e} lap={loss_lap.item():.4e} "
                f"a_mean={loss_alpha.item():.4e}) "
                f"r=[{r.min():.3f},{r.max():.3f}] "
                f"a=[{a.min():.3f},{a.max():.3f}] "
                f"[{time.time()-t0:.2f}s]"
            )

        # Periodic snapshot: render a few fixed views and dump npz
        if snapshot_every and snapshot_every > 0 and (it + 1) % snapshot_every == 0:
            with torch.no_grad():
                for vi in snap_idx:
                    R_v, t_v = poses[vi]
                    img_v, _ = render_shell_dispatch(
                        shell, Kc, R_v, t_v, H, W,
                        chunk_size=chunk_size, tile_size=tile_size,
                        tile_mode=tile_mode, use_checkpoint=False,
                    )
                    np_img = (img_v.clamp(0, 1).permute(1, 2, 0).cpu()
                              .numpy() * 255).astype(np.uint8)
                    cv2.imwrite(
                        str(snap_dir / f"iter{it+1:05d}_v{vi:03d}.png"),
                        cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR),
                    )
            save_shell_npz(shell, snap_dir / f"iter{it+1:05d}_shell.npz")
            print(f"[SNAPSHOT] saved iter {it+1} to {snap_dir}")

    # Final render + save
    if save_views:
        with torch.no_grad():
            imgs, _ = render_shell_views(shell, Kc, poses, img_size=img_res,
                                         chunk_size=chunk_size)
        for i, img in enumerate(imgs):
            np_img = (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(str(out_dir / f"recon_{i:03d}.png"),
                        cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR))

    save_shell_npz(shell, out_dir / "shell.npz")
    export_shell_ply(shell, out_dir / "shell.ply")
    print(f"Saved to {out_dir}")
    return shell


# ----------------- CLI -----------------

def _build_argparser():
    import argparse
    p = argparse.ArgumentParser(
        description="Differentiable spherical-shell Gaussian reconstruction "
                    "from an orbiting video.")
    p.add_argument("--video", required=True, help="Path to orbit video")
    p.add_argument("--orbit-period-frames", type=int, default=None,
                   help="Frames per full 2π orbit. If omitted, auto-estimate.")
    p.add_argument("--direction", type=int, default=None, choices=[-1, 1],
                   help="+1 CCW, -1 CW. If omitted, auto-estimate.")
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--frame-step", type=int, default=1)
    p.add_argument("--img-res", type=int, nargs=2, default=[128, 128],
                   metavar=("H", "W"))
    p.add_argument("--n-iters", type=int, default=2000)
    p.add_argument("--n-per-shell", type=int, default=8000)
    p.add_argument("--n-shells", type=int, default=1,
                   help="Number of concentric shells (use >1 for non-star-convex shapes)")
    p.add_argument("--init-radii", type=float, nargs="+", default=None,
                   help="Initial radius per shell (len must equal --n-shells)")
    p.add_argument("--anchors", type=float, nargs="+", default=None,
                   help="Flattened anchor coords [x0 y0 z0 x1 y1 z1 ...] for "
                        "the multi-anchor variant (len must equal 3*n_shells)")
    p.add_argument("--fov-y-deg", type=float, default=45.0)
    p.add_argument("--lr-radius", type=float, default=2e-2)
    p.add_argument("--lr-alpha", type=float, default=5e-2)
    p.add_argument("--lr-color", type=float, default=5e-2)
    p.add_argument("--lambda-lap", type=float, default=0.5)
    p.add_argument("--lambda-alpha", type=float, default=1e-3)
    p.add_argument("--chunk-size", type=int, default=2048)
    p.add_argument("--tile-size", type=int, default=0,
                   help="If >0, use tile-based rendering with per-Gaussian "
                        "bbox cull (e.g. 16). Big speedup for high G or "
                        "high resolution. 0 = dense renderer (default).")
    p.add_argument("--tile-mode", choices=["grid", "row"], default="grid",
                   help="Tile loop layout. 'grid' = per-(row,col) tile loop. "
                        "'row' = composite each tile-row in one pass — fewer "
                        "kernel launches, more GPU compute. Try 'row' for big "
                        "Gs / high res.")
    p.add_argument("--n-gpus", type=int, default=1,
                   help="Number of GPUs for data-parallel training "
                        "(views split across GPUs, grads summed onto cuda:0).")
    p.add_argument("--debug-sync", action="store_true",
                   help="Run per-GPU workers sequentially instead of in "
                        "threads. Slower but useful for debugging "
                        "multi-GPU hangs / weirdness on first run.")
    p.add_argument("--views-per-iter", type=int, default=None,
                   help="Random subsample of training views per iter "
                        "(default: use all views — much slower for many frames)")
    p.add_argument("--snapshot-every", type=int, default=0,
                   help="If >0, dump rendered views + shell.npz every N iters")
    p.add_argument("--snapshot-views", type=int, default=4,
                   help="How many evenly-spaced views to render per snapshot")
    p.add_argument("--init-npz", default=None,
                   help="Initialize shell from a saved .npz (e.g. culled checkpoint)")
    p.add_argument("--out-dir", default="shell_out")
    p.add_argument("--use-openai-bg-removal", action="store_true")
    # Depth-based floater culling during training
    p.add_argument("--cull-every", type=int, default=0,
                   help="Recompute floater suppress mask every N iters "
                        "(0 = disabled, no overhead)")
    p.add_argument("--cull-scale", type=float, default=2.36,
                   help="DA scale threshold: product < scale → floater")
    p.add_argument("--cull-min-views-frac", type=float, default=0.3,
                   help="Fraction of visible views to classify as floater")
    p.add_argument("--cull-da-device", default=None,
                   help="Device for DA inference (default: training device)")
    p.add_argument("--cull-far-plane", type=float, default=4.0,
                   help="Pure geometric frustum far plane for the cull "
                        "denominator. A view counts toward the floater-fraction "
                        "denominator only if the Gaussian's cam_z is within "
                        "[near, far_plane]. Default 4.0 ≈ orbit_radius (2.5) + "
                        "scene_radius (1.5). Lower values are stricter and "
                        "more aggressive on camera-near floaters.")
    p.add_argument("--alpha-cap", type=float, default=1.0,
                   help="Per-Gaussian alpha multiplier <=1.0. Lower values "
                        "prevent any single splat from saturating opacity, "
                        "letting gradient leak to occluded (back-of-object) "
                        "Gaussians. Try ~0.5.")
    p.add_argument("--mask-from-da", action="store_true",
                   help="Replace the color-based FG mask with one derived "
                        "from Depth Anything: pixels whose DA inv-depth is "
                        "below --da-mask-thresh are treated as background "
                        "and don't contribute to the loss. Catches gaps "
                        "(e.g. between sword and body) that color-based "
                        "background subtraction misses.")
    p.add_argument("--da-mask-thresh", type=float, default=0.3,
                   help="Absolute threshold on raw DA inv-depth (larger="
                        "closer). Default 0.3 keeps anything notably closer "
                        "than infinity. Tune via dump_da_debug.py.")
    p.add_argument("--da-mask-morph", type=int, default=5,
                   help="Morphological close+open kernel size for DA mask.")
    return p


if __name__ == "__main__":
    args = _build_argparser().parse_args()

    period = args.orbit_period_frames
    if period is None:
        print("[INFO] Auto-estimating orbit period...")
        result = estimate_orbit_period(args.video, start_frame=args.start_frame)
        # estimate_orbit_period returns (period, fps); accept either form
        period = result[0] if isinstance(result, (tuple, list)) else result
        print(f"[INFO] orbit_period_frames = {period}")

    direction = args.direction
    if direction is None:
        print("[INFO] Auto-estimating orbit direction...")
        direction = estimate_orbit_direction(args.video, start_frame=args.start_frame)
        print(f"[INFO] direction = {direction}")

    anchors = None
    if args.anchors is not None:
        if len(args.anchors) != 3 * args.n_shells:
            raise SystemExit(
                f"--anchors needs 3 * n_shells = {3*args.n_shells} values, "
                f"got {len(args.anchors)}")
        anchors = np.array(args.anchors, dtype=np.float32).reshape(args.n_shells, 3)

    train_shell_from_video(
        video_path=args.video,
        orbit_period_frames=period,
        direction=direction,
        start_frame=args.start_frame,
        frame_step=args.frame_step,
        n_per_shell=args.n_per_shell,
        n_shells=args.n_shells,
        init_radii=args.init_radii,
        anchors=anchors,
        init_npz=args.init_npz,
        img_res=tuple(args.img_res),
        n_iters=args.n_iters,
        fov_y_deg=args.fov_y_deg,
        out_dir=args.out_dir,
        lr_radius=args.lr_radius,
        lr_alpha=args.lr_alpha,
        lr_color=args.lr_color,
        lambda_lap=args.lambda_lap,
        lambda_alpha=args.lambda_alpha,
        chunk_size=args.chunk_size,
        tile_size=args.tile_size,
        tile_mode=args.tile_mode,
        n_gpus=args.n_gpus,
        debug_sync=args.debug_sync,
        views_per_iter=args.views_per_iter,
        snapshot_every=args.snapshot_every,
        snapshot_views=args.snapshot_views,
        use_openai_bg_removal=args.use_openai_bg_removal,
        cull_every=args.cull_every,
        cull_scale=args.cull_scale,
        cull_min_views_frac=args.cull_min_views_frac,
        alpha_cap=args.alpha_cap,
        mask_from_da=args.mask_from_da,
        da_mask_thresh=args.da_mask_thresh,
        da_mask_morph=args.da_mask_morph,
        cull_da_device=args.cull_da_device,
        cull_far_plane=args.cull_far_plane,
    )
