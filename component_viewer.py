#!/usr/bin/env python3
"""
Interactive Connected Component Viewer for Voxel Volumes

View, navigate, and edit connected components in reconstructed voxel volumes.
Uses pygame for interactive 3D visualization with component highlighting.
Supports both volumetric (NeRF-style) and cube (Minecraft-style) rendering.

Controls:
  Mouse Drag: Rotate camera
  Mouse Wheel: Zoom in/out
  Arrow Keys (Up/Down): Navigate through components
  V: Volumetric rendering mode
  C: Cube rendering mode
  D: Delete selected component
  R: Restore selected component
  A: Restore all deleted components
  W: Save edited volume (enter output path)
  T: Toggle showing only selected component
  S: Enter size threshold mode (delete all below size)
  +/-: Adjust sigma threshold
  [/]: Adjust color tolerance
  Space: Refresh component analysis
  ESC/Q: Quit

Usage:
  python component_viewer.py <npz_file> [--sigma-threshold 0.5] [--color-tolerance 0.1]
"""

import argparse
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import pygame
from pygame.locals import *
import torch
import torch.nn.functional as F

# Import our connected components analyzer
from analyze_connected_components import label_connected_components_3d


# ==================== Volumetric Rendering (from voxel_volume_viewer.py) ====================

def make_intrinsics(h, w, fov_y_deg=45.0, device="cpu"):
    fov_y = math.radians(fov_y_deg)
    fy = 0.5 * h / math.tan(0.5 * fov_y)
    fx = fy
    cx = w / 2.0
    cy = h / 2.0
    K = torch.tensor([[fx, 0, cx],
                      [0, fy, cy],
                      [0,  0,  1]], dtype=torch.float32, device=device)
    return K


def build_pose_from_eye(eye_np, target_np, device="cpu"):
    """Build world->camera rotation R, translation t that look from eye -> target."""
    eye = np.asarray(eye_np, dtype=np.float32)
    target = np.asarray(target_np, dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    forward = target - eye
    forward /= np.linalg.norm(forward) + 1e-8

    right = np.cross(forward, up)
    right /= np.linalg.norm(right) + 1e-8

    true_up = np.cross(right, forward)

    R = np.stack([right, true_up, forward], axis=0)
    t = -R @ eye

    R_t = torch.from_numpy(R).to(device)
    t_t = torch.from_numpy(t).to(device).view(3, 1)
    return R_t, t_t


def world_to_grid(pts_world, scene_radius=1.5):
    """Map world coordinates to [-1,1]^3."""
    return pts_world / scene_radius


def generate_rays(h, w, K, R, t, n_samples=64, near=0.5, far=2.5, device="cpu"):
    """Generate 3D sample points along rays."""
    ys, xs = torch.meshgrid(
        torch.linspace(0, h - 1, h, device=device),
        torch.linspace(0, w - 1, w, device=device),
        indexing="ij",
    )
    ones = torch.ones_like(xs)
    pix = torch.stack([xs, ys, ones], dim=-1)

    K_inv = torch.inverse(K)
    dirs_cam = (K_inv @ pix.reshape(-1, 3).T).T
    dirs_cam = dirs_cam / torch.norm(dirs_cam, dim=-1, keepdim=True)

    R = R.to(device)
    t = t.to(device)

    dirs_world = (R.transpose(0, 1) @ dirs_cam.T).T
    C = -(R.transpose(0, 1) @ t).reshape(1, 3)

    ts = torch.linspace(near, far, n_samples, device=device).view(-1, 1, 1)
    dirs_world = dirs_world.reshape(1, h, w, 3)
    C_exp = C.view(1, 1, 1, 3)

    pts = C_exp + ts[..., None] * dirs_world
    pts = pts.unsqueeze(0)
    return pts


def sample_volume(sigma, rgb, pts_world, scene_radius=1.5):
    """Sample volume at world points."""
    pts_grid = world_to_grid(pts_world, scene_radius)
    _, S, H, W, _ = pts_grid.shape
    grid = pts_grid.view(1, S, H, W, 3)

    sigma_samples = F.grid_sample(
        sigma, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    rgb_samples = F.grid_sample(
        rgb, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )

    sigma_samples = sigma_samples.squeeze(1)
    rgb_samples = rgb_samples.permute(0, 2, 3, 4, 1)
    return sigma_samples, rgb_samples


def volume_render(sigma_samples, rgb_samples, n_samples):
    """NeRF-style compositing along rays."""
    delta = 1.0 / n_samples
    alpha = 1.0 - torch.exp(-sigma_samples * delta)

    alpha_shifted = torch.cat(
        [torch.zeros_like(alpha[:, :1]), alpha[:, :-1]], dim=1
    )
    T = torch.cumprod(1.0 - alpha_shifted + 1e-10, dim=1)

    weights = T * alpha
    rgb_out = (weights.unsqueeze(-1) * rgb_samples).sum(dim=1)
    rgb_out = rgb_out.permute(0, 3, 1, 2)
    return rgb_out


# ==================== Cube Rendering (from voxel_volume_viewer.py) ====================

def rasterize_triangle(img, z_buffer, verts_2d, depths, color, H, W):
    """
    Rasterize a single triangle with depth testing.
    verts_2d: list of 3 (u, v) screen coordinates
    depths: list of 3 depth values
    color: RGB uint8 tuple
    """
    # Get bounding box
    us = [v[0] for v in verts_2d]
    vs = [v[1] for v in verts_2d]
    
    min_u = max(0, int(np.floor(min(us))))
    max_u = min(W - 1, int(np.ceil(max(us))))
    min_v = max(0, int(np.floor(min(vs))))
    max_v = min(H - 1, int(np.ceil(max(vs))))
    
    if min_u > max_u or min_v > max_v:
        return
    
    p0, p1, p2 = verts_2d
    z0, z1, z2 = depths
    
    # Precompute triangle edge functions for barycentric coordinates
    def edge_function(a, b, c):
        return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])
    
    area = edge_function(p0, p1, p2)
    if abs(area) < 1e-6:  # Degenerate triangle
        return
    
    # Scan over bounding box
    for v in range(min_v, max_v + 1):
        for u in range(min_u, max_u + 1):
            p = np.array([u + 0.5, v + 0.5])  # Pixel center
            
            # Compute barycentric coordinates
            w0 = edge_function(p1, p2, p)
            w1 = edge_function(p2, p0, p)
            w2 = edge_function(p0, p1, p)
            
            # Check if point is inside triangle
            if w0 >= 0 and w1 >= 0 and w2 >= 0:
                # Normalize barycentric coords
                w0 /= area
                w1 /= area
                w2 /= area
                
                # Interpolate depth
                z = w0 * z0 + w1 * z1 + w2 * z2
                
                # Depth test
                if z < z_buffer[v, u]:
                    z_buffer[v, u] = z
                    img[v, u] = color


class ComponentViewer:
    """Interactive viewer for connected components in voxel volumes."""
    
    def __init__(self, npz_path, sigma_threshold=0.5, color_tolerance=0.1,
                 window_size=(1024, 768)):
        """
        Initialize viewer.
        
        Args:
            npz_path: Path to NPZ file
            sigma_threshold: Initial threshold for occupied voxels
            color_tolerance: Initial color tolerance for connectivity
            window_size: Window dimensions (width, height)
        """
        # Load volume
        print(f"Loading volume: {npz_path}")
        data = np.load(npz_path)
        self.sigma_orig = data['sigma'].copy()  # Original, never modified
        self.rgb_orig = data['rgb'].copy()      # Original, never modified
        self.sigma = self.sigma_orig.copy()     # Working copy (can be edited)
        self.rgb = self.rgb_orig.copy()         # Working copy (can be edited)
        
        self.grid_size = self.sigma.shape
        print(f"Grid size: {self.grid_size}")
        
        # Thresholds
        self.sigma_threshold = sigma_threshold
        self.color_tolerance = color_tolerance
        
        # Component tracking
        self.components = []
        self.component_labels = None
        self.selected_component_idx = 0
        self.deleted_components = set()  # Set of component IDs that are deleted
        
        # View state
        self.show_only_selected = False
        self.render_mode = 'cubes'  # 'cubes' or 'volumetric'
        
        # Device for volumetric rendering
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Rendering device: {self.device}")
        
        # Camera state
        self.camera_angle_y = 0.0
        self.camera_angle_x = 0.3
        self.camera_radius = 3.0
        self.mouse_dragging = False
        self.last_mouse_pos = None
        
        # UI state
        self.size_threshold_mode = False
        self.size_threshold_input = ""
        self.save_path_mode = False
        self.save_path_input = ""
        
        # Track if changes have been made
        self.has_changes = False
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("Voxel Component Viewer")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)
        
        # Initial analysis
        self.analyze_components()
        
        print("\nViewer initialized!")
        print("Use arrow keys to navigate components, D to delete, R to restore")
    
    def analyze_components(self):
        """Analyze connected components with current thresholds."""
        print(f"\nAnalyzing components (sigma={self.sigma_threshold:.3f}, color_tol={self.color_tolerance})...")
        
        # Create occupancy mask
        occupied = self.sigma > self.sigma_threshold
        total_occupied = occupied.sum()
        
        if total_occupied == 0:
            print("No occupied voxels!")
            self.components = []
            self.component_labels = None
            return
        
        # Find components
        self.component_labels, num_components = label_connected_components_3d(
            occupied, rgb=self.rgb, color_tolerance=self.color_tolerance
        )
        
        # Analyze each component
        self.components = []
        for comp_id in range(1, num_components + 1):
            comp_mask = (self.component_labels == comp_id)
            size = comp_mask.sum()
            
            # Average color
            comp_rgb = self.rgb[comp_mask]
            avg_color = comp_rgb.mean(axis=0)
            avg_color_255 = (avg_color * 255).astype(int)
            
            # Bounding box
            coords = np.argwhere(comp_mask)
            bbox_min = coords.min(axis=0)
            bbox_max = coords.max(axis=0)
            
            self.components.append({
                'id': comp_id,
                'size': int(size),
                'avg_color': avg_color_255,
                'bbox': (bbox_min, bbox_max),
                'percentage': 100.0 * size / total_occupied,
                'deleted': comp_id in self.deleted_components
            })
        
        # Sort by size (descending)
        self.components.sort(key=lambda x: x['size'], reverse=True)
        
        print(f"Found {len(self.components)} components")
        if self.components:
            print(f"Largest: {self.components[0]['size']:,} voxels")
        
        # Clamp selected index
        if self.selected_component_idx >= len(self.components):
            self.selected_component_idx = max(0, len(self.components) - 1)
    
    def get_selected_component(self):
        """Get currently selected component info."""
        if 0 <= self.selected_component_idx < len(self.components):
            return self.components[self.selected_component_idx]
        return None
    
    def delete_selected_component(self):
        """Mark selected component as deleted and zero out its voxels."""
        comp = self.get_selected_component()
        if comp is None:
            return
        
        comp_id = comp['id']
        if comp_id in self.deleted_components:
            print(f"Component {comp_id} already deleted")
            return
        
        # Mark as deleted
        self.deleted_components.add(comp_id)
        comp['deleted'] = True
        
        # Zero out voxels
        comp_mask = (self.component_labels == comp_id)
        self.sigma[comp_mask] = 0.0
        
        self.has_changes = True
        print(f"Deleted component {comp_id} ({comp['size']:,} voxels)")
    
    def restore_selected_component(self):
        """Restore selected component from original data."""
        comp = self.get_selected_component()
        if comp is None:
            return
        
        comp_id = comp['id']
        if comp_id not in self.deleted_components:
            print(f"Component {comp_id} not deleted")
            return
        
        # Remove from deleted set
        self.deleted_components.remove(comp_id)
        comp['deleted'] = False
        
        # Restore voxels from original
        comp_mask = (self.component_labels == comp_id)
        self.sigma[comp_mask] = self.sigma_orig[comp_mask]
        self.rgb[comp_mask] = self.rgb_orig[comp_mask]
        
        self.has_changes = True
        print(f"Restored component {comp_id} ({comp['size']:,} voxels)")
    
    def restore_all_components(self):
        """Restore all deleted components."""
        if not self.deleted_components:
            print("No deleted components to restore")
            return
        
        count = len(self.deleted_components)
        self.deleted_components.clear()
        self.sigma = self.sigma_orig.copy()
        self.rgb = self.rgb_orig.copy()
        
        # Update component deleted status
        for comp in self.components:
            comp['deleted'] = False
        
        self.has_changes = True
        print(f"Restored {count} components")
    
    def delete_components_below_size(self, min_size):
        """Delete all components below specified size."""
        to_delete = [c for c in self.components if c['size'] < min_size and not c['deleted']]
        
        if not to_delete:
            print(f"No components below size {min_size}")
            return
        
        for comp in to_delete:
            comp_id = comp['id']
            self.deleted_components.add(comp_id)
            comp['deleted'] = True
            
            # Zero out voxels
            comp_mask = (self.component_labels == comp_id)
            self.sigma[comp_mask] = 0.0
        
        self.has_changes = True
        print(f"Deleted {len(to_delete)} components below size {min_size}")
    
    def render_volume_volumetric(self, img_size=(800, 600)):
        """
        Render volume using NeRF-style volumetric rendering.
        Returns RGB image as numpy array.
        """
        H_img, W_img = img_size
        scene_radius = 1.5
        fov_y_deg = 45.0
        n_samples = 64
        
        # Camera position
        cx = self.camera_radius * math.cos(self.camera_angle_x) * math.cos(self.camera_angle_y)
        cy = self.camera_radius * math.sin(self.camera_angle_x)
        cz = self.camera_radius * math.cos(self.camera_angle_x) * math.sin(self.camera_angle_y)
        eye = np.array([cx, cy, cz], dtype=np.float32)
        target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        # Prepare volume data
        sigma_work = self.sigma.copy()
        rgb_work = self.rgb.copy()
        
        # If showing only selected, zero out other components
        if self.show_only_selected:
            comp = self.get_selected_component()
            if comp:
                mask = (self.component_labels != comp['id'])
                sigma_work[mask] = 0.0
        
        # Convert to torch
        sigma_t = torch.from_numpy(sigma_work).float().to(self.device)
        rgb_t = torch.from_numpy(rgb_work).float().to(self.device)
        
        # Add batch and channel dimensions: [1, 1, D, H, W] and [1, 3, D, H, W]
        sigma_t = sigma_t.unsqueeze(0).unsqueeze(0)
        rgb_t = rgb_t.permute(3, 0, 1, 2).unsqueeze(0)
        
        # Setup camera
        K = make_intrinsics(H_img, W_img, fov_y_deg=fov_y_deg, device=self.device)
        R, t = build_pose_from_eye(eye, target, device=self.device)
        
        # Generate rays
        pts = generate_rays(H_img, W_img, K, R, t,
                           n_samples=n_samples,
                           near=0.1, far=5.0,
                           device=self.device)
        
        # Sample volume
        sigma_s, rgb_s = sample_volume(sigma_t, rgb_t, pts, scene_radius=scene_radius)
        
        # Render
        with torch.no_grad():
            rgb_img = volume_render(sigma_s, rgb_s, n_samples)  # [1, 3, H, W]
        
        # Convert to numpy
        img = rgb_img[0].permute(1, 2, 0).cpu().numpy()
        img = (img.clip(0, 1) * 255).astype(np.uint8)
        
        # Highlight selected component (add yellow tint to brighten)
        if not self.show_only_selected:
            comp = self.get_selected_component()
            if comp:
                # Create a highlight mask by rendering just the selected component
                sigma_selected = self.sigma.copy()
                mask = (self.component_labels != comp['id'])
                sigma_selected[mask] = 0.0
                
                sigma_sel_t = torch.from_numpy(sigma_selected).float().to(self.device)
                sigma_sel_t = sigma_sel_t.unsqueeze(0).unsqueeze(0)
                
                with torch.no_grad():
                    sigma_sel_s, _ = sample_volume(sigma_sel_t, rgb_t, pts, scene_radius=scene_radius)
                    # Use alpha to determine highlight strength
                    delta = 1.0 / n_samples
                    alpha = 1.0 - torch.exp(-sigma_sel_s * delta)
                    highlight_strength = alpha.sum(dim=1)[0].cpu().numpy()  # [H, W]
                
                # Apply yellow tint where selected component is visible
                highlight_strength = np.clip(highlight_strength * 100, 0, 1)
                yellow_tint = np.stack([highlight_strength * 80, highlight_strength * 80, 
                                       np.zeros_like(highlight_strength)], axis=-1)
                img = np.clip(img.astype(int) + yellow_tint.astype(int), 0, 255).astype(np.uint8)
        
        return img
    
    def render_volume_cubes(self, img_size=(800, 600)):
        """
        Render current volume from camera viewpoint using cube rendering.
        Returns RGB image as numpy array.
        """
        H_img, W_img = img_size
        scene_radius = 1.5
        fov_y_deg = 45.0
        
        # Camera position
        cx = self.camera_radius * math.cos(self.camera_angle_x) * math.cos(self.camera_angle_y)
        cy = self.camera_radius * math.sin(self.camera_angle_x)
        cz = self.camera_radius * math.cos(self.camera_angle_x) * math.sin(self.camera_angle_y)
        eye = np.array([cx, cy, cz], dtype=np.float32)
        target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        # Camera intrinsics
        fov_y = math.radians(fov_y_deg)
        fy = 0.5 * H_img / math.tan(0.5 * fov_y)
        fx = fy
        cx_i = W_img / 2.0
        cy_i = H_img / 2.0
        
        # Build camera R, t
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        forward = target - eye
        forward /= np.linalg.norm(forward) + 1e-8
        right = np.cross(forward, up)
        right /= np.linalg.norm(right) + 1e-8
        true_up = np.cross(right, forward)
        
        R = np.stack([right, true_up, forward], axis=0)
        t = -R @ eye
        
        D, H, W = self.sigma.shape
        
        # Get voxels to render
        if self.show_only_selected:
            comp = self.get_selected_component()
            if comp:
                mask = (self.component_labels == comp['id']) & (self.sigma > self.sigma_threshold)
            else:
                mask = np.zeros_like(self.sigma, dtype=bool)
        else:
            mask = self.sigma > self.sigma_threshold
        
        idxs = np.argwhere(mask)
        
        if idxs.shape[0] == 0:
            return np.zeros((H_img, W_img, 3), dtype=np.uint8)
        
        # World space voxel grid
        zs = np.linspace(-1, 1, D) * scene_radius
        ys = np.linspace(-1, 1, H) * scene_radius
        xs = np.linspace(-1, 1, W) * scene_radius
        voxel_size = 2.0 * scene_radius / D
        
        # Cube vertices (8 corners of unit cube)
        half_size = voxel_size * 0.5
        cube_verts = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # back face
            [-1, -1, 1],  [1, -1, 1],  [1, 1, 1],  [-1, 1, 1],   # front face
        ], dtype=np.float32) * half_size
        
        # Cube faces (6 faces, each defined by 4 vertex indices)
        cube_faces = np.array([
            [0, 1, 2, 3],  # back (-Z)
            [4, 7, 6, 5],  # front (+Z)
            [0, 4, 5, 1],  # bottom (-Y)
            [3, 2, 6, 7],  # top (+Y)
            [0, 3, 7, 4],  # left (-X)
            [1, 5, 6, 2],  # right (+X)
        ], dtype=int)
        
        # Face normals for lighting
        face_normals = np.array([
            [0, 0, -1], [0, 0, 1], [0, -1, 0], [0, 1, 0], [-1, 0, 0], [1, 0, 0]
        ], dtype=np.float32)
        
        # Create depth buffer and color buffer
        z_buffer = np.full((H_img, W_img), np.inf, dtype=np.float32)
        img = np.zeros((H_img, W_img, 3), dtype=np.uint8)
        
        # Get selected component ID for highlighting
        selected_comp = self.get_selected_component()
        selected_id = selected_comp['id'] if selected_comp else -1
        
        # Render each voxel as a cube
        for idx in idxs:
            z_i, y_i, x_i = idx
            voxel_center = np.array([xs[x_i], ys[y_i], zs[z_i]], dtype=np.float32)
            color = (self.rgb[z_i, y_i, x_i] * 255.0).astype(np.uint8)
            
            # Highlight selected component with yellow tint
            is_selected = (self.component_labels[z_i, y_i, x_i] == selected_id)
            if is_selected:
                color = np.clip(color.astype(int) + np.array([80, 80, 0]), 0, 255).astype(np.uint8)
            
            # Transform cube vertices to world space
            verts_world = cube_verts + voxel_center
            
            # Transform to camera space
            verts_cam = (R @ verts_world.T).T + t
            
            # Render each face
            for face_idx, face in enumerate(cube_faces):
                v0, v1, v2, v3 = verts_cam[face]
                
                # Backface culling: check if face normal points towards camera
                face_normal_world = face_normals[face_idx]
                face_normal_cam = R @ face_normal_world
                if face_normal_cam[2] >= 0:  # Face pointing away from camera
                    continue
                
                # Check if all vertices are in front of camera
                if v0[2] <= 0.01 or v1[2] <= 0.01 or v2[2] <= 0.01 or v3[2] <= 0.01:
                    continue
                
                # Project vertices to screen
                def project(v):
                    u = fx * (v[0] / v[2]) + cx_i
                    v_p = fy * (v[1] / v[2]) + cy_i
                    return np.array([u, v_p]), v[2]
                
                p0, z0 = project(v0)
                p1, z1 = project(v1)
                p2, z2 = project(v2)
                p3, z3 = project(v3)
                
                # Simple lighting: darken faces based on angle to camera
                light_dir = -forward  # Light from camera
                face_brightness = max(0.3, abs(np.dot(face_normal_world, light_dir)))
                lit_color = (color * face_brightness).astype(np.uint8)
                
                # Rasterize two triangles for this face
                for tri_verts, tri_depths in [([p0, p1, p2], [z0, z1, z2]), 
                                               ([p0, p2, p3], [z0, z2, z3])]:
                    rasterize_triangle(img, z_buffer, tri_verts, tri_depths, 
                                     lit_color, H_img, W_img)
        
        return img
    
    def render_volume(self, img_size=(800, 600)):
        """Render volume using current mode (cubes or volumetric)."""
        if self.render_mode == 'volumetric':
            return self.render_volume_volumetric(img_size)
        else:
            return self.render_volume_cubes(img_size)
    
    def draw_ui(self):
        """Draw UI overlay with component info and controls."""
        margin = 10
        line_height = 25
        y = margin
        
        # Background panel
        panel_width = 400
        panel_height = 300
        panel = pygame.Surface((panel_width, panel_height))
        panel.set_alpha(200)
        panel.fill((30, 30, 30))
        self.screen.blit(panel, (margin, margin))
        
        # Component info
        comp = self.get_selected_component()
        if comp:
            texts = [
                f"Component {self.selected_component_idx + 1}/{len(self.components)}",
                f"ID: {comp['id']}",
                f"Size: {comp['size']:,} voxels ({comp['percentage']:.2f}%)",
                f"Color: RGB({comp['avg_color'][0]}, {comp['avg_color'][1]}, {comp['avg_color'][2]})",
                f"Status: {'DELETED' if comp['deleted'] else 'Active'}",
                "",
                f"Render mode: {self.render_mode.upper()}",
                f"Sigma threshold: {self.sigma_threshold:.3f}",
                f"Color tolerance: {self.color_tolerance if np.isfinite(self.color_tolerance) else 'inf'}",
                "",
                f"Deleted components: {len(self.deleted_components)}",
            ]
        else:
            texts = [
                "No components found",
                "",
                f"Render mode: {self.render_mode.upper()}",
                f"Sigma threshold: {self.sigma_threshold:.3f}",
                f"Color tolerance: {self.color_tolerance if np.isfinite(self.color_tolerance) else 'inf'}",
            ]
        
        for i, text in enumerate(texts):
            surface = self.font_small.render(text, True, (255, 255, 255))
            self.screen.blit(surface, (margin + 10, y + i * line_height))
        
        # Controls help at bottom
        help_y = self.screen.get_height() - 120
        help_panel = pygame.Surface((600, 110))
        help_panel.set_alpha(200)
        help_panel.fill((30, 30, 30))
        self.screen.blit(help_panel, (margin, help_y))
        
        help_texts = [
            "Controls:",
            "↑/↓: Navigate  D: Delete  R: Restore  A: Restore All  W: Save",
            "T: Toggle view  V/C: Volume/Cube mode  S: Delete by size",
            "+/-: Sigma  [/]: Color  Space: Refresh  Drag: Rotate  Q/ESC: Quit"
        ]
        
        for i, text in enumerate(help_texts):
            surface = self.font_small.render(text, True, (200, 200, 200))
            self.screen.blit(surface, (margin + 10, help_y + 10 + i * 22))
        
        # Size threshold input mode
        if self.size_threshold_mode:
            input_panel = pygame.Surface((400, 80))
            input_panel.set_alpha(230)
            input_panel.fill((50, 50, 100))
            input_x = (self.screen.get_width() - 400) // 2
            input_y = (self.screen.get_height() - 80) // 2
            self.screen.blit(input_panel, (input_x, input_y))
            
            prompt = self.font.render("Delete components below size:", True, (255, 255, 255))
            self.screen.blit(prompt, (input_x + 20, input_y + 15))
            
            input_text = self.font.render(self.size_threshold_input + "_", True, (255, 255, 100))
            self.screen.blit(input_text, (input_x + 20, input_y + 45))
        
        # Save path input mode
        if self.save_path_mode:
            input_panel = pygame.Surface((500, 80))
            input_panel.set_alpha(230)
            input_panel.fill((50, 100, 50))
            input_x = (self.screen.get_width() - 500) // 2
            input_y = (self.screen.get_height() - 80) // 2
            self.screen.blit(input_panel, (input_x, input_y))
            
            prompt = self.font.render("Save edited volume as:", True, (255, 255, 255))
            self.screen.blit(prompt, (input_x + 20, input_y + 15))
            
            input_text = self.font_small.render(self.save_path_input + "_", True, (255, 255, 100))
            self.screen.blit(input_text, (input_x + 20, input_y + 45))
        
        # Unsaved changes indicator
        if self.has_changes:
            unsaved_text = self.font_small.render("* Unsaved changes", True, (255, 200, 100))
            self.screen.blit(unsaved_text, (self.screen.get_width() - 160, 10))
    
    def handle_events(self):
        """Handle pygame events. Returns False if should quit."""
        for event in pygame.event.get():
            if event.type == QUIT:
                return False
            
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE or event.key == K_q:
                    return False
                
                # Save path input mode
                if self.save_path_mode:
                    if event.key == K_RETURN:
                        if self.save_path_input.strip():
                            try:
                                self.save_edited_volume(self.save_path_input.strip())
                            except Exception as e:
                                print(f"Error saving: {e}")
                        self.save_path_mode = False
                        self.save_path_input = ""
                    
                    elif event.key == K_ESCAPE:
                        self.save_path_mode = False
                        self.save_path_input = ""
                    
                    elif event.key == K_BACKSPACE:
                        self.save_path_input = self.save_path_input[:-1]
                    
                    elif event.unicode and (event.unicode.isprintable() or event.unicode == '/'):
                        self.save_path_input += event.unicode
                    
                    continue
                
                # Size threshold input mode
                elif self.size_threshold_mode:
                    if event.key == K_RETURN:
                        try:
                            size = int(self.size_threshold_input)
                            self.delete_components_below_size(size)
                        except ValueError:
                            print("Invalid size input")
                        self.size_threshold_mode = False
                        self.size_threshold_input = ""
                    
                    elif event.key == K_ESCAPE:
                        self.size_threshold_mode = False
                        self.size_threshold_input = ""
                    
                    elif event.key == K_BACKSPACE:
                        self.size_threshold_input = self.size_threshold_input[:-1]
                    
                    elif event.unicode.isdigit():
                        self.size_threshold_input += event.unicode
                    
                    continue
                
                # Normal mode
                if event.key == K_UP:
                    self.selected_component_idx = max(0, self.selected_component_idx - 1)
                    comp = self.get_selected_component()
                    if comp:
                        print(f"Selected component {comp['id']} ({comp['size']:,} voxels)")
                
                elif event.key == K_DOWN:
                    self.selected_component_idx = min(len(self.components) - 1, 
                                                     self.selected_component_idx + 1)
                    comp = self.get_selected_component()
                    if comp:
                        print(f"Selected component {comp['id']} ({comp['size']:,} voxels)")
                
                elif event.key == K_d:
                    self.delete_selected_component()
                
                elif event.key == K_r:
                    self.restore_selected_component()
                
                elif event.key == K_a:
                    self.restore_all_components()
                
                elif event.key == K_t:
                    self.show_only_selected = not self.show_only_selected
                    print(f"Show only selected: {self.show_only_selected}")
                
                elif event.key == K_v:
                    self.render_mode = 'volumetric'
                    print(f"Render mode: VOLUMETRIC")
                
                elif event.key == K_c:
                    self.render_mode = 'cubes'
                    print(f"Render mode: CUBES")
                
                elif event.key == K_w:
                    self.save_path_mode = True
                    # Suggest a default filename
                    self.save_path_input = "edited_volume.npz"
                    print("Enter save path...")
                
                elif event.key == K_s:
                    self.size_threshold_mode = True
                    self.size_threshold_input = ""
                    print("Enter size threshold...")
                
                elif event.key == K_SPACE:
                    self.analyze_components()
                
                elif event.key == K_EQUALS or event.key == K_PLUS:
                    self.sigma_threshold = min(10.0, self.sigma_threshold + 0.1)
                    print(f"Sigma threshold: {self.sigma_threshold:.3f}")
                    self.analyze_components()
                
                elif event.key == K_MINUS:
                    self.sigma_threshold = max(0.0, self.sigma_threshold - 0.1)
                    print(f"Sigma threshold: {self.sigma_threshold:.3f}")
                    self.analyze_components()
                
                elif event.key == K_RIGHTBRACKET:
                    if np.isfinite(self.color_tolerance):
                        self.color_tolerance = min(2.0, self.color_tolerance + 0.05)
                    else:
                        self.color_tolerance = 0.5
                    print(f"Color tolerance: {self.color_tolerance:.3f}")
                    self.analyze_components()
                
                elif event.key == K_LEFTBRACKET:
                    if np.isfinite(self.color_tolerance):
                        self.color_tolerance = max(0.0, self.color_tolerance - 0.05)
                    else:
                        self.color_tolerance = 0.5
                    print(f"Color tolerance: {self.color_tolerance:.3f}")
                    self.analyze_components()
            
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    self.mouse_dragging = True
                    self.last_mouse_pos = event.pos
                
                elif event.button == 4:  # Scroll up
                    self.camera_radius = max(1.0, self.camera_radius - 0.2)
                
                elif event.button == 5:  # Scroll down
                    self.camera_radius = min(10.0, self.camera_radius + 0.2)
            
            elif event.type == MOUSEBUTTONUP:
                if event.button == 1:
                    self.mouse_dragging = False
            
            elif event.type == MOUSEMOTION:
                if self.mouse_dragging and self.last_mouse_pos:
                    dx = event.pos[0] - self.last_mouse_pos[0]
                    dy = event.pos[1] - self.last_mouse_pos[1]
                    
                    self.camera_angle_y += dx * 0.01
                    self.camera_angle_x = np.clip(self.camera_angle_x + dy * 0.01, 
                                                  -math.pi/2 + 0.1, math.pi/2 - 0.1)
                    
                    self.last_mouse_pos = event.pos
        
        return True
    
    def run(self):
        """Main viewer loop."""
        running = True
        
        while running:
            # Handle events
            running = self.handle_events()
            
            # Render volume
            volume_img = self.render_volume(img_size=(800, 600))
            
            # Convert to pygame surface
            volume_surface = pygame.surfarray.make_surface(
                np.transpose(volume_img, (1, 0, 2))
            )
            
            # Clear screen
            self.screen.fill((20, 20, 25))
            
            # Draw volume (centered)
            vol_rect = volume_surface.get_rect()
            vol_rect.center = (self.screen.get_width() // 2, self.screen.get_height() // 2)
            self.screen.blit(volume_surface, vol_rect)
            
            # Draw UI
            self.draw_ui()
            
            # Update display
            pygame.display.flip()
            self.clock.tick(30)
        
        pygame.quit()
        print("\nViewer closed")
    
    def save_edited_volume(self, output_path):
        """Save edited volume to NPZ file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, sigma=self.sigma, rgb=self.rgb)
        self.has_changes = False
        print(f"✓ Saved edited volume to: {output_path}")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Interactive Connected Component Viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls:
  Mouse Drag: Rotate camera
  Mouse Wheel: Zoom in/out
  Arrow Keys (Up/Down): Navigate through components
  D: Delete selected component
  R: Restore selected component
  A: Restore all deleted components
  T: Toggle showing only selected component
  S: Enter size threshold mode (delete all below size)
  +/-: Adjust sigma threshold
  [/]: Adjust color tolerance
  Space: Refresh component analysis
  ESC/Q: Quit

Example:
  python component_viewer.py video_voxel_out/recon_volume.npz
  python component_viewer.py volume.npz --sigma-threshold 0.3 --color-tolerance 0.1
        """
    )
    
    parser.add_argument("npz_file", help="Path to NPZ file with sigma and rgb arrays")
    parser.add_argument("--sigma-threshold", type=float, default=0.5,
                       help="Initial sigma threshold (default: 0.5)")
    parser.add_argument("--color-tolerance", type=float, default=0.1,
                       help="Initial color tolerance (default: 0.1, use inf for spatial only)")
    parser.add_argument("--output", type=str, default=None,
                       help="Save edited volume to this path on exit")
    
    args = parser.parse_args()
    
    # Check file exists
    if not Path(args.npz_file).exists():
        print(f"Error: File not found: {args.npz_file}")
        return 1
    
    # Create viewer
    viewer = ComponentViewer(
        args.npz_file,
        sigma_threshold=args.sigma_threshold,
        color_tolerance=args.color_tolerance
    )
    
    # Run
    viewer.run()
    
    # Save if requested or if there are unsaved changes
    if args.output:
        viewer.save_edited_volume(args.output)
    elif viewer.has_changes:
        print("\n" + "="*60)
        print("You have unsaved changes!")
        print("To save, restart with:")
        print(f"  python component_viewer.py {args.npz_file} --output cleaned_volume.npz")
        print("Or press W during the session to save interactively.")
        print("="*60)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

