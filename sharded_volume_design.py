"""
Design document and pseudo-code for sharded voxel grid across 8 GPUs.

This enables training 1024³ or larger volumes by distributing the model
across GPUs instead of copying it to each GPU.

Key idea: Model parallelism with sparse ray queries instead of dense volume copies.
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class ShardedVoxelVolume(nn.Module):
    """
    Voxel volume sharded across multiple GPUs.
    Each GPU owns a spatial partition of the volume.
    """
    
    def __init__(self, grid_size=1024, n_gpus=8, partition_axis='z'):
        """
        Args:
            grid_size: Total voxel grid dimension (assumes cube)
            n_gpus: Number of GPUs to shard across
            partition_axis: 'z' for slab partitioning, '3d' for block partitioning
        """
        super().__init__()
        self.grid_size = grid_size
        self.n_gpus = n_gpus
        self.partition_axis = partition_axis
        
        # Compute shard boundaries
        if partition_axis == 'z':
            # Slab partitioning along Z axis
            self.shard_size = grid_size // n_gpus
            self.shards = []
            for gpu_id in range(n_gpus):
                device = f"cuda:{gpu_id}"
                z_start = gpu_id * self.shard_size
                z_end = (gpu_id + 1) * self.shard_size
                
                # Create shard on this GPU
                # Each shard: [grid_size, grid_size, shard_size]
                shard = VoxelVolumeShard(
                    shape=(grid_size, grid_size, self.shard_size),
                    z_range=(z_start, z_end),
                    device=device
                )
                self.shards.append(shard)
        
        elif partition_axis == '3d':
            # 2×2×2 block partitioning
            assert n_gpus == 8, "3D partitioning requires exactly 8 GPUs"
            half = grid_size // 2
            self.shards = []
            
            gpu_id = 0
            for ix in range(2):
                for iy in range(2):
                    for iz in range(2):
                        device = f"cuda:{gpu_id}"
                        x_start, x_end = ix * half, (ix + 1) * half
                        y_start, y_end = iy * half, (iy + 1) * half
                        z_start, z_end = iz * half, (iz + 1) * half
                        
                        shard = VoxelVolumeShard(
                            shape=(half, half, half),
                            bbox=((x_start, x_end), (y_start, y_end), (z_start, z_end)),
                            device=device
                        )
                        self.shards.append(shard)
                        gpu_id += 1
    
    def sample_volume(self, pts_world):
        """
        Sample sigma and RGB at 3D world points.
        
        This is the KEY function that handles cross-GPU queries.
        
        Args:
            pts_world: [B, N, 3] points in world space [-scene_radius, scene_radius]
        
        Returns:
            sigma: [B, N] density values
            rgb: [B, N, 3] color values
        
        Implementation strategy:
        1. Determine which shard owns each point
        2. Group points by shard
        3. Query each shard on its GPU
        4. Gather results back to requesting GPU
        """
        B, N, _ = pts_world.shape
        device = pts_world.device  # Assume primary GPU
        
        # Convert world coords to grid indices
        pts_grid = self.world_to_grid(pts_world)  # [B, N, 3] in [0, grid_size]
        
        # Determine shard ownership for each point
        shard_ids = self.get_shard_ids(pts_grid)  # [B, N] integers in [0, n_gpus-1]
        
        # Initialize output tensors
        sigma_out = torch.zeros(B, N, device=device)
        rgb_out = torch.zeros(B, N, 3, device=device)
        
        # Query each shard in parallel
        from concurrent.futures import ThreadPoolExecutor
        
        def query_shard(shard_id):
            """Query a single shard for its points"""
            mask = (shard_ids == shard_id)  # [B, N]
            if not mask.any():
                return None
            
            # Extract points belonging to this shard
            pts_shard = pts_grid[mask]  # [K, 3] where K = num points in this shard
            
            # Move to shard's GPU and query
            shard = self.shards[shard_id]
            pts_shard_gpu = pts_shard.to(shard.device)
            
            with torch.cuda.device(shard.device):
                sigma_shard, rgb_shard = shard.sample(pts_shard_gpu)
            
            # Move results back to primary GPU
            return (mask, sigma_shard.to(device), rgb_shard.to(device))
        
        # Parallel queries to all shards
        with ThreadPoolExecutor(max_workers=self.n_gpus) as executor:
            futures = [executor.submit(query_shard, i) for i in range(self.n_gpus)]
            
            for future in futures:
                result = future.result()
                if result is not None:
                    mask, sigma_shard, rgb_shard = result
                    sigma_out[mask] = sigma_shard
                    rgb_out[mask] = rgb_shard
        
        return sigma_out, rgb_out
    
    def world_to_grid(self, pts_world):
        """Convert world coordinates to grid indices"""
        # pts_world in [-scene_radius, scene_radius]
        # → grid indices in [0, grid_size]
        scene_radius = 1.5
        pts_normalized = (pts_world + scene_radius) / (2 * scene_radius)  # [0, 1]
        pts_grid = pts_normalized * self.grid_size  # [0, grid_size]
        return pts_grid
    
    def get_shard_ids(self, pts_grid):
        """Determine which shard owns each point"""
        if self.partition_axis == 'z':
            # Slab partitioning: based on Z coordinate
            z_coords = pts_grid[..., 2]  # [B, N]
            shard_ids = (z_coords / self.shard_size).long()
            shard_ids = torch.clamp(shard_ids, 0, self.n_gpus - 1)
            return shard_ids
        
        elif self.partition_axis == '3d':
            # 3D block partitioning: based on all 3 coordinates
            half = self.grid_size // 2
            ix = (pts_grid[..., 0] >= half).long()  # [B, N]
            iy = (pts_grid[..., 1] >= half).long()
            iz = (pts_grid[..., 2] >= half).long()
            # Linearize 3D index to shard ID
            shard_ids = ix * 4 + iy * 2 + iz
            return shard_ids


class VoxelVolumeShard(nn.Module):
    """
    A single spatial partition of the voxel volume on one GPU.
    """
    
    def __init__(self, shape, device, **kwargs):
        super().__init__()
        self.shape = shape
        self.device = device
        
        # Learnable parameters for this shard
        D, H, W = shape
        self.density_logits = nn.Parameter(
            torch.randn(1, 1, D, H, W, device=device) * 0.1 - 5.0
        )
        self.color = nn.Parameter(
            torch.rand(1, 3, D, H, W, device=device)
        )
    
    def sample(self, pts_grid):
        """
        Sample this shard at grid coordinates.
        
        Args:
            pts_grid: [K, 3] points in grid space [0, grid_size]
        
        Returns:
            sigma: [K] density values
            rgb: [K, 3] color values
        """
        # Convert grid coordinates to normalized coords for F.grid_sample
        # F.grid_sample expects [-1, 1]
        D, H, W = self.shape
        pts_normalized = pts_grid / torch.tensor([W, H, D], device=self.device) * 2 - 1
        
        # Reshape for grid_sample: [1, K, 1, 1, 3]
        pts_normalized = pts_normalized.reshape(1, -1, 1, 1, 3)
        
        # Sample from this shard's tensors
        sigma_logits = torch.nn.functional.grid_sample(
            self.density_logits, pts_normalized,
            mode='bilinear', padding_mode='zeros', align_corners=False
        )
        rgb_sampled = torch.nn.functional.grid_sample(
            self.color, pts_normalized,
            mode='bilinear', padding_mode='zeros', align_corners=False
        )
        
        # Reshape output
        sigma = torch.nn.functional.softplus(sigma_logits.squeeze())  # [K]
        rgb = torch.sigmoid(rgb_sampled.squeeze().T)  # [K, 3]
        
        return sigma, rgb


# ============================================================
# Performance Analysis
# ============================================================

"""
1024³ VOXEL GRID ACROSS 8 GPUs

Memory per GPU:
  - 1024×1024×128 voxels per shard
  - Sigma: 128M × 4 bytes = 512 MB
  - RGB: 128M × 12 bytes = 1.5 GB
  - Total: ~2 GB per GPU ✅
  - A100 has 40GB → plenty of room

Communication per iteration:
  - 82 frames × 64×64 pixels = 337,920 rays
  - 64 samples/ray = 21,626,880 sample points
  - With slab partitioning: each ray crosses ~2 GPUs on average
  - Points per GPU query: ~5.4M points
  - Data transferred: 
      * Query: 5.4M × 12 bytes (xyz coords) = 65 MB
      * Response: 5.4M × 16 bytes (sigma + rgb) = 86 MB
  - Total: ~150 MB bidirectional per iteration
  
  Compare to copying entire volume: 16 GB × 8 = 128 GB ❌
  
  Speedup: 128 GB / 150 MB = 850× less data movement! ✅

Backward pass:
  - Gradients flow back to the GPU that owns each voxel
  - No explicit gradient gathering needed (PyTorch autograd handles it)
  - Each GPU's shard updates independently

Training time estimate:
  - Forward (rendering): ~0.15s (same as 128³)
  - Backward: ~0.05s (slightly more due to cross-GPU grad routing)
  - Total: ~0.20s/iteration
  - 8000 iterations: ~27 minutes ✅
  
RESULT: 1024³ resolution in ~27 minutes on 8 GPUs!
        (8× higher resolution in each dimension = 512× more voxels, same time)
"""


# ============================================================
# Integration with Existing Code
# ============================================================

def render_volume_sharded(sharded_vol, K, poses, img_size=(64, 64),
                         n_samples=64, scene_radius=1.5):
    """
    Render volume using sharded voxel grid.
    
    This is a drop-in replacement for render_volume() but uses
    ShardedVoxelVolume instead of VoxelVolume.
    """
    H, W = img_size
    device = "cuda:0"  # Primary/orchestrator GPU
    
    images = []
    for (R, t) in poses:
        # Generate rays (same as before)
        pts = generate_rays(H, W, K, R, t,
                          n_samples=n_samples,
                          near=0.1, far=5.0,
                          device=device)  # [1, H, W, n_samples, 3]
        
        # Sample from sharded volume (THIS IS THE MAGIC)
        # Automatically queries across GPUs as needed
        sigma_s, rgb_s = sharded_vol.sample_volume(pts)
        
        # Volume rendering (same as before)
        rgb_img = volume_render(sigma_s, rgb_s, n_samples)
        images.append(rgb_img[0])
    
    return images


# ============================================================
# Usage
# ============================================================

if __name__ == "__main__":
    # Example: Train 1024³ volume on 8 GPUs
    
    # Create sharded volume
    sharded_vol = ShardedVoxelVolume(
        grid_size=1024,
        n_gpus=8,
        partition_axis='z'  # or '3d' for 3D block partitioning
    )
    
    # Optimizer across all shards
    optimizer = torch.optim.Adam(sharded_vol.parameters(), lr=1e-2)
    
    # Training loop (same as before!)
    for it in range(8000):
        # Render all views
        pred_images = render_volume_sharded(
            sharded_vol, K, poses,
            img_size=(64, 64),
            n_samples=64,
            scene_radius=1.5
        )
        
        # Loss and backward (same as before)
        pred_stack = torch.stack(pred_images, dim=0)
        loss = F.mse_loss(pred_stack, gt_stack)  # + regularization
        
        optimizer.zero_grad()
        loss.backward()  # Gradients automatically routed to correct GPUs!
        optimizer.step()   # Each shard updates on its own GPU
    
    print("Training complete! 1024³ volume reconstructed.")

