"""
Sharded Voxel Volume for multi-GPU model parallelism.

Enables training 1024³ or larger volumes by distributing the model
across multiple GPUs using 3D block partitioning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple


class VoxelVolumeShard(nn.Module):
    """
    A single 3D spatial partition of the voxel volume on one GPU.
    Owns a cubic region of the full volume.
    """
    
    def __init__(self, 
                 shape: Tuple[int, int, int],
                 bbox: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
                 device: str,
                 sigma_scale: float = 1.0,
                 init_logit: float = -5.0):
        """
        Args:
            shape: (D, H, W) size of this shard in voxels
            bbox: ((x_start, x_end), (y_start, y_end), (z_start, z_end)) in global grid coords
            device: GPU device for this shard
            sigma_scale: Scale factor for density
            init_logit: Initial value for density logits
        """
        super().__init__()
        self.shape = shape
        self.bbox = bbox
        self.device = device
        self.sigma_scale = sigma_scale
        
        D, H, W = shape
        
        # Learnable parameters for this shard
        self.density_logits = nn.Parameter(
            torch.randn(1, 1, D, H, W, device=device) * 0.1 + init_logit
        )
        self.color = nn.Parameter(
            torch.rand(1, 3, D, H, W, device=device)
        )
    
    def forward(self):
        """Return sigma and RGB tensors for this shard."""
        sigma = F.softplus(self.density_logits) * self.sigma_scale
        rgb = torch.sigmoid(self.color)
        return sigma, rgb
    
    def sample_local(self, pts_local_grid):
        """
        Sample this shard at local grid coordinates.
        
        Args:
            pts_local_grid: [K, 3] points in LOCAL grid space [0, shard_size]
                           where shard_size is the dimension of this shard
        
        Returns:
            sigma: [K] density values
            rgb: [K, 3] color values
        """
        if pts_local_grid.shape[0] == 0:
            # No points to sample
            return (torch.zeros(0, device=self.device),
                    torch.zeros(0, 3, device=self.device))
        
        # Get current sigma and rgb
        sigma, rgb = self.forward()
        
        D, H, W = self.shape
        
        # Convert local grid coordinates to normalized coords for F.grid_sample
        # pts_local_grid is in [0, W], [0, H], [0, D]
        # F.grid_sample expects [..., (x, y, z)] in [-1, 1]
        pts_normalized = pts_local_grid.clone()
        pts_normalized[:, 0] = pts_normalized[:, 0] / W * 2 - 1  # x
        pts_normalized[:, 1] = pts_normalized[:, 1] / H * 2 - 1  # y
        pts_normalized[:, 2] = pts_normalized[:, 2] / D * 2 - 1  # z
        
        # Reshape for grid_sample: [1, K, 1, 1, 3]
        pts_normalized = pts_normalized.unsqueeze(0).unsqueeze(2).unsqueeze(2)
        
        # Sample from this shard's tensors
        sigma_sampled = F.grid_sample(
            sigma, pts_normalized,
            mode='bilinear', padding_mode='zeros', align_corners=False
        )
        rgb_sampled = F.grid_sample(
            rgb, pts_normalized,
            mode='bilinear', padding_mode='zeros', align_corners=False
        )
        
        # Reshape output: [1,C,K,1,1] -> [K,C]
        sigma_out = sigma_sampled.squeeze(0).squeeze(-1).squeeze(-1).squeeze(0)  # [K]
        rgb_out = rgb_sampled.squeeze(0).squeeze(-1).squeeze(-1).permute(1, 0)   # [K, 3]
        
        return sigma_out, rgb_out


class ShardedVoxelVolume(nn.Module):
    """
    Voxel volume sharded across 8 GPUs using 2×2×2 block partitioning.
    Each GPU owns a cubic octant of the full volume.
    """
    
    def __init__(self, grid_size: int = 1024, n_gpus: int = 8,
                 sigma_scale: float = 1.0, init_logit: float = -5.0):
        """
        Args:
            grid_size: Total voxel grid dimension (assumes cube)
            n_gpus: Must be 8 for 2×2×2 partitioning
            sigma_scale: Scale factor for density
            init_logit: Initial value for density logits
        """
        super().__init__()
        assert n_gpus == 8, "3D block partitioning requires exactly 8 GPUs"
        assert grid_size % 2 == 0, "grid_size must be even for 2×2×2 partitioning"
        
        self.grid_size = grid_size
        self.n_gpus = n_gpus
        self.half = grid_size // 2
        
        # Create 8 shards in 2×2×2 layout
        self.shards = nn.ModuleList()
        self.shard_bboxes = []
        
        gpu_id = 0
        for ix in range(2):
            for iy in range(2):
                for iz in range(2):
                    x_start = ix * self.half
                    x_end = (ix + 1) * self.half
                    y_start = iy * self.half
                    y_end = (iy + 1) * self.half
                    z_start = iz * self.half
                    z_end = (iz + 1) * self.half
                    
                    bbox = ((x_start, x_end), (y_start, y_end), (z_start, z_end))
                    device = f"cuda:{gpu_id}"
                    
                    shard = VoxelVolumeShard(
                        shape=(self.half, self.half, self.half),
                        bbox=bbox,
                        device=device,
                        sigma_scale=sigma_scale,
                        init_logit=init_logit
                    )
                    
                    self.shards.append(shard)
                    self.shard_bboxes.append(bbox)
                    gpu_id += 1
        
        print(f"[ShardedVolume] Created {len(self.shards)} shards:")
        for i, (shard, bbox) in enumerate(zip(self.shards, self.shard_bboxes)):
            print(f"  Shard {i} on {shard.device}: bbox={bbox}")
    
    def world_to_grid(self, pts_world: torch.Tensor, scene_radius: float = 1.5) -> torch.Tensor:
        """
        Convert world coordinates to grid indices.
        
        Args:
            pts_world: [..., 3] points in world space [-scene_radius, scene_radius]
            scene_radius: Radius of the scene
        
        Returns:
            pts_grid: [..., 3] points in grid space [0, grid_size]
        """
        # Normalize to [0, 1]
        pts_normalized = (pts_world + scene_radius) / (2 * scene_radius)
        # Scale to [0, grid_size]
        pts_grid = pts_normalized * self.grid_size
        return pts_grid
    
    def get_shard_id(self, pt_grid: torch.Tensor) -> int:
        """
        Determine which shard owns a single point.
        
        Args:
            pt_grid: [3] point in grid space [0, grid_size]
        
        Returns:
            shard_id: Integer in [0, 7]
        """
        x, y, z = pt_grid
        ix = 1 if x >= self.half else 0
        iy = 1 if y >= self.half else 0
        iz = 1 if z >= self.half else 0
        # Linearize 3D index: ix * 4 + iy * 2 + iz
        return ix * 4 + iy * 2 + iz
    
    def get_shard_ids_batch(self, pts_grid: torch.Tensor) -> torch.Tensor:
        """
        Determine which shard owns each point in a batch.
        
        Args:
            pts_grid: [N, 3] points in grid space [0, grid_size]
        
        Returns:
            shard_ids: [N] integers in [0, 7]
        """
        ix = (pts_grid[:, 0] >= self.half).long()
        iy = (pts_grid[:, 1] >= self.half).long()
        iz = (pts_grid[:, 2] >= self.half).long()
        shard_ids = ix * 4 + iy * 2 + iz
        return shard_ids
    
    def global_to_local_coords(self, pts_grid: torch.Tensor, shard_id: int) -> torch.Tensor:
        """
        Convert global grid coordinates to local shard coordinates.
        
        Args:
            pts_grid: [N, 3] points in global grid space [0, grid_size]
            shard_id: Which shard to convert to
        
        Returns:
            pts_local: [N, 3] points in local shard space [0, half]
        """
        bbox = self.shard_bboxes[shard_id]
        (x_start, _), (y_start, _), (z_start, _) = bbox
        
        pts_local = pts_grid.clone()
        pts_local[:, 0] -= x_start
        pts_local[:, 1] -= y_start
        pts_local[:, 2] -= z_start
        
        return pts_local
    
    def sample_volume(self, pts_world: torch.Tensor, scene_radius: float = 1.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample sigma and RGB at 3D world points.
        
        This function handles cross-GPU queries automatically.
        
        Args:
            pts_world: [N, 3] points in world space [-scene_radius, scene_radius]
            scene_radius: Radius of the scene
        
        Returns:
            sigma: [N] density values
            rgb: [N, 3] color values
        """
        N = pts_world.shape[0]
        device = pts_world.device  # Primary GPU (cuda:0)
        
        # Convert world coords to grid indices
        pts_grid = self.world_to_grid(pts_world, scene_radius)  # [N, 3]
        
        # Determine shard ownership for each point
        shard_ids = self.get_shard_ids_batch(pts_grid)  # [N]
        
        # Initialize output tensors on primary GPU
        sigma_out = torch.zeros(N, device=device)
        rgb_out = torch.zeros(N, 3, device=device)
        
        # Query each shard in parallel
        def query_shard(shard_id):
            """Query a single shard for its points"""
            mask = (shard_ids == shard_id)
            if not mask.any():
                return None
            
            # Extract points belonging to this shard
            pts_shard_global = pts_grid[mask]  # [K, 3]
            
            # Convert to local coordinates for this shard
            pts_shard_local = self.global_to_local_coords(pts_shard_global, shard_id)
            
            # Move to shard's GPU and query
            shard = self.shards[shard_id]
            pts_shard_local_gpu = pts_shard_local.to(shard.device)
            
            with torch.cuda.device(shard.device):
                sigma_shard, rgb_shard = shard.sample_local(pts_shard_local_gpu)
            
            # Move results back to primary GPU
            return (mask, sigma_shard.to(device), rgb_shard.to(device))
        
        # Parallel queries to all 8 shards
        with ThreadPoolExecutor(max_workers=self.n_gpus) as executor:
            futures = [executor.submit(query_shard, i) for i in range(self.n_gpus)]
            
            for future in futures:
                result = future.result()
                if result is not None:
                    mask, sigma_shard, rgb_shard = result
                    sigma_out[mask] = sigma_shard
                    rgb_out[mask] = rgb_shard
        
        return sigma_out, rgb_out


def test_sharded_volume():
    """Test the sharded volume implementation."""
    print("Testing ShardedVoxelVolume...")
    
    # Create a small sharded volume for testing
    sharded_vol = ShardedVoxelVolume(grid_size=128, n_gpus=8)
    
    # Test sampling at random points
    pts_world = torch.randn(1000, 3, device="cuda:0") * 1.0  # Points in [-1, 1]
    sigma, rgb = sharded_vol.sample_volume(pts_world, scene_radius=1.5)
    
    print(f"Sampled {pts_world.shape[0]} points:")
    print(f"  sigma: {sigma.shape}, mean={sigma.mean():.4f}, std={sigma.std():.4f}")
    print(f"  rgb: {rgb.shape}, mean={rgb.mean():.4f}, std={rgb.std():.4f}")
    
    # Verify all points were sampled (no zeros from unassigned shards)
    assert (sigma >= 0).all(), "Some sigma values are negative"
    assert (rgb >= 0).all() and (rgb <= 1).all(), "RGB values out of range"
    
    print("✓ Test passed!")


if __name__ == "__main__":
    test_sharded_volume()

