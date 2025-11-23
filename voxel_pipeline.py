#!/usr/bin/env python3
"""
Voxel Processing Pipeline Orchestrator

Orchestrates a sequence of voxel processing operations defined in a JSON config:
- Reconstruction from video (video_orbit_voxel_recon)
- Subdivision (subdivide_voxels)
- Hardening (differentiable_hardening)

Automatically handles GPU management and intermediate file passing.

Usage:
    python voxel_pipeline.py --config pipeline_config.json
"""

import argparse
import json
import math
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys

import numpy as np
import torch

# Import our local modules
import video_orbit_voxel_recon as recon_module
import subdivide_voxels as subdivide_module

# Import hardening components
from differentiable_hardening import (
    FlowBasedVoxelVolume,
    differentiable_harden
)


class VoxelPipeline:
    """
    Pipeline orchestrator for chaining voxel processing operations.
    """
    
    def __init__(self, config_path: str):
        """Load and validate pipeline configuration."""
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(self.config_path) as f:
            self.config = json.load(f)
        
        self.validate_config()
        
        # Setup GPU management
        self.n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.device = "cuda:0" if self.n_gpus > 0 else "cpu"
        
        # Working directory for intermediate files
        self.work_dir = None
        self.temp_dir = None
        
        print("=" * 70)
        print("VOXEL PROCESSING PIPELINE")
        print("=" * 70)
        print(f"Configuration: {self.config_path}")
        print(f"Available GPUs: {self.n_gpus}")
        if self.n_gpus > 0:
            for i in range(self.n_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"  GPU {i}: {gpu_name} ({gpu_mem:.1f} GB)")
        print("=" * 70)
    
    def validate_config(self):
        """Validate configuration structure."""
        required_keys = ["input", "output", "steps"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required key in config: {key}")
        
        # Validate input
        input_cfg = self.config["input"]
        if "type" not in input_cfg:
            raise ValueError("Input must specify 'type' (video, npz, or video_with_checkpoint)")
        if input_cfg["type"] not in ["video", "npz", "video_with_checkpoint"]:
            raise ValueError(f"Invalid input type: {input_cfg['type']}")
        
        # Validate paths based on type
        if input_cfg["type"] == "video_with_checkpoint":
            if "video_path" not in input_cfg:
                raise ValueError("video_with_checkpoint input must specify 'video_path'")
            if "checkpoint_path" not in input_cfg:
                raise ValueError("video_with_checkpoint input must specify 'checkpoint_path'")
        else:
            if "path" not in input_cfg:
                raise ValueError("Input must specify 'path'")
        
        # Validate steps
        if not isinstance(self.config["steps"], list):
            raise ValueError("'steps' must be a list")
        if len(self.config["steps"]) == 0:
            raise ValueError("Pipeline must have at least one step")
        
        for i, step in enumerate(self.config["steps"]):
            if "type" not in step:
                raise ValueError(f"Step {i} missing 'type'")
            if step["type"] not in ["reconstruct", "subdivide", "harden"]:
                raise ValueError(f"Invalid step type: {step['type']}")
            
            # First step validation
            if i == 0:
                if step["type"] != "reconstruct" and input_cfg["type"] != "npz":
                    raise ValueError(
                        "First step must be 'reconstruct' if input is not an NPZ file"
                    )
    
    def setup_working_directory(self):
        """Create temporary working directory for intermediate files."""
        self.temp_dir = tempfile.mkdtemp(prefix="voxel_pipeline_")
        self.work_dir = Path(self.temp_dir)
        print(f"\nWorking directory: {self.work_dir}")
    
    def cleanup_working_directory(self):
        """Remove temporary working directory."""
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
            print(f"\nCleaned up working directory: {self.work_dir}")
    
    def get_initial_npz(self) -> str:
        """Get or create the initial NPZ file."""
        input_cfg = self.config["input"]
        
        if input_cfg["type"] == "npz":
            # Input is already an NPZ file
            npz_path = Path(input_cfg["path"])
            if not npz_path.exists():
                raise FileNotFoundError(f"Input NPZ not found: {npz_path}")
            print(f"\nUsing existing NPZ: {npz_path}")
            return str(npz_path)
        
        elif input_cfg["type"] == "video":
            # First step must be reconstruction
            if self.config["steps"][0]["type"] != "reconstruct":
                raise ValueError("First step must be 'reconstruct' when input is video")
            
            # Will be handled by first step
            return None
        
        elif input_cfg["type"] == "video_with_checkpoint":
            # Validate checkpoint exists
            checkpoint_path = Path(input_cfg["checkpoint_path"])
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint NPZ not found: {checkpoint_path}")
            
            # First step must be reconstruction
            if self.config["steps"][0]["type"] != "reconstruct":
                raise ValueError("First step must be 'reconstruct' when input is video_with_checkpoint")
            
            # Will be handled by first step
            return None
        
        else:
            raise ValueError(f"Unknown input type: {input_cfg['type']}")
    
    def determine_gpu_strategy(self, grid_size: int, img_res: tuple) -> dict:
        """
        Determine optimal GPU usage strategy based on grid size and available GPUs.
        
        Returns dict with:
            - use_sharded: bool (use sharded multi-GPU for huge grids)
            - use_multigpu: bool (use multi-GPU rendering for high-res frames)
            - n_gpus_render: int (number of GPUs for rendering)
        """
        strategy = {
            "use_sharded": False,
            "use_multigpu": False,
            "n_gpus_render": 1
        }
        
        if self.n_gpus <= 1:
            return strategy
        
        # Sharded mode: for massive grids (512+ on 8 GPUs)
        # Memory estimate: grid_size^3 * 4 bytes * 2 (sigma + rgb) ≈ memory needed
        grid_mem_gb = (grid_size ** 3) * 4 * 2 / 1e9
        gpu_0_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        if grid_size >= 256 and self.n_gpus >= 8 and grid_mem_gb > gpu_0_mem_gb * 0.5:
            strategy["use_sharded"] = True
            print(f"  → Using SHARDED mode: {grid_size}³ grid across {self.n_gpus} GPUs")
            return strategy
        
        # Multi-GPU rendering: for high-resolution frames
        n_pixels = img_res[0] * img_res[1]
        if n_pixels >= 16384 and self.n_gpus > 1:  # 128×128 or larger
            strategy["use_multigpu"] = True
            strategy["n_gpus_render"] = self.n_gpus
            print(f"  → Using MULTI-GPU rendering: {self.n_gpus} GPUs for {img_res[0]}×{img_res[1]} frames")
        
        return strategy
    
    def run_reconstruct_step(
        self,
        step_config: Dict[str, Any],
        step_num: int,
        input_npz: Optional[str]
    ) -> str:
        """
        Run reconstruction step.
        
        Returns path to output NPZ.
        """
        print(f"\n{'='*70}")
        print(f"STEP {step_num}: RECONSTRUCTION")
        print(f"{'='*70}")
        
        params = step_config.get("params", {})
        
        # Get video path and optional checkpoint
        input_cfg = self.config["input"]
        checkpoint_npz = None
        
        if "video_path" not in params:
            if input_cfg["type"] == "video":
                params["video_path"] = input_cfg["path"]
            elif input_cfg["type"] == "video_with_checkpoint":
                params["video_path"] = input_cfg["video_path"]
                checkpoint_npz = input_cfg["checkpoint_path"]
            else:
                raise ValueError("Reconstruct step requires 'video_path' parameter")
        
        video_path = params["video_path"]
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Check if checkpoint specified in step params
        if "checkpoint_path" in params:
            checkpoint_npz = params["checkpoint_path"]
            if not Path(checkpoint_npz).exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_npz}")
        
        # If no explicit checkpoint specified but we have input from previous step, use it
        elif input_npz is not None:
            checkpoint_npz = input_npz
            print(f"Using previous step output as checkpoint: {checkpoint_npz}")
        
        # Default parameters
        start_frame = params.get("start_frame", 0)
        frame_step = params.get("frame_step", 1)
        grid_size = params.get("grid_size", 128)
        img_res = tuple(params.get("img_res", [256, 256]))
        n_samples = params.get("n_samples", 64)
        n_iters = params.get("n_iters", 8000)
        scene_radius = params.get("scene_radius", 1.5)
        fov_y_deg = params.get("fov", 45.0)
        
        # Regularization
        lambda_l1 = params.get("lambda_l1", 0.03)
        lambda_tv_sigma = params.get("lambda_tv_sigma", 0.002)
        lambda_tv_rgb = params.get("lambda_tv_rgb", 0.001)
        
        # Modes (support both hyphenated and underscored parameter names)
        use_neighbor_growth = params.get("neighbor_growth", params.get("neighbor-growth", False))
        use_sharded = params.get("sharded", False)
        enable_viewer = params.get("viewer", False)
        
        # Background removal (support both hyphenated and underscored parameter names)
        use_openai_bg_removal = params.get("openai_bg_removal", params.get("openai-bg-removal", False))
        openai_api_key = params.get("openai_api_key", params.get("openai-api-key", None))
        
        # Determine GPU strategy
        strategy = self.determine_gpu_strategy(grid_size, img_res)
        if strategy["use_sharded"]:
            use_sharded = True
        
        # Output directory for this step
        out_dir = self.work_dir / f"step_{step_num:02d}_reconstruct"
        
        print(f"Video: {video_path}")
        if checkpoint_npz:
            print(f"Resuming from checkpoint: {checkpoint_npz}")
        print(f"Grid size: {grid_size}³")
        print(f"Image resolution: {img_res[0]}×{img_res[1]}")
        print(f"Iterations: {n_iters}")
        if use_neighbor_growth:
            print(f"Mode: Neighbor growth")
        if use_openai_bg_removal:
            print(f"Background removal: OpenAI")
        print(f"Output: {out_dir}")
        
        # Auto-estimate orbit parameters
        print("\nAnalyzing video orbit...")
        orbit_period_frames, fps = recon_module.estimate_orbit_period(
            video_path, start_frame=start_frame
        )
        direction = recon_module.estimate_orbit_direction(
            video_path, start_frame=start_frame
        )
        
        # Run reconstruction
        recon_module.train_from_video(
            video_path=video_path,
            orbit_period_frames=orbit_period_frames,
            direction=direction,
            start_frame=start_frame,
            frame_step=frame_step,
            grid_size=grid_size,
            img_res=img_res,
            n_samples=n_samples,
            n_iters=n_iters,
            scene_radius=scene_radius,
            fov_y_deg=fov_y_deg,
            out_dir=str(out_dir),
            use_neighbor_growth=use_neighbor_growth,
            enable_viewer=enable_viewer,
            use_sharded=use_sharded,
            lambda_l1=lambda_l1,
            lambda_tv_sigma=lambda_tv_sigma,
            lambda_tv_rgb=lambda_tv_rgb,
            use_openai_bg_removal=use_openai_bg_removal,
            openai_api_key=openai_api_key,
            checkpoint_npz=checkpoint_npz,  # Pass checkpoint
        )
        
        output_npz = out_dir / "recon_volume.npz"
        if not output_npz.exists():
            raise RuntimeError(f"Reconstruction failed: {output_npz} not found")
        
        print(f"\n✓ Reconstruction complete: {output_npz}")
        return str(output_npz)
    
    def run_subdivide_step(
        self,
        step_config: Dict[str, Any],
        step_num: int,
        input_npz: str
    ) -> str:
        """
        Run subdivision step.
        
        Returns path to output NPZ.
        """
        print(f"\n{'='*70}")
        print(f"STEP {step_num}: SUBDIVISION")
        print(f"{'='*70}")
        
        params = step_config.get("params", {})
        subdivision = params.get("subdivision", 2)
        
        if subdivision < 1:
            raise ValueError(f"Subdivision must be >= 1, got {subdivision}")
        
        # Load input to get current size
        data = np.load(input_npz)
        sigma = data["sigma"]
        D, H, W = sigma.shape
        
        D_new = D * subdivision
        H_new = H * subdivision
        W_new = W * subdivision
        
        print(f"Input: {input_npz}")
        print(f"Current size: {D}×{H}×{W}")
        print(f"Subdivision factor: {subdivision}")
        print(f"New size: {D_new}×{H_new}×{W_new}")
        print(f"Voxel count: {D*H*W:,} → {D_new*H_new*W_new:,} ({subdivision**3}× increase)")
        
        # Output path
        output_npz = self.work_dir / f"step_{step_num:02d}_subdivide.npz"
        
        # Run subdivision
        rgb = data["rgb"]
        sigma_sub, rgb_sub = subdivide_module.subdivide_voxels(sigma, rgb, subdivision)
        
        # Save
        np.savez_compressed(output_npz, sigma=sigma_sub, rgb=rgb_sub)
        
        print(f"\n✓ Subdivision complete: {output_npz}")
        return str(output_npz)
    
    def run_harden_step(
        self,
        step_config: Dict[str, Any],
        step_num: int,
        input_npz: str
    ) -> str:
        """
        Run hardening step.
        
        Returns path to output NPZ.
        """
        print(f"\n{'='*70}")
        print(f"STEP {step_num}: HARDENING")
        print(f"{'='*70}")
        
        params = step_config.get("params", {})
        
        # Default parameters
        n_iters = params.get("n_iters", 1000)
        flow_lr = params.get("flow_lr", 0.01)
        variance_penalty_weight = params.get("variance_penalty", 0.1)
        flow_regularization_weight = params.get("flow_reg", 0.001)
        n_views = params.get("n_views", 8)
        img_res = tuple(params.get("img_res", [64, 64]))
        n_samples = params.get("n_samples", 64)
        scene_radius = params.get("scene_radius", 1.5)
        fov_y_deg = params.get("fov", 45.0)
        use_local_variance = params.get("use_local_variance", True)
        use_global_variance = params.get("use_global_variance", True)
        
        # Output directory for this step
        out_dir = self.work_dir / f"step_{step_num:02d}_harden"
        
        print(f"Input: {input_npz}")
        print(f"Iterations: {n_iters}")
        print(f"Variance penalty: {variance_penalty_weight}")
        print(f"Output: {out_dir}")
        
        # Run hardening
        differentiable_harden(
            npz_path=input_npz,
            out_dir=str(out_dir),
            n_iters=n_iters,
            flow_lr=flow_lr,
            variance_penalty_weight=variance_penalty_weight,
            flow_regularization_weight=flow_regularization_weight,
            n_views=n_views,
            img_res=img_res,
            n_samples=n_samples,
            scene_radius=scene_radius,
            fov_y_deg=fov_y_deg,
            use_local_variance=use_local_variance,
            use_global_variance=use_global_variance,
        )
        
        output_npz = out_dir / "hardened_final.npz"
        if not output_npz.exists():
            raise RuntimeError(f"Hardening failed: {output_npz} not found")
        
        print(f"\n✓ Hardening complete: {output_npz}")
        return str(output_npz)
    
    def run(self):
        """Execute the complete pipeline."""
        try:
            self.setup_working_directory()
            
            # Get initial input (if not starting from video)
            current_npz = self.get_initial_npz()
            
            # Execute each step
            for i, step in enumerate(self.config["steps"], start=1):
                step_type = step["type"]
                
                if step_type == "reconstruct":
                    current_npz = self.run_reconstruct_step(step, i, current_npz)
                
                elif step_type == "subdivide":
                    if current_npz is None:
                        raise RuntimeError("Subdivide step requires input NPZ")
                    current_npz = self.run_subdivide_step(step, i, current_npz)
                
                elif step_type == "harden":
                    if current_npz is None:
                        raise RuntimeError("Harden step requires input NPZ")
                    current_npz = self.run_harden_step(step, i, current_npz)
                
                else:
                    raise ValueError(f"Unknown step type: {step_type}")
            
            # Copy final output to destination
            output_path = Path(self.config["output"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            print(f"\n{'='*70}")
            print("PIPELINE COMPLETE")
            print(f"{'='*70}")
            print(f"Copying final output: {current_npz} → {output_path}")
            
            shutil.copy2(current_npz, output_path)
            
            # Report final file size
            final_size_mb = output_path.stat().st_size / 1e6
            print(f"Final output: {output_path} ({final_size_mb:.2f} MB)")
            
            # Load and report final statistics
            data = np.load(output_path)
            sigma = data["sigma"]
            rgb = data["rgb"]
            D, H, W = sigma.shape
            
            print(f"\nFinal volume statistics:")
            print(f"  Grid size: {D}×{H}×{W}")
            print(f"  Total voxels: {D*H*W:,}")
            print(f"  Non-zero voxels: {np.sum(sigma > 0.01):,}")
            print(f"  Sigma range: [{sigma.min():.4f}, {sigma.max():.4f}]")
            print(f"  RGB range: [{rgb.min():.4f}, {rgb.max():.4f}]")
            
            print(f"\n✓ Pipeline completed successfully!")
            print(f"  {len(self.config['steps'])} steps executed")
            print(f"  Final output: {output_path}")
            
        finally:
            # Always cleanup
            self.cleanup_working_directory()


def create_example_config(output_path: str):
    """Create an example configuration file."""
    example_config = {
        "input": {
            "type": "video",
            "path": "video.mp4"
        },
        "output": "final_volume.npz",
        "steps": [
            {
                "type": "reconstruct",
                "params": {
                    "start_frame": 0,
                    "grid_size": 64,
                    "n_iters": 8000,
                    "img_res": [256, 256],
                    "lambda_l1": 0.03,
                    "lambda_tv_sigma": 0.002,
                    "lambda_tv_rgb": 0.001
                }
            },
            {
                "type": "subdivide",
                "params": {
                    "subdivision": 2
                }
            },
            {
                "type": "harden",
                "params": {
                    "n_iters": 1000,
                    "variance_penalty": 0.1,
                    "flow_lr": 0.01
                }
            }
        ]
    }
    
    with open(output_path, "w") as f:
        json.dump(example_config, f, indent=2)
    
    print(f"Example configuration written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Voxel Processing Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run pipeline from config
  python voxel_pipeline.py --config pipeline.json
  
  # Generate example config
  python voxel_pipeline.py --create-example-config example.json

Example pipeline config (pipeline.json):
{
  "input": {
    "type": "video",
    "path": "video.mp4"
  },
  "output": "final_volume.npz",
  "steps": [
    {
      "type": "reconstruct",
      "params": {
        "grid_size": 64,
        "n_iters": 8000
      }
    },
    {
      "type": "subdivide",
      "params": {
        "subdivision": 2
      }
    },
    {
      "type": "harden",
      "params": {
        "n_iters": 1000,
        "variance_penalty": 0.1
      }
    }
  ]
}

Starting from existing NPZ:
{
  "input": {
    "type": "npz",
    "path": "existing_volume.npz"
  },
  "output": "processed_volume.npz",
  "steps": [
    {
      "type": "subdivide",
      "params": {"subdivision": 2}
    },
    {
      "type": "harden",
      "params": {"n_iters": 500}
    }
  ]
}

Resume from checkpoint (continue reconstruction):
{
  "input": {
    "type": "video_with_checkpoint",
    "video_path": "video.mp4",
    "checkpoint_path": "previous_recon.npz"
  },
  "output": "continued_volume.npz",
  "steps": [
    {
      "type": "reconstruct",
      "params": {"n_iters": 4000}
    }
  ]
}
        """
    )
    
    parser.add_argument(
        "--config", type=str,
        help="Path to pipeline configuration JSON file"
    )
    parser.add_argument(
        "--create-example-config", type=str, metavar="OUTPUT",
        help="Create an example configuration file at the specified path"
    )
    
    args = parser.parse_args()
    
    if args.create_example_config:
        create_example_config(args.create_example_config)
        return
    
    if not args.config:
        parser.error("Either --config or --create-example-config must be specified")
    
    # Run pipeline
    pipeline = VoxelPipeline(args.config)
    pipeline.run()


if __name__ == "__main__":
    main()

