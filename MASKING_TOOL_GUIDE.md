# Voxel Masking Tool Guide

## Overview

The interactive voxel masking tool lets you clean up your 3D reconstructions by manually removing unwanted voxels. This is perfect for:

- Removing white artifact "clouds"
- Cleaning up spurious density
- Removing background noise
- Sculpting the final reconstruction

## Usage

### Basic Usage

```bash
python voxel_masking_tool.py video_voxel_out/recon_volume.npz
```

This opens an interactive 3D viewer where you can mask/unmask voxels.

### Controls

#### Camera Controls
- **Arrow Keys**: Orbit camera (left/right for yaw, up/down for pitch)
- **+/-**: Zoom in/out
- **V**: Switch to volumetric rendering mode (default, recommended)
- **C**: Switch to cube rendering mode (debug view)

#### Masking Controls
- **Left Click**: **MASK** voxels along ray (removes them)
- **Right Click**: **UNMASK** voxels along ray (restores them to default density)
- **U**: Undo last operation (up to 20 operations)
- **Ctrl+S**: Save modified volume

#### Other
- **ESC**: Quit viewer

## How It Works

### Ray Casting

When you click on the screen:
1. A ray is cast from the camera through the clicked pixel
2. 256 sample points are generated along the ray
3. All voxels intersecting these points are modified
4. The effect is immediate and visible

### Masking (Left Click)

- Sets voxel density (σ) to **0.0** along the ray
- Effectively removes the voxels from rendering
- Use this to remove unwanted artifacts

### Unmasking (Right Click)

- Sets voxel density (σ) to **5.0** along the ray
- Restores voxels to a default visible state
- Useful if you accidentally masked too much

### Undo

- Stores the last 20 operations in memory
- Press **U** to undo the most recent mask/unmask
- Can undo multiple times in sequence

## Workflow

### 1. Initial View

```bash
python voxel_masking_tool.py video_voxel_out/recon_volume.npz
```

The tool starts in volumetric rendering mode (like NeRF) which gives you the best view of the overall structure.

### 2. Identify Artifacts

Rotate the camera using arrow keys to find:
- White clouds at top/bottom
- Isolated noise voxels
- Funnel shapes or other artifacts

### 3. Remove Artifacts

- Position the camera so the artifact is visible
- **Left click** through the artifact to cast a masking ray
- Repeat from different angles if needed
- The artifact should disappear immediately

### 4. Fine Tuning

- Volumetric mode (**V**) is recommended for understanding the structure
- Switch to cube mode (**C**) only if you need to see individual voxels
- Use **U** to undo mistakes
- Zoom in (**+**) for detailed work

### 5. Save Result

Press **Ctrl+S** to save. The modified volume is saved as:
```
video_voxel_out/recon_volume_masked.npz
```

### 6. View Final Result

```bash
python voxel_volume_viewer.py video_voxel_out/recon_volume_masked.npz
```

## Tips & Tricks

### Effective Masking

1. **Use Multiple Angles**: Artifacts often require removal from multiple viewpoints
   - Rotate to see the artifact from different sides
   - Mask from 2-3 angles for complete removal

2. **Volumetric Mode (Default)**: The NeRF-style rendering shows you exactly what the reconstruction looks like
   - Best for understanding overall structure
   - Shows transparency and depth correctly
   - Use this for 95% of your masking work

3. **Cube Mode (Debug)**: Switch to cube mode (**C**) only if you need to debug
   - Shows individual voxel positions
   - Useful for finding isolated noise voxels
   - Less intuitive for understanding the shape

4. **Undo is Your Friend**: Don't hesitate to undo (**U**) if you remove too much

### Common Patterns

**Removing Top/Bottom Clouds:**
```
1. Orbit to side view (arrow keys)
2. Position camera so you're looking horizontally through the cloud
3. Left click through the cloud area
4. Rotate slightly and repeat
```

**Removing Isolated Noise:**
```
1. Zoom in (+) to see the noise clearly
2. Left click directly on the noise point
3. The ray will pass through and remove it
```

**Removing Funnel Artifacts:**
```
1. Orbit to look down the funnel axis
2. Left click along the funnel walls
3. Rotate 90 degrees and repeat
4. The funnel should collapse inward
```

### Keyboard Shortcuts Summary

| Key | Action |
|-----|--------|
| ← → | Orbit left/right |
| ↑ ↓ | Orbit up/down |
| + - | Zoom in/out |
| V | Volume rendering (default) |
| C | Cube rendering (debug) |
| Left Click | Mask (remove) |
| Right Click | Unmask (restore) |
| U | Undo |
| Ctrl+S | Save |
| ESC | Quit |

## Technical Details

### Ray Sampling

- **256 samples** per ray for complete coverage
- **Near**: 0.1 world units
- **Far**: 5.0 world units
- **Scene radius**: 1.5 world units

### Undo Stack

- Stores up to **20** previous states
- Each state is a full copy of the sigma volume
- Memory usage: ~20 × volume_size

### Save Format

Saved files use the same `.npz` format as the original:
```python
{
    'sigma': np.array [D, H, W],  # Modified density
    'rgb': np.array [D, H, W, 3]  # Original colors (unchanged)
}
```

### Performance

- **CPU-based**: Runs on CPU for immediate feedback
- **Interactive**: 30 FPS target framerate
- **Volume Mode (Default)**: High-quality NeRF-style rendering with 128 samples per ray
- **Cube Mode**: Faster but less intuitive, mainly for debugging

**Note**: Volumetric rendering may be slower on very large volumes (>256³), but gives the best visualization

## Example Session

```bash
# 1. Start masking tool
$ python voxel_masking_tool.py video_voxel_out/recon_volume.npz
Loading video_voxel_out/recon_volume.npz...
Volume shape: (128, 128, 128)
Active voxels (σ>0.5): 45231

# 2. Interactive masking (in viewer)
#    - Rotate to see artifacts
#    - Left click to remove them
#    - U to undo mistakes
#    - Ctrl+S to save

Masked ray at pixel (400, 200)
Masked ray at pixel (410, 205)
Undone
Masked ray at pixel (405, 203)
Saved to video_voxel_out/recon_volume_masked.npz

# 3. View cleaned result
$ python voxel_volume_viewer.py video_voxel_out/recon_volume_masked.npz
```

## Troubleshooting

### "Nothing happens when I click"

- Make sure you're clicking inside the viewer window
- Check that the camera is positioned to see voxels
- Try switching to cube mode (**C**) to see if voxels are present

### "Undo doesn't work"

- Undo only works if you've made at least one mask/unmask operation
- Maximum 20 undo levels

### "Viewer is slow"

- Volumetric rendering uses 128 samples per ray for high quality
- For very large volumes (>256³), you may experience slowdown
- If needed, switch to cube mode (**C**) for faster rendering
- Consider working with a filtered/downsampled version first

### "I removed too much"

- Press **U** to undo
- Right click to unmask and restore voxels
- If you saved, you'll need to reload the original file

## Advanced Usage

### Batch Processing

After manual masking, you can still use automated tools:

```bash
# 1. Manual cleanup with masking tool
python voxel_masking_tool.py recon_volume.npz
# (save as recon_volume_masked.npz)

# 2. Further automated filtering
python filter_white_noise.py recon_volume_masked.npz --min-percent 1.0

# 3. Final result
python voxel_volume_viewer.py recon_volume_masked_filtered.npz
```

### Multiple Passes

For complex reconstructions, work in passes:

1. **Pass 1**: Remove large obvious artifacts (clouds, funnels)
   - Save as `recon_volume_pass1.npz`
   
2. **Pass 2**: Remove medium-sized noise
   - Load `recon_volume_pass1.npz`
   - Save as `recon_volume_pass2.npz`
   
3. **Pass 3**: Fine detail cleanup
   - Load `recon_volume_pass2.npz`
   - Save as `recon_volume_final.npz`

## Next Steps

After masking:
1. View the result: `python voxel_volume_viewer.py video_voxel_out/recon_volume_masked.npz`
2. Export to PLY: Use the masked `.npz` file in your export pipeline
3. Analyze: `python analyze_connected_components.py video_voxel_out/recon_volume_masked.npz`

## See Also

- `voxel_volume_viewer.py` - View-only viewer
- `filter_white_noise.py` - Automated white noise filtering
- `analyze_connected_components.py` - Component analysis
- `PARAMETER_GUIDE.md` - Reconstruction parameters

