# Component Viewer Guide

Interactive tool for viewing, navigating, and editing connected components in voxel volumes.

## Quick Start

```bash
# Basic usage (uses color tolerance 0.1 by default)
python component_viewer.py video_voxel_out/recon_volume.npz

# With custom thresholds
python component_viewer.py volume.npz --sigma-threshold 0.3 --color-tolerance 0.05

# Spatial connectivity only (no color separation)
python component_viewer.py volume.npz --color-tolerance inf

# Save edited volume on exit
python component_viewer.py volume.npz --output cleaned_volume.npz
```

## Features

### 🎨 Dual Rendering Modes
- **Cube Mode (C)**: Minecraft-style voxel cubes with proper depth buffering, triangle rasterization, backface culling, and lighting
- **Volumetric Mode (V)**: NeRF-style volume rendering with ray marching and alpha compositing
- Same rendering quality as `voxel_volume_viewer.py`
- Smooth mouse-based camera rotation
- Toggle between modes on-the-fly

### 🔍 Component Navigation
- Components are automatically sorted by size (largest first)
- Navigate through them with **Up/Down** arrow keys
- Selected component is highlighted in yellow in the 3D view
- Component info shown in top-left panel

### 🗑️ Component Deletion & Saving
- **D** - Delete currently selected component
- **S** - Delete all components below a specified size
- **R** - Restore currently selected component
- **A** - Restore all deleted components
- **W** - Save edited volume (prompts for output path)

### 🎨 Threshold Adjustment
- **+/-** - Adjust sigma threshold (affects what's considered "occupied")
- **[/]** - Adjust color tolerance (affects component connectivity)
- **Space** - Refresh analysis with new thresholds

### 👁️ View Controls
- **Mouse Drag** - Rotate camera around the volume
- **Mouse Wheel** - Zoom in/out
- **V** - Switch to volumetric rendering mode (NeRF-style)
- **C** - Switch to cube rendering mode (Minecraft-style)
- **T** - Toggle showing only selected component (isolate view)
- **Q/ESC** - Quit

## Workflow Examples

### Remove Small Spurious Components

1. Open volume: `python component_viewer.py volume.npz`
2. Navigate through components to inspect them
3. Press **S** to enter size threshold mode
4. Type minimum size (e.g., `100`) and press **Enter**
5. All components below that size are deleted
6. Press **A** if you want to restore them
7. Press **W** to save, enter filename (e.g., `cleaned.npz`), press **Enter**
8. Quit with **Q**

### Separate Objects by Color

1. Open with color tolerance: `python component_viewer.py volume.npz --color-tolerance 0.1`
2. Navigate through components (now separated by color)
3. Delete unwanted colored regions with **D**
4. Save cleaned volume: restart with `--output cleaned.npz`

### Manual Component Cleanup

1. Open volume: `python component_viewer.py volume.npz`
2. Press **Down** to navigate to next component
3. Press **D** to delete if unwanted
4. Press **R** if you deleted by mistake
5. Continue through all components
6. Press **W**, enter output path, press **Enter** to save

### Isolate and Inspect

1. Navigate to component of interest
2. Press **T** to toggle isolation (show only this component)
3. Rotate with mouse to inspect from all angles
4. Press **T** again to show all components

## UI Elements

### Top-Left Panel
Shows information about currently selected component:
- Component number and total count
- Component ID
- Size in voxels and percentage of total
- Average RGB color
- Status (Active or DELETED)
- Current thresholds
- Number of deleted components

### Bottom-Left Panel
Quick reference of all keyboard controls

### Size Threshold Input
When you press **S**, a dialog appears:
- Type the minimum size (digits only)
- Press **Enter** to delete all components below that size
- Press **ESC** to cancel

## Parameters

### Sigma Threshold
- Controls which voxels are considered "occupied"
- Higher values = only denser voxels (fewer components, smaller)
- Lower values = more voxels included (more components, larger)
- Adjust with **+/-** keys

### Color Tolerance
- Controls how similar colors must be to be in same component
- `0.1` (default) = moderate color similarity (recommended)
- `0.05` = strict similarity required
- `inf` = spatial connectivity only (ignores color completely)
- Adjust with **[/]** keys
- After adjusting, press **Space** to refresh

## Tips & Tricks

### Finding Floating Artifacts
1. Set low sigma threshold to see all voxels
2. Navigate through components by size
3. Small isolated components are likely artifacts
4. Use **S** to delete all small components at once

### Color-Based Separation
1. By default, color tolerance is 0.1 (moderate separation)
2. If objects are still merged, decrease tolerance with **[**
3. If objects are too fragmented, increase tolerance with **]**
4. Press **Space** after each adjustment to refresh
5. For spatial-only connectivity, increase to `inf` with **]**

### Inspection Workflow
1. Use **T** to isolate each component
2. Toggle between **V** (volumetric) and **C** (cube) modes to see different representations
3. Rotate with mouse to view from all angles
4. Decide if component should be kept or deleted
5. Press **T** again to return to full view

### Rendering Mode Tips
- **Cube Mode (C)**: Better for seeing individual voxel structure, faster
- **Volumetric Mode (V)**: Better for smooth appearance, shows density gradients
- Selected component highlighted in yellow in both modes
- Try both modes to get best understanding of your data

### Undo Mistakes
- Single component: Press **R** to restore
- All components: Press **A** to restore all
- No changes are permanent until you save

## Saving Edited Volumes

### Option 1: Interactive Save (Recommended)
Press **W** during the session:
1. Press **W** key
2. Type output path (e.g., `cleaned_volume.npz`)
3. Press **Enter**
4. See "✓ Saved" confirmation

### Option 2: Command Line (Auto-save on exit)
```bash
python component_viewer.py volume.npz --output cleaned_volume.npz
```
Automatically saves when you quit.

### Unsaved Changes Warning
- "*Unsaved changes*" indicator appears in top-right when you make edits
- Warning message shown on exit if changes aren't saved
- Press **W** anytime to save current state

## Performance Notes

### Large Volumes
- For 256³ or larger volumes, rendering may be slow
- Component analysis is fast but rendering takes time
- Consider using lower sigma threshold to reduce voxels rendered

### Many Components
- If analysis finds 100+ components, navigation works but may be slow
- Use color tolerance to reduce fragmentation
- Or use size threshold to delete small ones quickly

## Keyboard Reference

| Key | Action |
|-----|--------|
| ↑/↓ | Navigate components |
| V | Volumetric mode |
| C | Cube mode |
| D | Delete selected |
| R | Restore selected |
| A | Restore all |
| W | Save edited volume |
| T | Toggle isolation |
| S | Delete by size |
| +/- | Adjust sigma |
| [/] | Adjust color tolerance |
| Space | Refresh analysis |
| Q/ESC | Quit |

## Common Issues

### "No components found"
- Sigma threshold too high
- Press **-** several times to lower threshold
- Press **Space** to refresh

### Too many tiny components
- Sigma threshold too low, or
- Color tolerance too strict
- Press **S** to delete small components
- Or press **]** to increase color tolerance

### Component highlighting not visible
- Camera might be at wrong angle
- Try zooming out with mouse wheel
- Try different camera angles with mouse drag

### Changes not saving
- Must use `--output` flag when launching viewer
- Or manually save using other scripts after noting deleted IDs

## See Also

- `analyze_connected_components.py` - Batch analysis without GUI
- `filter_voxel_components.py` - Script-based component filtering
- `PARAMETER_GUIDE.md` - Understanding reconstruction parameters

