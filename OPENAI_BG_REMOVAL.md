# OpenAI Background Removal Guide

## Overview

The reconstruction pipeline now supports using OpenAI's DALL-E 3 to intelligently remove backgrounds and white artifacts from video frames before reconstruction. This is particularly useful for:

1. **White Island Artifacts**: Isolated white regions that are not connected to the image border
2. **Complex Backgrounds**: Scenes where traditional background subtraction struggles
3. **Inconsistent Lighting**: Videos with varying lighting conditions

## How It Works

The system uses an AI-powered mask generation approach:

1. **Vision Analysis**: Each frame is analyzed using GPT-4o Vision to identify the main subject/object, detect white artifacts, and determine optimal masking parameters
2. **Mask Generation**: Based on AI guidance, an improved alpha mask is created that removes background and white islands
3. **Original Preservation**: The original image is kept intact - only the alpha mask is modified

**IMPORTANT**: This preserves the original image geometry, camera angles, and multi-view consistency required for 3D reconstruction!

All frames are processed in parallel (default: 4 workers) to maximize throughput.

## Setup

### 1. Install Dependencies

```bash
pip install openai "pillow>=9.1.0" requests
```

**Note**: Pillow 9.1.0+ is recommended for best compatibility, though older versions are supported with fallback.

### 2. Get an OpenAI API Key

Sign up at https://platform.openai.com/ and create an API key.

### 3. Set Your API Key

**Option A: Environment Variable (Recommended)**
```bash
export OPENAI_API_KEY="sk-your-key-here"
```

**Option B: Command-Line Argument**
```bash
python video_orbit_voxel_recon.py video.mp4 0 --openai-bg-removal --openai-api-key "sk-your-key-here"
```

## Usage

### Basic Usage

```bash
# Set API key
export OPENAI_API_KEY="sk-your-key-here"

# Run reconstruction with OpenAI background removal
python video_orbit_voxel_recon.py video.mp4 0 --openai-bg-removal
```

### Advanced Usage

```bash
# High-resolution with OpenAI background removal
python video_orbit_voxel_recon.py video.mp4 0 \
  --openai-bg-removal \
  --img-res 512 512 \
  --n-iters 16000 \
  --grid-size 256

# With custom regularization
python video_orbit_voxel_recon.py video.mp4 0 \
  --openai-bg-removal \
  --lambda-l1 0.02 \
  --lambda-tv-sigma 0.001
```

## Cost Considerations

OpenAI background removal uses **one API call per frame**:
1. **GPT-4o Vision** (~$0.01 per image for high-detail analysis)

**Estimated cost**: ~$0.01 per frame

For an 82-frame video (default full orbit):
- **Total cost**: ~$0.82
- **Processing time**: 1-2 minutes (depending on API speed)

### Cost Optimization Tips

1. **Use frame step**: Process fewer frames
   ```bash
   python video_orbit_voxel_recon.py video.mp4 0 --openai-bg-removal --frame-step 2
   ```
   This halves the number of frames (41 frames = ~$0.41)

2. **Test first**: Run without OpenAI to see if traditional background subtraction works
   ```bash
   python video_orbit_voxel_recon.py video.mp4 0
   ```

3. **Spot check**: Manually inspect a few frames to decide if AI removal is necessary

4. **Cost is minimal**: At ~$0.82 for a full orbit, the cost is very reasonable for significantly improved quality

## When to Use OpenAI Background Removal

### ✅ Use It When:
- Traditional background subtraction leaves white artifacts
- The object has similar colors to the background
- You have isolated white "islands" not connected to borders
- The background is complex or non-uniform
- Lighting changes throughout the video

### ❌ Skip It When:
- Traditional background subtraction works well
- The background is uniform and distinct from the object
- Cost is a major concern
- Processing time is critical

## Fallback Behavior

If OpenAI background removal fails for any reason:
- Individual frames fall back to keeping the original image with full opacity
- Processing continues with remaining frames
- The reconstruction will still complete

## Debugging

Enable detailed output to see processing progress:
```bash
python video_orbit_voxel_recon.py video.mp4 0 --openai-bg-removal 2>&1 | tee recon.log
```

The system will print:
- Number of frames being processed
- Progress updates every 10 frames
- Any errors with specific frame indices

## Comparison

### Traditional Background Subtraction
- **Speed**: Fast (seconds)
- **Cost**: Free
- **Quality**: Good for uniform backgrounds
- **Limitations**: Struggles with complex scenes, leaves artifacts
- **Geometry**: Preserves original

### OpenAI Background Removal
- **Speed**: Moderate (1-2 minutes for 82 frames)
- **Cost**: ~$0.01/frame
- **Quality**: Excellent, AI-powered mask generation
- **Geometry**: **Preserves original** (critical for 3D reconstruction!)
- **Limitations**: Requires API key, minimal cost, depends on API availability

## Technical Details

### Image Processing Pipeline

1. **Frame Extraction**: Frames are extracted from video at native resolution
2. **AI Analysis**: GPT-4o Vision analyzes each frame to identify:
   - Main object location and boundaries
   - Color ranges to distinguish object from background
   - Isolated white "islands" that should be removed
   - Optimal masking parameters
3. **Mask Generation**: An improved alpha mask is created using:
   - AI-guided thresholding
   - Morphological operations to remove noise
   - Connected component analysis to remove isolated regions
   - Edge smoothing for natural transitions
4. **Original Preservation**: RGB channels from the original image are kept intact
5. **RGBA Output**: Original RGB + AI-generated alpha = final RGBA frame
6. **Resize**: Images are resized to training resolution preserving aspect ratio

### Parallel Processing

- Default: 4 parallel workers
- Configurable in code (see `remove_backgrounds_parallel` function)
- More workers = faster but may hit API rate limits

### API Models Used

- **Vision Only**: `gpt-4o` with high-detail image analysis
- **No Image Generation**: Masks are computed locally based on AI guidance
- **Cost-Effective**: Only pays for vision analysis, not image generation

## Troubleshooting

### "module 'PIL.Image' has no attribute 'Resampling'"
Your Pillow version is too old. Upgrade it:
```bash
pip install --upgrade "pillow>=9.1.0"
```
The code has a fallback for older versions, but this error shouldn't occur with the latest version.

### "OpenAI not available"
Install the package:
```bash
pip install openai
```

### "Authentication Error"
Check your API key:
```bash
echo $OPENAI_API_KEY
```

### "Rate Limit Error"
Reduce parallel workers or add delays between requests (requires code modification)

### "Poor Results"
Try adjusting the prompt in the `remove_background_openai` function for your specific use case

## Next Steps

After reconstruction with OpenAI background removal:
1. View results: `python voxel_volume_viewer.py video_voxel_out/recon_volume.npz`
2. Filter artifacts: `python filter_white_noise.py video_voxel_out/recon_volume.npz`
3. Analyze components: `python analyze_connected_components.py video_voxel_out/recon_volume.npz`

## References

- OpenAI API Documentation: https://platform.openai.com/docs
- DALL-E 3: https://platform.openai.com/docs/guides/images
- GPT-4 Vision: https://platform.openai.com/docs/guides/vision

