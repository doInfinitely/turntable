"""Monocular depth estimation via DepthAnything (HuggingFace).

Returns a relative depth map calibrated to the same convention used
elsewhere in the pipeline: 1.0 = scene centre (camera distance),
< 1.0 = closer to camera, > 1.0 = farther.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch


def _pipeline_device(device: str) -> int:
    if device == "cuda":
        return 0
    if device.startswith("cuda:"):
        return int(device.split(":", 1)[1])
    return -1


class DepthAnythingEstimator:
    """Wrap a HuggingFace DepthAnything model.

    DepthAnything outputs *inverse* depth (larger value = closer).
    We invert and rescale so the result matches our pipeline's
    "1.0 = reference distance" convention.
    """

    def __init__(
        self,
        model_id: str = "depth-anything/Depth-Anything-V2-Small-hf",
        device: str | None = None,
    ):
        from transformers import pipeline
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.pipe = pipeline(
            task="depth-estimation",
            model=model_id,
            device=_pipeline_device(device),
        )

    def estimate_inv(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Raw inverse-depth from DA (no 1/x, no normalization).

        Larger = closer. Background / sky pixels are near zero. Returned at
        the same resolution as the input frame.
        """
        from PIL import Image
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        result = self.pipe(pil_img)
        inv_depth = result["predicted_depth"]
        if isinstance(inv_depth, torch.Tensor):
            inv_depth = inv_depth.detach().cpu().numpy()
        if inv_depth.ndim == 3:
            inv_depth = inv_depth[0]

        H, W = frame_bgr.shape[:2]
        if inv_depth.shape != (H, W):
            inv_depth = cv2.resize(inv_depth, (W, H),
                                   interpolation=cv2.INTER_LINEAR)
        return inv_depth.astype(np.float32)

    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Estimate depth for a single frame.

        Args:
            frame_bgr: BGR uint8 frame ``(H, W, 3)``.

        Returns:
            depth: ``(H, W)`` float32, in pipeline units (1.0 = scene
            centre, smaller = closer, larger = farther).
        """
        inv_depth = self.estimate_inv(frame_bgr)

        # Invert: depth ∝ 1 / inv_depth.  Add small epsilon for safety.
        depth = 1.0 / (inv_depth + 1e-6)

        # Normalise so the median pixel maps to 1.0 (scene centre).
        # This anchors the relative depth to our pipeline convention.
        median = np.median(depth)
        if median > 0:
            depth = depth / median

        return depth.astype(np.float32)
