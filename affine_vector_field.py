# affine_vector_field.py
import cv2
import numpy as np
import math


def _candidate_affines(dx_range, dy_range, scales, angles, shears):
    """
    Build a list of candidate (dx,dy,s,ang,sh,M) affine transforms.
    M is 2x3.
    """
    affines = []
    for dx in dx_range:
        for dy in dy_range:
            for s in scales:
                for ang in angles:
                    for sh in shears:
                        theta = math.radians(ang)

                        a = s * math.cos(theta)
                        d = s * math.sin(theta)
                        b = -s * math.sin(theta) + sh
                        e =  s * math.cos(theta) + sh
                        c = dx
                        f = dy

                        M = np.array([[a, b, c],
                                      [d, e, f]], dtype=np.float32)
                        affines.append((dx, dy, s, ang, sh, M))
    return affines


def _best_affine_for_window(
    win,
    img_next,
    x,
    y,
    affines,
    reg_lambda=1e-4,
    tie_eps=1e-9,
):
    """
    For a given window in img1 (win), search in img2 with candidate affines.

    Cost = MSE + reg_lambda * ||(dx,dy)||^2.
    If costs are (near) ties, prefer the smaller ||(dx,dy)|| so
    whitespace tends to get a null vector.
    """
    h, w = win.shape[:2]
    H2, W2 = img_next.shape[:2]

    best_cost = float("inf")
    best_err = float("inf")
    best_params = None
    best_norm2 = float("inf")

    for (dx, dy, s, ang, sh, M) in affines:
        patch = cv2.warpAffine(
            img_next,
            M,
            (W2, H2),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

        patch_win = patch[y:y + h, x:x + w]
        if patch_win.shape != win.shape:
            continue

        diff = patch_win.astype(np.float32) - win.astype(np.float32)
        mse = np.mean(diff * diff)

        norm2 = dx * dx + dy * dy
        cost = mse + reg_lambda * norm2

        if cost < best_cost - tie_eps:
            best_cost = cost
            best_err = mse
            best_params = (dx, dy, s, ang, sh, M)
            best_norm2 = norm2
        elif abs(cost - best_cost) <= tie_eps:
            # tie: prefer shorter motion vector
            if norm2 < best_norm2:
                best_cost = cost
                best_err = mse
                best_params = (dx, dy, s, ang, sh, M)
                best_norm2 = norm2

    return best_err, best_params


def dense_motion_field(
    img1,
    img2,
    window=32,
    stride=16,
    search=16,
    scales=None,
    angles=None,
    shears=None,
    reg_lambda=1e-4,
):
    """
    Compute a coarse dense affine-based motion field between img1 and img2.

    img1, img2: grayscale uint8 arrays (H,W)
    window: window size
    stride: step between window centers
    search: +/- search range in pixels for dx, dy
    scales: list of scales to try
    angles: list of rotation angles (degrees)
    shears: list of shear values
    reg_lambda: weight for ||(dx,dy)||^2 regularization.

    Returns:
      flow:   (Hc, Wc, 2)  with (dx, dy) per window
      errors: (Hc, Wc)     best match MSE per window
      affmats: Hc x Wc list of (dx,dy,s,ang,sh,M) or None
    """
    if scales is None:
        scales = [1.0]
    if angles is None:
        angles = [0.0]
    if shears is None:
        shears = [0.0]

    H, W = img1.shape[:2]

    dx_range = range(-search, search + 1, 4)
    dy_range = range(-search, search + 1, 4)
    affines = _candidate_affines(dx_range, dy_range, scales, angles, shears)

    out_h = (H - window) // stride + 1
    out_w = (W - window) // stride + 1

    flow = np.zeros((out_h, out_w, 2), dtype=np.float32)
    errors = np.zeros((out_h, out_w), dtype=np.float32)
    affmats = [[None for _ in range(out_w)] for _ in range(out_h)]

    for i, y in enumerate(range(0, H - window + 1, stride)):
        for j, x in enumerate(range(0, W - window + 1, stride)):
            win = img1[y:y + window, x:x + window]

            best_err, params = _best_affine_for_window(
                win,
                img2,
                x,
                y,
                affines,
                reg_lambda=reg_lambda,
            )

            if params is None:
                flow[i, j] = (0.0, 0.0)
            else:
                dx, dy, s, ang, sh, M = params
                flow[i, j] = (dx, dy)

            errors[i, j] = best_err
            affmats[i][j] = params

    return flow, errors, affmats

