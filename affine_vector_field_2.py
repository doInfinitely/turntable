import cv2
import numpy as np
import math


# ------------------------------
# Generate candidate affine transforms
# ------------------------------

def candidate_affines(dx_range, dy_range, scales, angles, shears):
    """
    Returns list of 2x3 affine matrices.
    """
    affines = []
    for dx in dx_range:
        for dy in dy_range:
            for s in scales:
                for ang in angles:
                    for sh in shears:
                        theta = math.radians(ang)

                        # Affine matrix components
                        a = s * math.cos(theta)
                        b = -s * math.sin(theta) + sh
                        c = dx
                        d = s * math.sin(theta)
                        e =  s * math.cos(theta) + sh
                        f = dy

                        M = np.array([[a, b, c],
                                      [d, e, f]], dtype=np.float32)

                        affines.append((dx, dy, s, ang, sh, M))
    return affines


# ------------------------------
# Compute best affine per window
# ------------------------------

def best_affine_for_window(win, img_next, x, y, affines):
    h, w = win.shape[:2]
    best_err = float("inf")
    best_params = None

    for (dx, dy, s, ang, sh, M) in affines:
        # Apply affine transform to image_next, sample only the window region
        patch = cv2.warpAffine(img_next, M, (img_next.shape[1], img_next.shape[0]),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)

        patch_win = patch[y:y+h, x:x+w]

        if patch_win.shape != win.shape:
            continue

        err = np.mean((patch_win.astype(float) - win.astype(float)) ** 2)

        if err < best_err:
            best_err = err
            best_params = (dx, dy, s, ang, sh, M)

    return best_err, best_params


# ------------------------------
# Dense vector field computation
# ------------------------------

def compute_affine_field(img1, img2,
                         window=32, stride=16,
                         search=16,
                         scales=[1.0],
                         angles=[-5, 0, 5],
                         shears=[0]):
    """
    Returns:
      flow: (H/stride, W/stride, 2) dx,dy
      affine_params: list of local affine matrices
      error_map: matching error
    """

    # Precompute affine candidates
    dx_range = range(-search, search+1, 4)
    dy_range = range(-search, search+1, 4)
    affines = candidate_affines(dx_range, dy_range, scales, angles, shears)

    H, W = img1.shape[:2]

    out_h = (H - window) // stride + 1
    out_w = (W - window) // stride + 1

    flow   = np.zeros((out_h, out_w, 2), dtype=np.float32)
    errors = np.zeros((out_h, out_w),    dtype=np.float32)
    affmats = [[None for _ in range(out_w)] for _ in range(out_h)]

    for i, y in enumerate(range(0, H-window+1, stride)):
        for j, x in enumerate(range(0, W-window+1, stride)):
            win = img1[y:y+window, x:x+window]

            best_err, params = best_affine_for_window(win, img2, x, y, affines)

            if params is None:
                flow[i, j] = (0, 0)
            else:
                dx, dy, s, ang, sh, M = params
                flow[i, j] = (dx, dy)

            errors[i, j] = best_err
            affmats[i][j] = params

    return flow, errors, affmats

img1 = cv2.imread("frame_000.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("frame_001.png", cv2.IMREAD_GRAYSCALE)

flow, errors, mats = compute_affine_field(img1, img2)

# Visualize motion vectors
vis = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
step = 10
for i in range(0, flow.shape[0], 2):
    for j in range(0, flow.shape[1], 2):
        dx, dy = flow[i, j]
        y = i*16 + 16
        x = j*16 + 16
        cv2.arrowedLine(vis, (x, y), (int(x+dx), int(y+dy)), (0,255,0), 1)

cv2.imshow("flow", vis)
cv2.waitKey(0)
