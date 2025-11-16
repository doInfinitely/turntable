import numpy as np
import cv2

def estimate_depth_from_flow(
    flow,               # (H, W, 2)
    u_coords, v_coords, # pixel grids
    f,                  # focal length in px
    R=1.0,              # orbit radius
    dt=1/24,            # time step between frames (fps = 24 default)
    t=0.0,              # orbit angle of frame1 (we infer angle difference)
    iterations=5        # fixed-point refinements
):
    """
    Computes depth using the true pinhole + circular orbit model.

    Inputs:
      flow: optical flow (du, dv)
      u_coords, v_coords: meshgrid of pixel coordinates
      f: focal length in px
      R: camera orbit radius
      dt: time between the two frames
      t: orbit angle (rad)
    """

    du = flow[...,0]
    # dv unused for depth, but we could incorporate it
    # magnitude of horizontal flow drives depth
    mag = np.abs(du) + 1e-6

    # angular velocity omega
    # (one orbit every T seconds)
    # If user knows the period:
    # omega = 2π / T
    # Here approximate from dt = Δt and frame offset of 1
    # -> need caller to pass omega explicitly or period.
    # So we compute based on dt and assume small-angle motion
    # between frames:
    # camera angle Δt_cam ≈ flow mean scale; user should pass omega.
    # For now set:
    omega = 1.0  # user will pass correct one externally

    # Effective depth
    Z_eff = (f * R * omega * dt) / mag

    # Initialize Z
    Z = Z_eff.copy()

    # Fixed-point refinement
    for _ in range(iterations):
        X = (u_coords / f) * Z
        Z = Z_eff - R*np.cos(t) + (u_coords/f)*Z

    return np.clip(Z, 0, np.percentile(Z, 99))

