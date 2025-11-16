import sys
import math
import pygame
import numpy as np


# ---------- PLY LOADING ----------

def load_ply_points(path):
    """
    Load an ASCII PLY file with vertex x,y,z and (optionally) r,g,b,a.
    Returns:
      positions: (N,3) float32 in normalized coords
      colors:    (N,3) uint8  in [0,255]
    """
    with open(path, "r") as f:
        lines = f.readlines()

    # Parse header
    header_ended = False
    vertex_count = None
    properties = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        i += 1

        if line.startswith("element vertex"):
            parts = line.split()
            vertex_count = int(parts[-1])
        elif line.startswith("property"):
            parts = line.split()
            if len(parts) >= 3:
                properties.append(parts[-1])
        elif line == "end_header":
            header_ended = True
            break

    if not header_ended or vertex_count is None:
        raise RuntimeError("Invalid PLY header in %s" % path)

    def prop_idx(name):
        return properties.index(name) if name in properties else None

    idx_x = prop_idx("x")
    idx_y = prop_idx("y")
    idx_z = prop_idx("z")
    idx_r = prop_idx("red")
    idx_g = prop_idx("green")
    idx_b = prop_idx("blue")
    idx_a = prop_idx("alpha")

    if idx_x is None or idx_y is None or idx_z is None:
        raise RuntimeError("PLY must have x,y,z vertex properties")

    positions = []
    colors = []

    for v in range(vertex_count):
        if i >= len(lines):
            break
        parts = lines[i].strip().split()
        i += 1
        if len(parts) < 3:
            continue

        x = float(parts[idx_x])
        y = float(parts[idx_y])
        z = float(parts[idx_z])
        positions.append((x, y, z))

        # Default white
        r = g = b = 255
        if idx_r is not None and idx_g is not None and idx_b is not None:
            r = int(float(parts[idx_r]))
            g = int(float(parts[idx_g]))
            b = int(float(parts[idx_b]))

        # Optional alpha: treat near-zero alpha as invisible-black
        if idx_a is not None:
            a = int(float(parts[idx_a]))
            if a < 10:
                r = g = b = 0

        colors.append((r, g, b))

    positions = np.array(positions, dtype=np.float32)
    colors = np.array(colors, dtype=np.uint8)

    # Normalize to roughly [-1,1]^3 (helps with arbitrary scales)
    max_abs = np.max(np.abs(positions))
    if max_abs > 0:
        positions = positions / max_abs

    return positions, colors


# ---------- CAMERA & PROJECTION ----------

def look_at(eye, target, up=np.array([0, 1, 0], dtype=np.float32)):
    """
    Build a right-handed view transform with +Z pointing *forward*
    (toward target). This matches the z>0 visibility test in projection.
    Returns R (3x3), t (3,)
    """
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)

    forward = target - eye
    forward /= np.linalg.norm(forward) + 1e-8

    right = np.cross(forward, up)
    right /= np.linalg.norm(right) + 1e-8

    true_up = np.cross(right, forward)

    # world -> camera: rows are camera basis vectors
    # x_cam = right·(X - eye), y_cam = true_up·(X - eye), z_cam = forward·(X - eye)
    R = np.stack([right, true_up, forward], axis=0)
    t = -R @ eye
    return R, t


def project_points(positions, colors, eye, target, fov_deg, img_w, img_h):
    """
    Project 3D points with a pinhole camera.
    Returns xs, ys (pixel coords), cols, depths.
    """
    R, t = look_at(eye, target)
    pts_cam = (R @ positions.T).T + t  # (N,3)

    z = pts_cam[:, 2]
    valid = z > 0.05          # points in front of camera
    pts_cam = pts_cam[valid]
    cols = colors[valid]
    z = z[valid]

    if pts_cam.shape[0] == 0:
        return np.array([]), np.array([]), np.zeros((0, 3), dtype=np.uint8), np.array([])

    # Intrinsics from vertical FOV
    fov_rad = math.radians(fov_deg)
    fy = 0.5 * img_h / math.tan(0.5 * fov_rad)
    fx = fy
    cx = img_w / 2.0
    cy = img_h / 2.0

    x_cam = pts_cam[:, 0]
    y_cam = pts_cam[:, 1]

    xs = fx * (x_cam / z) + cx
    ys = fy * (y_cam / z) + cy

    return xs, ys, cols, z


# ---------- MAIN VIEWER ----------

def main(ply_path):
    positions, colors = load_ply_points(ply_path)
    print(f"Loaded {positions.shape[0]} points from {ply_path}")

    pygame.init()
    width, height = 800, 800
    screen = pygame.display.set_mode((width, height))
    pygame.display.setcaption = pygame.display.set_caption("PLY Voxel Viewer")

    clock = pygame.time.Clock()

    # Camera orbit params
    radius = 3.0
    angle_y = 0.0  # yaw (left-right)
    angle_x = 0.0  # pitch (up-down)
    fov_deg = 45.0
    zoom_speed = 0.1
    orbit_speed = 0.03

    running = True
    while running:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()

        # Orbit controls
        if keys[pygame.K_LEFT]:
            angle_y -= orbit_speed
        if keys[pygame.K_RIGHT]:
            angle_y += orbit_speed
        if keys[pygame.K_UP]:
            angle_x = max(angle_x - orbit_speed, -math.pi / 2 + 0.1)
        if keys[pygame.K_DOWN]:
            angle_x = min(angle_x + orbit_speed,  math.pi / 2 - 0.1)

        # Zoom
        if keys[pygame.K_EQUALS] or keys[pygame.K_PLUS]:
            radius = max(0.5, radius - zoom_speed)
        if keys[pygame.K_MINUS]:
            radius += zoom_speed

        # Camera position from spherical coords
        cx = radius * math.cos(angle_x) * math.cos(angle_y)
        cy = radius * math.sin(angle_x)
        cz = radius * math.cos(angle_x) * math.sin(angle_y)
        eye = np.array([cx, cy, cz], dtype=np.float32)
        target = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        xs, ys, cols, depth = project_points(
            positions, colors, eye, target, fov_deg, width, height
        )

        screen.fill((0, 0, 0))

        if xs.size > 0:
            # Back-to-front sorting (so nearer overwrites farther)
            order = np.argsort(depth)  # far → near
            xs = xs[order]
            ys = ys[order]
            cols = cols[order]

            # Thin out if there are too many points
            max_points = 80000
            if xs.shape[0] > max_points:
                step = xs.shape[0] // max_points
                xs = xs[::step]
                ys = ys[::step]
                cols = cols[::step]

            for x, y, c in zip(xs, ys, cols):
                ix = int(x)
                iy = int(y)
                if 0 <= ix < width and 0 <= iy < height:
                    screen.set_at((ix, iy), (int(c[0]), int(c[1]), int(c[2])))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ply_voxel_viewer.py <path_to_ply>")
        sys.exit(1)
    main(sys.argv[1])

