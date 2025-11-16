#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Optional: Poisson mesh post-processing
try:
    import open3d as o3d
except Exception:
    o3d = None


def run(cmd, cwd=None):
    print(">>", " ".join(str(c) for c in cmd))
    p = subprocess.run(cmd, cwd=cwd)
    if p.returncode != 0:
        print(f"Command failed with code {p.returncode}")
        sys.exit(p.returncode)


def require_in_path(exe: str):
    from shutil import which
    if which(exe) is None:
        print(f"Error: `{exe}` not found in PATH. Please install it or add it to PATH.")
        sys.exit(1)


def extract_frames(video: Path, img_dir: Path, fps: int):
    img_dir.mkdir(parents=True, exist_ok=True)
    # Extract frames as 000001.jpg, 000002.jpg, ...
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video),
        "-vf", f"fps={fps}",
        "-q:v", "2",
        str(img_dir / "%06d.jpg"),
    ]
    run(cmd)


def colmap_feature_match(db: Path, img_dir: Path, sequential: bool = True):
    # 1) Feature extraction (CPU-only, no GPU flags)
    run([
        "colmap", "feature_extractor",
        "--database_path", str(db),
        "--image_path", str(img_dir),
        "--ImageReader.single_camera", "1",
        "--ImageReader.default_focal_length_factor", "1.2",
    ])

    # 2) Matching
    if sequential:
        run([
            "colmap", "sequential_matcher",
            "--database_path", str(db),
        ])
    else:
        run([
            "colmap", "exhaustive_matcher",
            "--database_path", str(db),
        ])


def colmap_sparse_reconstruction(
    db: Path,
    img_dir: Path,
    sparse_dir: Path,
    dense_dir: Path,
):
    sparse_dir.mkdir(parents=True, exist_ok=True)
    dense_dir.mkdir(parents=True, exist_ok=True)

    # 3) Sparse SfM (mapper) — use only core flags that exist on all builds
    run([
        "colmap", "mapper",
        "--database_path", str(db),
        "--image_path", str(img_dir),
        "--output_path", str(sparse_dir),
    ])

    # Find the largest model directory inside sparse_dir
    candidates = [p for p in sparse_dir.iterdir() if p.is_dir()]
    if not candidates:
        print("No sparse model produced. Check input frames / matches.")
        sys.exit(1)

    model_dir = max(
        candidates,
        key=lambda p: sum(1 for _ in p.iterdir()),
    )
    print("Using sparse model:", model_dir)

    # 4) Undistort images for dense reconstruction
    run([
        "colmap", "image_undistorter",
        "--image_path", str(img_dir),
        "--input_path", str(model_dir),
        "--output_path", str(dense_dir),
        "--output_type", "COLMAP",
    ])


def colmap_dense_reconstruction(workspace_dir: Path, fused_ply_out: Path):
    # 5) PatchMatch stereo
    run([
        "colmap", "patch_match_stereo",
        "--workspace_path", str(workspace_dir),
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency", "1",
    ])

    # 6) Stereo fusion → fused point cloud
    run([
        "colmap", "stereo_fusion",
        "--workspace_path", str(workspace_dir),
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",
        "--output_path", str(fused_ply_out),
    ])


def poisson_mesh_from_pointcloud(
    ply_in: Path,
    ply_out: Path,
    depth: int = 9,
    simplify_to: int = 300000,
):
    if o3d is None:
        print("Open3D not installed; skipping mesh.")
        return

    print(f"Meshing {ply_in} -> {ply_out} (Poisson depth={depth})")
    pcd = o3d.io.read_point_cloud(str(ply_in))
    if not pcd.has_normals():
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(20)

    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)

    # Crop to the bounding box of the original points
    bbox = pcd.get_axis_aligned_bounding_box()
    mesh = mesh.crop(bbox)

    # Simplify mesh if too dense
    if simplify_to and len(mesh.triangles) > simplify_to:
        mesh = mesh.simplify_quadric_decimation(simplify_to)

    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(ply_out), mesh)
    print("Wrote mesh:", ply_out)


def main():
    ap = argparse.ArgumentParser(
        description="Reconstruct 3D from a single orbit video using COLMAP and export PLY."
    )
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--fps", type=int, default=4, help="Frame extraction rate (frames/sec)")
    ap.add_argument("--no_gpu", action="store_true",
                    help="(Ignored in this CPU-only script, kept for compatibility)")
    ap.add_argument("--exhaustive", action="store_true",
                    help="Use exhaustive matcher instead of sequential")
    ap.add_argument("--min_track_len", type=int, default=5,
                    help="(Ignored; kept only for CLI compatibility)")
    ap.add_argument("--mesh", action="store_true",
                    help="Also export a Poisson mesh .ply via Open3D")
    args = ap.parse_args()

    require_in_path("ffmpeg")
    require_in_path("colmap")

    out_root = args.out
    imgs = out_root / "images"
    db = out_root / "database.db"
    sparse = out_root / "sparse"
    dense = out_root / "dense"

    # Prepare / clean
    out_root.mkdir(parents=True, exist_ok=True)
    if imgs.exists():
        shutil.rmtree(imgs)
    if db.exists():
        db.unlink()
    if sparse.exists():
        shutil.rmtree(sparse)
    if dense.exists():
        shutil.rmtree(dense)

    # 1) Extract frames from video
    extract_frames(args.video, imgs, args.fps)

    # 2) Features + matching
    colmap_feature_match(
        db,
        imgs,
        sequential=not args.exhaustive,
    )

    # 3) Sparse reconstruction + undistortion
    colmap_sparse_reconstruction(
        db,
        imgs,
        sparse,
        dense,
    )

    # 4) Dense reconstruction → fused point cloud
    fused_ply = dense / "fused.ply"
    colmap_dense_reconstruction(dense, fused_ply)
    print("Dense point cloud:", fused_ply)

    # 5) Optional: Poisson mesh
    if args.mesh:
        mesh_out = out_root / "mesh_poisson.ply"
        poisson_mesh_from_pointcloud(fused_ply, mesh_out)

    print("\nDone.")
    print("Primary output point cloud:", fused_ply)
    if args.mesh:
        print("Mesh output:", mesh_out)


if __name__ == "__main__":
    main()

