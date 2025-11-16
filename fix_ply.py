from plyfile import PlyData, PlyElement
import numpy as np

input_path = "recon/sparse/0/points3D_colmap.ply"
output_path = "recon/sparse/0/points3D_blender.ply"

# 1) Read COLMAP's binary PLY
ply = PlyData.read(input_path)
verts = ply["vertex"]

# COLMAP usually names these properties:
# x, y, z, red, green, blue, error, track_length
# We'll keep only x, y, z, red, green, blue.
x = np.asarray(verts["x"], dtype=np.float32)
y = np.asarray(verts["y"], dtype=np.float32)
z = np.asarray(verts["z"], dtype=np.float32)

# Try to get color if present; otherwise default to white
if all(name in verts.data.dtype.names for name in ("red", "green", "blue")):
    r = np.asarray(verts["red"], dtype=np.uint8)
    g = np.asarray(verts["green"], dtype=np.uint8)
    b = np.asarray(verts["blue"], dtype=np.uint8)
else:
    r = np.full_like(x, 255, dtype=np.uint8)
    g = np.full_like(x, 255, dtype=np.uint8)
    b = np.full_like(x, 255, dtype=np.uint8)

# 2) Build a new clean vertex array: x,y,z,red,green,blue
vertex_array = np.empty(len(x), dtype=[
    ("x", "f4"),
    ("y", "f4"),
    ("z", "f4"),
    ("red", "u1"),
    ("green", "u1"),
    ("blue", "u1"),
])
vertex_array["x"] = x
vertex_array["y"] = y
vertex_array["z"] = z
vertex_array["red"] = r
vertex_array["green"] = g
vertex_array["blue"] = b

vertex_element = PlyElement.describe(vertex_array, "vertex")

# 3) Write as ASCII PLY (Blender-friendly)
clean_ply = PlyData([vertex_element], text=True)
clean_ply.write(output_path)

print("Wrote Blender-safe PLY:", output_path) 
