import matplotlib
matplotlib.use("Agg")
import os
import time
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt

os.makedirs("output", exist_ok=True)
sample = cv2.imread("images/sample.jpg")


def show(ax, im, title=""):
    ax.imshow(im if im.ndim == 2 else cv2.cvtColor(im, cv2.COLOR_BGR2RGB), cmap="gray")
    ax.set_title(title)
    ax.axis("off")


def poly_area(pts):
    pts = np.asarray(pts, float)
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def order_points(pts):
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.float32([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]])


def find_quad(image):
    g = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cs, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cs, key=cv2.contourArea)
    approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
    if len(approx) != 4:
        approx = cv2.boxPoints(cv2.minAreaRect(c))
    return order_points(approx)


def quad_size(q):
    tl, tr, br, bl = q
    w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    return w, h


view_paths = sorted(glob.glob("images/colmap_input/*.jpg"))
fig, axes = plt.subplots(2, 4, figsize=(18, 7))
for ax, path in zip(axes.ravel(), view_paths[::3]):
    show(ax, cv2.imread(path), os.path.basename(path))
plt.tight_layout()
plt.savefig("output/q4_input_images.png", dpi=110)
plt.close()
print("Input images:", len(view_paths))


import shutil
import pycolmap

pycolmap.logging.minloglevel = 2
image_dir = "images/colmap_input"
workspace = "output/colmap_workspace"
if os.path.exists(workspace):
    shutil.rmtree(workspace)
os.makedirs(workspace + "/sparse")
database = workspace + "/database.db"

t0 = time.time()
pycolmap.extract_features(database, image_dir, camera_mode=pycolmap.CameraMode.SINGLE)
print(f"Feature detection done in {time.time() - t0:.0f}s")
t1 = time.time()
pycolmap.match_exhaustive(database)
print(f"Feature matching done in {time.time() - t1:.0f}s")
t2 = time.time()
maps = pycolmap.incremental_mapping(database, image_dir, workspace + "/sparse")
rec = max(maps.values(), key=lambda m: m.num_reg_images())
print(f"Camera pose estimation done in {time.time() - t2:.0f}s, reconstructions: {len(maps)}")
print(rec.summary())

cam = next(iter(rec.cameras.values()))
print("Estimated camera:", cam.model.name, np.round(cam.params, 2), "| known focal 800, centre (480, 360)")
rec.write(workspace + "/sparse")
rec.export_PLY("output/sparse_pointcloud_colmap.ply")


by_name = {im.name: im for im in rec.images.values()}
name_a, name_b = "view_01.jpg", "view_02.jpg"
im_a, im_b = by_name[name_a], by_name[name_b]
img_a = cv2.imread(os.path.join(image_dir, name_a))
img_b = cv2.imread(os.path.join(image_dir, name_b))

keypoints = np.array([pt.xy for pt in im_a.points2D])
pairs = []
for p3 in rec.points3D.values():
    seen = {e.image_id: e.point2D_idx for e in p3.track.elements}
    if im_a.image_id in seen and im_b.image_id in seen:
        pairs.append((im_a.points2D[seen[im_a.image_id]].xy, im_b.points2D[seen[im_b.image_id]].xy))

canvas = np.hstack([img_a, img_b])
off = img_a.shape[1]
for x, y in keypoints[::3]:
    cv2.circle(canvas, (int(x), int(y)), 2, (0, 255, 0), -1)
for (xa, ya), (xb, yb) in pairs[::6]:
    cv2.line(canvas, (int(xa), int(ya)), (int(xb) + off, int(yb)), (0, 200, 255), 1)

plt.figure(figsize=(18, 7))
plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title(f"{name_a} and {name_b}: {len(keypoints)} keypoints, {len(pairs)} verified matches")
plt.tight_layout()
plt.savefig("output/q4_features_matches.png", dpi=110)
plt.close()
print(len(keypoints), "keypoints detected in", name_a, "and", len(pairs), "verified matches with", name_b)


def gt_centre(i):
    th = 2 * np.pi * i / 24
    rad = 5.0 + 0.4 * np.sin(3 * th)
    return np.array([rad * np.cos(th), rad * np.sin(th), 2.0 + 1.3 * (i % 2)])


def fit_similarity(src, dst):
    ms, md = src.mean(axis=0), dst.mean(axis=0)
    A, B = src - ms, dst - md
    U, S, Vt = np.linalg.svd(B.T @ A)
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / np.sum(A ** 2)
    t = md - s * R @ ms
    return s, R, t


est_c, gt_c = [], []
for im in rec.images.values():
    idx = int(im.name.split("_")[1].split(".")[0]) - 1
    est_c.append(im.projection_center())
    gt_c.append(gt_centre(idx))
est_c, gt_c = np.array(est_c), np.array(gt_c)
s_fit, R_fit, t_fit = fit_similarity(est_c, gt_c)
est_aligned = (s_fit * (R_fit @ est_c.T)).T + t_fit
pose_err = np.linalg.norm(est_aligned - gt_c, axis=1)
print(f"Registered images: {rec.num_reg_images()}/24")
print(f"Camera centre error after similarity alignment: mean {pose_err.mean() * 100:.2f} cm, max {pose_err.max() * 100:.2f} cm (orbit radius about 5 units)")

fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(*gt_c.T, s=60, facecolors="none", edgecolors="tab:green", label="Known camera positions")
ax.scatter(*est_aligned.T, s=25, marker="^", color="tab:red", label="COLMAP estimated poses")
ax.set_title("Camera poses")
ax.legend()
plt.tight_layout()
plt.savefig("output/q4_camera_poses.png", dpi=120)
plt.close()


import open3d as o3d

pts_raw = np.array([p.xyz for p in rec.points3D.values()])
col_raw = np.array([p.color for p in rec.points3D.values()]) / 255.0
pts_world = (s_fit * (R_fit @ pts_raw.T)).T + t_fit

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts_world)
pcd.colors = o3d.utility.Vector3dVector(col_raw)
n_before = len(pcd.points)
pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
pcd, _ = pcd.remove_radius_outlier(nb_points=6, radius=0.35)
o3d.io.write_point_cloud("output/pointcloud.ply", pcd)
print(f"Point cloud: {n_before} points -> {len(pcd.points)} after outlier removal")

boxes = [((-0.8, 0.8), (-0.6, 0.6), 1.4), ((0.9, 1.5), (-1.1, -0.5), 0.8)]


def box_distance(P, xr, yr, h):
    lo = np.array([xr[0], yr[0], 0.0])
    hi = np.array([xr[1], yr[1], h])
    outside = np.linalg.norm(np.maximum(np.maximum(lo - P, P - hi), 0), axis=1)
    inside = np.min(np.minimum(P - lo, hi - P), axis=1)
    return np.where(outside > 0, outside, inside)


P = np.asarray(pcd.points)
d_true = np.minimum(np.abs(P[:, 2]), np.min([box_distance(P, *b) for b in boxes], axis=0))
print(f"Distance to the true surfaces: median {np.median(d_true) * 100:.2f} cm, 90th percentile {np.percentile(d_true, 90) * 100:.2f} cm, within 5 cm: {100 * np.mean(d_true < 0.05):.1f}%")

fig = plt.figure(figsize=(16, 7))
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.scatter(*P.T, c=np.asarray(pcd.colors), s=2)
ax1.set_title("Reconstructed point cloud")
ax2 = fig.add_subplot(1, 2, 2)
ax2.hist(d_true * 100, bins=60, color="tab:blue")
ax2.set_xlabel("distance to true surface (cm)")
ax2.set_ylabel("points")
ax2.set_title("Reconstruction accuracy")
plt.tight_layout()
plt.savefig("output/q4_pointcloud.png", dpi=120)
plt.close()


pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.4, max_nn=40))
pcd.orient_normals_consistent_tangent_plane(30)
nrm = np.asarray(pcd.normals)
on_ground = np.abs(np.asarray(pcd.points)[:, 2]) < 0.05
if nrm[on_ground, 2].mean() < 0:
    pcd.normals = o3d.utility.Vector3dVector(-nrm)

mesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
density = np.asarray(density)
mesh.remove_vertices_by_mask(density < np.quantile(density, 0.06))
mesh = mesh.crop(pcd.get_axis_aligned_bounding_box())
mesh.remove_degenerate_triangles()
mesh.remove_unreferenced_vertices()
mesh.compute_vertex_normals()
o3d.io.write_triangle_mesh("output/mesh_poisson.ply", mesh)
o3d.io.write_triangle_mesh("output/mesh_poisson.obj", mesh)
print(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles, watertight: {mesh.is_watertight()}")

verts = np.asarray(mesh.vertices)
tris = np.asarray(mesh.triangles)
fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=tris, cmap="viridis", linewidth=0.05, edgecolor="gray")
ax.set_title("Poisson surface mesh")
fig.colorbar(surf, ax=ax, shrink=0.5, label="height")
plt.tight_layout()
plt.savefig("output/q4_mesh.png", dpi=120)
plt.close()
