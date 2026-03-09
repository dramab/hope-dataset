#!/usr/bin/env python3
"""
3D Object Placement Planner (HOPE-Video dataset)

For **every** object in a scene, compute valid collision-free positions on the
main horizontal support surface (table / floor).  Placement candidates are
filtered for camera visibility, clustered with DBSCAN, and the most
representative position per cluster is kept.

Algorithm per object
  1. Visibility check – all 8 OBB corners project inside the image
  2. Grid preparation  – all objects' OBBs filled as OCCUPIED;
                         target OBB set to FREE ("simulated removal")
  3. FFT collision search at fixed table-Z height (XY safety margin)
  4. Camera-visibility filter on each candidate placement
  5. DBSCAN clustering → one representative per cluster (most free space)

Coordinate conventions (HOPE-Video)
  - World / object coordinates : cm,  Z-up
  - Object pose in annotation  : object-canonical → camera frame  (4×4)
  - Camera extrinsics           : world → camera  (translation in m → ×100)
  - To get object→world         : E_c2w @ pose
"""

import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, generate_binary_structure
from scipy.signal import fftconvolve
from sklearn.cluster import DBSCAN

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401

# ═══════════════════════════════════════════════════════════════════════════════
#  Adjustable parameters
# ═══════════════════════════════════════════════════════════════════════════════

SAFETY_MARGIN_CM    = 2.0
DBSCAN_EPS_CM       = 5.0
DBSCAN_MIN_SAMPLES  = 1
VIS_MARGIN_PX       = 30            # placement bbox may extend this far outside image

FREE, OCCUPIED, UNKNOWN = 0, 1, 2
WORLD_UP = np.array([0.0, 0.0, 1.0])

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_data")

BOX_EDGES = [
    (0, 1), (2, 3), (4, 5), (6, 7),
    (0, 2), (1, 3), (4, 6), (5, 7),
    (0, 4), (1, 5), (2, 6), (3, 7),
]

# Dark-theme palette (matching reference visualisation)
CLR_BG      = "#1A1A2E"
CLR_PANEL   = "#0D0D1A"
CLR_ORIG    = "#FF6D00"
CLR_PLACE   = "#00E676"
CLR_CAM     = "#FFEB3B"
CLR_ARROW   = "#FFD54F"
CLR_OCC     = "#78909C"
CLR_FREE    = "#4CAF50"


# ═══════════════════════════════════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_scene_data(data_dir: str):
    d = Path(data_dir)
    grid = np.load(d / "occupancy_grid.npy")
    with open(d / "grid_meta.json") as f:
        grid_meta = json.load(f)
    with open(d / "meta_data.json") as f:
        annotations = json.load(f)
    rgb = np.asarray(Image.open(d / "0000_rgb.jpg"), dtype=np.uint8)
    return grid, grid_meta, annotations, rgb


# ═══════════════════════════════════════════════════════════════════════════════
#  Coordinate helpers
# ═══════════════════════════════════════════════════════════════════════════════

def camera_transforms(annotations):
    E_w2c = np.array(annotations["camera"]["extrinsics"], dtype=np.float64)
    E_w2c[:3, 3] *= 100.0
    E_c2w = np.linalg.inv(E_w2c)
    return E_w2c, E_c2w


def obj_to_world(pose, E_c2w):
    return E_c2w @ np.asarray(pose, dtype=np.float64)


def transform_points(pts, T):
    h = np.hstack([np.asarray(pts, dtype=np.float64),
                    np.ones((len(pts), 1))])
    return (T @ h.T).T[:, :3]


def get_bbox_corners(bbox3d):
    mn, mx = np.array(bbox3d[:3]), np.array(bbox3d[3:])
    corners = []
    for zi in range(2):
        for yi in range(2):
            for xi in range(2):
                corners.append([[mn[0], mx[0]][xi],
                                [mn[1], mx[1]][yi],
                                [mn[2], mx[2]][zi]])
    return np.array(corners, dtype=np.float64)


def world_to_voxel(pts, vp):
    o = np.asarray(vp["origin"], dtype=np.float64)
    return np.floor((np.asarray(pts, dtype=np.float64) - o)
                    / float(vp["voxel_size"])).astype(int)


def voxel_to_world(idx, vp):
    o = np.asarray(vp["origin"], dtype=np.float64)
    return o + (np.asarray(idx, dtype=np.float64) + 0.5) * float(vp["voxel_size"])


def project_world(pts_world, K, E_w2c):
    """World → image.  Returns (u, v, z_cam) arrays."""
    pts_h   = np.hstack([np.asarray(pts_world, dtype=np.float64),
                          np.ones((len(pts_world), 1))])
    pts_cam = (E_w2c @ pts_h.T).T[:, :3]
    fx, fy  = K[0, 0], K[1, 1]
    cx, cy  = K[0, 2], K[1, 2]
    z = pts_cam[:, 2]
    safe_z = np.where(z > 0, z, np.nan)
    u = pts_cam[:, 0] / safe_z * fx + cx
    v = pts_cam[:, 1] / safe_z * fy + cy
    return u, v, z


# ═══════════════════════════════════════════════════════════════════════════════
#  OBB voxelization
# ═══════════════════════════════════════════════════════════════════════════════

def voxelize_obb(bbox3d, T_obj2world, vp, grid_shape):
    origin = np.asarray(vp["origin"], dtype=np.float64)
    vs = float(vp["voxel_size"])
    cw = transform_points(get_bbox_corners(bbox3d), T_obj2world)
    lo = np.maximum(np.floor((cw.min(0) - vs - origin) / vs).astype(int), 0)
    hi = np.minimum(np.ceil((cw.max(0) + vs - origin) / vs).astype(int),
                    np.array(grid_shape))
    ranges = [np.arange(lo[d], hi[d]) for d in range(3)]
    if any(len(r) == 0 for r in ranges):
        return np.empty((0, 3), dtype=int)
    gi, gj, gk = np.meshgrid(*ranges, indexing="ij")
    idx = np.stack([gi.ravel(), gj.ravel(), gk.ravel()], axis=1)
    centres = origin + (idx + 0.5) * vs
    co = transform_points(centres, np.linalg.inv(T_obj2world))
    bmin, bmax = np.array(bbox3d[:3]), np.array(bbox3d[3:])
    return idx[np.all((co >= bmin) & (co <= bmax), axis=1)]


def obb_corners_world(bbox3d, T_obj2world):
    """OBB corners in world coordinates."""
    return transform_points(get_bbox_corners(bbox3d), T_obj2world)


# ═══════════════════════════════════════════════════════════════════════════════
#  Visibility checks
# ═══════════════════════════════════════════════════════════════════════════════

def is_fully_visible(bbox3d, pose, fx, fy, cx, cy, img_w, img_h):
    """True if all 8 OBB corners project within image bounds (Z > 0)."""
    corners_cam = transform_points(get_bbox_corners(bbox3d),
                                   np.asarray(pose, dtype=np.float64))
    if np.any(corners_cam[:, 2] <= 0):
        return False
    uv = np.stack([fx * corners_cam[:, 0] / corners_cam[:, 2] + cx,
                   fy * corners_cam[:, 1] / corners_cam[:, 2] + cy], axis=1)
    return bool(np.all(uv[:, 0] >= 0) and np.all(uv[:, 0] < img_w) and
                np.all(uv[:, 1] >= 0) and np.all(uv[:, 1] < img_h))


def filter_visible_placements(positions_xy, landing_z, obj_voxels,
                               bbox3d, T_obj2world, E_w2c, K,
                               img_w, img_h, vp,
                               margin_px=VIS_MARGIN_PX):
    """
    Keep only placements whose placed OBB projects (mostly) inside the image.
    A tolerance of *margin_px* pixels is allowed beyond the image boundary to
    account for small offsets due to discretised landing height.
    Vectorised over all candidates.
    """
    if len(positions_xy) == 0:
        return positions_xy

    vs = float(vp["voxel_size"])
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    vmin_abs = obj_voxels.min(axis=0)

    pose_cam = E_w2c @ T_obj2world
    corners_cam_orig = transform_points(get_bbox_corners(bbox3d), pose_cam)
    R_w2c = E_w2c[:3, :3]

    N = len(positions_xy)
    anchors = np.column_stack([positions_xy,
                                np.full(N, landing_z)])   # (N, 3)
    dv = (anchors - vmin_abs[None, :]).astype(np.float64) * vs
    dc = (R_w2c @ dv.T).T                                 # (N, 3) cam deltas

    all_cam = corners_cam_orig[None, :, :] + dc[:, None, :]

    Z = all_cam[:, :, 2]
    z_ok = np.all(Z > 0, axis=1)

    safe_Z = np.where(Z > 0, Z, 1.0)
    U = all_cam[:, :, 0] / safe_Z * fx + cx
    V = all_cam[:, :, 1] / safe_Z * fy + cy

    m = margin_px
    u_ok = np.all(U >= -m, axis=1) & np.all(U < img_w + m, axis=1)
    v_ok = np.all(V >= -m, axis=1) & np.all(V < img_h + m, axis=1)

    keep = z_ok & u_ok & v_ok
    return positions_xy[keep]


# ═══════════════════════════════════════════════════════════════════════════════
#  Grid preparation
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_grid(grid_base, all_obj_voxels, target_idx):
    g = grid_base.copy()
    gs = g.shape
    for i, vox in enumerate(all_obj_voxels):
        if len(vox) == 0:
            continue
        m = ((vox[:, 0] >= 0) & (vox[:, 0] < gs[0]) &
             (vox[:, 1] >= 0) & (vox[:, 1] < gs[1]) &
             (vox[:, 2] >= 0) & (vox[:, 2] < gs[2]))
        v = vox[m]
        if len(v) == 0:
            continue
        if i == target_idx:
            g[v[:, 0], v[:, 1], v[:, 2]] = FREE
        else:
            g[v[:, 0], v[:, 1], v[:, 2]] = OCCUPIED
    return g


# ═══════════════════════════════════════════════════════════════════════════════
#  Table surface detection
# ═══════════════════════════════════════════════════════════════════════════════

def detect_table_z(grid):
    occ = (grid == OCCUPIED)
    z_counts = occ.sum(axis=(0, 1))
    if z_counts.max() == 0:
        return None
    return int(np.argmax(z_counts))


# ═══════════════════════════════════════════════════════════════════════════════
#  Placement search (table-surface only, FFT convolution)
# ═══════════════════════════════════════════════════════════════════════════════

def dilate_obstacles_xy(grid, margin_voxels):
    occ = (grid == OCCUPIED) | (grid == UNKNOWN)
    if margin_voxels <= 0:
        return occ
    s2d = generate_binary_structure(2, 1)
    s3d = np.zeros((3, 3, 3), dtype=bool)
    s3d[:, :, 1] = s2d
    return binary_dilation(occ, structure=s3d, iterations=margin_voxels)


def find_table_placements(grid_work, obj_voxels, table_z,
                          safety_margin_voxels, vp):
    Gx, Gy, Gz = grid_work.shape

    vmin = obj_voxels.min(axis=0)
    rel = obj_voxels - vmin
    osize = rel.max(axis=0) + 1
    ox, oy, oz = osize

    obj_mask = np.zeros(tuple(osize), dtype=np.float32)
    obj_mask[rel[:, 0], rel[:, 1], rel[:, 2]] = 1.0

    landing_z = table_z + 1
    if landing_z + oz > Gz or landing_z < 0:
        return np.empty((0, 2), dtype=int), {"error": "object_too_tall"}

    obstacle = dilate_obstacles_xy(grid_work, safety_margin_voxels)\
                   .astype(np.float32)
    collision = fftconvolve(obstacle,
                            obj_mask[::-1, ::-1, ::-1],
                            mode="valid")
    nx, ny, nz = collision.shape
    iz = landing_z
    if iz < 0 or iz >= nz:
        return np.empty((0, 2), dtype=int), {"error": "landing_z_oob"}

    free_mask = (np.abs(collision[:, :, iz]) < 0.5)
    ys, xs = np.where(free_mask)
    positions_xy = np.stack([ys, xs], axis=1)

    meta = {
        "total_xy": int(nx * ny),
        "valid_raw": len(positions_xy),
        "object_size_voxels": osize.tolist(),
        "landing_z": landing_z,
        "table_z": table_z,
    }
    return positions_xy, meta


# ═══════════════════════════════════════════════════════════════════════════════
#  DBSCAN clustering + representative selection
# ═══════════════════════════════════════════════════════════════════════════════

def cluster_placements(positions_xy, grid_work, obj_voxels,
                       landing_z, vp, eps_cm, min_samples):
    if len(positions_xy) == 0:
        return np.empty((0, 3), dtype=int), []

    vs = float(vp["voxel_size"])
    pts_w = voxel_to_world(
        np.column_stack([positions_xy,
                         np.full(len(positions_xy), landing_z)]),
        vp)[:, :2]

    labels = DBSCAN(eps=eps_cm, min_samples=min_samples).fit_predict(pts_w)
    unique = sorted(set(labels) - {-1})

    vmin = obj_voxels.min(axis=0)
    osize = (obj_voxels - vmin).max(axis=0) + 1
    Gx, Gy, Gz = grid_work.shape

    reps, infos = [], []
    for lbl in unique:
        members = positions_xy[labels == lbl]
        best_score, best_pos = -1, members[0]
        for pos in members:
            ix, iy = int(pos[0]), int(pos[1])
            pad = 3
            region = grid_work[
                max(ix - pad, 0): min(ix + osize[0] + pad, Gx),
                max(iy - pad, 0): min(iy + osize[1] + pad, Gy),
                max(landing_z - 1, 0): min(landing_z + osize[2] + 1, Gz)]
            score = int((region == FREE).sum())
            if score > best_score:
                best_score, best_pos = score, pos

        anchor = np.array([int(best_pos[0]), int(best_pos[1]), landing_z])
        reps.append(anchor)
        infos.append({
            "cluster_id": int(lbl),
            "size": len(members),
            "anchor_voxel": anchor.tolist(),
            "anchor_world_cm": voxel_to_world(anchor, vp).tolist(),
            "free_score": best_score,
        })

    arr = np.array(reps, dtype=int) if reps else np.empty((0, 3), dtype=int)
    return arr, infos


# ═══════════════════════════════════════════════════════════════════════════════
#  Visualisation  (dark-theme, two-panel, reference style)
# ═══════════════════════════════════════════════════════════════════════════════

def _draw_bbox_2d(ax, corners_world, K, E_w2c,
                  color, lw=2.0, label=None, alpha=1.0):
    u, v, z = project_world(corners_world, K, E_w2c)
    drawn = False
    for i, j in BOX_EDGES:
        if z[i] <= 0 or z[j] <= 0:
            continue
        lbl = label if (not drawn and label) else None
        ax.plot([u[i], u[j]], [v[i], v[j]],
                color=color, lw=lw, alpha=alpha, label=lbl)
        drawn = True


def _draw_bbox_3d(ax, corners_world, color, lw=1.5, label=None, alpha=1.0):
    for idx, (i, j) in enumerate(BOX_EDGES):
        lbl = label if idx == 0 else None
        ax.plot([corners_world[i, 0], corners_world[j, 0]],
                [corners_world[i, 1], corners_world[j, 1]],
                [corners_world[i, 2], corners_world[j, 2]],
                color=color, lw=lw, alpha=alpha, label=lbl)


def save_placement_vis(rgb, obj_name, bbox3d, T_obj2world,
                       K, E_w2c, vp,
                       cam_origin, cluster_reps, cluster_infos,
                       obj_voxels, grid, out_path):
    """
    Two-panel dark-theme figure:
      Left  – RGB + orange (original) + green (placements) 3-D bboxes
      Right – 3-D world view with point cloud, bboxes, camera, arrows
    """
    vs = float(vp["voxel_size"])
    vmin_abs = obj_voxels.min(axis=0) if len(obj_voxels) else np.zeros(3)
    corners_obj = get_bbox_corners(bbox3d)
    orig_world  = obb_corners_world(bbox3d, T_obj2world)
    orig_ctr    = orig_world.mean(axis=0)
    n_reps      = len(cluster_reps)
    img_h, img_w = rgb.shape[:2]

    fig = plt.figure(figsize=(18, 7))
    fig.patch.set_facecolor(CLR_BG)

    # ── Left panel: RGB image ────────────────────────────────────────────
    ax_rgb = fig.add_axes([0.02, 0.06, 0.47, 0.88])
    ax_rgb.imshow(rgb)

    _draw_bbox_2d(ax_rgb, orig_world, K, E_w2c,
                  color=CLR_ORIG, lw=2.5,
                  label=f"Original: {obj_name}")

    for k, rep in enumerate(cluster_reps):
        delta_w = (rep - vmin_abs).astype(np.float64) * vs
        T_new = T_obj2world.copy()
        T_new[:3, 3] += delta_w
        placed_world = transform_points(corners_obj, T_new)
        lbl = f"Placement #{k} (of {n_reps})" if k == 0 else None
        _draw_bbox_2d(ax_rgb, placed_world, K, E_w2c,
                      color=CLR_PLACE, lw=2.0, label=lbl, alpha=0.85)

    # Label annotations
    u_o, v_o, z_o = project_world(orig_ctr[None], K, E_w2c)
    if z_o[0] > 0:
        ax_rgb.annotate(
            obj_name, xy=(u_o[0], v_o[0]),
            xytext=(u_o[0] + 20, v_o[0] - 25),
            color=CLR_ORIG, fontsize=9, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=CLR_ORIG, lw=1.2))

    # Arrows from original center to each placement center + text label
    for k, rep in enumerate(cluster_reps):
        delta_w = (rep - vmin_abs).astype(np.float64) * vs
        T_new = T_obj2world.copy()
        T_new[:3, 3] += delta_w
        p_ctr = transform_points(corners_obj, T_new).mean(axis=0)
        u_p, v_p, z_p = project_world(p_ctr[None], K, E_w2c)
        if z_o[0] > 0 and z_p[0] > 0:
            ax_rgb.annotate(
                "", xy=(u_p[0], v_p[0]), xytext=(u_o[0], v_o[0]),
                arrowprops=dict(arrowstyle="->", color=CLR_ARROW,
                                lw=1.5, connectionstyle="arc3,rad=0.15"))
            ax_rgb.text(u_p[0] + 8, v_p[0] - 8,
                        f"#{k}", color=CLR_PLACE,
                        fontsize=8, fontweight="bold")

    ax_rgb.set_xlim(0, img_w)
    ax_rgb.set_ylim(img_h, 0)
    ax_rgb.set_title("RGB Image — 3-D Bbox Projection\n"
                     "Orange = current position  |  Green = sample placements",
                     color="white", fontsize=11, pad=8)
    ax_rgb.legend(loc="upper right", fontsize=8,
                  facecolor=CLR_PANEL, labelcolor="white", framealpha=0.85)
    ax_rgb.axis("off")

    # ── Right panel: 3-D world view ──────────────────────────────────────
    ax3 = fig.add_axes([0.52, 0.04, 0.46, 0.92], projection="3d")
    ax3.set_facecolor(CLR_PANEL)

    # Background point cloud
    rng = np.random.default_rng(42)
    occ_idx  = np.argwhere(grid == OCCUPIED)
    free_idx = np.argwhere(grid == FREE)
    if len(occ_idx) > 3000:
        occ_idx = occ_idx[rng.choice(len(occ_idx), 3000, replace=False)]
    if len(free_idx) > 2000:
        free_idx = free_idx[rng.choice(len(free_idx), 2000, replace=False)]
    if len(free_idx):
        fw = voxel_to_world(free_idx, vp)
        ax3.scatter(fw[:, 0], fw[:, 1], fw[:, 2],
                    c=CLR_FREE, s=1, alpha=0.10, depthshade=False)
    if len(occ_idx):
        ow = voxel_to_world(occ_idx, vp)
        ax3.scatter(ow[:, 0], ow[:, 1], ow[:, 2],
                    c=CLR_OCC, s=2, alpha=0.40, depthshade=False)

    # Camera marker
    ax3.scatter(*cam_origin, c=CLR_CAM, s=80, marker="*",
                zorder=10, label="Camera", depthshade=False)

    # Original bbox (orange)
    _draw_bbox_3d(ax3, orig_world, color=CLR_ORIG, lw=2.5,
                  label=f"Original: {obj_name}")

    # Placement bboxes + displacement arrows
    for k, rep in enumerate(cluster_reps):
        delta_w = (rep - vmin_abs).astype(np.float64) * vs
        T_new = T_obj2world.copy()
        T_new[:3, 3] += delta_w
        placed_world = transform_points(corners_obj, T_new)
        lbl = f"Placement #{k}" if k == 0 else None
        _draw_bbox_3d(ax3, placed_world, color=CLR_PLACE, lw=2.0,
                      label=lbl, alpha=0.85)
        p_ctr = placed_world.mean(axis=0)
        lbl_q = "displacement" if k == 0 else None
        ax3.quiver(orig_ctr[0], orig_ctr[1], orig_ctr[2],
                   *(p_ctr - orig_ctr),
                   color=CLR_ARROW, lw=1.5, arrow_length_ratio=0.15,
                   label=lbl_q)

    ax3.set_xlabel("X (cm)", color="white", fontsize=8)
    ax3.set_ylabel("Y (cm)", color="white", fontsize=8)
    ax3.set_zlabel("Z (cm)", color="white", fontsize=8)
    ax3.tick_params(colors="white", labelsize=7)
    for pane in (ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#333355")
    ax3.set_title(f"3-D World View\n"
                  f"Orange → {obj_name} (current)   "
                  f"Green → Placements",
                  color="white", fontsize=10, pad=6)
    ax3.legend(fontsize=8, loc="upper left",
               facecolor=CLR_PANEL, labelcolor="white",
               framealpha=0.85, markerscale=3)

    # ── Info text box ────────────────────────────────────────────────────
    lines = [f"Target  {obj_name}"]
    for k, ci in enumerate(cluster_infos):
        cw = np.array(ci["anchor_world_cm"])
        delta = cw - voxel_to_world(vmin_abs, vp)
        dist = float(np.linalg.norm(delta))

        dw = (np.array(ci["anchor_voxel"]) - vmin_abs).astype(np.float64) * vs
        T_tmp = T_obj2world.copy()
        T_tmp[:3, 3] += dw
        pw = transform_points(corners_obj, T_tmp)
        aabb_min, aabb_max = pw.min(0), pw.max(0)

        lines.append(
            f"  Placement #{k}:  Δ=({delta[0]:+.1f}, {delta[1]:+.1f}, "
            f"{delta[2]:+.1f}) cm   |Δ|={dist:.1f} cm   "
            f"cluster_size={ci['size']}")
        lines.append(
            f"    AABB  x:[{aabb_min[0]:.1f},{aabb_max[0]:.1f}]  "
            f"y:[{aabb_min[1]:.1f},{aabb_max[1]:.1f}]  "
            f"z:[{aabb_min[2]:.1f},{aabb_max[2]:.1f}]")

    fig.text(0.535, 0.03, "\n".join(lines),
             fontsize=8, color="white", family="monospace",
             verticalalignment="bottom",
             bbox=dict(facecolor=CLR_PANEL, edgecolor="#444466",
                       boxstyle="round,pad=0.5", alpha=0.9))

    fig.text(0.5, 0.97,
             f"Placement Planning — Sample Visualisation  ({obj_name})",
             ha="center", va="top", color="white",
             fontsize=13, fontweight="bold")

    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main(data_dir=None,
         safety_margin_cm=SAFETY_MARGIN_CM,
         dbscan_eps_cm=DBSCAN_EPS_CM,
         dbscan_min_samples=DBSCAN_MIN_SAMPLES):

    if data_dir is None:
        data_dir = DATA_DIR

    bar = "=" * 66
    print(f"\n{bar}")
    print("  HOPE · 3-D Placement Planner  (all objects)")
    print(f"  safety_margin  = {safety_margin_cm} cm")
    print(f"  dbscan_eps     = {dbscan_eps_cm} cm")
    print(f"  data_dir       = {data_dir}")
    print(f"{bar}\n")

    # ── load ─────────────────────────────────────────────────────────────
    grid, grid_meta, annotations, rgb = load_scene_data(data_dir)
    vp = grid_meta["voxel_params"]
    vs = float(vp["voxel_size"])
    gs = tuple(grid_meta["grid_shape"])
    img_h, img_w = rgb.shape[:2]

    E_w2c, E_c2w = camera_transforms(annotations)
    cam_origin = E_c2w[:3, 3]

    K = np.array(annotations["camera"]["intrinsics"], dtype=np.float64)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    objects = annotations["objects"]
    n_obj = len(objects)
    print(f"Objects : {n_obj}  |  Grid {gs}  |  Voxel {vs} cm\n")

    # ── voxelize all objects ─────────────────────────────────────────────
    all_voxels, all_T = [], []
    for obj in objects:
        pose = np.array(obj["pose"], dtype=np.float64)
        T = obj_to_world(pose, E_c2w)
        vox = voxelize_obb(obj["bbox3d"], T, vp, gs) \
            if "bbox3d" in obj else np.empty((0, 3), dtype=int)
        all_voxels.append(vox)
        all_T.append(T)

    margin_voxels = int(math.ceil(safety_margin_cm / vs))

    # ── process each object ──────────────────────────────────────────────
    all_results = {}

    for oi, obj in enumerate(objects):
        name = obj["class"]
        pose = np.array(obj["pose"], dtype=np.float64)
        bbox3d = obj["bbox3d"]
        T = all_T[oi]

        print(f"── [{oi+1}/{n_obj}] {name} {'─'*(50-len(name))}")

        # 1. Visibility check on original object
        if not is_fully_visible(bbox3d, pose, fx, fy, cx, cy, img_w, img_h):
            print("   SKIP: not fully visible in image")
            all_results[name] = {"status": "not_visible"}
            continue

        target_vox = all_voxels[oi]
        if len(target_vox) == 0:
            print("   SKIP: no voxels")
            all_results[name] = {"status": "no_voxels"}
            continue

        # 2. Prepare grid
        grid_work = prepare_grid(grid, all_voxels, oi)

        # 3. Detect table
        table_z = detect_table_z(grid_work)
        if table_z is None:
            print("   SKIP: no table detected")
            all_results[name] = {"status": "no_table"}
            continue

        # 4. Collision search
        pos_xy, meta = find_table_placements(
            grid_work, target_vox, table_z, margin_voxels, vp)
        landing_z = meta.get("landing_z", table_z + 1)
        n_raw = meta.get("valid_raw", 0)

        # 5. Filter placements for camera visibility
        pos_xy = filter_visible_placements(
            pos_xy, landing_z, target_vox,
            bbox3d, T, E_w2c, K, img_w, img_h, vp)
        n_vis = len(pos_xy)

        print(f"   table_z={table_z}  landing_z={landing_z}  "
              f"mask={meta.get('object_size_voxels', '?')}  "
              f"raw={n_raw}  visible={n_vis}")

        # 6. Cluster
        reps, c_infos = cluster_placements(
            pos_xy, grid_work, target_vox,
            landing_z, vp, dbscan_eps_cm, dbscan_min_samples)
        print(f"   clusters={len(reps)}")

        # 7. Visualise
        obj_dir = os.path.join(data_dir, f"placement_{name}")
        os.makedirs(obj_dir, exist_ok=True)

        vis_path = os.path.join(obj_dir, "placement_vis.png")
        save_placement_vis(
            rgb, name, bbox3d, T, K, E_w2c, vp,
            cam_origin, reps, c_infos,
            target_vox, grid, vis_path)
        print(f"   [✓] {vis_path}")

        all_results[name] = {
            "status": "ok",
            "table_z": table_z,
            "landing_z": landing_z,
            "raw_valid": n_raw,
            "visible_valid": n_vis,
            "clusters": c_infos,
        }

    # ── JSON summary ─────────────────────────────────────────────────────
    summary = {
        "safety_margin_cm": safety_margin_cm,
        "dbscan_eps_cm":    dbscan_eps_cm,
        "voxel_size_cm":    vs,
        "objects": all_results,
    }
    json_path = os.path.join(data_dir, "placement_result.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[✓] JSON summary: {json_path}")
    print(f"{bar}\n")


if __name__ == "__main__":
    main()
