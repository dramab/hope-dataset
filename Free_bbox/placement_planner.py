#!/usr/bin/env python3
"""
3D Object Placement Planner v2 (HOPE-Video dataset)

For **every** object in a scene, compute valid collision-free positions on the
main horizontal support surface.  Placement candidates are filtered for
physical stability, camera visibility, and occlusion before being clustered.

Improvements over v1
  - Yaw rotation : objects rotated around Z-axis (24 discrete angles by default)
  - Robust surface detection via morphological opening + connected components
  - Physical stability : support-ratio ≥ 0.9 and COM-projection check
  - Occlusion awareness : Z-buffer depth comparison
  - Optional GPU acceleration with CuPy (graceful CPU fallback)

Algorithm per object
  1. Visibility check – all 8 OBB corners project inside the image
  2. Grid preparation  – all objects' OBBs filled as OCCUPIED;
                         target OBB set to FREE ("simulated removal")
  3. Surface detection – morphological opening + connected components
                         → largest horizontal support surface
  4. Depth-buffer construction from remaining scene geometry
  5. FFT collision search across (X, Y, θ) configuration space
     with per-layer 2D convolution and optional CuPy GPU backend
  6. Physical-stability filter (support ratio + COM projection)
  7. Camera-visibility filter on each candidate placement
  8. Occlusion filter via Z-buffer depth comparison
  9. DBSCAN clustering → one representative per cluster (most free space)

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
from scipy.ndimage import (binary_dilation, binary_opening,
                           generate_binary_structure, label)
from scipy.signal import fftconvolve
from sklearn.cluster import DBSCAN

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401

# ── GPU backend (optional) ───────────────────────────────────────────────────
try:
    import cupy as cp
    from cupyx.scipy.signal import fftconvolve as gpu_fftconvolve
    HAS_GPU = True
    print("GPU backend loaded")
except ImportError:
    HAS_GPU = False

# ═══════════════════════════════════════════════════════════════════════════════
#  Adjustable parameters
# ═══════════════════════════════════════════════════════════════════════════════

SAFETY_MARGIN_CM     = 0.5
DBSCAN_EPS_CM        = 5.0
DBSCAN_MIN_SAMPLES   = 1
VIS_MARGIN_PX        = 30

YAW_STEPS            = 24           # 360° / 24 = 15° per step
MIN_SURFACE_AREA_CM2 = 50.0
MIN_SUPPORT_RATIO    = 1.0
OCCLUSION_THRESHOLD  = 0.3          # fraction of OBB corners behind scene
STABILITY_CHUNK_SIZE = 2000         # max candidates per chunk in stability filter

FREE, OCCUPIED, UNKNOWN = 0, 1, 2
WORLD_UP = np.array([0.0, 0.0, 1.0])

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_data")

BOX_EDGES = [
    (0, 1), (2, 3), (4, 5), (6, 7),
    (0, 2), (1, 3), (4, 6), (5, 7),
    (0, 4), (1, 5), (2, 6), (3, 7),
]

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
    """World -> image.  Returns (u, v, z_cam) arrays."""
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


def rotation_z_3x3(angle):
    """3x3 rotation matrix for *angle* (radians) around the Z axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


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
#  Support-surface detection (morphological + connected components)
# ═══════════════════════════════════════════════════════════════════════════════

def detect_support_surfaces(grid, vp, min_area_cm2=MIN_SURFACE_AREA_CM2):
    """
    Find the largest horizontal support surface via morphological opening and
    connected-component labelling on each Z slice.

    Returns (table_z, surface_mask_2d) or (None, None).
    """
    vs = float(vp["voxel_size"])
    min_voxels = max(1, int(min_area_cm2 / (vs * vs)))

    occ = (grid == OCCUPIED)
    struct_2d = np.ones((3, 3), dtype=bool)

    # Z-axis pruning: only scan layers with enough occupied voxels
    z_counts = occ.sum(axis=(0, 1))
    active_zs = np.where(z_counts >= min_voxels)[0]

    best_z, best_area, best_mask = None, 0, None

    for z in active_zs:
        slice_2d = occ[:, :, z]
        opened = binary_opening(slice_2d, structure=struct_2d)
        labeled, n_features = label(opened)
        for comp_id in range(1, n_features + 1):
            component = (labeled == comp_id)
            area = int(component.sum())
            if area >= min_voxels and area > best_area:
                best_z = int(z)
                best_area = area
                best_mask = component

    if best_z is None:
        return None, None
    return best_z, best_mask


def detect_table_z(grid):
    """Legacy wrapper kept for backward compatibility."""
    occ = (grid == OCCUPIED)
    z_counts = occ.sum(axis=(0, 1))
    if z_counts.max() == 0:
        return None
    return int(np.argmax(z_counts))


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


def filter_visible_placements(candidates, landing_z,
                               bbox3d, T_obj2world, E_w2c, K,
                               img_w, img_h, vp, yaw_data,
                               margin_px=VIS_MARGIN_PX):
    """
    Keep only placements whose placed OBB projects (mostly) inside the image.
    Vectorised per yaw-angle batch.
    """
    if len(candidates) == 0:
        return candidates

    vs = float(vp["voxel_size"])
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    R_w2c = E_w2c[:3, :3]
    corners_canonical = get_bbox_corners(bbox3d)

    keep = np.zeros(len(candidates), dtype=bool)
    yaw_angles  = yaw_data["yaw_angles"]
    vmin_rots   = yaw_data["vmin_rot_abs"]
    T_rot_list  = yaw_data["T_rotated"]

    for yaw_idx in range(len(yaw_angles)):
        mask = candidates[:, 2] == yaw_idx
        if not mask.any():
            continue

        batch    = candidates[mask]
        T_rot    = T_rot_list[yaw_idx]
        vmin_rot = vmin_rots[yaw_idx]

        corners_cam_base = transform_points(corners_canonical, E_w2c @ T_rot)

        N = len(batch)
        anchors = np.column_stack([batch[:, :2].astype(np.float64),
                                   np.full(N, landing_z, dtype=np.float64)])
        delta_cam = (R_w2c @ ((anchors - vmin_rot) * vs).T).T

        all_cam = corners_cam_base[None, :, :] + delta_cam[:, None, :]

        Z      = all_cam[:, :, 2]
        z_ok   = np.all(Z > 0, axis=1)
        safe_Z = np.where(Z > 0, Z, 1.0)
        U = all_cam[:, :, 0] / safe_Z * fx + cx
        V = all_cam[:, :, 1] / safe_Z * fy + cy

        m = margin_px
        u_ok = np.all(U >= -m, axis=1) & np.all(U < img_w + m, axis=1)
        v_ok = np.all(V >= -m, axis=1) & np.all(V < img_h + m, axis=1)

        keep[mask] = z_ok & u_ok & v_ok

    return candidates[keep]


# ═══════════════════════════════════════════════════════════════════════════════
#  Physical stability (support ratio + COM projection)
# ═══════════════════════════════════════════════════════════════════════════════

def filter_stable_placements(candidates, yaw_data, table_mask_2d,
                              min_support_ratio=MIN_SUPPORT_RATIO):
    """
    Keep placements whose bottom-face footprint is sufficiently supported by
    the table surface and whose centre-of-mass projects onto the support area.
    Vectorised per yaw-angle batch.
    """
    if len(candidates) == 0 or table_mask_2d is None:
        return candidates

    Gx, Gy = table_mask_2d.shape
    keep = np.zeros(len(candidates), dtype=bool)
    footprints = yaw_data["footprints"]

    chunk_sz = STABILITY_CHUNK_SIZE

    for yaw_idx, footprint in enumerate(footprints):
        mask = candidates[:, 2] == yaw_idx
        if not mask.any() or len(footprint) == 0:
            continue

        batch  = candidates[mask]
        n_foot = len(footprint)
        com_rel = footprint.mean(axis=0).astype(np.float64)

        batch_keep = np.zeros(len(batch), dtype=bool)

        for c0 in range(0, len(batch), chunk_sz):
            c1 = min(c0 + chunk_sz, len(batch))
            chunk = batch[c0:c1]

            # (C, F, 2) – absolute grid coords of each footprint voxel
            placed = chunk[:, None, :2] + footprint[None, :, :]

            in_bounds = ((placed[:, :, 0] >= 0) & (placed[:, :, 0] < Gx) &
                         (placed[:, :, 1] >= 0) & (placed[:, :, 1] < Gy))

            cl_x = np.clip(placed[:, :, 0], 0, Gx - 1).astype(int)
            cl_y = np.clip(placed[:, :, 1], 0, Gy - 1).astype(int)

            on_table  = table_mask_2d[cl_x, cl_y]
            supported = in_bounds & on_table
            ratio     = supported.sum(axis=1) / n_foot

            com   = com_rel + chunk[:, :2].astype(np.float64)
            com_i = np.round(com[:, 0]).astype(int)
            com_j = np.round(com[:, 1]).astype(int)
            com_in = ((com_i >= 0) & (com_i < Gx) &
                      (com_j >= 0) & (com_j < Gy))
            com_ok = com_in & table_mask_2d[np.clip(com_i, 0, Gx - 1),
                                            np.clip(com_j, 0, Gy - 1)]

            batch_keep[c0:c1] = (ratio >= min_support_ratio) & com_ok

        keep[mask] = batch_keep

    return candidates[keep]


# ═══════════════════════════════════════════════════════════════════════════════
#  Depth buffer & occlusion filtering
# ═══════════════════════════════════════════════════════════════════════════════

def build_depth_buffer(grid_work, vp, K, E_w2c, img_w, img_h):
    """
    Build a per-pixel minimum-depth buffer from all OCCUPIED voxels in
    *grid_work* (which already has the target object removed).
    """
    occ_idx = np.argwhere(grid_work == OCCUPIED)
    depth_buf = np.full((img_h, img_w), np.inf, dtype=np.float64)

    if len(occ_idx) == 0:
        return depth_buf

    occ_world = voxel_to_world(occ_idx, vp)
    u, v, z_cam = project_world(occ_world, K, E_w2c)

    valid = ((z_cam > 0) &
             np.isfinite(u) & np.isfinite(v) &
             (u >= 0) & (u < img_w) &
             (v >= 0) & (v < img_h))

    u_int   = np.clip(np.round(u[valid]).astype(int), 0, img_w - 1)
    v_int   = np.clip(np.round(v[valid]).astype(int), 0, img_h - 1)
    z_valid = z_cam[valid]

    np.minimum.at(depth_buf, (v_int, u_int), z_valid)
    return depth_buf


def filter_occluded_placements(candidates, landing_z,
                                bbox3d, T_obj2world,
                                depth_buffer, K, E_w2c, vp,
                                yaw_data, img_w, img_h,
                                occlusion_threshold=OCCLUSION_THRESHOLD):
    """
    Remove placements where more than *occlusion_threshold* fraction of the
    OBB corners are behind existing scene geometry (Z-buffer comparison).
    Vectorised per yaw-angle batch.
    """
    if len(candidates) == 0:
        return candidates

    vs = float(vp["voxel_size"])
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    R_w2c = E_w2c[:3, :3]
    corners_canonical = get_bbox_corners(bbox3d)

    keep = np.zeros(len(candidates), dtype=bool)
    yaw_angles = yaw_data["yaw_angles"]
    vmin_rots  = yaw_data["vmin_rot_abs"]
    T_rot_list = yaw_data["T_rotated"]

    for yaw_idx in range(len(yaw_angles)):
        mask = candidates[:, 2] == yaw_idx
        if not mask.any():
            continue

        batch    = candidates[mask]
        T_rot    = T_rot_list[yaw_idx]
        vmin_rot = vmin_rots[yaw_idx]

        corners_cam_base = transform_points(corners_canonical, E_w2c @ T_rot)

        N = len(batch)
        anchors = np.column_stack([batch[:, :2].astype(np.float64),
                                   np.full(N, landing_z, dtype=np.float64)])
        delta_cam = (R_w2c @ ((anchors - vmin_rot) * vs).T).T

        all_cam = corners_cam_base[None, :, :] + delta_cam[:, None, :]

        Z     = all_cam[:, :, 2]
        z_pos = Z > 0
        safe_Z = np.where(z_pos, Z, 1.0)
        U = all_cam[:, :, 0] / safe_Z * fx + cx
        V = all_cam[:, :, 1] / safe_Z * fy + cy

        U_int = np.clip(np.round(U).astype(int), 0, img_w - 1)
        V_int = np.clip(np.round(V).astype(int), 0, img_h - 1)

        scene_depth = depth_buffer[V_int, U_int]

        occluded  = z_pos & (Z > scene_depth + 1e-3)
        occ_ratio = occluded.sum(axis=1) / 8.0

        keep[mask] = occ_ratio <= occlusion_threshold

    return candidates[keep]


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


def prepare_grid_base(grid, all_obj_voxels):
    """One-time grid preparation: mark every object OBB as OCCUPIED."""
    g = grid.copy()
    gs = g.shape
    for vox in all_obj_voxels:
        if len(vox) == 0:
            continue
        m = ((vox[:, 0] >= 0) & (vox[:, 0] < gs[0]) &
             (vox[:, 1] >= 0) & (vox[:, 1] < gs[1]) &
             (vox[:, 2] >= 0) & (vox[:, 2] < gs[2]))
        v = vox[m]
        if len(v):
            g[v[:, 0], v[:, 1], v[:, 2]] = OCCUPIED
    return g


def grid_remove_object(grid, voxels):
    """In-place: set *voxels* to FREE, return (valid_voxels, saved_values)."""
    if len(voxels) == 0:
        return voxels, np.array([], dtype=grid.dtype)
    gs = grid.shape
    m = ((voxels[:, 0] >= 0) & (voxels[:, 0] < gs[0]) &
         (voxels[:, 1] >= 0) & (voxels[:, 1] < gs[1]) &
         (voxels[:, 2] >= 0) & (voxels[:, 2] < gs[2]))
    v = voxels[m]
    if len(v) == 0:
        return v, np.array([], dtype=grid.dtype)
    saved = grid[v[:, 0], v[:, 1], v[:, 2]].copy()
    grid[v[:, 0], v[:, 1], v[:, 2]] = FREE
    return v, saved


def grid_restore_object(grid, voxels, saved_values):
    """In-place: restore previously saved voxel values."""
    if len(voxels) > 0:
        grid[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = saved_values


# ═══════════════════════════════════════════════════════════════════════════════
#  Placement search  (configuration space: X, Y, theta)
# ═══════════════════════════════════════════════════════════════════════════════

def dilate_obstacles_xy(grid, margin_voxels):
    occ = (grid == OCCUPIED) | (grid == UNKNOWN)
    if margin_voxels <= 0:
        return occ
    s2d = generate_binary_structure(2, 1)
    s3d = np.zeros((3, 3, 3), dtype=bool)
    s3d[:, :, 1] = s2d
    return binary_dilation(occ, structure=s3d, iterations=margin_voxels)


def _compute_collision_slice(obstacle, obj_mask, landing_z, use_gpu=False):
    """
    2D collision map at *landing_z* via per-layer 2D convolutions.
    Only the layers actually occupied by the object mask are convolved,
    avoiding a full 3D FFT and reducing memory.
    """
    Gx, Gy, Gz = obstacle.shape
    ox, oy, oz = obj_mask.shape
    nx, ny = Gx - ox + 1, Gy - oy + 1

    if nx <= 0 or ny <= 0:
        return None

    if use_gpu and HAS_GPU:
        collision = cp.zeros((nx, ny), dtype=cp.float32)
        for dz in range(oz):
            z_idx = landing_z + dz
            if z_idx < 0 or z_idx >= Gz:
                continue
            obj_slice = obj_mask[:, :, dz]
            if obj_slice.sum() == 0:
                continue
            obs_gpu  = cp.asarray(obstacle[:, :, z_idx].astype(np.float32))
            mask_gpu = cp.asarray(
                np.ascontiguousarray(obj_slice[::-1, ::-1]).astype(np.float32))
            collision += gpu_fftconvolve(obs_gpu, mask_gpu, mode="valid")
        return cp.asnumpy(collision)
    else:
        collision = np.zeros((nx, ny), dtype=np.float32)
        for dz in range(oz):
            z_idx = landing_z + dz
            if z_idx < 0 or z_idx >= Gz:
                continue
            obj_slice = obj_mask[:, :, dz]
            if obj_slice.sum() == 0:
                continue
            collision += fftconvolve(
                obstacle[:, :, z_idx].astype(np.float32),
                obj_slice[::-1, ::-1].astype(np.float32),
                mode="valid")
        return collision


def find_table_placements(grid_work, bbox3d, T_obj2world,
                          table_z, safety_margin_voxels, vp,
                          yaw_steps=YAW_STEPS, use_gpu=False,
                          table_mask_2d=None):
    """
    Search for collision-free placements in (X, Y, theta) configuration space.

    For each discretised yaw angle the OBB is re-voxelised with the rotated
    transform and a per-layer 2D FFT convolution is run at the landing height.

    When *table_mask_2d* is provided the obstacle volume is cropped to the
    table bounding-box (+ padding) before FFT, drastically reducing compute.

    Returns
    -------
    candidates : (N, 3) int array  –  columns (grid_x, grid_y, yaw_index)
    meta       : dict with summary statistics
    yaw_data   : dict with per-yaw precomputed data for downstream filters
    """
    Gx, Gy, Gz = grid_work.shape
    grid_shape  = (Gx, Gy, Gz)
    landing_z   = table_z + 1

    obstacle = dilate_obstacles_xy(grid_work, safety_margin_voxels) \
                   .astype(np.float32)

    # ROI cropping: restrict FFT to the table region + padding
    if table_mask_2d is not None:
        t_rows, t_cols = np.where(table_mask_2d)
        if len(t_rows) > 0:
            bbox_diag = np.linalg.norm(
                np.array(bbox3d[3:]) - np.array(bbox3d[:3]))
            vs = float(vp["voxel_size"])
            max_obj_vx = int(np.ceil(bbox_diag / vs)) + 2
            pad = max_obj_vx + safety_margin_voxels
            roi_x0 = max(int(t_rows.min()) - pad, 0)
            roi_x1 = min(int(t_rows.max()) + 1 + pad, Gx)
            roi_y0 = max(int(t_cols.min()) - pad, 0)
            roi_y1 = min(int(t_cols.max()) + 1 + pad, Gy)
        else:
            roi_x0, roi_y0, roi_x1, roi_y1 = 0, 0, Gx, Gy
    else:
        roi_x0, roi_y0, roi_x1, roi_y1 = 0, 0, Gx, Gy

    obstacle_roi = obstacle[roi_x0:roi_x1, roi_y0:roi_y1, :]
    roi_Gx, roi_Gy = obstacle_roi.shape[0], obstacle_roi.shape[1]

    bbox_center      = (np.array(bbox3d[:3]) + np.array(bbox3d[3:])) / 2.0
    obj_center_world = (T_obj2world @ np.append(bbox_center, 1))[:3]

    yaw_angles = np.linspace(0, 2 * np.pi, yaw_steps, endpoint=False)

    all_candidates   = []
    yaw_rel_voxels   = []
    yaw_vmin_rot     = []
    yaw_T_rotated    = []
    yaw_footprints   = []
    valid_yaw_count  = 0
    total_raw        = 0

    _empty_vox  = np.empty((0, 3), dtype=int)
    _empty_foot = np.empty((0, 2), dtype=int)
    _zero3      = np.zeros(3, dtype=np.float64)

    for yaw_idx, angle in enumerate(yaw_angles):
        R_yaw = rotation_z_3x3(angle)

        T_rot = T_obj2world.copy()
        T_rot[:3, :3] = R_yaw @ T_obj2world[:3, :3]
        T_rot[:3, 3]  = obj_center_world + R_yaw @ (
            T_obj2world[:3, 3] - obj_center_world)

        rot_voxels = voxelize_obb(bbox3d, T_rot, vp, grid_shape)

        if len(rot_voxels) == 0:
            yaw_rel_voxels.append(_empty_vox)
            yaw_vmin_rot.append(_zero3.copy())
            yaw_T_rotated.append(T_rot)
            yaw_footprints.append(_empty_foot)
            continue

        vmin_rot = rot_voxels.min(axis=0)
        rel_rot  = rot_voxels - vmin_rot
        osize    = rel_rot.max(axis=0) + 1
        ox, oy, oz = int(osize[0]), int(osize[1]), int(osize[2])

        # Early pruning: rotated AABB exceeds ROI / grid
        if (ox > roi_Gx or oy > roi_Gy or
                landing_z + oz > Gz or landing_z < 0):
            yaw_rel_voxels.append(rel_rot)
            yaw_vmin_rot.append(vmin_rot.astype(np.float64))
            yaw_T_rotated.append(T_rot)
            yaw_footprints.append(_empty_foot)
            continue

        obj_mask = np.zeros((ox, oy, oz), dtype=np.float32)
        obj_mask[rel_rot[:, 0], rel_rot[:, 1], rel_rot[:, 2]] = 1.0

        collision_2d = _compute_collision_slice(
            obstacle_roi, obj_mask, landing_z, use_gpu=use_gpu)

        # Bottom footprint (Z == 0 in relative coords -> landing_z in grid)
        bottom = rel_rot[rel_rot[:, 2] == 0][:, :2]

        if collision_2d is None:
            yaw_rel_voxels.append(rel_rot)
            yaw_vmin_rot.append(vmin_rot.astype(np.float64))
            yaw_T_rotated.append(T_rot)
            yaw_footprints.append(bottom)
            continue

        free_mask = np.abs(collision_2d) < 0.5
        ys, xs = np.where(free_mask)
        # Map ROI-local coords back to global grid coords
        ys += roi_x0
        xs += roi_y0
        n_free = len(ys)
        total_raw += n_free

        if n_free > 0:
            yaw_col = np.full(n_free, yaw_idx, dtype=int)
            all_candidates.append(np.stack([ys, xs, yaw_col], axis=1))
            valid_yaw_count += 1

        yaw_rel_voxels.append(rel_rot)
        yaw_vmin_rot.append(vmin_rot.astype(np.float64))
        yaw_T_rotated.append(T_rot)
        yaw_footprints.append(bottom)

    candidates = (np.vstack(all_candidates) if all_candidates
                  else np.empty((0, 3), dtype=int))

    yaw_data = {
        "yaw_angles":   yaw_angles,
        "rel_voxels":   yaw_rel_voxels,
        "vmin_rot_abs": yaw_vmin_rot,
        "T_rotated":    yaw_T_rotated,
        "footprints":   yaw_footprints,
    }

    meta = {
        "total_xy":         int(Gx * Gy),
        "valid_raw":        total_raw,
        "yaw_steps":        yaw_steps,
        "valid_yaw_angles": valid_yaw_count,
        "landing_z":        landing_z,
        "table_z":          table_z,
    }

    return candidates, meta, yaw_data


# ═══════════════════════════════════════════════════════════════════════════════
#  DBSCAN clustering + representative selection
# ═══════════════════════════════════════════════════════════════════════════════

def cluster_placements(candidates, grid_work, yaw_data,
                       landing_z, vp, eps_cm, min_samples):
    """
    Cluster in 2D world XY.  For each cluster pick the representative with the
    highest surrounding free-space score.

    *candidates* columns: (grid_x, grid_y, yaw_index).
    """
    if len(candidates) == 0:
        return np.empty((0, 3), dtype=int), []

    vs = float(vp["voxel_size"])
    anchors_3d = np.column_stack([candidates[:, :2],
                                  np.full(len(candidates), landing_z)])
    pts_w = voxel_to_world(anchors_3d, vp)[:, :2]

    labels = DBSCAN(eps=eps_cm, min_samples=min_samples).fit_predict(pts_w)
    unique = sorted(set(labels) - {-1})

    Gx, Gy, Gz = grid_work.shape
    rel_voxels_list = yaw_data["rel_voxels"]
    yaw_angles      = yaw_data["yaw_angles"]

    reps, infos = [], []
    for lbl in unique:
        members = candidates[labels == lbl]
        best_score, best_pos = -1, members[0]

        for pos in members:
            ix, iy, yi = int(pos[0]), int(pos[1]), int(pos[2])
            rv    = rel_voxels_list[yi]
            osize = rv.max(axis=0) + 1 if len(rv) > 0 else np.array([1, 1, 1])
            pad   = 3
            region = grid_work[
                max(ix - pad, 0): min(ix + osize[0] + pad, Gx),
                max(iy - pad, 0): min(iy + osize[1] + pad, Gy),
                max(landing_z - 1, 0): min(landing_z + osize[2] + 1, Gz)]
            score = int((region == FREE).sum())
            if score > best_score:
                best_score, best_pos = score, pos

        anchor = np.array([int(best_pos[0]), int(best_pos[1]),
                           int(best_pos[2])], dtype=int)
        reps.append(anchor)

        anchor_3d = np.array([anchor[0], anchor[1], landing_z])
        infos.append({
            "cluster_id":     int(lbl),
            "size":           len(members),
            "anchor_voxel":   anchor_3d.tolist(),
            "anchor_world_cm": voxel_to_world(anchor_3d, vp).tolist(),
            "yaw_index":      int(anchor[2]),
            "yaw_degrees":    float(np.degrees(yaw_angles[anchor[2]])),
            "free_score":     best_score,
        })

    arr = np.array(reps, dtype=int) if reps else np.empty((0, 3), dtype=int)
    return arr, infos


# ═══════════════════════════════════════════════════════════════════════════════
#  Placed-object transform helper
# ═══════════════════════════════════════════════════════════════════════════════

def compute_placed_transform(T_obj2world, bbox3d, anchor_xy,
                              landing_z, yaw_data, yaw_idx, vp):
    """
    4x4 object->world transform for a placed object, accounting for yaw
    rotation and grid-space displacement.
    """
    vs       = float(vp["voxel_size"])
    T_rot    = yaw_data["T_rotated"][yaw_idx]
    vmin_rot = yaw_data["vmin_rot_abs"][yaw_idx]

    anchor_3d   = np.array([anchor_xy[0], anchor_xy[1], landing_z],
                           dtype=np.float64)
    delta_world = (anchor_3d - vmin_rot) * vs

    T_placed = T_rot.copy()
    T_placed[:3, 3] += delta_world
    return T_placed


# ═══════════════════════════════════════════════════════════════════════════════
#  Visualisation  (dark-theme, two-panel)
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
                       obj_voxels, grid, out_path,
                       yaw_data, landing_z):
    """
    Two-panel dark-theme figure.
      Left  – RGB + orange (original) + green (placements) 3-D bboxes
      Right – 3-D world view with point cloud, bboxes, camera, arrows
    Placement bboxes are drawn with their respective yaw rotation.
    """
    vs = float(vp["voxel_size"])
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
        yaw_idx = int(rep[2])
        T_placed = compute_placed_transform(
            T_obj2world, bbox3d, rep[:2], landing_z, yaw_data, yaw_idx, vp)
        placed_world = transform_points(corners_obj, T_placed)
        lbl = f"Placement #{k} (of {n_reps})" if k == 0 else None
        _draw_bbox_2d(ax_rgb, placed_world, K, E_w2c,
                      color=CLR_PLACE, lw=2.0, label=lbl, alpha=0.85)

    u_o, v_o, z_o = project_world(orig_ctr[None], K, E_w2c)
    if z_o[0] > 0:
        ax_rgb.annotate(
            obj_name, xy=(u_o[0], v_o[0]),
            xytext=(u_o[0] + 20, v_o[0] - 25),
            color=CLR_ORIG, fontsize=9, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=CLR_ORIG, lw=1.2))

    for k, rep in enumerate(cluster_reps):
        yaw_idx = int(rep[2])
        T_placed = compute_placed_transform(
            T_obj2world, bbox3d, rep[:2], landing_z, yaw_data, yaw_idx, vp)
        p_ctr = transform_points(corners_obj, T_placed).mean(axis=0)
        u_p, v_p, z_p = project_world(p_ctr[None], K, E_w2c)
        if z_o[0] > 0 and z_p[0] > 0:
            ax_rgb.annotate(
                "", xy=(u_p[0], v_p[0]), xytext=(u_o[0], v_o[0]),
                arrowprops=dict(arrowstyle="->", color=CLR_ARROW,
                                lw=1.5, connectionstyle="arc3,rad=0.15"))
            yaw_deg = np.degrees(yaw_data["yaw_angles"][yaw_idx])
            ax_rgb.text(u_p[0] + 8, v_p[0] - 8,
                        f"#{k} ({yaw_deg:.0f}\u00b0)", color=CLR_PLACE,
                        fontsize=8, fontweight="bold")

    ax_rgb.set_xlim(0, img_w)
    ax_rgb.set_ylim(img_h, 0)
    ax_rgb.set_title("RGB Image \u2014 3-D Bbox Projection\n"
                     "Orange = current  |  Green = placements (with yaw)",
                     color="white", fontsize=11, pad=8)
    ax_rgb.legend(loc="upper right", fontsize=8,
                  facecolor=CLR_PANEL, labelcolor="white", framealpha=0.85)
    ax_rgb.axis("off")

    # ── Right panel: 3-D world view ──────────────────────────────────────
    ax3 = fig.add_axes([0.52, 0.04, 0.46, 0.92], projection="3d")
    ax3.set_facecolor(CLR_PANEL)

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

    ax3.scatter(*cam_origin, c=CLR_CAM, s=80, marker="*",
                zorder=10, label="Camera", depthshade=False)

    _draw_bbox_3d(ax3, orig_world, color=CLR_ORIG, lw=2.5,
                  label=f"Original: {obj_name}")

    for k, rep in enumerate(cluster_reps):
        yaw_idx = int(rep[2])
        T_placed = compute_placed_transform(
            T_obj2world, bbox3d, rep[:2], landing_z, yaw_data, yaw_idx, vp)
        placed_world = transform_points(corners_obj, T_placed)
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
                  f"Orange \u2192 {obj_name} (current)   "
                  f"Green \u2192 Placements",
                  color="white", fontsize=10, pad=6)
    ax3.legend(fontsize=8, loc="upper left",
               facecolor=CLR_PANEL, labelcolor="white",
               framealpha=0.85, markerscale=3)

    # ── Info text box ────────────────────────────────────────────────────
    lines = [f"Target  {obj_name}"]
    for k, ci in enumerate(cluster_infos):
        rep_k     = cluster_reps[k]
        yaw_idx_k = int(rep_k[2])
        T_pl = compute_placed_transform(
            T_obj2world, bbox3d, rep_k[:2], landing_z,
            yaw_data, yaw_idx_k, vp)
        pw = transform_points(corners_obj, T_pl)
        aabb_min, aabb_max = pw.min(0), pw.max(0)

        disp = pw.mean(0) - orig_ctr
        dist = float(np.linalg.norm(disp))

        lines.append(
            f"  #{k}: yaw={ci['yaw_degrees']:.0f}\u00b0  "
            f"\u0394=({disp[0]:+.1f}, {disp[1]:+.1f}, {disp[2]:+.1f}) cm  "
            f"|\u0394|={dist:.1f} cm  cluster={ci['size']}")
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
             f"Placement Planning \u2014 Sample Visualisation  ({obj_name})",
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
         dbscan_min_samples=DBSCAN_MIN_SAMPLES,
         yaw_steps=YAW_STEPS,
         min_support_ratio=MIN_SUPPORT_RATIO,
         occlusion_threshold=OCCLUSION_THRESHOLD,
         use_gpu=None):

    if data_dir is None:
        data_dir = DATA_DIR

    if use_gpu is None:
        use_gpu = HAS_GPU

    bar = "=" * 66
    print(f"\n{bar}")
    print("  HOPE \u00b7 3-D Placement Planner v2  (all objects)")
    print(f"  safety_margin    = {safety_margin_cm} cm")
    print(f"  dbscan_eps       = {dbscan_eps_cm} cm")
    print(f"  yaw_steps        = {yaw_steps}  ({360 / yaw_steps:.0f}\u00b0 per step)")
    print(f"  min_support      = {min_support_ratio}")
    print(f"  occlusion_thresh = {occlusion_threshold}")
    print(f"  GPU              = {'ON' if use_gpu else 'OFF'}"
          f"{'  (CuPy)' if use_gpu else ''}")
    print(f"  data_dir         = {data_dir}")
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
    grid_base = prepare_grid_base(grid, all_voxels)

    for oi, obj in enumerate(objects):
        name   = obj["class"]
        pose   = np.array(obj["pose"], dtype=np.float64)
        bbox3d = obj["bbox3d"]
        T      = all_T[oi]

        sep = "\u2500" * (50 - len(name))
        print(f"\u2500\u2500 [{oi + 1}/{n_obj}] {name} {sep}")

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

        # 2. In-place: remove target object from grid
        tv, saved = grid_remove_object(grid_base, target_vox)
        try:
            # 3. Detect support surface
            table_z, table_mask_2d = detect_support_surfaces(grid_base, vp)
            if table_z is None:
                print("   SKIP: no support surface detected")
                all_results[name] = {"status": "no_surface"}
                continue

            # 4. Build depth buffer (target already removed in grid_base)
            depth_buffer = build_depth_buffer(
                grid_base, vp, K, E_w2c, img_w, img_h)

            # 5. Collision search across (X, Y, theta)
            candidates, meta, yaw_data = find_table_placements(
                grid_base, bbox3d, T, table_z, margin_voxels, vp,
                yaw_steps=yaw_steps, use_gpu=use_gpu,
                table_mask_2d=table_mask_2d)
            landing_z = meta["landing_z"]
            n_raw     = meta["valid_raw"]

            print(f"   table_z={table_z}  landing_z={landing_z}  "
                  f"yaw_angles={meta['valid_yaw_angles']}/{yaw_steps}  "
                  f"raw={n_raw}")

            # 6. Physical stability
            candidates = filter_stable_placements(
                candidates, yaw_data, table_mask_2d,
                min_support_ratio=min_support_ratio)
            n_stable = len(candidates)
            print(f"   stable={n_stable}")

            # 7. Camera visibility
            candidates = filter_visible_placements(
                candidates, landing_z,
                bbox3d, T, E_w2c, K, img_w, img_h, vp, yaw_data)
            n_vis = len(candidates)
            print(f"   visible={n_vis}")

            # 8. Occlusion filter
            candidates = filter_occluded_placements(
                candidates, landing_z, bbox3d, T,
                depth_buffer, K, E_w2c, vp, yaw_data,
                img_w, img_h,
                occlusion_threshold=occlusion_threshold)
            n_unocc = len(candidates)
            print(f"   unoccluded={n_unocc}")

            # 9. Cluster
            reps, c_infos = cluster_placements(
                candidates, grid_base, yaw_data,
                landing_z, vp, dbscan_eps_cm, dbscan_min_samples)
            print(f"   clusters={len(reps)}")

            # 10. Visualise
            vis_path = os.path.join(data_dir, f"placement_{name}.png")
            save_placement_vis(
                rgb, name, bbox3d, T, K, E_w2c, vp,
                cam_origin, reps, c_infos,
                target_vox, grid, vis_path,
                yaw_data, landing_z)
            print(f"   [\u2713] {vis_path}")
        finally:
            grid_restore_object(grid_base, tv, saved)

        all_results[name] = {
            "status":        "ok",
            "table_z":       table_z,
            "landing_z":     landing_z,
            "raw_candidates": n_raw,
            "stable":        n_stable,
            "visible":       n_vis,
            "unoccluded":    n_unocc,
            "clusters":      c_infos,
        }

    # ── JSON summary ─────────────────────────────────────────────────────
    summary = {
        "safety_margin_cm":   safety_margin_cm,
        "dbscan_eps_cm":      dbscan_eps_cm,
        "yaw_steps":          yaw_steps,
        "min_support_ratio":  min_support_ratio,
        "occlusion_threshold": occlusion_threshold,
        "gpu_used":           use_gpu,
        "voxel_size_cm":      vs,
        "objects":            all_results,
    }
    json_path = os.path.join(data_dir, "placement_result.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[\u2713] JSON summary: {json_path}")
    print(f"{bar}\n")


if __name__ == "__main__":
    main()
