#!/usr/bin/env python3
"""
RGB-D → 3D Occupancy Grid + 3D Point Cloud via Ray-Casting (HOPE-Video dataset)

Voxel state definitions:
  FREE (0)     – ray marched through this voxel before hitting the surface
  OCCUPIED (1) – depth measurement hit, surface is present here
  UNKNOWN (2)  – never traversed by any ray (occluded or outside camera frustum)

Unit convention (HOPE-Video, per README errata):
  - World / object coordinates  : cm
  - Camera extrinsic translation: stored in m → corrected to cm (×100)
  - Depth pixel values           : raw uint16 mm × 0.98042517 ÷ 10 → cm
"""

import numpy as np
import json
import os
from pathlib import Path
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3-D projection)

# ═══════════════════════════════════════════════════════════════════════════════
#  Adjustable parameters
# ═══════════════════════════════════════════════════════════════════════════════

VOXEL_SIZE   = 1.0               # cm  ← tune as needed (e.g. 0.5 or 2.0)
DEPTH_SCALE  = 0.98042517 / 10.0 # raw uint16 (mm) → cm
PIXEL_STRIDE = 4                 # sample 1 pixel per N×N block (speed vs density)
GRID_PADDING = 10.0              # cm extra margin around the scene bounding box

FREE, OCCUPIED, UNKNOWN = 0, 1, 2

# Colour palette:  Free=green  Occupied=red  Unknown=grey
PALETTE = ["#4CAF50", "#F44336", "#9E9E9E"]

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_data")


# ═══════════════════════════════════════════════════════════════════════════════
#  Voxel coordinate utilities
# ═══════════════════════════════════════════════════════════════════════════════

def make_voxel_params(grid_min: np.ndarray, voxel_size: float) -> dict:
    """
    Build a voxel parameter dict used for coordinate ↔ index conversion.

    Schema
    ------
    {
        "voxel_size": float,          # edge length of one voxel (cm)
        "origin":     [x, y, z]       # world-space corner of voxel (0,0,0) (cm)
    }

    Conversion rules
    ----------------
    world → index : i = floor((p - origin) / voxel_size)   (integer, truncated)
    index → world : p = origin + (i + 0.5) * voxel_size    (voxel centre)
    """
    return {
        "voxel_size": float(voxel_size),
        "origin":     grid_min.tolist(),   # [x_min, y_min, z_min] in cm
    }


def world_to_voxel(points: np.ndarray, voxel_params: dict) -> np.ndarray:
    """
    Convert world-space coordinates (cm) to integer voxel indices (i, j, k).

    Parameters
    ----------
    points       : (..., 3) float array  — world coordinates in cm
    voxel_params : dict from make_voxel_params()

    Returns
    -------
    indices : (..., 3) int array  — voxel indices (may be out-of-grid-bounds)
    """
    origin     = np.asarray(voxel_params["origin"],     dtype=np.float64)
    voxel_size = float(voxel_params["voxel_size"])
    return np.floor((np.asarray(points, dtype=np.float64) - origin)
                    / voxel_size).astype(int)


def voxel_to_world(indices: np.ndarray, voxel_params: dict) -> np.ndarray:
    """
    Convert voxel indices (i, j, k) to the world-space centre of that voxel (cm).

    Parameters
    ----------
    indices      : (..., 3) int array   — voxel indices
    voxel_params : dict from make_voxel_params()

    Returns
    -------
    centres : (..., 3) float array  — world coordinates of voxel centres (cm)
    """
    origin     = np.asarray(voxel_params["origin"],     dtype=np.float64)
    voxel_size = float(voxel_params["voxel_size"])
    return origin + (np.asarray(indices, dtype=np.float64) + 0.5) * voxel_size


# ═══════════════════════════════════════════════════════════════════════════════
#  I/O helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_scene(data_dir: str):
    d = Path(data_dir)
    with open(d / "0000.json") as f:
        annots = json.load(f)
    rgb   = np.asarray(Image.open(d / "0000_rgb.jpg"),   dtype=np.uint8)
    depth = np.asarray(Image.open(d / "0000_depth.png"), dtype=np.float32)
    return annots, rgb, depth


def parse_camera(annots: dict):
    """
    Returns (fx, fy, cx, cy, R_c2w, cam_origin_cm).
    Applies README errata: extrinsics translation is in m, converted to cm.
    Extrinsics convention: world-to-camera → invert to get camera-to-world.
    """
    K  = np.array(annots["camera"]["intrinsics"], dtype=np.float64)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    E_w2c = np.array(annots["camera"]["extrinsics"], dtype=np.float64)
    E_w2c[:3, -1] *= 100          # README errata: m → cm
    E_c2w = np.linalg.inv(E_w2c)

    R_c2w      = E_c2w[:3, :3]    # rotation: camera → world
    cam_origin = E_c2w[:3,  3]    # camera centre in world (cm)
    return fx, fy, cx, cy, R_c2w, cam_origin


def save_ply(path: str, points: np.ndarray, colors: np.ndarray):
    """Save a coloured point cloud as ASCII PLY."""
    n = len(points)
    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    data = np.hstack([points.astype(np.float32), colors.astype(np.float32)])
    with open(path, "w") as f:
        f.write(header)
        np.savetxt(f, data, fmt="%.4f %.4f %.4f %d %d %d")
    print(f"[✓] Point cloud saved   : {path}  ({n:,} pts)")


# ═══════════════════════════════════════════════════════════════════════════════
#  Point cloud (depth un-projection)
# ═══════════════════════════════════════════════════════════════════════════════

def depth_to_pointcloud(depth_raw, rgb, fx, fy, cx, cy,
                        R_c2w, cam_origin, stride):
    """
    Back-project every (stride)-th valid pixel into world-space 3-D points.
    Returns (pts_world [N,3], colors [N,3]).
    """
    h, w = depth_raw.shape
    dcm  = depth_raw * DEPTH_SCALE

    u_arr  = np.arange(0, w, stride)
    v_arr  = np.arange(0, h, stride)
    uu, vv = np.meshgrid(u_arr, v_arr)
    uu, vv = uu.ravel(), vv.ravel()

    d    = dcm[vv, uu]
    mask = d > 0
    uu, vv, d = uu[mask], vv[mask], d[mask]

    # Camera-space coordinates (pinhole model, depth = Z)
    pts_cam = np.stack([
        (uu - cx) / fx * d,
        (vv - cy) / fy * d,
        d
    ], axis=1)                                          # (N, 3)

    # Camera → world
    pts_world = (R_c2w @ pts_cam.T).T + cam_origin     # (N, 3)
    colors    = rgb[vv, uu].astype(np.uint8)            # (N, 3)
    return pts_world, colors


# ═══════════════════════════════════════════════════════════════════════════════
#  Occupancy grid via ray-casting
# ═══════════════════════════════════════════════════════════════════════════════

def build_occupancy_grid(depth_raw, fx, fy, cx, cy, R_c2w, cam_origin,
                         voxel_size=VOXEL_SIZE, stride=PIXEL_STRIDE):
    """
    Builds a 3-D occupancy grid using ray-casting.

    Algorithm
    ---------
    For every sampled pixel:
      1. Compute the surface point P_world from the depth value.
      2. Cast a ray from the camera origin toward P_world.
      3. Each voxel the ray passes through *before* P_world is marked FREE.
      4. The voxel containing P_world is marked OCCUPIED.
      5. Voxels never reached by any ray remain UNKNOWN.

    State priority:  OCCUPIED > FREE > UNKNOWN
    (a voxel already marked OCCUPIED is never downgraded to FREE)
    """
    h, w  = depth_raw.shape
    dcm   = depth_raw * DEPTH_SCALE
    step  = voxel_size * 0.5              # sub-voxel step → denser coverage

    # ── Sampled pixel grid ────────────────────────────────────────────────────
    u_arr  = np.arange(0, w, stride)
    v_arr  = np.arange(0, h, stride)
    uu, vv = np.meshgrid(u_arr, v_arr)
    uu, vv = uu.ravel(), vv.ravel()

    d    = dcm[vv, uu]
    mask = d > 0
    uu, vv, d = uu[mask], vv[mask], d[mask]   # (N,)

    # ── Surface points in world space ─────────────────────────────────────────
    pts_cam = np.stack([
        (uu - cx) / fx * d,
        (vv - cy) / fy * d,
        d
    ], axis=1)                                          # (N, 3)
    surface = (R_c2w @ pts_cam.T).T + cam_origin        # (N, 3) world frame

    # ── Grid initialisation ───────────────────────────────────────────────────
    all_pts   = np.vstack([surface, cam_origin[None]])
    grid_min  = all_pts.min(0) - GRID_PADDING
    grid_max  = all_pts.max(0) + GRID_PADDING
    grid_shape = np.ceil((grid_max - grid_min) / voxel_size).astype(int)

    print(f"  Grid min  (cm) : {grid_min.round(1)}")
    print(f"  Grid max  (cm) : {grid_max.round(1)}")
    print(f"  Grid shape     : {tuple(grid_shape)}  "
          f"({grid_shape.prod():,} voxels total)")
    print(f"  Active rays    : {len(d):,}")

    grid = np.full(tuple(grid_shape), UNKNOWN, dtype=np.uint8)

    # ── Ray vectors (camera origin → surface, world frame) ───────────────────
    ray_vecs  = surface - cam_origin                    # (N, 3)
    ray_lens  = np.linalg.norm(ray_vecs, axis=1)        # (N,)  Euclidean dist
    ray_dirs  = ray_vecs / ray_lens[:, None]            # (N, 3) unit vectors

    # ── Helper functions ──────────────────────────────────────────────────────
    def to_idx(pts):
        """Convert world-space points (cm) to integer voxel indices."""
        return np.floor((pts - grid_min) / voxel_size).astype(int)

    def in_bounds(idx):
        """Boolean mask for voxel indices within grid bounds."""
        return (
            (idx[:, 0] >= 0) & (idx[:, 0] < grid_shape[0]) &
            (idx[:, 1] >= 0) & (idx[:, 1] < grid_shape[1]) &
            (idx[:, 2] >= 0) & (idx[:, 2] < grid_shape[2])
        )

    # ── Step along rays → mark FREE ───────────────────────────────────────────
    # t ranges [0, max_ray_length) in cm; step = voxel_size/2
    max_t  = float(ray_lens.max())
    t_vals = np.arange(0.0, max_t, step)
    n_t    = len(t_vals)
    print(f"  Max ray length : {max_t:.1f} cm  |  ray steps : {n_t}")

    for k, t in enumerate(t_vals):
        # Only advance rays that have not yet reached their surface
        active = ray_lens > t           # (N,) boolean
        if not active.any():
            break

        pts = cam_origin + t * ray_dirs[active]   # (M, 3)
        idx = to_idx(pts)                          # (M, 3)
        ok  = in_bounds(idx)
        i   = idx[ok]                              # valid in-bounds indices

        # State-priority: only update UNKNOWN voxels to FREE
        # (OCCUPIED voxels are never downgraded)
        cur = grid[i[:, 0], i[:, 1], i[:, 2]]
        sel = i[cur == UNKNOWN]
        if len(sel):
            grid[sel[:, 0], sel[:, 1], sel[:, 2]] = FREE

        if k % 40 == 0:
            print(f"    step {k:4d}/{n_t}  t={t:6.1f} cm  "
                  f"active_rays={active.sum():,}")

    # ── Mark surface voxels → OCCUPIED ───────────────────────────────────────
    # OCCUPIED unconditionally overrides FREE/UNKNOWN (highest priority)
    sidx = to_idx(surface)
    ok   = in_bounds(sidx)
    s    = sidx[ok]
    grid[s[:, 0], s[:, 1], s[:, 2]] = OCCUPIED

    total  = grid.size
    counts = {
        "free":     int((grid == FREE).sum()),
        "occupied": int((grid == OCCUPIED).sum()),
        "unknown":  int((grid == UNKNOWN).sum()),
    }
    print(
        f"  FREE={counts['free']:,} ({counts['free']/total*100:.1f}%)  "
        f"OCCUPIED={counts['occupied']:,} ({counts['occupied']/total*100:.1f}%)  "
        f"UNKNOWN={counts['unknown']:,} ({counts['unknown']/total*100:.1f}%)"
    )
    return grid, grid_min, grid_max, counts


# ═══════════════════════════════════════════════════════════════════════════════
#  Occupancy grid → point cloud PLY
# ═══════════════════════════════════════════════════════════════════════════════

# RGB colours assigned to each voxel state (uint8)
STATE_COLORS = {
    FREE:     np.array([76,  175,  80], dtype=np.uint8),   # green
    OCCUPIED: np.array([244,  67,  54], dtype=np.uint8),   # red
    UNKNOWN:  np.array([158, 158, 158], dtype=np.uint8),   # grey
}


def save_occupancy_ply(path: str, grid: np.ndarray, grid_min: np.ndarray,
                       voxel_size: float,
                       include_unknown: bool = False,
                       states: list = None):
    """
    Export the occupancy grid as a coloured point cloud (voxel centres).

    Parameters
    ----------
    path            : output .ply file path
    grid            : uint8 array of shape (Nx, Ny, Nz), values 0/1/2
    grid_min        : world-space origin of the grid (cm)
    voxel_size      : edge length of each voxel (cm)
    include_unknown : if True, also write UNKNOWN voxels (grey); can be large
    states          : explicit list of states to export (overrides include_unknown
                      when provided).  E.g. ``[OCCUPIED]`` for occupied-only.
    """
    if states is not None:
        states_to_export = list(states)
    else:
        states_to_export = [FREE, OCCUPIED]
        if include_unknown:
            states_to_export.append(UNKNOWN)

    all_pts, all_cols = [], []
    for state in states_to_export:
        idx = np.argwhere(grid == state)          # (M, 3) voxel indices
        if len(idx) == 0:
            continue
        # voxel centre = grid_min + (index + 0.5) * voxel_size
        centres = grid_min + (idx + 0.5) * voxel_size   # (M, 3)
        colour  = np.tile(STATE_COLORS[state], (len(idx), 1))
        all_pts.append(centres)
        all_cols.append(colour)

    if not all_pts:
        print("[!] No voxels to export.")
        return

    pts  = np.vstack(all_pts).astype(np.float32)
    cols = np.vstack(all_cols).astype(np.uint8)
    n    = len(pts)

    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    data = np.hstack([pts, cols.astype(np.float32)])
    with open(path, "w") as f:
        f.write(header)
        np.savetxt(f, data, fmt="%.4f %.4f %.4f %d %d %d")

    state_names = {FREE: "Free", OCCUPIED: "Occupied", UNKNOWN: "Unknown"}
    label = "+".join(state_names.get(s, str(s)) for s in states_to_export)
    print(f"[✓] Occupancy PLY saved : {path}  ({n:,} pts, {label})")


# ═══════════════════════════════════════════════════════════════════════════════
#  Visualisation
# ═══════════════════════════════════════════════════════════════════════════════

def visualize_overview(rgb, depth_raw, pts_world, colors, out_path):
    """
    4-panel overview: RGB | depth map | top-view (XZ) | side-view (XY) point cloud.
    """
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle("Scene Overview", fontsize=13)

    # 1. RGB
    axes[0].imshow(rgb)
    axes[0].set_title("RGB Image")
    axes[0].axis("off")

    # 2. Depth map (display in cm)
    dcm = depth_raw * DEPTH_SCALE
    im  = axes[1].imshow(dcm, cmap="plasma", interpolation="nearest")
    axes[1].set_title("Depth Map (cm)")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="cm")

    # 3. Point cloud top-view (X–Z plane)
    axes[2].scatter(pts_world[:, 0], pts_world[:, 2],
                    c=colors / 255.0, s=0.5, alpha=0.6)
    axes[2].set_xlabel("X (cm)"); axes[2].set_ylabel("Z (cm)")
    axes[2].set_title("Point Cloud — Top View (XZ)")
    axes[2].set_aspect("equal")

    # 4. Point cloud side-view (X–Y plane)
    axes[3].scatter(pts_world[:, 0], pts_world[:, 1],
                    c=colors / 255.0, s=0.5, alpha=0.6)
    axes[3].set_xlabel("X (cm)"); axes[3].set_ylabel("Y (cm)")
    axes[3].set_title("Point Cloud — Side View (XY)")
    axes[3].set_aspect("equal")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[✓] Overview saved      : {out_path}")


def visualize_slices(grid, grid_min, voxel_size, out_path):
    """
    Three orthogonal slices through the centroid of occupied voxels.
    Colour: Green=Free  Red=Occupied  Grey=Unknown
    """
    occ = np.argwhere(grid == OCCUPIED)
    if len(occ) == 0:
        print("[!] No occupied voxels — skipping slice plot")
        return

    cx_v, cy_v, cz_v = occ.mean(0).astype(int)
    cmap = ListedColormap(PALETTE)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Occupancy Grid — Orthogonal Slices\n"
        "■ Green = Free   ■ Red = Occupied   ■ Grey = Unknown",
        fontsize=12)

    panels = [
        (grid[:, :, cz_v].T,
         f"XY slice  z = {grid_min[2] + cz_v * voxel_size:.1f} cm",
         "X (cm)", "Y (cm)"),
        (grid[:, cy_v, :].T,
         f"XZ slice  y = {grid_min[1] + cy_v * voxel_size:.1f} cm",
         "X (cm)", "Z (cm)"),
        (grid[cx_v, :, :].T,
         f"YZ slice  x = {grid_min[0] + cx_v * voxel_size:.1f} cm",
         "Y (cm)", "Z (cm)"),
    ]

    for ax, (img, title, xl, yl) in zip(axes, panels):
        im = ax.imshow(img, origin="lower", cmap=cmap, vmin=0, vmax=2,
                       interpolation="nearest", aspect="auto")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xl); ax.set_ylabel(yl)

    plt.colorbar(im, ax=axes[-1], ticks=[0, 1, 2],
                 label="0 = Free  |  1 = Occupied  |  2 = Unknown")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[✓] Slices saved        : {out_path}")


def visualize_3d(grid, grid_min, voxel_size, out_path, max_pts=6000):
    """
    3-D scatter: Occupied (red, all) + sampled Free (green, up to max_pts).
    Unknown voxels are omitted to keep the plot readable.
    """
    occ  = np.argwhere(grid == OCCUPIED)
    free = np.argwhere(grid == FREE)

    rng = np.random.default_rng(42)
    if len(free) > max_pts:
        free = free[rng.choice(len(free), max_pts, replace=False)]
    if len(occ)  > max_pts:
        occ  = occ[rng.choice(len(occ),  max_pts, replace=False)]

    def to_world(idx):
        return grid_min + idx * voxel_size

    fig = plt.figure(figsize=(12, 9))
    ax  = fig.add_subplot(111, projection="3d")

    if len(free):
        fw = to_world(free)
        ax.scatter(fw[:, 0], fw[:, 1], fw[:, 2],
                   c=PALETTE[0], s=1, alpha=0.15, label=f"Free ({len(free):,})")
    if len(occ):
        ow = to_world(occ)
        ax.scatter(ow[:, 0], ow[:, 1], ow[:, 2],
                   c=PALETTE[1], s=6, alpha=0.9,  label=f"Occupied ({len(occ):,})")

    ax.set_xlabel("X (cm)"); ax.set_ylabel("Y (cm)"); ax.set_zlabel("Z (cm)")
    ax.set_title("3D Occupancy Grid — Ray-Casting Result\n"
                 "Green = Free  |  Red = Occupied  |  Unknown not shown",
                 fontsize=11)
    ax.legend(markerscale=5, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[✓] 3-D scatter saved   : {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main(data_dir=None, voxel_size=VOXEL_SIZE, pixel_stride=PIXEL_STRIDE):
    if data_dir is None:
        data_dir = DATA_DIR

    bar = "=" * 62
    print(f"\n{bar}")
    print(f"  HOPE · Occupancy Grid Builder")
    print(f"  voxel_size   = {voxel_size} cm")
    print(f"  pixel_stride = {pixel_stride}")
    print(f"  data_dir     = {data_dir}")
    print(f"{bar}\n")

    # ── Load ──────────────────────────────────────────────────────────────────
    annots, rgb, depth_raw = load_scene(data_dir)
    fx, fy, cx, cy, R_c2w, cam_origin = parse_camera(annots)
    print(f"Camera origin in world (cm): {cam_origin.round(2)}\n")

    # ── Point cloud ───────────────────────────────────────────────────────────
    print("Generating point cloud …")
    pts, colors = depth_to_pointcloud(
        depth_raw, rgb, fx, fy, cx, cy, R_c2w, cam_origin, pixel_stride)
    save_ply(os.path.join(data_dir, "point_cloud.ply"), pts, colors)

    # ── Overview visualisation ────────────────────────────────────────────────
    visualize_overview(rgb, depth_raw, pts, colors,
                       os.path.join(data_dir, "scene_overview.png"))

    # ── Occupancy grid ────────────────────────────────────────────────────────
    print("\nBuilding occupancy grid …")
    grid, grid_min, grid_max, counts = build_occupancy_grid(
        depth_raw, fx, fy, cx, cy, R_c2w, cam_origin,
        voxel_size=voxel_size, stride=pixel_stride)

    grid_path = os.path.join(data_dir, "occupancy_grid.npy")
    np.save(grid_path, grid)
    print(f"[✓] Grid saved          : {grid_path}")

    # Export occupancy grid as coloured point cloud (Free=green, Occupied=red)
    save_occupancy_ply(
        os.path.join(data_dir, "occupancy_grid.ply"),
        grid, grid_min, voxel_size,
        include_unknown=False)
    # With UNKNOWN voxels included (larger file)
    save_occupancy_ply(
        os.path.join(data_dir, "occupancy_grid_full.ply"),
        grid, grid_min, voxel_size,
        include_unknown=True)
    # Occupied voxels only (smallest file; useful for obstacle visualisation)
    save_occupancy_ply(
        os.path.join(data_dir, "occupied_only.ply"),
        grid, grid_min, voxel_size,
        states=[OCCUPIED])

    # ── Voxel params (coordinate ↔ index conversion) ──────────────────────────
    vp = make_voxel_params(grid_min, voxel_size)
    print(f"\nvoxel_params = {vp}")

    # Quick round-trip sanity check
    _test_pt  = np.array([grid_min + np.array(grid.shape) / 2 * voxel_size])
    _test_ijk = world_to_voxel(_test_pt, vp)
    _test_rec = voxel_to_world(_test_ijk, vp)
    print(f"  round-trip check  world→voxel→world  Δ = "
          f"{np.abs(_test_pt - _test_rec).max():.4f} cm  (should be < voxel_size/2)")

    meta = {
        "voxel_params":   vp,                          # ← primary lookup dict
        "voxel_size_cm":  voxel_size,                  # redundant, kept for compat
        "grid_min_cm":    grid_min.tolist(),
        "grid_max_cm":    grid_max.tolist(),
        "grid_shape":     list(map(int, grid.shape)),
        "states":         {"0": "Free", "1": "Occupied", "2": "Unknown"},
        "voxel_counts":   counts,
        "depth_scale":    DEPTH_SCALE,
        "pixel_stride":   pixel_stride,
    }
    meta_path = os.path.join(data_dir, "grid_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[✓] Metadata saved      : {meta_path}")

    # ── Grid visualisations ───────────────────────────────────────────────────
    print("\nGenerating visualisations …")
    visualize_slices(grid, grid_min, voxel_size,
                     os.path.join(data_dir, "occupancy_slices.png"))
    visualize_3d(grid, grid_min, voxel_size,
                 os.path.join(data_dir, "occupancy_3d.png"))

    # ── Summary ───────────────────────────────────────────────────────────────
    outputs = [
        "point_cloud.ply",
        "occupancy_grid.npy",
        "occupancy_grid.ply",
        "occupancy_grid_full.ply",
        "occupied_only.ply",
        "grid_meta.json",
        "scene_overview.png",
        "occupancy_slices.png",
        "occupancy_3d.png",
    ]
    print(f"\n{bar}")
    print("Output files in test_data/:")
    for fname in outputs:
        fp = os.path.join(data_dir, fname)
        if os.path.exists(fp):
            print(f"  {fname:<30s}  {os.path.getsize(fp) / 1024:8.1f} KB")
    print(bar + "\n")


if __name__ == "__main__":
    main()
