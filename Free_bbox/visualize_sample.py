#!/usr/bin/env python3
"""
Visualise one sampled placement bbox overlaid on the original RGB image.

Layout (saved as placement_sample_vis.png):
  Left  : RGB image with the original object's 3-D bbox (orange) and the
           selected placement bbox (green) both projected onto the image.
  Right : 3-D world-space view of the same two boxes, plus the point cloud.

The script picks the median-indexed placement for TomatoSauce so the
selected position is representative rather than an extremal one.
"""

import sys, json, os
import numpy as np
from pathlib import Path
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

ROOT     = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "test_data"
OUT_DIR  = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT))
from build_occupancy import world_to_voxel, voxel_to_world, FREE, OCCUPIED, UNKNOWN

# ─── Configuration ─────────────────────────────────────────────────────────────
TARGET_CLASS  = "TomatoSauce"   # which object to visualise
SAMPLE_IDX    = None            # None → pick the median placement automatically
W, H          = 640, 480

# ─── Helpers ───────────────────────────────────────────────────────────────────

def load_everything():
    with open(DATA_DIR / "meta_data.json") as f:
        meta_ann = json.load(f)
    with open(DATA_DIR / "grid_meta.json") as f:
        grid_meta = json.load(f)
    with open(OUT_DIR / "placement_results.json") as f:
        results = json.load(f)

    grid  = np.load(DATA_DIR / "occupancy_grid.npy")
    rgb   = np.asarray(Image.open(DATA_DIR / "0000_rgb.jpg"), dtype=np.uint8)

    K     = np.array(meta_ann["camera"]["intrinsics"], dtype=np.float64)
    E_w2c = np.array(meta_ann["camera"]["extrinsics"], dtype=np.float64)
    E_w2c[:3, -1] *= 100
    E_c2w = np.linalg.inv(E_w2c)

    vp = grid_meta["voxel_params"]
    return meta_ann, grid_meta, grid, rgb, K, E_w2c, E_c2w, results, vp


def aabb_corners(aabb):
    """8 corners of an AABB given as [xmin,ymin,zmin, xmax,ymax,zmax]."""
    mn, mx = np.array(aabb[:3]), np.array(aabb[3:])
    return np.array([
        [mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]],
        [mx[0], mx[1], mn[2]], [mn[0], mx[1], mn[2]],
        [mn[0], mn[1], mx[2]], [mx[0], mn[1], mx[2]],
        [mx[0], mx[1], mx[2]], [mn[0], mx[1], mx[2]],
    ])  # (8, 3)


EDGES = [(0,1),(1,2),(2,3),(3,0),
         (4,5),(5,6),(6,7),(7,4),
         (0,4),(1,5),(2,6),(3,7)]


def project_points(pts_world, K, E_w2c):
    """
    Project Nx3 world points onto the image.
    Returns (u, v, z_cam) arrays; z_cam used to skip behind-camera points.
    """
    pts_h   = np.hstack([pts_world, np.ones((len(pts_world), 1))])  # (N,4)
    pts_cam = (E_w2c @ pts_h.T).T[:, :3]                           # (N,3)
    fx, fy  = K[0, 0], K[1, 1]
    cx, cy  = K[0, 2], K[1, 2]
    z = pts_cam[:, 2]
    # Avoid division by zero for behind-camera points
    safe_z  = np.where(z > 0, z, np.nan)
    u = pts_cam[:, 0] / safe_z * fx + cx
    v = pts_cam[:, 1] / safe_z * fy + cy
    return u, v, z


def draw_bbox_on_ax(ax, corners_world, K, E_w2c, color, lw=2.0, label=None,
                    alpha=1.0):
    """
    Draw the 12 edges of a 3-D bbox onto a 2-D image axis.
    Edges where any endpoint is behind the camera (z≤0) are skipped.
    """
    u, v, z = project_points(corners_world, K, E_w2c)
    drawn = False
    for idx, (i, j) in enumerate(EDGES):
        if z[i] <= 0 or z[j] <= 0:
            continue
        lbl = label if (not drawn and label) else None
        ax.plot([u[i], u[j]], [v[i], v[j]],
                color=color, lw=lw, alpha=alpha, label=lbl)
        drawn = True


def draw_bbox_3d(ax, corners_world, color, lw=1.5, label=None, alpha=1.0):
    """Draw 12 edges of a 3-D bbox on a 3-D axes."""
    for idx, (i, j) in enumerate(EDGES):
        lbl = label if idx == 0 else None
        ax.plot([corners_world[i,0], corners_world[j,0]],
                [corners_world[i,1], corners_world[j,1]],
                [corners_world[i,2], corners_world[j,2]],
                color=color, lw=lw, alpha=alpha, label=lbl)


def obb_corners_world(obj, E_c2w):
    from build_occupancy import world_to_voxel, voxel_to_world
    pose_cam  = np.array(obj["pose"], dtype=np.float64)
    pose_world = E_c2w @ pose_cam
    bd = obj["bbox3d"]
    xs, ys, zs = [bd[0], bd[3]], [bd[1], bd[4]], [bd[2], bd[5]]
    corners_local = np.array([[x,y,z,1.] for x in xs for y in ys for z in zs])
    return (pose_world @ corners_local.T).T[:, :3]


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    (meta_ann, grid_meta, grid, rgb, K,
     E_w2c, E_c2w, results, vp) = load_everything()

    # ── Find the target object result ─────────────────────────────────────────
    target_result = next(
        (r for r in results if r["target_obj"] == TARGET_CLASS), None)
    if target_result is None or target_result["n_placements"] == 0:
        print(f"[!] No placements found for {TARGET_CLASS}.")
        return

    placements = target_result["placements"]
    n          = len(placements)
    idx        = n // 2 if SAMPLE_IDX is None else SAMPLE_IDX
    sample     = placements[idx]
    print(f"Visualising {TARGET_CLASS}  placement [{idx}/{n}]")
    print(f"  Original obj AABB  : (see below)")
    print(f"  Placement AABB     : {[round(x,1) for x in sample['aabb_world_cm']]}")
    print(f"  Placement center   : {[round(x,1) for x in sample['center_world_cm']]}")

    # ── Object geometry ───────────────────────────────────────────────────────
    obj_lookup    = {o["class"]: o for o in meta_ann["objects"]}
    obj           = obj_lookup[TARGET_CLASS]
    orig_corners  = obb_corners_world(obj, E_c2w)               # (8,3) world
    orig_aabb     = orig_corners.min(0).tolist() + orig_corners.max(0).tolist()
    place_corners = aabb_corners(sample["aabb_world_cm"])        # (8,3) world

    # ── Build figure ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 7))
    fig.patch.set_facecolor("#1A1A2E")

    # ─── Panel 1: RGB image with projected bboxes ─────────────────────────────
    ax_rgb = fig.add_axes([0.02, 0.06, 0.47, 0.88])   # [left, bottom, w, h]
    ax_rgb.imshow(rgb)

    # Original object bbox — orange, thick
    draw_bbox_on_ax(ax_rgb, orig_corners, K, E_w2c,
                    color="#FF6D00", lw=2.5,
                    label=f"Original: {TARGET_CLASS}")

    # Placement bbox — cyan-green, dashed
    draw_bbox_on_ax(ax_rgb, place_corners, K, E_w2c,
                    color="#00E676", lw=2.0,
                    label=f"Placement #{idx} (of {n})")

    # Annotate original center
    u_o, v_o, z_o = project_points(orig_corners.mean(0, keepdims=True), K, E_w2c)
    if z_o[0] > 0:
        ax_rgb.annotate(
            TARGET_CLASS,
            xy=(u_o[0], v_o[0]), xytext=(u_o[0]+20, v_o[0]-25),
            color="#FF6D00", fontsize=9, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#FF6D00", lw=1.2)
        )

    # Annotate placement center
    u_p, v_p, z_p = project_points(
        np.array([sample["center_world_cm"]]), K, E_w2c)
    if z_p[0] > 0:
        ax_rgb.annotate(
            f"Placement\n#{idx}",
            xy=(u_p[0], v_p[0]), xytext=(u_p[0]+25, v_p[0]+30),
            color="#00E676", fontsize=9, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#00E676", lw=1.2)
        )

    ax_rgb.set_xlim(0, W); ax_rgb.set_ylim(H, 0)
    ax_rgb.set_title("RGB Image — 3-D Bbox Projection\n"
                     "Orange = current position  |  Green = sample placement",
                     color="white", fontsize=11, pad=8)
    ax_rgb.legend(loc="upper right", fontsize=8,
                  facecolor="#0D0D1A", labelcolor="white", framealpha=0.85)
    ax_rgb.axis("off")

    # ─── Panel 2: 3-D world-space view ───────────────────────────────────────
    ax3d = fig.add_axes([0.52, 0.04, 0.46, 0.92], projection="3d")
    ax3d.set_facecolor("#0D0D1A")

    # Background point cloud (occupied + free)
    rng      = np.random.default_rng(42)
    occ_idx  = np.argwhere(grid == OCCUPIED)
    free_idx = np.argwhere(grid == FREE)
    if len(occ_idx)  > 3000: occ_idx  = occ_idx[rng.choice(len(occ_idx),  3000, replace=False)]
    if len(free_idx) > 2000: free_idx = free_idx[rng.choice(len(free_idx), 2000, replace=False)]
    ow = voxel_to_world(occ_idx,  vp)
    fw = voxel_to_world(free_idx, vp)
    if len(fw): ax3d.scatter(fw[:,0], fw[:,1], fw[:,2], c="#4CAF50", s=1,
                              alpha=0.10, depthshade=False)
    if len(ow): ax3d.scatter(ow[:,0], ow[:,1], ow[:,2], c="#78909C", s=2,
                              alpha=0.40, depthshade=False)

    # Camera position
    cam_w = -np.linalg.inv(E_w2c[:3,:3]) @ E_w2c[:3, 3]
    ax3d.scatter(*cam_w, c="#FFEB3B", s=80, marker="*", zorder=10,
                 label="Camera", depthshade=False)

    # Original object bbox — orange
    draw_bbox_3d(ax3d, orig_corners,  color="#FF6D00", lw=2.5,
                 label=f"Original: {TARGET_CLASS}")

    # Placement bbox — green
    draw_bbox_3d(ax3d, place_corners, color="#00E676", lw=2.0,
                 label=f"Placement #{idx}")

    # Arrow: original → placement
    orig_ctr   = np.array(orig_corners.mean(0))
    place_ctr  = np.array(sample["center_world_cm"])
    ax3d.quiver(orig_ctr[0],  orig_ctr[1],  orig_ctr[2],
                *(place_ctr - orig_ctr),
                color="#FFD54F", lw=1.5, arrow_length_ratio=0.15,
                label="displacement")

    ax3d.set_xlabel("X (cm)", color="white", fontsize=8)
    ax3d.set_ylabel("Y (cm)", color="white", fontsize=8)
    ax3d.set_zlabel("Z (cm)", color="white", fontsize=8)
    ax3d.tick_params(colors="white", labelsize=7)
    ax3d.xaxis.pane.fill = False
    ax3d.yaxis.pane.fill = False
    ax3d.zaxis.pane.fill = False
    ax3d.xaxis.pane.set_edgecolor("#333355")
    ax3d.yaxis.pane.set_edgecolor("#333355")
    ax3d.zaxis.pane.set_edgecolor("#333355")
    ax3d.set_title(f"3-D World View\n"
                   f"Orange → {TARGET_CLASS} (current)   "
                   f"Green → Placement #{idx}",
                   color="white", fontsize=10, pad=6)
    legend = ax3d.legend(fontsize=8, loc="upper left",
                          facecolor="#0D0D1A", labelcolor="white",
                          framealpha=0.85, markerscale=3)

    # ── Displacement info box ─────────────────────────────────────────────────
    delta   = place_ctr - orig_ctr
    disp_cm = np.linalg.norm(delta)
    info    = (f"Target  : {TARGET_CLASS}\n"
               f"Placement: #{idx}  (of {n} valid)\n"
               f"Δ = ({delta[0]:+.1f}, {delta[1]:+.1f}, {delta[2]:+.1f}) cm\n"
               f"|Δ| = {disp_cm:.1f} cm\n\n"
               f"Placement AABB (world, cm):\n"
               f"  x: [{sample['aabb_world_cm'][0]:.1f}, {sample['aabb_world_cm'][3]:.1f}]\n"
               f"  y: [{sample['aabb_world_cm'][1]:.1f}, {sample['aabb_world_cm'][4]:.1f}]\n"
               f"  z: [{sample['aabb_world_cm'][2]:.1f}, {sample['aabb_world_cm'][5]:.1f}]")
    fig.text(0.535, 0.03, info,
             fontsize=8.5, color="white", family="monospace",
             verticalalignment="bottom",
             bbox=dict(facecolor="#0D0D1A", edgecolor="#444466",
                       boxstyle="round,pad=0.5", alpha=0.9))

    # ── Title ─────────────────────────────────────────────────────────────────
    fig.text(0.5, 0.97,
             f"Placement Planning — Sample Visualisation  ({TARGET_CLASS})",
             ha="center", va="top", color="white", fontsize=13, fontweight="bold")

    out_path = OUT_DIR / "placement_sample_vis.png"
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved: {out_path}  ({out_path.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()