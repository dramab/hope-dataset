#!/usr/bin/env python3
"""
Batch processing pipeline for HOPE-Video dataset.

Processes scenes from hope_video/ through 3 steps:
  Step 1 (3D_bbox)   : Compute per-object bbox3d from mesh + visualize on RGB
  Step 2 (occupancy) : Build 3D occupancy grid from RGB-D via ray-casting
  Step 3 (placement) : Plan collision-free placements on table surface

Output directory structure:
  output/
  └── scene_XXXX/
      ├── 3D_bbox/
      │   └── {frame_id}/   -> meta_data.json, bbox_3d_vis.jpg
      ├── occupancy/
      │   └── {frame_id}/   -> occupancy_grid.npy, grid_meta.json, *.ply, *.png
      └── placement/
          └── {frame_id}/   -> placement_result.json, placement_{Obj}/placement_vis.png

Usage:
  conda activate hope
  python batch_process.py
"""

import sys
import os
import gc
import json
import time
import math
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon

import trimesh

# ═══════════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════════

ROOT_DIR       = Path(__file__).resolve().parent
HOPE_VIDEO_DIR = ROOT_DIR / "hope_video"
MESH_DIR       = ROOT_DIR / "meshes" / "full"
OUTPUT_DIR     = ROOT_DIR / "output"

FRAME_STEP = 60

# Frames to skip (key: scene folder name, value: list of frame IDs).
# Fill in problematic frames that cause OOM or excessive computation.
SKIP_FRAMES = {
    "scene_0006": [60],       # example: skip frames 120 and 180
    "scene_0008": [0],
    # "scene_0012": [0, 60, 300],     # example: skip these specific frames
}

# Add module directories so we can import existing helpers
sys.path.insert(0, str(ROOT_DIR / "Free_bbox"))
sys.path.insert(0, str(ROOT_DIR / "3D_bbox"))

from build_occupancy import (
    parse_camera, depth_to_pointcloud, build_occupancy_grid,
    save_ply, save_occupancy_ply, make_voxel_params,
    visualize_overview, visualize_slices, visualize_3d,
    DEPTH_SCALE, VOXEL_SIZE, PIXEL_STRIDE,
    FREE, OCCUPIED, UNKNOWN,
)

from placement_planner import (
    camera_transforms, obj_to_world, voxelize_obb,
    get_bbox_corners, transform_points, voxel_to_world,
    is_fully_visible, filter_visible_placements,
    prepare_grid, prepare_grid_base,
    grid_remove_object, grid_restore_object,
    detect_support_surfaces, find_table_placements,
    filter_stable_placements, build_depth_buffer,
    filter_occluded_placements, cluster_placements,
    save_placement_vis, compute_placed_transform,
    SAFETY_MARGIN_CM, DBSCAN_EPS_CM, DBSCAN_MIN_SAMPLES,
    YAW_STEPS, MIN_SUPPORT_RATIO, OCCLUSION_THRESHOLD, HAS_GPU,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  Constants for bbox3d visualisation (from visualize_bbox3d.py)
# ═══════════════════════════════════════════════════════════════════════════════

WORLD_UP     = np.array([0.0, 0.0, 1.0])
BOTTOM_COLOR = "orange"
MM_TO_CM     = 0.1

BOX_EDGES = [
    (0, 1), (2, 3), (4, 5), (6, 7),
    (0, 2), (1, 3), (4, 6), (5, 7),
    (0, 4), (1, 5), (2, 6), (3, 7),
]

CONTACT_FACE_CORNERS = {
    (0, +1): [1, 3, 7, 5],  (0, -1): [0, 2, 6, 4],
    (1, +1): [2, 3, 7, 6],  (1, -1): [0, 1, 5, 4],
    (2, +1): [4, 5, 7, 6],  (2, -1): [0, 1, 3, 2],
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 1 helpers: 3D bbox generation + visualisation
# ═══════════════════════════════════════════════════════════════════════════════

_bbox3d_cache = {}


def _compute_bbox3d(class_name):
    """Compute canonical AABB for an object class (cached per class)."""
    if class_name in _bbox3d_cache:
        return _bbox3d_cache[class_name]

    mesh_path = MESH_DIR / f"{class_name}.obj"
    if not mesh_path.exists():
        print(f"    [WARNING] mesh not found: {mesh_path}")
        return None

    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    verts = np.array(mesh.vertices, dtype=np.float64) * MM_TO_CM
    bbox3d = verts.min(axis=0).tolist() + verts.max(axis=0).tolist()
    _bbox3d_cache[class_name] = bbox3d
    return bbox3d


def _corners_from_bbox3d(bbox3d: list) -> np.ndarray:
    """8 corners of an AABB, shape (8, 3)."""
    mn_x, mn_y, mn_z, mx_x, mx_y, mx_z = bbox3d
    corners = []
    for zi in range(2):
        for yi in range(2):
            for xi in range(2):
                corners.append([
                    [mn_x, mx_x][xi],
                    [mn_y, mx_y][yi],
                    [mn_z, mx_z][zi],
                ])
    return np.array(corners, dtype=np.float64)


def _contact_face_indices(pose: np.ndarray, extrinsics: np.ndarray) -> list:
    """Determine which AABB face is the contact (bottom) face."""
    R_pose = pose[:3, :3]
    down_cam = extrinsics[:3, :3] @ (-WORLD_UP)
    down_obj = R_pose.T @ down_cam
    axis = int(np.argmax(np.abs(down_obj)))
    sign = int(np.sign(down_obj[axis]))
    return CONTACT_FACE_CORNERS[(axis, sign)]


def step1_generate_meta_and_vis(annot_path: str, rgb_path: str,
                                out_dir: str) -> dict:
    """
    Step 1: Add bbox3d to every object and render 3D bbox overlay.

    Returns the enriched annotation dict (meta_data).
    """
    os.makedirs(out_dir, exist_ok=True)

    with open(annot_path) as f:
        annots = json.load(f)

    for obj in annots["objects"]:
        bbox3d = _compute_bbox3d(obj["class"])
        if bbox3d is not None:
            obj["bbox3d"] = bbox3d

    meta_path = os.path.join(out_dir, "meta_data.json")
    with open(meta_path, "w") as f:
        json.dump(annots, f, indent=4)

    # ── Visualise bbox3d on RGB ──────────────────────────────────────────
    K  = np.array(annots["camera"]["intrinsics"], dtype=np.float64)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    extrinsics = np.array(annots["camera"]["extrinsics"], dtype=np.float64)

    img = np.array(Image.open(rgb_path))
    fig, ax = plt.subplots(1, 1, figsize=(10, 7.5))
    ax.imshow(img)
    ax.axis("off")

    cmap = plt.get_cmap("tab10")
    legend_handles = []

    for idx, obj in enumerate(annots["objects"]):
        bbox3d = obj.get("bbox3d")
        pose   = obj.get("pose")
        if bbox3d is None or pose is None:
            continue

        cls      = obj["class"]
        color    = cmap(idx % 10)
        pose_mat = np.array(pose, dtype=np.float64)

        corners_obj = _corners_from_bbox3d(bbox3d)
        ones = np.ones((8, 1))
        corners_cam = (pose_mat @ np.hstack([corners_obj, ones]).T).T[:, :3]
        X, Y, Z = corners_cam[:, 0], corners_cam[:, 1], corners_cam[:, 2]
        corners_2d = np.stack([fx * X / Z + cx, fy * Y / Z + cy], axis=1)

        for i, j in BOX_EDGES:
            ax.plot([corners_2d[i, 0], corners_2d[j, 0]],
                    [corners_2d[i, 1], corners_2d[j, 1]],
                    color=color, linewidth=1.5, alpha=0.9)

        contact_idx = _contact_face_indices(pose_mat, extrinsics)
        poly = Polygon(corners_2d[contact_idx], closed=True,
                       facecolor=BOTTOM_COLOR, edgecolor=BOTTOM_COLOR,
                       alpha=0.35, linewidth=2.0, zorder=3)
        ax.add_patch(poly)

        center_obj = (np.array(bbox3d[:3]) + np.array(bbox3d[3:])) / 2.0
        contact_ctr = corners_obj[contact_idx].mean(axis=0)
        two = np.stack([center_obj, contact_ctr])
        two_h = np.hstack([two, np.ones((2, 1))])
        two_cam = (pose_mat @ two_h.T).T[:, :3]
        two_2d = np.stack([
            fx * two_cam[:, 0] / two_cam[:, 2] + cx,
            fy * two_cam[:, 1] / two_cam[:, 2] + cy,
        ], axis=1)
        ax.plot([two_2d[0, 0], two_2d[1, 0]],
                [two_2d[0, 1], two_2d[1, 1]],
                color=BOTTOM_COLOR, linewidth=1.5,
                linestyle="--", alpha=0.9, zorder=4)

        z_vals = corners_cam[:, 2]
        near_mask = z_vals <= np.partition(z_vals, 4)[4]
        near_2d = corners_2d[near_mask]
        ax.text(near_2d[:, 0].mean(), near_2d[:, 1].min() - 5, cls,
                color=color, fontsize=7, fontweight="bold",
                ha="center", va="bottom",
                bbox=dict(fc="black", alpha=0.4, pad=1.5, edgecolor="none"))

        legend_handles.append(mpatches.Patch(color=color, label=cls))

    legend_handles.append(
        mpatches.Patch(color=BOTTOM_COLOR, alpha=0.6, label="contact face"))
    ax.legend(handles=legend_handles, loc="upper right",
              fontsize=7, framealpha=0.6)

    plt.tight_layout(pad=0)
    vis_path = os.path.join(out_dir, "bbox_3d_vis.jpg")
    plt.savefig(vis_path, dpi=150, bbox_inches="tight")
    plt.close()

    return annots


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 2: Build occupancy grid
# ═══════════════════════════════════════════════════════════════════════════════

def step2_build_occupancy(annot_path: str, rgb_path: str,
                          depth_path: str, out_dir: str):
    """Step 2: Build occupancy grid from RGB-D + save all artefacts."""
    os.makedirs(out_dir, exist_ok=True)

    with open(annot_path) as f:
        annots = json.load(f)
    rgb   = np.asarray(Image.open(rgb_path),   dtype=np.uint8)
    depth = np.asarray(Image.open(depth_path), dtype=np.float32)

    fx, fy, cx, cy, R_c2w, cam_origin = parse_camera(annots)

    # Point cloud
    pts, colors = depth_to_pointcloud(
        depth, rgb, fx, fy, cx, cy, R_c2w, cam_origin, PIXEL_STRIDE)
    save_ply(os.path.join(out_dir, "point_cloud.ply"), pts, colors)

    visualize_overview(rgb, depth, pts, colors,
                       os.path.join(out_dir, "scene_overview.png"))

    # Occupancy grid
    grid, grid_min, grid_max, counts = build_occupancy_grid(
        depth, fx, fy, cx, cy, R_c2w, cam_origin,
        voxel_size=VOXEL_SIZE, stride=PIXEL_STRIDE)

    np.save(os.path.join(out_dir, "occupancy_grid.npy"), grid)

    save_occupancy_ply(os.path.join(out_dir, "occupancy_grid.ply"),
                       grid, grid_min, VOXEL_SIZE, include_unknown=False)
    save_occupancy_ply(os.path.join(out_dir, "occupied_only.ply"),
                       grid, grid_min, VOXEL_SIZE, states=[OCCUPIED])

    vp = make_voxel_params(grid_min, VOXEL_SIZE)
    meta = {
        "voxel_params":  vp,
        "voxel_size_cm": VOXEL_SIZE,
        "grid_min_cm":   grid_min.tolist(),
        "grid_max_cm":   grid_max.tolist(),
        "grid_shape":    list(map(int, grid.shape)),
        "states":        {"0": "Free", "1": "Occupied", "2": "Unknown"},
        "voxel_counts":  counts,
        "depth_scale":   DEPTH_SCALE,
        "pixel_stride":  PIXEL_STRIDE,
    }
    with open(os.path.join(out_dir, "grid_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    visualize_slices(grid, grid_min, VOXEL_SIZE,
                     os.path.join(out_dir, "occupancy_slices.png"))
    visualize_3d(grid, grid_min, VOXEL_SIZE,
                 os.path.join(out_dir, "occupancy_3d.png"))


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 3: Placement planning
# ═══════════════════════════════════════════════════════════════════════════════

def step3_placement(occupancy_dir: str, meta_data_path: str,
                    rgb_path: str, out_dir: str,
                    yaw_steps=YAW_STEPS,
                    min_support_ratio=MIN_SUPPORT_RATIO,
                    occlusion_threshold=OCCLUSION_THRESHOLD,
                    use_gpu=None):
    """Step 3: Find collision-free placements for every visible object."""
    os.makedirs(out_dir, exist_ok=True)

    if use_gpu is None:
        use_gpu = HAS_GPU

    grid = np.load(os.path.join(occupancy_dir, "occupancy_grid.npy"))
    with open(os.path.join(occupancy_dir, "grid_meta.json")) as f:
        grid_meta = json.load(f)
    with open(meta_data_path) as f:
        annotations = json.load(f)
    rgb = np.asarray(Image.open(rgb_path), dtype=np.uint8)

    vp = grid_meta["voxel_params"]
    vs = float(vp["voxel_size"])
    gs = tuple(grid_meta["grid_shape"])
    img_h, img_w = rgb.shape[:2]

    E_w2c, E_c2w = camera_transforms(annotations)
    cam_origin = E_c2w[:3, 3]
    K  = np.array(annotations["camera"]["intrinsics"], dtype=np.float64)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    objects = annotations["objects"]

    all_voxels, all_T = [], []
    for obj in objects:
        pose = np.array(obj["pose"], dtype=np.float64)
        T = obj_to_world(pose, E_c2w)
        vox = (voxelize_obb(obj["bbox3d"], T, vp, gs)
               if "bbox3d" in obj
               else np.empty((0, 3), dtype=int))
        all_voxels.append(vox)
        all_T.append(T)

    margin_voxels = int(math.ceil(SAFETY_MARGIN_CM / vs))
    all_results = {}
    grid_base = prepare_grid_base(grid, all_voxels)

    for oi, obj in enumerate(objects):
        name   = obj["class"]
        pose   = np.array(obj["pose"], dtype=np.float64)
        bbox3d = obj.get("bbox3d")
        T      = all_T[oi]

        if bbox3d is None:
            continue

        if not is_fully_visible(bbox3d, pose, fx, fy, cx, cy, img_w, img_h):
            continue

        target_vox = all_voxels[oi]
        if len(target_vox) == 0:
            continue

        tv, saved = grid_remove_object(grid_base, target_vox)
        try:
            table_z, table_mask_2d = detect_support_surfaces(grid_base, vp)
            if table_z is None:
                continue

            depth_buffer = build_depth_buffer(
                grid_base, vp, K, E_w2c, img_w, img_h)

            candidates, meta, yaw_data = find_table_placements(
                grid_base, bbox3d, T, table_z, margin_voxels, vp,
                yaw_steps=yaw_steps, use_gpu=use_gpu,
                table_mask_2d=table_mask_2d)
            landing_z = meta["landing_z"]

            candidates = filter_stable_placements(
                candidates, yaw_data, table_mask_2d,
                min_support_ratio=min_support_ratio)

            candidates = filter_visible_placements(
                candidates, landing_z,
                bbox3d, T, E_w2c, K, img_w, img_h, vp, yaw_data)

            candidates = filter_occluded_placements(
                candidates, landing_z, bbox3d, T,
                depth_buffer, K, E_w2c, vp, yaw_data,
                img_w, img_h,
                occlusion_threshold=occlusion_threshold)

            reps, c_infos = cluster_placements(
                candidates, grid_base, yaw_data,
                landing_z, vp, DBSCAN_EPS_CM, DBSCAN_MIN_SAMPLES)

            if len(c_infos) == 0:
                continue

            save_placement_vis(
                rgb, name, bbox3d, T, K, E_w2c, vp,
                cam_origin, reps, c_infos, target_vox, grid,
                os.path.join(out_dir, f"placement_{name}.png"),
                yaw_data, landing_z)

            all_results[name] = {
                "status":         "ok",
                "table_z":        table_z,
                "landing_z":      landing_z,
                "raw_candidates": meta.get("valid_raw", 0),
                "stable":         len(candidates),
                "clusters":       c_infos,
            }
        finally:
            grid_restore_object(grid_base, tv, saved)

    summary = {
        "safety_margin_cm":    SAFETY_MARGIN_CM,
        "dbscan_eps_cm":       DBSCAN_EPS_CM,
        "yaw_steps":           yaw_steps,
        "min_support_ratio":   min_support_ratio,
        "occlusion_threshold": occlusion_threshold,
        "gpu_used":            use_gpu,
        "voxel_size_cm":       vs,
        "objects":             all_results,
    }
    with open(os.path.join(out_dir, "placement_result.json"), "w") as f:
        json.dump(summary, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
#  Batch loop
# ═══════════════════════════════════════════════════════════════════════════════

def get_frame_ids(scene_dir, step):
    """Return sorted frame IDs sampled every *step* frames."""
    ids = sorted(
        int(f.stem) for f in scene_dir.glob("*.json") if f.stem.isdigit()
    )
    return [fid for fid in ids if fid % step == 0]


def is_frame_done(scene_out: Path, frame_str: str) -> bool:
    """Check whether all 3 steps have already finished for this frame."""
    return (
        (scene_out / "3D_bbox"   / frame_str / "meta_data.json").exists() and
        (scene_out / "occupancy" / frame_str / "occupancy_grid.npy").exists() and
        (scene_out / "placement" / frame_str / "placement_result.json").exists()
    )


def main():
    scenes = sorted(HOPE_VIDEO_DIR.glob("scene_*"))
    if not scenes:
        print(f"[ERROR] No scenes found in {HOPE_VIDEO_DIR}")
        return

    total_frames = sum(len(get_frame_ids(s, FRAME_STEP)) for s in scenes)

    print(f"\n{'=' * 70}")
    print(f"  HOPE-Video Batch Processor")
    print(f"  Scenes       : {len(scenes)}")
    print(f"  Frame step   : every {FRAME_STEP} frames")
    print(f"  Total frames : {total_frames}")
    print(f"  Output       : {OUTPUT_DIR}")
    print(f"{'=' * 70}\n")

    t_total = time.time()
    processed, skipped, failed = 0, 0, 0
    global_idx = 0

    for scene_dir in scenes:
        scene_name = scene_dir.name
        frame_ids  = get_frame_ids(scene_dir, FRAME_STEP)
        scene_out  = OUTPUT_DIR / scene_name

        print(f"\n{'─' * 60}")
        print(f"  Scene: {scene_name}  ({len(frame_ids)} frames)")
        print(f"{'─' * 60}")

        for fid in frame_ids:
            global_idx += 1
            frame_str = f"{fid:04d}"
            tag = f"[{global_idx}/{total_frames}] {scene_name}/{frame_str}"

            # Skip already-completed frames
            if is_frame_done(scene_out, frame_str):
                print(f"  {tag}  SKIP (already done)")
                skipped += 1
                continue

            # Skip frames explicitly listed in SKIP_FRAMES
            if fid in SKIP_FRAMES.get(scene_name, []):
                print(f"  {tag}  SKIP (in SKIP_FRAMES list)")
                skipped += 1
                continue

            # Verify input files exist
            annot_path = scene_dir / f"{frame_str}.json"
            rgb_path   = scene_dir / f"{frame_str}_rgb.jpg"
            depth_path = scene_dir / f"{frame_str}_depth.png"

            if not all(p.exists() for p in [annot_path, rgb_path, depth_path]):
                print(f"  {tag}  SKIP (missing input files)")
                skipped += 1
                continue

            print(f"\n  {tag}")

            bbox_dir      = str(scene_out / "3D_bbox"   / frame_str)
            occupancy_dir = str(scene_out / "occupancy" / frame_str)
            placement_dir = str(scene_out / "placement" / frame_str)

            try:
                # ── Step 1: 3D bbox ──────────────────────────────────────
                t0 = time.time()
                meta_data = step1_generate_meta_and_vis(
                    str(annot_path), str(rgb_path), bbox_dir)
                print(f"    Step 1 (3D_bbox)   : {time.time() - t0:.1f}s")

                # ── Step 2: occupancy grid ───────────────────────────────
                t0 = time.time()
                step2_build_occupancy(
                    str(annot_path), str(rgb_path), str(depth_path),
                    occupancy_dir)
                print(f"    Step 2 (occupancy) : {time.time() - t0:.1f}s")

                # ── Step 3: placement planning ───────────────────────────
                t0 = time.time()
                meta_data_path = os.path.join(bbox_dir, "meta_data.json")
                step3_placement(
                    occupancy_dir, meta_data_path, str(rgb_path),
                    placement_dir)
                print(f"    Step 3 (placement) : {time.time() - t0:.1f}s")

                processed += 1

            except Exception as e:
                print(f"    [ERROR] {e}")
                traceback.print_exc()
                failed += 1

            finally:
                plt.close("all")
                gc.collect()

    elapsed = time.time() - t_total
    print(f"\n{'=' * 70}")
    print(f"  DONE")
    print(f"  processed : {processed}")
    print(f"  skipped   : {skipped}")
    print(f"  failed    : {failed}")
    print(f"  time      : {elapsed:.0f}s ({elapsed / 60:.1f}min)")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
