"""
根据 test_data/0000.json 的元数据，结合 meshes/full/ 下的 3D mesh，
计算每个物体的：
  - bbox3d：物体规范坐标系下的 AABB（单位 cm，pose 无关）
  - dimensions：世界坐标系下的 AABB extents（单位 cm，pose 相关，侧放时正确变化）
  - bottom_offset：物体原点到接触面沿世界重力方向的距离（单位 cm，pose 相关）
生成 test_data/meta_data.json。

注意：
- pose 的平移单位是 cm。
- meshes/full/ 的顶点单位是 mm，需乘以 0.1 转为 cm。
- bbox3d 格式：[min_x, min_y, min_z, max_x, max_y, max_z]（cm，物体规范坐标系）
- dimensions 格式：{"dim_x": float, "dim_y": float, "dim_z": float}（cm，世界坐标系 XYZ 轴）
- bottom_offset：float（cm），物体原点到接触面在重力方向上的距离
- WORLD_UP 可按需改为 [0,0,1]（Z-up）等其他惯例
"""

import json
import os
import numpy as np
import trimesh

ANNOT_PATH  = "../test_data/0000.json"
MESH_DIR    = "../meshes/full"
OUTPUT_PATH = "../test_data/meta_data.json"

MM_TO_CM = 0.1   # full mesh 顶点单位为 mm，转为与 pose 一致的 cm

# 世界坐标系的 "上" 方向
# HOPE 数据集（Baxter 机器人 / ROS 惯例）使用 Z-up
# 若数据集使用 Y-up，改为 np.array([0.0, 1.0, 0.0])
WORLD_UP = np.array([0.0, 0.0, 1.0])


def load_vertices(mesh_path: str) -> np.ndarray:
    """加载 mesh 顶点，缩放为 cm，shape (N, 3)。"""
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    return np.array(mesh.vertices, dtype=np.float64) * MM_TO_CM


def compute_bbox3d_canonical(verts: np.ndarray) -> list:
    """
    在物体规范坐标系下计算 AABB（pose 无关，per-class 固定值）。
    返回 [min_x, min_y, min_z, max_x, max_y, max_z]（cm）
    """
    min_xyz = verts.min(axis=0)
    max_xyz = verts.max(axis=0)
    return min_xyz.tolist() + max_xyz.tolist()


def compute_scene_properties(verts_obj: np.ndarray,
                              pose: np.ndarray,
                              extrinsics: np.ndarray) -> tuple[dict, float]:
    """
    计算 pose 相关的世界系维度和底部偏移量。

    dimensions 用旋转将物体顶点映射到世界系后做 AABB（translation 不影响 extents）：
        R_WO = R_extrinsics.T @ R_pose
        verts_world = R_WO @ verts_obj.T
        extents = AABB(verts_world)

    bottom_offset 在相机坐标系中计算（单位统一为 cm，避免外参平移单位问题）：
        down_cam = R_extrinsics @ (-WORLD_UP)
        bottom_offset = max(verts_cam @ down_cam) - (t_pose @ down_cam)

    Args:
        verts_obj:  物体规范坐标系顶点 (N, 3)，单位 cm
        pose:       4×4 物体→相机变换矩阵（平移单位 cm）
        extrinsics: 4×4 世界→相机变换矩阵

    Returns:
        dimensions:    {"dim_x": float, "dim_y": float, "dim_z": float}（cm，世界系）
        bottom_offset: float（cm）
    """
    R_pose       = pose[:3, :3]
    t_pose       = pose[:3,  3]       # 物体原点在相机系中的位置，cm
    R_extrinsics = extrinsics[:3, :3]

    # ── dimensions: 在世界系做 AABB（只用旋转，translation 不影响 extents）──────
    R_WO        = R_extrinsics.T @ R_pose           # 物体系 → 世界系（旋转部分）
    verts_world = (R_WO @ verts_obj.T).T            # (N, 3)
    extents     = verts_world.max(axis=0) - verts_world.min(axis=0)

    dimensions = {
        "dim_x": round(float(extents[0]), 4),
        "dim_y": round(float(extents[1]), 4),
        "dim_z": round(float(extents[2]), 4),
    }

    # ── bottom_offset: 在相机坐标系中计算，单位 cm ────────────────────────────
    # 世界 "下" 方向映射到相机系
    down_cam = R_extrinsics @ (-WORLD_UP)            # 单位向量

    # 物体所有顶点变换到相机系
    verts_cam = (R_pose @ verts_obj.T).T + t_pose   # (N, 3), cm

    # 各顶点在 "down_cam" 方向上的投影值（越大越靠近地面）
    proj_verts  = verts_cam  @ down_cam              # (N,)
    proj_origin = t_pose     @ down_cam              # 物体原点的投影

    # bottom_offset = 最低顶点投影 - 原点投影（距离，>0 表示顶点在原点"下方"）
    bottom_offset = round(float(np.max(proj_verts) - proj_origin), 4)

    return dimensions, bottom_offset


def main():
    with open(ANNOT_PATH, "r") as f:
        annots = json.load(f)

    extrinsics = np.array(annots["camera"]["extrinsics"], dtype=np.float64)

    missing = []
    for obj in annots["objects"]:
        cls       = obj["class"]
        mesh_path = os.path.join(MESH_DIR, f"{cls}.obj")

        if not os.path.exists(mesh_path):
            print(f"[WARNING] mesh not found: {mesh_path}, skipping {cls}")
            missing.append(cls)
            continue

        verts = load_vertices(mesh_path)
        pose  = np.array(obj["pose"], dtype=np.float64)

        bbox3d                    = compute_bbox3d_canonical(verts)
        dimensions, bottom_offset = compute_scene_properties(verts, pose, extrinsics)

        obj["bbox3d"]        = bbox3d
        obj["dimensions"]    = dimensions
        obj["bottom_offset"] = bottom_offset

        print(
            f"  {cls}: "
            f"dim_world=({dimensions['dim_x']:.2f},{dimensions['dim_y']:.2f},{dimensions['dim_z']:.2f}) cm  "
            f"bottom_offset={bottom_offset:.4f} cm"
        )

    if missing:
        print(f"\n[WARNING] {len(missing)} object(s) skipped: {missing}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(annots, f, indent=4)

    print(f"\n[DONE] meta_data.json saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
