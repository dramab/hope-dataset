"""
根据 test_data/0000.json 的元数据，结合 meshes/full/ 下的 3D mesh，
计算每个物体的：
  - bbox3d：物体规范坐标系下的 AABB（单位 cm，pose 无关）
生成 test_data/meta_data.json。

注意：
- pose 的平移单位是 cm。
- meshes/full/ 的顶点单位是 mm，需乘以 0.1 转为 cm。
- bbox3d 格式：[min_x, min_y, min_z, max_x, max_y, max_z]（cm，物体规范坐标系）
"""

import json
import os
import numpy as np
import trimesh

ANNOT_PATH  = "../test_data/0000.json"
MESH_DIR    = "../meshes/full"
OUTPUT_PATH = "../test_data/meta_data.json"

MM_TO_CM = 0.1   # full mesh 顶点单位为 mm，转为与 pose 一致的 cm

def load_vertices(mesh_path: str) -> np.ndarray:
    """加载 mesh 顶点，缩放为 cm，shape (N, 3)。"""
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    return np.array(mesh.vertices, dtype=np.float64) * MM_TO_CM

def compute_bbox3d_canonical(verts: np.ndarray) -> list:
    """
    在物体规范坐标系下计算 aabb（pose 无关，per-class 固定值）。
    返回 [min_x, min_y, min_z, max_x, max_y, max_z]（cm）
    """
    min_xyz = verts.min(axis=0)
    max_xyz = verts.max(axis=0)
    return min_xyz.tolist() + max_xyz.tolist()

def main():
    with open(ANNOT_PATH, "r") as f:
        annots = json.load(f)

    missing = []
    for obj in annots["objects"]:
        cls       = obj["class"]
        mesh_path = os.path.join(MESH_DIR, f"{cls}.obj")

        if not os.path.exists(mesh_path):
            print(f"[WARNING] mesh not found: {mesh_path}, skipping {cls}")
            missing.append(cls)
            continue

        verts = load_vertices(mesh_path)
        bbox3d = compute_bbox3d_canonical(verts)
        obj["bbox3d"] = bbox3d

        print(
            f"  {cls}: "
            f"bbox3d=[{', '.join(f'{b:.2f}' for b in bbox3d)}]"
        )

    if missing:
        print(f"\n[WARNING] {len(missing)} object(s) skipped: {missing}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(annots, f, indent=4)

    print(f"\n[DONE] meta_data.json saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
