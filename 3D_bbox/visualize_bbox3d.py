"""
将 test_data/meta_data.json 中每个物体的 3D bbox（物体坐标系 AABB，单位 cm）
用 pose 变换到相机坐标系后投影到 test_data/0000_rgb.jpg 上。

接触面（底面）根据 WORLD_UP + 外参 + pose 动态确定，
正确处理侧放物体（接触面不再固定为 min_y 面）。

保存结果到 test_data/bbox_3d_0000.png。
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
from PIL import Image

META_PATH   = "../test_data/meta_data.json"
IMAGE_PATH  = "../test_data/0000_rgb.jpg"
OUTPUT_PATH = "../test_data/bbox_3d_0000.png"

BOTTOM_COLOR = "orange"

# 世界坐标系 "上" 方向，与 generate_meta_data.py 保持一致
# HOPE 数据集（Baxter 机器人 / ROS 惯例）使用 Z-up
WORLD_UP = np.array([0.0, 0.0, 1.0])

# 12 条 bbox 边（顶点索引对）
# 角点编码：index = zi*4 + yi*2 + xi，取值 0/1 对应 min/max
#   0:(min_x,min_y,min_z) 1:(max_x,min_y,min_z) 2:(min_x,max_y,min_z) 3:(max_x,max_y,min_z)
#   4:(min_x,min_y,max_z) 5:(max_x,min_y,max_z) 6:(min_x,max_y,max_z) 7:(max_x,max_y,max_z)
BOX_EDGES = [
    (0, 1), (2, 3), (4, 5), (6, 7),   # 沿 X 轴的 4 条边
    (0, 2), (1, 3), (4, 6), (5, 7),   # 沿 Y 轴的 4 条边
    (0, 4), (1, 5), (2, 6), (3, 7),   # 沿 Z 轴的 4 条边
]

# 接触面 4 个角点（按顺序构成四边形），按（dominant axis，符号）索引
# 例如 (1, -1)：down_obj 主轴为 Y、方向为 -Y → 接触面是 min_y 面（竖直时的底面）
CONTACT_FACE_CORNERS = {
    (0, +1): [1, 3, 7, 5],   # max_x 面
    (0, -1): [0, 2, 6, 4],   # min_x 面
    (1, +1): [2, 3, 7, 6],   # max_y 面
    (1, -1): [0, 1, 5, 4],   # min_y 面（竖直时的底面）
    (2, +1): [4, 5, 7, 6],   # max_z 面
    (2, -1): [0, 1, 3, 2],   # min_z 面
}


def get_corners_object(bbox3d: list) -> np.ndarray:
    """从物体坐标系 bbox3d 生成 8 个角点，shape (8, 3)。"""
    min_x, min_y, min_z, max_x, max_y, max_z = bbox3d
    corners = []
    for zi in range(2):
        for yi in range(2):
            for xi in range(2):
                corners.append([
                    [min_x, max_x][xi],
                    [min_y, max_y][yi],
                    [min_z, max_z][zi],
                ])
    return np.array(corners, dtype=np.float64)  # (8, 3)


def transform_to_camera(points_obj: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """用 pose 矩阵将物体坐标系点集变换到相机坐标系，shape (N, 3)。"""
    ones = np.ones((len(points_obj), 1))
    return (pose @ np.hstack([points_obj, ones]).T).T[:, :3]


def project_points(points_3d: np.ndarray, fx, fy, cx, cy) -> np.ndarray:
    """将相机坐标系下的 3D 点（N,3）投影到像素坐标（N,2）。"""
    X, Y, Z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
    return np.stack([fx * X / Z + cx, fy * Y / Z + cy], axis=1)


def get_contact_face_indices(pose: np.ndarray, extrinsics: np.ndarray) -> list:
    """
    根据 WORLD_UP、外参、pose 动态确定接触面的 4 个角点索引。

    原理：
      down_obj = R_pose.T @ R_extrinsics @ (-WORLD_UP)
      主导轴（|down_obj| 最大的分量）决定接触哪个 AABB 面。
    """
    R_pose       = pose[:3, :3]
    R_extrinsics = extrinsics[:3, :3]

    down_cam = R_extrinsics @ (-WORLD_UP)     # 世界下方向 → 相机系
    down_obj = R_pose.T    @ down_cam         # 相机系 → 物体规范系

    axis = int(np.argmax(np.abs(down_obj)))
    sign = int(np.sign(down_obj[axis]))

    return CONTACT_FACE_CORNERS[(axis, sign)]


def main():
    with open(META_PATH, "r") as f:
        meta = json.load(f)

    K          = np.array(meta["camera"]["intrinsics"])
    fx, fy     = K[0, 0], K[1, 1]
    cx, cy     = K[0, 2], K[1, 2]
    extrinsics = np.array(meta["camera"]["extrinsics"], dtype=np.float64)

    img = np.array(Image.open(IMAGE_PATH))
    fig, ax = plt.subplots(1, 1, figsize=(10, 7.5))
    ax.imshow(img)
    ax.axis("off")

    cmap = plt.get_cmap("tab10")
    legend_handles = []

    for idx, obj in enumerate(meta["objects"]):
        cls           = obj["class"]
        bbox3d        = obj.get("bbox3d")
        pose          = obj.get("pose")
        bottom_offset = obj.get("bottom_offset")

        if bbox3d is None or pose is None:
            print(f"[WARNING] {cls} 缺少 bbox3d 或 pose，跳过")
            continue

        color    = cmap(idx % 10)
        pose_mat = np.array(pose, dtype=np.float64)

        # ── 1. 8 角点：物体系 → 相机系 → 像素 ──────────────────────────────
        corners_obj = get_corners_object(bbox3d)                        # (8, 3)
        corners_cam = transform_to_camera(corners_obj, pose_mat)        # (8, 3)
        corners_2d  = project_points(corners_cam, fx, fy, cx, cy)      # (8, 2)

        # ── 2. 绘制 12 条边 ─────────────────────────────────────────────────
        for i, j in BOX_EDGES:
            ax.plot([corners_2d[i, 0], corners_2d[j, 0]],
                    [corners_2d[i, 1], corners_2d[j, 1]],
                    color=color, linewidth=1.5, alpha=0.9)

        # ── 3. 动态确定接触面，橙色填充 ─────────────────────────────────────
        contact_idx = get_contact_face_indices(pose_mat, extrinsics)    # [4 indices]
        contact_2d  = corners_2d[contact_idx]                           # (4, 2)
        poly = Polygon(contact_2d, closed=True,
                       facecolor=BOTTOM_COLOR, edgecolor=BOTTOM_COLOR,
                       alpha=0.35, linewidth=2.0, zorder=3)
        ax.add_patch(poly)

        # ── 4. 虚线：bbox 中心 → 接触面中心 + 偏移量标注 ─────────────────────
        center_obj       = (np.array(bbox3d[:3]) + np.array(bbox3d[3:])) / 2.0
        contact_face_obj = corners_obj[contact_idx].mean(axis=0)

        two_pts_cam = transform_to_camera(
            np.stack([center_obj, contact_face_obj]), pose_mat)
        two_pts_2d  = project_points(two_pts_cam, fx, fy, cx, cy)

        ax.plot([two_pts_2d[0, 0], two_pts_2d[1, 0]],
                [two_pts_2d[0, 1], two_pts_2d[1, 1]],
                color=BOTTOM_COLOR, linewidth=1.5,
                linestyle="--", alpha=0.9, zorder=4)

        if bottom_offset is not None:
            ax.annotate(
                f"{bottom_offset:.2f} cm",
                xy=two_pts_2d[1],
                xytext=(two_pts_2d[1, 0] + 5, two_pts_2d[1, 1] + 5),
                fontsize=6, color=BOTTOM_COLOR,
                bbox=dict(fc="black", alpha=0.35, pad=1, edgecolor="none"),
            )

        # ── 5. 类别标注：放在最近面投影中心上方 ─────────────────────────────
        z_vals    = corners_cam[:, 2]
        near_mask = z_vals <= np.partition(z_vals, 4)[4]
        near_2d   = corners_2d[near_mask]
        ax.text(near_2d[:, 0].mean(), near_2d[:, 1].min() - 5, cls,
                color=color, fontsize=7, fontweight="bold",
                ha="center", va="bottom",
                bbox=dict(fc="black", alpha=0.4, pad=1.5, edgecolor="none"))

        legend_handles.append(mpatches.Patch(color=color, label=cls))

    legend_handles.append(
        mpatches.Patch(color=BOTTOM_COLOR, alpha=0.6, label="contact face / offset"))
    ax.legend(handles=legend_handles, loc="upper right",
              fontsize=7, framealpha=0.6)

    plt.tight_layout(pad=0)
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[DONE] 已保存到 {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
