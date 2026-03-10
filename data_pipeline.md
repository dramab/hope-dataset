# 数据标注流程说明文档 (Data Pipeline Documentation)

本文档详细说明了本项目的数据标注流程、输入输出数据格式以及程序运行方式。

## 1. 标注数据的原始输入

程序的输入数据主要来自 `hope_video` 文件夹下的场景数据和 `meshes` 文件夹下的物体模型。

*   **场景数据** (`hope_video/scene_XXXX/`):
    *   `{frame_id}.json`: 原始标注文件，包含相机参数（内参、外参）和物体位姿（Pose）信息。
    *   `{frame_id}_rgb.jpg`: RGB 彩色图像。
    *   `{frame_id}_depth.png`: 深度图像。
*   **物体模型** (`meshes/full/`):
    *   `{class_name}.obj`: 各类物体的 3D 网格模型文件（用于计算 3D 包围盒）。

## 2. 标注的输出

程序运行后会在 `output` 目录下生成处理结果，按场景 (`scene_XXXX`) 和帧号分类。

输出目录结构如下：
```text
output/
└── scene_XXXX/
    ├── 3D_bbox/             # 第一步：3D 包围盒结果
    │   └── {frame_id}/
    │       ├── meta_data.json      # 增加 bbox3d 字段后的标注文件
    │       └── bbox_3d_vis.jpg     # 3D 包围盒可视化图
    ├── occupancy/           # 第二步：占据栅格结果
    │   └── {frame_id}/
    │       ├── occupancy_grid.npy  # 3D 占据栅格数据 (NumPy)
    │       ├── grid_meta.json      # 栅格元数据（原点、尺寸等）
    │       ├── point_cloud.ply     # 原始点云文件
    │       ├── occupancy_grid.ply  # 栅格可视化模型
    │       └── *.png               # 概览和切片可视化图
    └── placement/           # 第三步：放置点规划结果
        └── {frame_id}/
            ├── placement_result.json  # 最终放置点结果汇总
            └── placement_{Obj}.png    # 针对每个物体的放置可视化图
```

## 3. 标注文件内容解释

核心的输出文件是 `placement/` 目录下的 `placement_result.json`，其主要字段解释如下：

*   **全局配置参数**:
    *   `safety_margin_cm`: 安全边距（厘米）。
    *   `yaw_steps`: 旋转角度采样数（例如 24 表示每 15 度采样一次）。
    *   `voxel_size_cm`: 体素大小（厘米）。
    *   `min_support_ratio`: 最小支撑比例（用于稳定性检查）。

*   **objects**: 包含场景中每个物体的处理结果。
    *   `{ClassName}`: 物体类别名称。
        *   `status`: 处理状态 (`ok` 表示成功, `not_visible` 表示不可见等)。
        *   `table_z`: 检测到的桌面高度（体素索引）。
        *   `landing_z`: 物体放置高度（体素索引）。
        *   `raw_candidates`: 原始无碰撞候选点数量。
        *   `stable`: 通过稳定性检测的候选点数量。
        *   `visible`: 投影在图像范围内的候选点数量。
        *   `unoccluded`: 未被遮挡的候选点数量。
        *   `clusters`: 聚类后的最终推荐放置点列表。
            *   `cluster_id`: 聚类 ID。
            *   `size`: 该聚类包含的候选点数量。
            *   `anchor_voxel`: 推荐放置点的体素坐标 `[x, y, z]`。
            *   `anchor_world_cm`: 推荐放置点的世界坐标 `[x, y, z]` (cm)。
            *   `yaw_index`: 旋转角度索引。
            *   `yaw_degrees`: 旋转角度（度）。
            *   `free_score`: 周围自由空间评分（越高越好）。

## 4. 整个标注流程

数据处理流程由 `batch_process.py` 驱动，分为三个主要步骤：

1.  **Step 1: 3D Bounding Box (3D 包围盒生成)**
    *   读取物体 Mesh 模型，计算其轴对齐包围盒 (AABB)。
    *   结合原始标注中的 Pose，计算物体在世界坐标系下的包围盒。
    *   生成可视化的 RGB 图片 (`bbox_3d_vis.jpg`) 以验证对齐情况。

2.  **Step 2: Occupancy Grid (占据栅格构建)**
    *   利用 RGB 和 Depth 图，结合相机内参，将深度图反投影为 3D 点云。
    *   将场景体素化（Voxelization），构建 3D 占据栅格 (`occupancy_grid.npy`)。
    *   状态分为：`Free` (空闲), `Occupied` (占据), `Unknown` (未知)。

3.  **Step 3: Placement Planning (放置点规划)**
    *   **移除目标**: 在栅格中暂时移除当前目标物体（"模拟拿走"）。
    *   **平面检测**: 识别场景中最大的水平支撑面（通常是桌面）。
    *   **碰撞检测**: 在桌面上搜索可放置该物体的空闲位置（使用 FFT 卷积加速），并尝试不同的旋转角度 (Yaw)。
    *   **过滤筛选**:
        *   **物理稳定性**: 检查物体底部是否有足够的支撑。
        *   **可见性**: 确保放置后的物体在相机视野内。
        *   **遮挡剔除**: 利用 Z-buffer 检查放置位置是否被前景物体遮挡。
    *   **聚类**: 对所有可行点进行 DBSCAN 聚类，每个聚类选出一个最佳代表点。
    *   **可视化**: 生成包含原物体和推荐放置点的可视化图像。

## 5. 应该怎么启动程序？

### 环境准备
确保已经安装并激活了项目所需的 Python 环境（例如 `hope`）：
```bash
conda activate hope
```

### 运行批处理程序
在项目根目录下运行 `batch_process.py` 脚本：
```bash
python batch_process.py
```

该脚本会自动扫描 `hope_video` 目录下的所有场景，按设定的帧间隔（默认 60 帧）进行处理。

*   **跳过特定帧**: 如果某些帧有问题，可以在 `batch_process.py` 中的 `SKIP_FRAMES` 字典中配置要跳过的帧。
*   **断点续传**: 程序会自动跳过已经处理完成的帧 (`is_frame_done` 检查)，支持中断后继续运行。
