from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import open3d as o3d
import trimesh

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

gt_bbox_dir = Path("data/dataset/bbox3d_labels")
predictions_dir = Path("data/prediction/event_localization3d/sound_source_bbox3d/")
excluded_recordings = [5, 20]
n_bins = 50

category_scores = [[], [], []]
category_names = ["Sawing", "Chiseling", "Drilling"]


def get_category_id(rec_id):
    if 1 <= rec_id <= 10:
        return 0
    elif 11 <= rec_id <= 16:
        return 1
    elif 17 <= rec_id <= 22:
        return 2


sample_box = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(
    o3d.geometry.OrientedBoundingBox(np.zeros(3), np.eye(3), np.ones(3))
)


def corner_pts_to_mesh(pts: np.ndarray):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    obb = pc.get_minimal_oriented_bounding_box(True)
    mesh_box = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(obb)
    tm_box = trimesh.Trimesh(
        vertices=np.asarray(mesh_box.vertices), faces=np.asarray(mesh_box.triangles)
    )
    return tm_box


def trimesh_iou_3d(box1_pts: np.ndarray, box2_pts: np.ndarray) -> float:
    """
    Computes the exact intersection volume of two 3D oriented bounding boxes.

    Args:
        box1: o3d.geometry.OrientedBoundingBox
        box2: o3d.geometry.OrientedBoundingBox

    Returns:
        The intersection volume of the two bounding boxes.
    """

    tm_box1 = corner_pts_to_mesh(box1_pts)
    tm_box2 = corner_pts_to_mesh(box2_pts)

    try:
        intersection_mesh = tm_box1.intersection(tm_box2, engine="manifold")
    except Exception as e:
        return 0.0
    intersect_vol = intersection_mesh.volume
    if intersect_vol < 1e-10:
        return 0.0
    union_vol = tm_box1.volume + tm_box2.volume - intersect_vol
    return intersect_vol / union_vol


sensitivity_analysis_r_values = [7.5, 15, 22.5, 30, 45, 60]
sensitivity_analysis_r = {
    f"r{r},w200,ds1": "1_{:03d}_Movie2D_3dBboxes" + f"_r{r:.0f}_minWeight200.npy"
    for r in sensitivity_analysis_r_values
}

sensitivity_analysis_w_values = [50, 100, 150, 200, 300, 400, 800]
sensitivity_analysis_w = {
    f"r30,w{w},ds1": "1_{:03d}_Movie2D_3dBboxes" + f"_r30_minWeight{w}.npy"
    for w in sensitivity_analysis_w_values
}

sensitivity_analysis_ds_values = [1, 2, 3, 4, 6, 8]
sensitivity_analysis_ds = {
    f"r30,w{200/s/s},ds{s*s}": "1_{:03d}_Movie2D_3dBboxes"
    + (f"_r30_minWeight{np.floor(200/s/s):.0f}_ds{s}.npy" if s > 1 else "_r30_minWeight200.npy")
    for s in sensitivity_analysis_ds_values
}
sensitivity_analysis_ds_values = [s * s for s in sensitivity_analysis_ds_values]

all_ablation_studies = [
    ("DBSCAN Radius [mm]", sensitivity_analysis_r, sensitivity_analysis_r_values),
    ("DBSCAN Min Weight", sensitivity_analysis_w, sensitivity_analysis_w_values),
    ("Point Cloud Downsampling Factor", sensitivity_analysis_ds, sensitivity_analysis_ds_values),
]

for name, ablation_study, tested_values in all_ablation_studies:
    # computer average IoU score and average 3D BBox Distance
    average_ious = []
    average_bbox_center_errors = []
    for approach, approach_template in ablation_study.items():
        ious = []
        bbox_center_errors = []
        for i in range(1, 23):
            if i in excluded_recordings:
                continue
            gt_bboxes = np.load(gt_bbox_dir / f"1_{i:03d}_Movie2D_3dBboxes.npy")
            N_frames = gt_bboxes.shape[0]
            pred_bboxes = np.load(predictions_dir / approach_template.format(i))
            assert pred_bboxes.shape[0] == N_frames
            for j in range(N_frames):
                has_gt = not np.all(gt_bboxes[j] == 0)
                has_pred = not np.all(pred_bboxes[j] == 0)
                if has_gt and has_pred:
                    ious.append(trimesh_iou_3d(pred_bboxes[j], gt_bboxes[j]))
                    bbox_center_errors.append(
                        np.linalg.norm(pred_bboxes[j].mean(0) - gt_bboxes[j].mean(0))
                    )
                else:
                    ious.append(0.0)
        average_ious.append(np.mean(ious))
        average_bbox_center_errors.append(
            np.mean(bbox_center_errors) if len(bbox_center_errors) > 0 else np.nan
        )

    plt.figure(figsize=(4, 2))
    plt.plot(tested_values, average_ious, lw=2)
    plt.xlabel(name)
    # plt.xticks(tested_values)
    plt.ylabel("Average IoU Score")
    # plt.title(f"Sensitivity Analysis on {name}")
    plt.ylim(bottom=0, top=0.3)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"data/evaluation/ablation_study_{name.replace(' ', '_')}.pdf")
    plt.show()

    if False:
        plt.figure(figsize=(6, 6))
        plt.plot(tested_values, average_bbox_center_errors)
        plt.xlabel(name)
        plt.ylabel("Average 3D BBox Center Error")
        plt.title(f"Sensitivity Analysis on {name}")
        plt.grid()
        plt.tight_layout()
        plt.show()
