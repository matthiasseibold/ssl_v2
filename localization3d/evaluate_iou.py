from collections import defaultdict

import numpy as np
import matplotlib as mpl
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
category_colors = [mpl.colormaps["Set1"](i) for i in [1, 2, 0]]


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


approach_filename_templates = {f"r30,w400": "1_{:03d}_Movie2D_3dBboxes" + f"_r30_minWeight400.npy"}

# compute IoU scores
category_ious = defaultdict(lambda: defaultdict(list))
recording_ious = defaultdict(list)
# compute bbox center distances
category_bbox_center_errs = defaultdict(lambda: defaultdict(list))
recording_bbox_center_errs = defaultdict(list)

MAX_DIST_ERROR = 500.0

for i in range(1, 23):
    if i in excluded_recordings:
        continue
    gt_bboxes = np.load(gt_bbox_dir / f"1_{i:03d}_Movie2D_3dBboxes.npy")
    N_frames = gt_bboxes.shape[0]
    for approach, approach_template in approach_filename_templates.items():
        ious = []
        bbox_center_errors = []
        pred_bboxes = np.load(predictions_dir / approach_template.format(i))
        assert pred_bboxes.shape[0] == N_frames
        for j in range(N_frames):
            has_gt = not np.all(gt_bboxes[j] == 0)
            has_pred = not np.all(pred_bboxes[j] == 0)
            if has_gt and has_pred:
                ious.append(trimesh_iou_3d(pred_bboxes[j], gt_bboxes[j]))
                bbox_center_errors.append(
                    min(
                        MAX_DIST_ERROR,
                        np.linalg.norm(pred_bboxes[j].mean(0) - gt_bboxes[j].mean(0)),
                    )
                )
            elif has_pred:
                ious.append(-2.0)  # false positive
            elif has_gt:
                ious.append(-1.0)  # false negative
        category_ious[approach][get_category_id(i)].extend(ious)
        recording_ious[approach].append(ious)
        category_bbox_center_errs[approach][get_category_id(i)].extend(bbox_center_errors)
        recording_bbox_center_errs[approach].append(bbox_center_errors)

# evaluate and generate plots
for approach, _ in approach_filename_templates.items():
    plt.figure(figsize=(4, 2))
    plt.hist(
        recording_ious[approach],
        bins=n_bins,
        label=[i for i in range(1, 23) if i not in excluded_recordings],
        stacked=True,
    )
    plt.xlabel("3D Bounding Box IoU")
    plt.ylabel("Number of Samples")
    plt.title(f"IoU Scores per Recording: {approach}")
    plt.xlim(0, 1)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

    for i in range(3):
        # false positives have assigned score of -2.0, so move them to 0.0
        raw_category_scores = np.array(category_ious[approach][i])
        category_scores[i] = np.maximum(0, raw_category_scores)
        print(
            f"{approach}\t {category_names[i]}\t IoU: {np.mean(category_scores[i]):.2f} +- {np.std(category_scores[i]):.2f}"
        )

        for thres in np.arange(0.05, 0.95, step=0.05):
            tp = (raw_category_scores >= thres).sum()
            fp = (
                np.logical_and(0 <= raw_category_scores, raw_category_scores < thres).sum()
                + (raw_category_scores == -2.0).sum()
            )
            fn = (
                np.logical_and(0 <= raw_category_scores, raw_category_scores < thres).sum()
                + (raw_category_scores == -1.0).sum()
            )

            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0
            print(
                f"{approach}\t {category_names[i]}\t IoU Threshold {thres:.2f}:\tPrecision {precision:.2f},\tRecall {recall:.2f},\tF1 Score {f1:.2f}"
            )

    all_scores = np.concatenate([scores for _, scores in category_ious[approach].items()])
    print(f"{approach}\t Overall\t IoU: {np.mean(all_scores):.2f} +- {np.std(all_scores):.2f}")
    for thres in np.arange(0.05, 0.95, step=0.05):
        tp = (all_scores >= thres).sum()
        fp = np.logical_and(0 <= all_scores, all_scores < thres).sum() + (all_scores == -2.0).sum()
        fn = np.logical_and(0 <= all_scores, all_scores < thres).sum() + (all_scores == -1.0).sum()
        # print(f"{approach}\t Overall\t IoU Threshold {thres:.2f} {tp=}, {fp=}, {fn=}")

        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        f1 = 2 * (precision * recall) / (precision + recall)
        print(
            f"{approach}\t Overall\t IoU Threshold {thres:.2f}:\tPrecision {precision:.2f},\tRecall {recall:.2f},\tF1 Score {f1:.2f}"
        )

    plt.figure(figsize=(4, 2))
    plt.hist(
        category_scores, bins=n_bins, label=category_names, color=category_colors, stacked=True
    )
    plt.xlabel("3D Bounding Box IoU")
    plt.ylabel("Number of Samples")
    # plt.title(f"IoU Scores per Category: {approach}")
    plt.xlim(0, 0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"data/evaluation/bbox_iou_per_category.pdf")
    plt.show()

# evaluate and generate plots for bbox center errors
for approach, _ in approach_filename_templates.items():
    plt.figure(figsize=(4, 2))
    plt.hist(
        recording_bbox_center_errs[approach],
        bins=n_bins,
        label=[i for i in range(1, 23) if i not in excluded_recordings],
        stacked=True,
    )
    plt.xlabel("3D Bounding Box Center Error [mm]")
    plt.ylabel("Number of Samples")
    plt.title(f"3D Bounding Box Center Errors per Recording: {approach}")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

    for i in range(3):
        category_scores[i] = np.array(category_bbox_center_errs[approach][i])
        print(
            f"{approach}\t {category_names[i]:10s}\t 3D BBox Center Error: {np.mean(category_scores[i]):.2f} +- {np.std(category_scores[i]):.2f}"
        )

    all_scores = np.concatenate([category_bbox_center_errs[approach][i] for i in range(3)])
    print(
        f"{approach}\t Overall\t 3D BBox Center Error: {np.mean(all_scores):.2f} +- {np.std(all_scores):.2f}"
    )

    plt.figure(figsize=(4, 2))
    plt.hist(
        category_scores, bins=n_bins, label=category_names, color=category_colors, stacked=True
    )
    plt.xlabel("3D Bounding Box Center Error [mm]")
    plt.ylabel("Number of Samples")
    # plt.title(f"3D Bounding Box Center Errors per Category: {approach}")
    plt.xlim(left=0, right=MAX_DIST_ERROR)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"data/evaluation/bbox_center_error_per_category.pdf")
    plt.show()
