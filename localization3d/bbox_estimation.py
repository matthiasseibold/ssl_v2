import time
from pathlib import Path
import json
import numpy as np
import cv2
import pyzed.sl as sl
from tqdm import tqdm
import open3d as o3d
from argparse import ArgumentParser
from inout import load_calibration, load_ftk, read_ftk_pose, get_object_info
from utils import (
    get_scene_category,
    project_points,
    calculate_color_score,
    find_weighted_cluster_centers,
)

webcam_image_roi = [240, 20, 1600, 1030]  # xyxy format

zed_to_ocv_coordinate_frame = np.zeros((4, 4))
zed_to_ocv_coordinate_frame[0, 0] = 1.0
zed_to_ocv_coordinate_frame[1, 2] = -1.0
zed_to_ocv_coordinate_frame[2, 1] = 1.0
zed_to_ocv_coordinate_frame[3, 3] = 1.0


def estimate_sound_sources(args):
    dataset_path = Path(args.dataset_path)
    assert dataset_path.is_dir(), f"Path not found: {args.dataset_path}"
    synchronization_path = dataset_path / "rocsync.json"
    assert synchronization_path.is_file(), f"Path not found: {synchronization_path}"
    out_path = Path(args.out_path)

    projection_latencies = []
    localization_latencies = []

    # input streams
    webcam_rec_path = dataset_path / "webcam" / f"1_{args.scene_id:03d}_Movie2D_image.avi"
    rgbd_rec_path = dataset_path / "rgbd" / f"1_{args.scene_id:03d}_Movie2D_rgbd.svo2"
    ftk_rec_path = dataset_path / "tracking_system" / f"1_{args.scene_id:03d}_Movie2D_tracking.csv"

    # generate output suffix
    dbscan_r = args.dbscan_radius
    dbscan_min_weight = args.dbscan_min_weight
    output_suffix = f"_r{dbscan_r:.0f}_minWeight{dbscan_min_weight:.0f}"

    # ZED point cloud resolution (H*W)
    zed_resolution_downsample_factor = args.rgbd_downsampling_factor
    if zed_resolution_downsample_factor > 1:
        zed_target_resolution = sl.Resolution(
            int(1920 / zed_resolution_downsample_factor),
            int(1080 / zed_resolution_downsample_factor),
        )
        output_suffix += f"_ds{zed_resolution_downsample_factor}"
    else:
        zed_target_resolution = sl.Resolution(0, 0)

    # create output directories
    (dataset_path / "bbox3d_labels").mkdir(parents=True, exist_ok=True)
    (out_path / "sound_source_bbox3d").mkdir(parents=True, exist_ok=True)
    (out_path / "webcam_rgb" / webcam_rec_path.stem).mkdir(parents=True, exist_ok=True)
    (out_path / "webcam_heatmap" / webcam_rec_path.stem).mkdir(parents=True, exist_ok=True)
    (out_path / "stereo_rgb_left" / webcam_rec_path.stem).mkdir(parents=True, exist_ok=True)
    (out_path / "stereo_rgb_right" / webcam_rec_path.stem).mkdir(parents=True, exist_ok=True)
    (out_path / "rgbd_pointcloud" / webcam_rec_path.stem).mkdir(parents=True, exist_ok=True)

    (
        webcam_pose,
        zed_left_pose,
        ftk_pose,
        webcam_clock_offset,
        zed_clock_offset,
        ftk_clock_offset,
        webcam_K,
        webcam_dist_coeffs,
    ) = load_calibration(dataset_path)
    synchronization_info = json.load(synchronization_path.open("r"))
    (
        meshes,
        object_marker_poses,
        mesh_per_category,
        mesh_diameters,
        mesh_extent_range,
        expected_mesh_extent,
    ) = get_object_info(dataset_path / "object_models")

    # load bbox detections (only for event frame ids)
    event_frame_path = Path(args.event_frames) / f"{webcam_rec_path.stem[:13]}.csv"
    event_frames = np.loadtxt(event_frame_path, delimiter=",")[:, 0]
    # load ftk
    scene_category = get_scene_category(args.scene_id)
    ftk_frames = load_ftk(ftk_rec_path, list(object_marker_poses[scene_category].keys()))
    # load webcam
    webcam_rec = cv2.VideoCapture(webcam_rec_path, cv2.CAP_FFMPEG)
    webcam_sync = synchronization_info.get(webcam_rec_path.name, None)
    assert webcam_sync is not None
    assert webcam_rec.get(cv2.CAP_PROP_FRAME_COUNT) == webcam_sync["n_frames"]
    webcam_frame_steps = (webcam_sync["last_frame"] - webcam_sync["first_frame"]) / (
        webcam_sync["n_frames"] - 1
    )
    # load heatmap
    webcam_heatmap_path = (
        webcam_rec_path.parent.parent
        / "webcam_acoustic_heatmap"
        / f"{webcam_rec_path.stem[:13]}_heatmap.avi"
    )
    assert webcam_heatmap_path.is_file(), f"Path does not exist: {webcam_rec_path}"
    webcam_heatmap_rec = cv2.VideoCapture(webcam_heatmap_path, cv2.CAP_FFMPEG)
    # load zed
    zed_sync = synchronization_info.get(rgbd_rec_path.name, None)
    assert zed_sync is not None
    zed_frame_steps = (zed_sync["last_frame"] - zed_sync["first_frame"]) / (
        zed_sync["n_frames"] - 1
    )
    sl_init = sl.InitParameters(
        depth_mode=sl.DEPTH_MODE.NEURAL,
        coordinate_units=sl.UNIT.MILLIMETER,
        coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP,
    )
    sl_init.set_from_svo_file(str(rgbd_rec_path))
    zed_rec = sl.Camera()
    if zed_rec.open(sl_init) != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError("Error opening SL")

    pbar_total = (
        webcam_rec.get(cv2.CAP_PROP_FRAME_COUNT)
        if not args.only_event_frames
        else len(event_frames)
    )
    pbar = tqdm(total=pbar_total, desc="Frame")

    gt_bboxes = []
    pred_3d_bboxes = []
    mesh_key = None
    pred_3d_bbox = None
    gt_bbox = None
    event_idx = 0
    while webcam_rec.isOpened():
        if args.only_event_frames:
            if event_idx >= len(event_frames):
                break
            if not webcam_rec.set(cv2.CAP_PROP_POS_FRAMES, event_frames[event_idx]):
                print("ERROR setting ref_frame_id webcam_rec")
                break
            is_event_frame = True
            event_idx += 1
        ref_frame_id = int(webcam_rec.get(cv2.CAP_PROP_POS_FRAMES))
        if not webcam_rec.set(cv2.CAP_PROP_POS_FRAMES, ref_frame_id):
            print("ERROR setting ref_frame_id")
            break
        ret, ref_frame = webcam_rec.read()
        if not ret:
            print("ERROR reading ref_frame")
            break
        if not webcam_heatmap_rec.set(cv2.CAP_PROP_POS_FRAMES, ref_frame_id):
            print("ERROR setting ref_frame_id heatmap")
            break
        ret, ref_heatmap = webcam_heatmap_rec.read()
        if not ret:
            print("ERROR reading ref_heatmap")
            break

        if not args.only_event_frames:
            event_idx = np.searchsorted(event_frames, ref_frame_id)
            is_event_frame = (
                0 <= event_idx < len(event_frames) and event_frames[event_idx] == ref_frame_id
            )

        ref_timestamp = webcam_frame_steps * ref_frame_id + webcam_sync["first_frame"]  # in ms
        webcam_rgb_path = out_path / "webcam_rgb" / webcam_rec_path.stem / f"{ref_frame_id:06d}.jpg"
        if args.save_raw_frames and not webcam_rgb_path.exists():
            cv2.imwrite(webcam_rgb_path, ref_frame)
        webcam_heatmap_frame_path = (
            out_path / "webcam_heatmap" / webcam_rec_path.stem / f"{ref_frame_id:06d}.jpg"
        )
        if args.save_raw_frames and is_event_frame and not webcam_heatmap_frame_path.exists():
            cv2.imwrite(webcam_heatmap_frame_path, ref_heatmap)
        # grab ZED frame
        zed_frame_id = int(round((ref_timestamp - zed_sync["first_frame"]) / zed_frame_steps, 0))
        if zed_frame_id < 0 or zed_frame_id >= zed_sync["n_frames"]:
            continue
        zed_rec.set_svo_position(zed_frame_id)
        if zed_rec.grab() != sl.ERROR_CODE.SUCCESS:
            continue
        zed_frame_time = zed_rec.get_timestamp(sl.TIME_REFERENCE.IMAGE).data_ns / 1e3  # in ms
        zed_left_rgb_path = (
            out_path / "stereo_rgb_left" / webcam_rec_path.stem / f"{ref_frame_id:06d}.jpg"
        )
        rgb_left = sl.Mat()
        zed_rec.retrieve_image(rgb_left, sl.VIEW.LEFT, resolution=zed_target_resolution)
        if args.save_raw_frames and is_event_frame and not zed_left_rgb_path.exists():
            rgb_left.write(str(zed_left_rgb_path))
        zed_right_rgb_path = (
            out_path / "stereo_rgb_right" / webcam_rec_path.stem / f"{ref_frame_id:06d}.jpg"
        )
        if args.save_raw_frames and is_event_frame and not zed_right_rgb_path.exists():
            rgb_right = sl.Mat()
            zed_rec.retrieve_image(rgb_right, sl.VIEW.RIGHT, resolution=zed_target_resolution)
            rgb_right.write(str(zed_right_rgb_path))
        zed_point_cloud_path = (
            out_path / "rgbd_pointcloud" / webcam_rec_path.stem / f"{ref_frame_id:06d}.ply"
        )
        point_cloud = sl.Mat()
        zed_rec.retrieve_measure(
            point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, resolution=zed_target_resolution
        )
        if args.save_raw_frames and is_event_frame and not zed_point_cloud_path.exists():
            point_cloud.write(str(zed_point_cloud_path))
        # project ZED point cloud into webcam
        start_time = time.perf_counter()
        xyz_np = point_cloud.get_data()[:, :, :3].reshape(-1, 3)
        valid_mask = np.isfinite(xyz_np).all(axis=1)
        xyz_np = xyz_np[valid_mask]
        xyz_hom = np.concatenate([xyz_np, np.ones((xyz_np.shape[0], 1))], axis=1).T
        # change of coordinate system definition from zed to opencv
        xyz_hom = zed_to_ocv_coordinate_frame @ xyz_hom
        proj_pts = project_points(xyz_hom[:3].T, webcam_pose, webcam_K, webcam_dist_coeffs).T

        proj_pts_int = proj_pts.astype(int)
        pts_in_webcam = proj_pts_int[0] >= webcam_image_roi[0]
        pts_in_webcam &= proj_pts_int[0] < webcam_image_roi[2]
        pts_in_webcam &= proj_pts_int[1] >= webcam_image_roi[1]
        pts_in_webcam &= proj_pts_int[1] < webcam_image_roi[3]
        proj_pts_int = proj_pts_int[:, pts_in_webcam]
        xyz_trunc = xyz_hom[:3].T

        projection_latencies.append(time.perf_counter() - start_time)
        start_time = time.perf_counter()

        if is_event_frame:
            pred_3d_bbox = None  # reset for this event frame
            point_scores = calculate_color_score(
                ref_heatmap[proj_pts_int[1], proj_pts_int[0], ::-1]
            )
            cluster_centers, labels = find_weighted_cluster_centers(
                xyz_trunc[pts_in_webcam], point_scores, dbscan_r, dbscan_min_weight
            )
            unique_labels, label_counts = np.unique(labels[labels != -1], return_counts=True)
            sort_idx = np.argsort(label_counts)
            cluster_centers = cluster_centers[sort_idx]
            for i in range(len(cluster_centers[:1])):
                cluster_filter = labels == i
                cluster_pc = o3d.geometry.PointCloud()
                cluster_pc.points = o3d.utility.Vector3dVector(
                    xyz_trunc[pts_in_webcam][cluster_filter]
                )
                cluster_bbox = cluster_pc.get_minimal_oriented_bounding_box(True)
                extents = np.asarray(cluster_bbox.extent)
                extents = np.minimum(
                    extents, mesh_extent_range[mesh_per_category[scene_category]][1]
                )
                extents = np.maximum(
                    extents, mesh_extent_range[mesh_per_category[scene_category]][0]
                )
                pred_3d_bbox = o3d.geometry.OrientedBoundingBox(
                    cluster_bbox.center, cluster_bbox.R, extents
                )
                localization_latencies.append(time.perf_counter() - start_time)

        ftk_query_time = zed_frame_time - zed_clock_offset + ftk_clock_offset
        ftk_frame_idx = np.searchsorted(ftk_frames[:, 0], ftk_query_time, side="left")
        if ftk_frame_idx >= ftk_frames.shape[0]:
            ftk_frame_idx = ftk_frames.shape[0] - 1
        ftk_time = ftk_frames[ftk_frame_idx, 0]
        if abs(ftk_query_time - ftk_time) < 45000:
            idx = ftk_frame_idx
            while idx < ftk_frames.shape[0] and ftk_frames[idx, 0] == ftk_time:
                marker_id = int(ftk_frames[idx, 1])
                if marker_id in object_marker_poses[scene_category]:
                    mesh_key, mesh_pose = object_marker_poses[scene_category][marker_id]
                    mesh = o3d.geometry.TriangleMesh(meshes[mesh_key])
                    marker_pose = read_ftk_pose(ftk_frames[idx])
                    mesh_pose = np.linalg.inv(ftk_pose) @ marker_pose @ mesh_pose
                    mesh.transform(mesh_pose)
                    gt_bbox = mesh.get_minimal_oriented_bounding_box(True)
                idx += 1
            if idx >= ftk_frames.shape[0]:
                break

        if is_event_frame:
            if gt_bbox is not None:
                gt_bboxes.append(np.asarray(gt_bbox.get_box_points()))
            else:
                gt_bboxes.append(np.zeros((8, 3)))
            if pred_3d_bbox is not None:
                pred_3d_bboxes.append(np.asarray(pred_3d_bbox.get_box_points()))
            else:
                pred_3d_bboxes.append(np.zeros((8, 3)))

        pbar.update()

    if len(gt_bboxes) > 0:
        gt_bboxes = np.stack(gt_bboxes)
    np.save(
        dataset_path / "bbox3d_labels" / f"1_{args.scene_id:03d}_Movie2D_3dBboxes.npy", gt_bboxes
    )
    if len(pred_3d_bboxes) > 0:
        pred_3d_bboxes = np.stack(pred_3d_bboxes)
    np.save(
        out_path
        / "sound_source_bbox3d"
        / f"1_{args.scene_id:03d}_Movie2D_3dBboxes{output_suffix}.npy",
        pred_3d_bboxes,
    )
    pbar.close()

    projection_latencies = np.array(projection_latencies) * 1000  # convert to ms
    localization_latencies = np.array(localization_latencies) * 1000  # convert to ms
    print(
        f"Projection Latency: {np.mean(projection_latencies):.2f} +- {np.std(projection_latencies):.2f} ms"
    )
    print(
        f"Localization Latency: {np.mean(localization_latencies):.2f} +- {np.std(localization_latencies):.2f} ms"
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", default="data/dataset/", help="Path to the dataset")
    parser.add_argument("--scene_id", required=True, type=int, help="Scene to process")
    parser.add_argument(
        "--out_path",
        default="data/prediction/event_localization3d/",
        help="Where to save the predictions",
    )
    parser.add_argument(
        "--save_raw_frames", action="store_true", help="Save input frames used for predictions"
    )
    parser.add_argument(
        "--event_frames",
        type=str,
        help="Path to event frame detections.",
    )
    parser.add_argument(
        "--only_event_frames",
        action="store_true",
        help="Predict sound source only on frames with detected acoustic events.",
    )
    parser.add_argument(
        "--dbscan_radius", type=float, default=30.0, help="Radius for DBSCAN clustering"
    )
    parser.add_argument(
        "--dbscan_min_weight",
        type=int,
        default=400,
        help="Cluster minimal weight threshold for DBSCAN clustering",
    )
    parser.add_argument(
        "--rgbd_downsampling_factor",
        type=int,
        default=1,
        help="Downsampling factor for the width _and_ height of the input RGB-D image representing the 3D scene.",
    )
    args = parser.parse_args()

    estimate_sound_sources(args)
