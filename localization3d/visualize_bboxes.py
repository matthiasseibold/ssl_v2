from pathlib import Path
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pyzed.sl as sl
from tqdm import tqdm
import open3d as o3d
from argparse import ArgumentParser
from inout import load_calibration
from utils import (
    get_scene_category,
    project_points,
    calculate_color_score,
    iou_3d,
)

visualization_config = {
    "eye": np.array([-1000.0, -800.0, 200.0]),
    "lookat": np.array([11, -179.0, 1492]),
    "up": np.array([0.0, -1.0, 0.0]),
    "bg_color": np.array([1.0, 1.0, 1.0, 1.0]),
    "field_of_view": 40,
    "bbox_decay_frame_count": -1,
    "W": 1920,
    "H": 1080,
}

webcam_image_roi = [240, 20, 1600, 1030]  # xyxy format

pc_mat = o3d.visualization.rendering.MaterialRecord()
pc_mat.shader = "defaultUnlit"

gt_bbox_mat = o3d.visualization.rendering.MaterialRecord()
gt_bbox_mat.shader = "unlitLine"
gt_bbox_mat.line_width = 5.0
gt_base_color = np.array([0.0, 1.0, 0.0, 1.0])  # RGBA

pred_bbox_mat = o3d.visualization.rendering.MaterialRecord()
pred_bbox_mat.shader = "unlitLine"
pred_bbox_mat.line_width = 5.0
pred_bbox_base_color = np.array([0.0, 0.0, 1.0, 1.0])  # RGBA

sphere_mat = o3d.visualization.rendering.MaterialRecord()
sphere_mat.shader = "defaultLit"
gt_sphere_base_color = np.array([0.0, 1.0, 0.0, 0.7])  # RGBA with transparency
pred_sphere_base_color = np.array([0.0, 0.0, 1.0, 0.7])  # RGBA with transparency

zed_to_ocv_coordinate_frame = np.zeros((4, 4))
zed_to_ocv_coordinate_frame[0, 0] = 1.0
zed_to_ocv_coordinate_frame[1, 2] = -1.0
zed_to_ocv_coordinate_frame[2, 1] = 1.0
zed_to_ocv_coordinate_frame[3, 3] = 1.0


def compute_animated_eye(frame_idx, mode, amplitude, period, base_eye, lookat, up):
    """Return an animated camera eye position for the given rendered-frame index.

    Both modes keep `lookat` and `up` fixed; only the eye position moves.

    Args:
        frame_idx: sequential index of the rendered frame (0-based).
        mode: "orbit_up" or "orbit_view".
        amplitude: for orbit_up – half-angle in degrees; for orbit_view – radius
                   as a fraction of the eye-to-lookat distance.
        period: number of frames per full oscillation cycle.
        base_eye: original static camera position (3-vector).
        lookat: the look-at point (3-vector).
        up: camera up vector (3-vector).
    """
    theta = 2 * np.pi * frame_idx / period
    if mode == "orbit_up":
        # Oscillate ±amplitude degrees around the up axis (sine wave, not full 360°)
        angle = np.radians(amplitude) * np.sin(theta)
        offset = base_eye - lookat
        up_norm = up / np.linalg.norm(up)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        # Rodrigues' rotation formula
        rotated = (
            offset * cos_a
            + np.cross(up_norm, offset) * sin_a
            + up_norm * np.dot(up_norm, offset) * (1 - cos_a)
        )
        return lookat + rotated
    elif mode == "orbit_view":
        # Circle in the plane orthogonal to the viewing direction
        view = lookat - base_eye
        dist = np.linalg.norm(view)
        view_norm = view / dist
        up_norm = up / np.linalg.norm(up)
        right = np.cross(view_norm, up_norm)
        right /= np.linalg.norm(right)
        plane_up = np.cross(right, view_norm)
        plane_up /= np.linalg.norm(plane_up)
        radius = amplitude * dist
        return base_eye + radius * (np.cos(theta) * right + np.sin(theta) * plane_up)
    return base_eye


def visualize_sound_sources(args):
    dataset_path = Path(args.dataset_path)
    assert dataset_path.is_dir(), f"Path not found: {args.dataset_path}"
    synchronization_path = dataset_path / "rocsync.json"
    assert synchronization_path.is_file(), f"Path not found: {synchronization_path}"
    out_path = Path(args.out_path)
    video_out_path = out_path / "visualization"

    # input streams
    webcam_rec_path = dataset_path / "webcam" / f"1_{args.scene_id:03d}_Movie2D_image.avi"
    rgbd_rec_path = dataset_path / "rgbd" / f"1_{args.scene_id:03d}_Movie2D_rgbd.svo2"

    # reconstruct output suffix to find the right pred bbox NPY file
    dbscan_r = args.dbscan_radius
    dbscan_min_weight = args.dbscan_min_weight
    output_suffix = f"_r{dbscan_r:.0f}_minWeight{dbscan_min_weight:.0f}"

    zed_resolution_downsample_factor = args.rgbd_downsampling_factor
    if zed_resolution_downsample_factor > 1:
        zed_target_resolution = sl.Resolution(
            int(1920 / zed_resolution_downsample_factor),
            int(1080 / zed_resolution_downsample_factor),
        )
        output_suffix += f"_ds{zed_resolution_downsample_factor}"
    else:
        zed_target_resolution = sl.Resolution(0, 0)

    # create visualization output directory
    (video_out_path / f"1_{args.scene_id:03d}_Movie2D{output_suffix}").mkdir(
        parents=True, exist_ok=True
    )

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

    # load saved bboxes from disk
    pred_bboxes_path = (
        out_path
        / "sound_source_bbox3d"
        / f"1_{args.scene_id:03d}_Movie2D_3dBboxes{output_suffix}.npy"
    )
    gt_bboxes_path = dataset_path / "bbox3d_labels" / f"1_{args.scene_id:03d}_Movie2D_3dBboxes.npy"
    assert (
        pred_bboxes_path.is_file()
    ), f"Pred bboxes not found: {pred_bboxes_path}. Run bbox_estimation.py first."
    assert (
        gt_bboxes_path.is_file()
    ), f"GT bboxes not found: {gt_bboxes_path}. Run bbox_estimation.py first."
    pred_3d_bboxes_arr = np.load(pred_bboxes_path)  # (N_event_frames, 8, 3)
    gt_bboxes_arr = np.load(gt_bboxes_path)  # (N_event_frames, 8, 3)

    # load event frame IDs
    event_frame_path = Path(args.event_frames) / f"{webcam_rec_path.stem[:13]}.csv"
    event_frames = np.loadtxt(event_frame_path, delimiter=",")[:, 0]
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
    assert webcam_heatmap_path.is_file(), f"Path does not exist: {webcam_heatmap_path}"
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

    if args.render_to_file:
        renderer = o3d.visualization.rendering.OffscreenRenderer(
            width=visualization_config["W"], height=visualization_config["H"]
        )
        renderer.scene.set_background(visualization_config["bg_color"])
        renderer.scene.show_skybox(False)

    pbar_total = (
        webcam_rec.get(cv2.CAP_PROP_FRAME_COUNT)
        if not args.only_event_frames
        else len(event_frames)
    )
    pbar = tqdm(total=pbar_total, desc="Frame")
    pred_3d_bbox_age = visualization_config["bbox_decay_frame_count"]
    gt_bbox_age = visualization_config["bbox_decay_frame_count"]

    pred_3d_bbox = None
    gt_bbox = None
    gt_sphere_diameter = None  # Will be computed from GT bbox
    event_array_idx = 0
    event_idx = 0
    frame_render_idx = 0
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
        # grab ZED frame
        zed_frame_id = int(round((ref_timestamp - zed_sync["first_frame"]) / zed_frame_steps, 0))
        if zed_frame_id < 0 or zed_frame_id >= zed_sync["n_frames"]:
            continue
        zed_rec.set_svo_position(zed_frame_id)
        if zed_rec.grab() != sl.ERROR_CODE.SUCCESS:
            continue
        rgb_left = sl.Mat()
        zed_rec.retrieve_image(rgb_left, sl.VIEW.LEFT, resolution=zed_target_resolution)
        point_cloud = sl.Mat()
        zed_rec.retrieve_measure(
            point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, resolution=zed_target_resolution
        )
        # project ZED point cloud into webcam and build heatmap-colored point cloud
        xyz_np = point_cloud.get_data()[:, :, :3].reshape(-1, 3)
        rgb_np = rgb_left.get_data()[:, :, [2, 1, 0]].reshape(-1, 3)  # BGR to RGB
        valid_mask = np.isfinite(xyz_np).all(axis=1)
        xyz_np = xyz_np[valid_mask]
        rgb_np = rgb_np[valid_mask]
        xyz_hom = np.concatenate([xyz_np, np.ones((xyz_np.shape[0], 1))], axis=1).T
        # change of coordinate system definition from zed to opencv
        xyz_hom = zed_to_ocv_coordinate_frame @ xyz_hom
        proj_pts = project_points(xyz_hom[:3].T, webcam_pose, webcam_K, webcam_dist_coeffs).T

        geometries = []
        if 0 <= visualization_config["bbox_decay_frame_count"] <= gt_bbox_age:
            gt_bbox = None
        if 0 <= visualization_config["bbox_decay_frame_count"] <= pred_3d_bbox_age:
            pred_3d_bbox = None
        # assign heatmap colors to point cloud
        proj_pts_int = proj_pts.astype(int)
        pts_in_webcam = proj_pts_int[0] >= webcam_image_roi[0]
        pts_in_webcam &= proj_pts_int[0] < webcam_image_roi[2]
        pts_in_webcam &= proj_pts_int[1] >= webcam_image_roi[1]
        pts_in_webcam &= proj_pts_int[1] < webcam_image_roi[3]
        proj_pts_int = proj_pts_int[:, pts_in_webcam]
        xyz_trunc = xyz_hom[:3].T
        # blend heatmap into colored point cloud
        heatmap_colors = ref_heatmap[proj_pts_int[1], proj_pts_int[0], ::-1]
        rgb_colors = rgb_np.copy()
        non_white_heatmap = np.any(heatmap_colors < 224, axis=1)
        heatmap_idx = np.arange(xyz_trunc.shape[0])[pts_in_webcam][non_white_heatmap]
        relative_age = 0.7
        rgb_colors[heatmap_idx, :] = (
            relative_age * heatmap_colors[non_white_heatmap, :].astype(float)
            + (1.0 - relative_age) * rgb_colors[heatmap_idx, :].astype(float)
        ).astype(np.uint8)
        heatmap_colors = rgb_colors

        if is_event_frame and event_array_idx < len(pred_3d_bboxes_arr):
            pred_corners = pred_3d_bboxes_arr[event_array_idx]
            if not np.all(pred_corners == 0):
                pred_3d_bbox = o3d.geometry.OrientedBoundingBox.create_from_points(
                    o3d.utility.Vector3dVector(pred_corners)
                )
                pred_3d_bbox_age = 0
            else:
                pred_3d_bbox = None
                pred_3d_bbox_age = visualization_config["bbox_decay_frame_count"]
            gt_corners = gt_bboxes_arr[event_array_idx]
            if not np.all(gt_corners == 0):
                gt_bbox = o3d.geometry.OrientedBoundingBox.create_from_points(
                    o3d.utility.Vector3dVector(gt_corners)
                )
                gt_bbox_age = 0
                # Compute sphere diameter from GT bbox (max extent)
                if (args.show_gt_sphere or args.show_pred_sphere) and gt_sphere_diameter is None:
                    gt_extent = gt_bbox.extent
                    gt_sphere_diameter = np.max(gt_extent)
            else:
                gt_bbox = None
                gt_bbox_age = visualization_config["bbox_decay_frame_count"]
            event_array_idx += 1

        pcd_heatmap = o3d.geometry.PointCloud()
        pcd_heatmap.points = o3d.utility.Vector3dVector(xyz_trunc)
        # Use heatmap-blended colors if enabled, otherwise use original RGB colors
        if args.show_heatmap:
            pcd_heatmap.colors = o3d.utility.Vector3dVector(heatmap_colors / 255.0)
        else:
            pcd_heatmap.colors = o3d.utility.Vector3dVector(rgb_np / 255.0)
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=75.0)
        geometries.append({"name": "ZED left", "geometry": origin, "material": pc_mat})
        webcam_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0)
        webcam_origin.transform(webcam_pose)
        geometries.append({"name": "Webcam", "geometry": webcam_origin, "material": pc_mat})
        geometries.append({"name": "Point Cloud", "geometry": pcd_heatmap, "material": pc_mat})

        # Render predicted sphere
        if pred_3d_bbox is not None and args.show_pred_sphere and gt_sphere_diameter is not None:
            if 0 <= visualization_config["bbox_decay_frame_count"]:
                relative_age = pred_3d_bbox_age / visualization_config["bbox_decay_frame_count"]
            else:
                relative_age = 0.0
            fade_factor = 1.0 - relative_age
            sphere_color = pred_sphere_base_color.copy()
            sphere_color[3] = sphere_color[3] * fade_factor
            sphere_mat_pred = o3d.visualization.rendering.MaterialRecord()
            sphere_mat_pred.shader = "defaultLitTransparency"
            sphere_mat_pred.base_color = sphere_color
            pred_center = pred_3d_bbox.center
            pred_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=gt_sphere_diameter / 2.0)
            pred_sphere.compute_vertex_normals()
            pred_sphere.translate(pred_center)
            pred_sphere.paint_uniform_color(sphere_color[:3])
            geometries.append(
                {"name": "Pred Sphere", "geometry": pred_sphere, "material": sphere_mat_pred}
            )

        # Render GT sphere
        if gt_bbox is not None and args.show_gt_sphere and gt_sphere_diameter is not None:
            if 0 <= visualization_config["bbox_decay_frame_count"]:
                relative_age = gt_bbox_age / visualization_config["bbox_decay_frame_count"]
            else:
                relative_age = 0.0
            fade_factor = 1.0 - relative_age
            sphere_color = gt_sphere_base_color.copy()
            sphere_color[3] = sphere_color[3] * fade_factor
            sphere_mat_gt = o3d.visualization.rendering.MaterialRecord()
            sphere_mat_gt.shader = "defaultLitTransparency"
            sphere_mat_gt.base_color = sphere_color
            gt_center = gt_bbox.center
            gt_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=gt_sphere_diameter / 2.0)
            gt_sphere.compute_vertex_normals()
            gt_sphere.translate(gt_center)
            gt_sphere.paint_uniform_color(sphere_color[:3])
            geometries.append(
                {"name": "GT Sphere", "geometry": gt_sphere, "material": sphere_mat_gt}
            )

        # Render predicted bounding box
        if pred_3d_bbox is not None and args.show_pred_bbox:
            if 0 <= visualization_config["bbox_decay_frame_count"]:
                relative_age = pred_3d_bbox_age / visualization_config["bbox_decay_frame_count"]
            else:
                relative_age = 0.0
            base_color = relative_age * np.ones(4) + (1.0 - relative_age) * pred_bbox_base_color
            base_color[3] = 1.0
            pred_bbox_mat.base_color = base_color
            line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(pred_3d_bbox)
            line_set.paint_uniform_color(base_color[:3])
            geometries.append(
                {"name": "Pred BBox", "geometry": line_set, "material": pred_bbox_mat}
            )

        # Render GT bounding box
        if gt_bbox is not None and args.show_gt_bbox:
            if 0 <= visualization_config["bbox_decay_frame_count"]:
                relative_age = gt_bbox_age / visualization_config["bbox_decay_frame_count"]
            else:
                relative_age = 0.0
            base_color = relative_age * np.ones(4) + (1.0 - relative_age) * gt_base_color
            base_color[3] = 1.0
            gt_bbox_mat.base_color = base_color
            line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(gt_bbox)
            line_set.paint_uniform_color(base_color[:3])
            geometries.append({"name": "GT BBox", "geometry": line_set, "material": gt_bbox_mat})

        if args.render_webcam:
            rows, cols = 1, 3
            plt.rcParams["figure.figsize"] = (15 * cols, 13 * rows)
            plt.subplot(rows, cols, 1)
            plt.title("Webcam")
            plt.imshow(ref_frame[:, :, ::-1])
            plt.subplot(rows, cols, 2)
            plt.title("Heatmap")
            plt.imshow(ref_heatmap[:, :, ::-1])
            plt.subplot(rows, cols, 3)
            plt.title("Scoremap")
            score_heatmap = np.zeros_like(ref_heatmap, dtype=np.float32)
            score_heatmap[
                webcam_image_roi[1] : webcam_image_roi[3],
                webcam_image_roi[0] : webcam_image_roi[2],
                :,
            ] = calculate_color_score(
                ref_heatmap[
                    webcam_image_roi[1] : webcam_image_roi[3],
                    webcam_image_roi[0] : webcam_image_roi[2],
                    ::-1,
                ]
            )[
                ..., None
            ]
            plt.imshow((255.0 * score_heatmap).astype(np.uint8))
            plt.show()

        if args.interactive:
            print(
                f"IoU Scores: {iou_3d(np.asarray(pred_3d_bbox.get_box_points()), np.asarray(gt_bbox.get_box_points())) if pred_3d_bbox is not None and gt_bbox is not None else -1:.3f}"
            )
            o3d.visualization.draw(
                geometries,
                title="Sound Source Localization",
                eye=visualization_config["eye"],
                lookat=visualization_config["lookat"],
                up=visualization_config["up"],
                field_of_view=visualization_config["field_of_view"],
                width=visualization_config["W"],
                height=visualization_config["H"],
                show_skybox=False,
                bg_color=visualization_config["bg_color"],
            )

        if args.render_to_file:
            renderer.scene.clear_geometry()
            for g in geometries:
                renderer.scene.add_geometry(g["name"], g["geometry"], g["material"])
            # Compute (animated) camera eye position
            eye = visualization_config["eye"]
            if args.camera_motion:
                eye = compute_animated_eye(
                    frame_render_idx,
                    args.camera_motion,
                    args.camera_motion_amplitude,
                    args.camera_motion_period,
                    visualization_config["eye"],
                    visualization_config["lookat"],
                    visualization_config["up"],
                )
            # Call setup_camera only after all geometry has been added!
            renderer.setup_camera(
                center=visualization_config["lookat"],
                eye=eye,
                up=visualization_config["up"],
                vertical_field_of_view=visualization_config["field_of_view"],
            )
            img = renderer.render_to_image()
            o3d.io.write_image(
                video_out_path
                / f"1_{args.scene_id:03d}_Movie2D{output_suffix}"
                / f"{ref_frame_id:06d}.jpg",
                img,
            )

        gt_bbox_age += 1
        pred_3d_bbox_age += 1
        frame_render_idx += 1
        pbar.update()

    pbar.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", default="data/dataset/", help="Path to the dataset")
    parser.add_argument("--scene_id", required=True, type=int, help="Scene to visualize")
    parser.add_argument(
        "--out_path",
        default="data/prediction/event_localization3d/",
        help="Where the predictions were saved (and where visualization output goes)",
    )
    parser.add_argument("--render_webcam", action="store_true", help="Show matplotlib debug plots")
    parser.add_argument(
        "--render_to_file", action="store_true", help="Render visualization frames to disk"
    )
    parser.add_argument(
        "--event_frames",
        type=str,
        help="Path to event frame detections.",
    )
    parser.add_argument(
        "--only_event_frames",
        action="store_true",
        help="Visualize only frames with detected acoustic events.",
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Show per-frame interactive visualization"
    )
    parser.add_argument(
        "--dbscan_radius",
        type=float,
        default=30.0,
        help="DBSCAN radius used during estimation (to locate the correct NPY file)",
    )
    parser.add_argument(
        "--dbscan_min_weight",
        type=int,
        default=400,
        help="DBSCAN min weight used during estimation (to locate the correct NPY file)",
    )
    parser.add_argument(
        "--rgbd_downsampling_factor",
        type=int,
        default=1,
        help="Downsampling factor used during estimation (to locate the correct NPY file)",
    )
    parser.add_argument(
        "--show_heatmap",
        action="store_true",
        help="Show heatmap overlay on point cloud (default: hidden)",
    )
    parser.add_argument(
        "--show_gt_bbox",
        action="store_true",
        help="Show ground truth bounding box (default: hidden)",
    )
    parser.add_argument(
        "--show_pred_bbox",
        action="store_true",
        help="Show predicted bounding box (default: hidden)",
    )
    parser.add_argument(
        "--show_gt_sphere",
        action="store_true",
        help="Show GT as semitransparent sphere (default: hidden)",
    )
    parser.add_argument(
        "--show_pred_sphere",
        action="store_true",
        help="Show prediction as semitransparent sphere (default: hidden)",
    )
    parser.add_argument(
        "--camera_motion",
        choices=["orbit_up", "orbit_view"],
        default=None,
        help=(
            "Animate the camera position across frames. "
            "orbit_up: oscillate around the up/gravity axis (turntable). "
            "orbit_view: trace a small circle in the plane perpendicular to the view direction."
        ),
    )
    parser.add_argument(
        "--camera_motion_amplitude",
        type=float,
        default=30.0,
        help=(
            "orbit_up: half-angle of oscillation in degrees (default 30). "
            "orbit_view: radius as fraction of camera-to-lookat distance (default 0.05)."
        ),
    )
    parser.add_argument(
        "--camera_motion_period",
        type=int,
        default=120,
        help="Number of rendered frames per full oscillation cycle (default 120).",
    )
    args = parser.parse_args()

    visualize_sound_sources(args)
