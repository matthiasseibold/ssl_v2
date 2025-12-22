import json
import numpy as np
import open3d as o3d
from pathlib import Path
from typing import Sequence


def read_matrix(calib, keys, default=None):
    for k in keys:
        if calib is None:
            break
        calib = calib.get(k, None)
    if calib is None:
        return default
    return np.array([s.split(",") for s in calib.split(";")]).astype(float)


def load_calibration(dataset_path: Path):
    webcam_path = dataset_path / "webcam" / "calibration_info.json"
    zed_path = dataset_path / "rgbd" / "calibration_info.json"
    ftk_path = dataset_path / "tracking_system" / "calibration_info.json"

    assert webcam_path.is_file(), f"File not found: {webcam_path}"
    assert zed_path.is_file(), f"File not found: {zed_path}"
    assert ftk_path.is_file(), f"File not found: {ftk_path}"

    webcam_calib = json.load(webcam_path.open("r"))
    zed_calib = json.load(zed_path.open("r"))
    ftk_calib = json.load(ftk_path.open("r"))

    zed_device_pose = read_matrix(zed_calib, ["calibration", "deviceExtrinsics", "data"], np.eye(4))
    zed_left_sensor_pose = read_matrix(
        zed_calib, ["calibration", "sensors", "rgb_left", "extrinsics", "data"], np.eye(4)
    )
    zed_left_pose = zed_left_sensor_pose @ zed_device_pose

    webcam_device_pose = read_matrix(
        webcam_calib, ["calibration", "deviceExtrinsics", "data"], np.eye(4)
    )
    webcam_sensor_pose = read_matrix(
        webcam_calib, ["calibration", "sensors", "default", "extrinsics", "data"], np.eye(4)
    )
    webcam_pose = webcam_sensor_pose @ webcam_device_pose

    ftk_device_pose = read_matrix(ftk_calib, ["calibration", "deviceExtrinsics", "data"], np.eye(4))
    ftk_sensor_pose = read_matrix(
        ftk_calib, ["calibration", "sensors", "default", "extrinsics", "data"], np.eye(4)
    )
    ftk_pose = ftk_sensor_pose @ ftk_device_pose

    # zed_left_pose is our world coordinate frame
    zed_left_inv_pose = np.linalg.inv(zed_left_pose)
    webcam_pose = webcam_pose @ zed_left_inv_pose
    ftk_pose = ftk_pose @ zed_left_inv_pose
    zed_left_pose = np.eye(4)

    webcam_clock_offset = webcam_calib.get("calibration", {}).get("clockOffsetUs", None)
    zed_clock_offset = zed_calib.get("calibration", {}).get("clockOffsetUs", None)
    ftk_clock_offset = ftk_calib.get("calibration", {}).get("clockOffsetUs", None)

    webcam_K = read_matrix(
        webcam_calib, ["calibration", "sensors", "default", "intrinsics", "data"], np.eye(3)
    )
    webcam_dist_coeffs = read_matrix(
        webcam_calib,
        ["calibration", "sensors", "default", "distortionCoefficients", "data"],
        np.zeros(5),
    )

    return (
        webcam_pose,
        zed_left_pose,
        ftk_pose,
        webcam_clock_offset,
        zed_clock_offset,
        ftk_clock_offset,
        webcam_K,
        webcam_dist_coeffs,
    )


def get_object_info(objects_dir: Path):
    arthrex_1607 = np.array(
        [
            [0.08563, -0.1071, 0.9906, -7.517],
            [0.8428, 0.538, -0.01466, -106.1],
            [-0.5314, 0.8361, 0.1364, 36.01],
            [0, 0, 0, 1],
        ]
    )
    arthrex_16070 = np.array(
        [
            [0.7229, -0.05464, 0.6888, 39.25],
            [-0.5911, -0.5651, 0.5755, 32.98],
            [0.3578, -0.8232, -0.4408, -86.72],
            [0, 0, 0, 1],
        ]
    )
    chisel_1608 = np.array(
        [
            [0.4854, -0.09175, -0.8695, -91.91],
            [0.5929, -0.6964, 0.4044, 90.64],
            [-0.6426, -0.7118, -0.2836, 24.35],
            [0, 0, 0, 1],
        ]
    )
    chisel_16080 = np.array(
        [
            [0.7081, -0.01893, 0.7059, 91.74],
            [-0.7054, 0.02713, 0.7083, 51.24],
            [-0.03256, -0.9995, 0.005859, 32.12],
            [0, 0, 0, 1],
        ]
    )

    meshes = {
        "Arthrex_saw_lowres": o3d.io.read_triangle_mesh(
            objects_dir / "Arthrex_saw_lowres_nomarker.ply"
        ),
        "Arthrex_drill_lowres": o3d.io.read_triangle_mesh(
            objects_dir / "Arthrex_drill_lowres_nomarker.ply"
        ),
        "Chisel_lowres": o3d.io.read_triangle_mesh(objects_dir / "Chisel_contact_bbox_5cm.ply"),
    }
    mesh_per_category = {
        "sawing": "Arthrex_saw_lowres",
        "drilling": "Arthrex_drill_lowres",
        "chiseling": "Chisel_lowres",
    }
    mesh_extent_range = {
        k: (
            np.min(m.get_oriented_bounding_box().extent),
            np.max(m.get_oriented_bounding_box().extent),
        )
        for k, m in meshes.items()
    }
    expected_mesh_extent = {
        k: 0.5 * np.sum(m.get_oriented_bounding_box().extent) for k, m in meshes.items()
    }
    mesh_diameters = {
        k: np.linalg.norm(m.get_oriented_bounding_box().extent) for k, m in meshes.items()
    }
    object_marker_poses = {
        "sawing": {
            1607: ("Arthrex_saw_lowres", arthrex_1607),
            16070: ("Arthrex_saw_lowres", arthrex_16070),
        },
        "drilling": {
            1607: ("Arthrex_drill_lowres", arthrex_1607),
            16070: ("Arthrex_drill_lowres", arthrex_16070),
        },
        "chiseling": {
            1608: ("Chisel_lowres", chisel_1608),
            16080: ("Chisel_lowres", chisel_16080),
        },
    }

    return (
        meshes,
        object_marker_poses,
        mesh_per_category,
        mesh_diameters,
        mesh_extent_range,
        expected_mesh_extent,
    )


def read_ftk_pose(line_np):
    pose = np.eye(4)
    pose[:3, 3] = line_np[2:5]
    pose[:3, :3] = line_np[5:14].reshape(3, 3)
    return pose


def load_ftk(
    gt_file: Path,
    filter_by_marker_ids: Sequence[int] = None,
):
    lines = []
    with gt_file.open("r") as f:
        for line in f:
            stripped_line = line.strip()
            if stripped_line:  # Avoid adding empty lists for blank lines
                split_row = stripped_line.split(",")
                if len(split_row) >= 16:
                    lines.append(split_row[:16])
    gt = np.array(lines)
    # For markers:
    # hostTime, deviceTime, m, markerId, posX, posY, posZ, rot00, rot01, rot02, rot10, rot11, rot12, rot20, rot21, rot22, registrationError, fiducial0Idx, fiducial1Idx, ...
    # 1627577420878487, 115348993058, m, 9930, 598.013, -112.324, 1287.12, -0.368266, -0.110821, 0.923092, 0.830708, -0.485078, 0.273174, 0.417498, 0.867421, 0.270698, 0.0612486
    # Filter fiducials, remove hostTime, marker/fiducial indicator
    gt = gt[gt[:, 2] == "m", 1:]
    gt = np.delete(gt, 1, 1).astype(float)
    # deviceTime, markerId, posX, posY, posZ, rot00, rot01, rot02, rot10, rot11, rot12, rot20, rot21, rot22, registrationError, fiducial0Idx, fiducial1Idx, ...
    gt = gt[gt[:, 0].argsort()]
    # filter by marker ids
    if filter_by_marker_ids is not None:
        id_filter = np.isin(gt[:, 1], filter_by_marker_ids)
        gt = gt[id_filter, :]
    return gt
