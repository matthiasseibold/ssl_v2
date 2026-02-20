from pathlib import Path
import ctypes
import json
import queue
import subprocess
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pyzed.sl as sl
from tqdm import tqdm
import open3d as o3d
import numba.cuda as cuda
from argparse import ArgumentParser
from inout import load_calibration
from utils import (
    get_scene_category,
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


def _load_pc_kernel():
    """Compile _pc_kernel.cu with nvcc (if .so is absent or stale) and return
    a ctypes-callable launcher.

    numba's JIT can't link PTX 8.5 against the current CUDA 12.2 driver.
    Compiling directly to a native .so via nvcc bypasses the driver's PTX JIT
    entirely: the cubin is loaded directly into the CUDA context, no PTX
    version negotiation required.
    """
    kernel_dir = Path(__file__).parent
    cu_path = kernel_dir / "_pc_kernel.cu"
    so_path = kernel_dir / "_pc_kernel.so"

    needs_compile = not so_path.exists() or cu_path.stat().st_mtime > so_path.stat().st_mtime
    if needs_compile:
        print(f"[pc_kernel] Compiling {cu_path.name} with nvcc …", flush=True)
        result = subprocess.run(
            [
                "nvcc",
                "-arch=sm_75",  # RTX 2080 Super Max-Q
                "--shared",
                "-Xcompiler",
                "-fPIC",
                "-O3",
                "-o",
                str(so_path),
                str(cu_path),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"nvcc compilation failed:\n{result.stderr}")
        print("[pc_kernel] Compilation done.", flush=True)

    lib = ctypes.CDLL(str(so_path))
    fn = lib.launch_process_pc
    fn.restype = ctypes.c_int
    fn.argtypes = [
        ctypes.c_ulonglong,  # xyzrgba_ptr  (float*, device)
        ctypes.c_ulonglong,  # bgra_ptr      (uint8*, device)
        ctypes.c_ulonglong,  # heatmap_ptr   (uint8*, device)
        ctypes.c_int,
        ctypes.c_int,  # H_wc, W_wc
        ctypes.c_ulonglong,  # R_ptr  (float[9], device, row-major)
        ctypes.c_ulonglong,  # t_ptr  (float[3], device)
        ctypes.c_double,
        ctypes.c_double,  # fx, fy
        ctypes.c_double,
        ctypes.c_double,  # cx, cy
        ctypes.c_int,
        ctypes.c_int,  # roi_x0, roi_y0
        ctypes.c_int,
        ctypes.c_int,  # roi_x1, roi_y1
        ctypes.c_float,  # blend
        ctypes.c_int,  # show_heatmap (0 or 1)
        ctypes.c_ulonglong,  # xyz_out_ptr (float*, device)
        ctypes.c_ulonglong,  # rgb_out_ptr (float*, device)
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,  # N, grid_size, block_size
    ]
    return fn


def compute_animated_eye(frame_idx, mode, amplitude, period, period_shift, base_eye, lookat, up):
    """Return an animated camera eye position for the given rendered-frame index.

    Both modes keep `lookat` and `up` fixed; only the eye position moves.

    Args:
        frame_idx: sequential index of the rendered frame (0-based).
        mode: "orbit_up" or "orbit_view".
        amplitude: for orbit_up – half-angle in degrees; for orbit_view – radius
                   as a fraction of the eye-to-lookat distance.
        period: number of frames per full oscillation cycle.
        period_shift: fractional shift of oscillation cycle.
        base_eye: original static camera position (3-vector).
        lookat: the look-at point (3-vector).
        up: camera up vector (3-vector).
    """
    theta = 2 * np.pi * (period_shift + frame_idx / period)
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

    # Static geometry — created once and reused across all frames
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=75.0)
    webcam_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0)
    webcam_origin.transform(webcam_pose)

    if args.render_to_file:
        # Add static geometry to renderer once (never removed)
        renderer.scene.add_geometry("ZED left", origin, pc_mat)
        renderer.scene.add_geometry("Webcam", webcam_origin, pc_mat)
        # Pre-created sphere materials (reused across frames)
        sphere_mat_pred_r = o3d.visualization.rendering.MaterialRecord()
        sphere_mat_pred_r.shader = "defaultLitTransparency"
        sphere_mat_gt_r = o3d.visualization.rendering.MaterialRecord()
        sphere_mat_gt_r.shader = "defaultLitTransparency"
        # Persistent sphere meshes — translated in-place rather than recreated
        pred_sphere_geom = None
        gt_sphere_geom = None
        pred_sphere_at = None  # current center of pred_sphere_geom
        gt_sphere_at = None
        # Names currently present in the renderer scene
        scene_has: set = {"ZED left", "Webcam"}
        # Background thread for JPEG writes
        write_executor = ThreadPoolExecutor(max_workers=1)

    # Build frame specs before handing cameras to the producer thread.
    # In only_event_frames mode: list of (webcam_frame_id, event_idx).
    # In normal mode: list of (frame_id, None) for all webcam frames.
    total_webcam_frames = int(webcam_rec.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.only_event_frames:
        frame_specs = [(int(event_frames[i]), i) for i in range(len(event_frames))]
    else:
        frame_specs = [(i, None) for i in range(total_webcam_frames)]

    # Pre-compute constant projection parameters (float32, pinhole — no distortion).
    # Max error vs full cv2.projectPoints is <0.003 px: imperceptible for colour lookup.
    _R_f32 = webcam_pose[:3, :3].astype(np.float32)
    _t_f32 = webcam_pose[:3, 3].astype(np.float32)
    _fx = float(webcam_K[0, 0])
    _fy = float(webcam_K[1, 1])
    _cx = float(webcam_K[0, 2])
    _cy = float(webcam_K[1, 2])

    # Upload camera params to GPU once — read-only constant for every frame.
    d_R = cuda.to_device(_R_f32)
    d_t = cuda.to_device(_t_f32)

    # Compile (if needed) and load the CUDA C point-cloud kernel.
    _launch_pc = _load_pc_kernel()

    # Limit frames processed (useful for timing / debugging).
    if args.max_frames is not None:
        frame_specs = frame_specs[: args.max_frames]

    # Producer: owns all camera I/O and GPU point-cloud processing.
    def _producer(frame_specs, q):
        # GPU output buffers — lazily allocated on the first frame, then reused.
        _n_px = 0
        _d_xyz = _d_rgb = None

        # Thread pool for parallel webcam + heatmap reads.
        # cv2.VideoCapture.read() calls into FFmpeg C code and releases the GIL,
        # so two workers reading separate VideoCapture objects genuinely overlap.
        read_pool = ThreadPoolExecutor(max_workers=2)

        # Track each cap's next expected frame to skip the seek for sequential access.
        # In whole-recording mode frames are 0,1,2,... so seeks are entirely avoided.
        # In only_event_frames mode seeks still occur only when the frame is non-sequential.
        _cap_next: dict[int, int | None] = {}

        def _seek_read(cap, frame_id):
            """Seek only when the requested frame is not the next sequential one."""
            cap_id = id(cap)
            if _cap_next.get(cap_id) != frame_id:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            t0 = time.perf_counter()
            ret, frm = cap.read()
            elapsed_ms = 1e3 * (time.perf_counter() - t0)
            _cap_next[cap_id] = (frame_id + 1) if ret else None
            return ret, frm, elapsed_ms

        # Track the next expected ZED frame to skip set_svo_position for sequential access.
        # In SVO playback mode grab() auto-advances the read cursor, so sequential frames
        # never need an explicit seek — mirroring the _cap_next optimization for VideoCapture.
        _zed_next: int | None = None

        event_array_idx = 0
        for webcam_frame_id, _ in frame_specs:
            t_frame_start = time.perf_counter()

            # Submit heatmap read (always needed for CUDA kernel).
            # Webcam frame is only needed for --render_webcam debug plots; skip the
            # 127 ms decode when not needed.
            f_webcam = (
                read_pool.submit(_seek_read, webcam_rec, webcam_frame_id)
                if args.render_webcam
                else None
            )
            f_heatmap = read_pool.submit(_seek_read, webcam_heatmap_rec, webcam_frame_id)

            # While reads run in background: compute ZED frame ID and grab+retrieve.
            # ZED SDK C calls also release the GIL, so this overlaps with reads.
            ref_timestamp = webcam_frame_steps * webcam_frame_id + webcam_sync["first_frame"]
            zed_frame_id = int(
                round((ref_timestamp - zed_sync["first_frame"]) / zed_frame_steps, 0)
            )
            t_before_grab = t_after_grab = t_retrieve_done = time.perf_counter()
            t_set_svo_ms = 0.0
            zed_ok = False
            if 0 <= zed_frame_id < zed_sync["n_frames"]:
                if zed_frame_id != _zed_next:
                    t0 = time.perf_counter()
                    zed_rec.set_svo_position(zed_frame_id)
                    t_set_svo_ms = 1e3 * (time.perf_counter() - t0)
                t_before_grab = time.perf_counter()
                if zed_rec.grab() == sl.ERROR_CODE.SUCCESS:
                    t_after_grab = time.perf_counter()
                    pc_gpu = sl.Mat()
                    rgb_gpu = sl.Mat()
                    zed_rec.retrieve_measure(
                        pc_gpu, sl.MEASURE.XYZRGBA, sl.MEM.GPU, resolution=zed_target_resolution
                    )
                    zed_rec.retrieve_image(
                        rgb_gpu, sl.VIEW.LEFT, sl.MEM.GPU, resolution=zed_target_resolution
                    )
                    t_retrieve_done = time.perf_counter()
                    zed_ok = True
                    _zed_next = zed_frame_id + 1
                else:
                    _zed_next = None  # grab failed; don't assume cursor position
            else:
                _zed_next = None  # out-of-range frame; reset cursor assumption

            # Wait for webcam/heatmap futures.
            if f_webcam is not None:
                ret_wc, ref_frame, t_wc_read_ms = f_webcam.result()
            else:
                # Not reading webcam: validity check via frame-count bounds only.
                ret_wc = 0 <= webcam_frame_id < total_webcam_frames
                ref_frame = None
                t_wc_read_ms = 0.0
            ret_hm, ref_heatmap, t_hm_read_ms = f_heatmap.result()
            t_webcam_done = time.perf_counter()

            if not zed_ok or not ret_wc or not ret_hm:
                continue

            ref_frame_id = webcam_frame_id

            # Determine whether this is an event frame.
            if args.only_event_frames:
                is_event_frame = True
            else:
                event_idx = np.searchsorted(event_frames, ref_frame_id)
                is_event_frame = (
                    0 <= event_idx < len(event_frames) and event_frames[event_idx] == ref_frame_id
                )

            # --- GPU pipeline ---
            # All per-pixel work (isfinite, coord transform, projection, heatmap
            # blend) runs in a single CUDA kernel.  Invalid points get NaN xyz
            # written by the kernel — no CPU compaction step required.
            t_np0 = time.perf_counter()
            H_z, W_z = pc_gpu.get_height(), pc_gpu.get_width()
            n_px = H_z * W_z

            # Lazy-allocate persistent output buffers (reused across frames).
            if n_px != _n_px:
                _d_xyz = cuda.device_array((n_px, 3), dtype=np.float32)
                _d_rgb = cuda.device_array((n_px, 3), dtype=np.float32)
                _n_px = n_px

            # Get raw device pointers — ZED GPU buffers are zero-copy (no PCIe).
            xyzrgba_ptr = pc_gpu.get_pointer(sl.MEM.GPU)
            bgra_ptr = rgb_gpu.get_pointer(sl.MEM.GPU)

            # Upload webcam heatmap to GPU (BGR uint8, one per frame).
            H_wc, W_wc = ref_heatmap.shape[:2]
            d_heatmap = cuda.to_device(np.ascontiguousarray(ref_heatmap))

            # Extract device addresses for all GPU arrays.
            def _dptr(arr):
                return arr.__cuda_array_interface__["data"][0]

            blk = 256
            grd = (n_px + blk - 1) // blk
            err = _launch_pc(
                xyzrgba_ptr,
                bgra_ptr,
                _dptr(d_heatmap),
                H_wc,
                W_wc,
                _dptr(d_R),
                _dptr(d_t),
                _fx,
                _fy,
                _cx,
                _cy,
                webcam_image_roi[0],
                webcam_image_roi[1],
                webcam_image_roi[2],
                webcam_image_roi[3],
                ctypes.c_float(0.7),
                int(args.show_heatmap),
                _dptr(_d_xyz),
                _dptr(_d_rgb),
                n_px,
                grd,
                blk,
            )
            if err:
                raise RuntimeError(f"CUDA kernel failed (cudaError_t={err})")
            cuda.synchronize()
            t_kernel_done = time.perf_counter()

            # Download full arrays — no CPU compaction (invalid pts have NaN xyz).
            xyz_h = _d_xyz.copy_to_host()
            rgb_h = _d_rgb.copy_to_host()
            t_download_done = time.perf_counter()

            item_np_breakdown = {
                "_t_gpu_kernel_ms": 1e3 * (t_kernel_done - t_np0),
                "_t_gpu_download_ms": 1e3 * (t_download_done - t_kernel_done),
                "_t_gpu_filter_ms": 0.0,  # eliminated (NaN sentinel approach)
            }

            # Read pred/gt corners from pre-loaded read-only arrays.
            pred_corners = None
            gt_corners = None
            if is_event_frame and event_array_idx < len(pred_3d_bboxes_arr):
                pred_corners = pred_3d_bboxes_arr[event_array_idx].copy()
                gt_corners = gt_bboxes_arr[event_array_idx].copy()
                event_array_idx += 1

            item = {
                "ref_frame_id": ref_frame_id,
                "is_event_frame": is_event_frame,
                "xyz_trunc": xyz_h,  # (N, 3) float32 — OCV frame; NaN where invalid
                "heatmap_colors": rgb_h,  # (N, 3) float32
                "pred_corners": pred_corners,  # (8, 3) or None
                "gt_corners": gt_corners,  # (8, 3) or None
                # timing (ms) for this frame's producer work
                "_t_webcam_ms": 1e3 * (t_webcam_done - t_frame_start),
                "_t_wc_read_ms": t_wc_read_ms,
                "_t_hm_read_ms": t_hm_read_ms,
                "_t_set_svo_ms": t_set_svo_ms,
                "_t_grab_ms": 1e3 * (t_after_grab - t_before_grab),
                "_t_retrieve_ms": 1e3 * (t_retrieve_done - t_after_grab),
                "_t_numpy_ms": 1e3 * (t_download_done - t_np0),
                "_t_producer_total_ms": 1e3 * (t_download_done - t_frame_start),
                **item_np_breakdown,
            }
            if args.render_webcam:
                item["ref_frame"] = ref_frame
                item["ref_heatmap"] = ref_heatmap

            q.put(item)

        read_pool.shutdown(wait=False)
        q.put(None)  # sentinel — signals main thread to stop

    frame_q = queue.Queue(maxsize=2)
    producer_thread = threading.Thread(target=_producer, args=(frame_specs, frame_q), daemon=True)
    producer_thread.start()

    pbar_total = len(frame_specs)
    pbar = tqdm(total=pbar_total, desc="Frame")
    pred_3d_bbox_age = visualization_config["bbox_decay_frame_count"]
    gt_bbox_age = visualization_config["bbox_decay_frame_count"]

    pred_3d_bbox = None
    gt_bbox = None
    gt_sphere_diameter = None  # Will be computed from GT bbox
    frame_render_idx = 0

    # Timing accumulators (all in ms).
    timing: dict[str, list[float]] = defaultdict(list)

    while True:
        t_get_start = time.perf_counter()
        item = frame_q.get()
        t_get_done = time.perf_counter()
        if item is None:
            break
        timing["queue_wait_ms"].append(1e3 * (t_get_done - t_get_start))
        timing["producer_webcam_ms"].append(item["_t_webcam_ms"])
        timing["wc_read_ms"].append(item["_t_wc_read_ms"])
        timing["hm_read_ms"].append(item["_t_hm_read_ms"])
        timing["set_svo_ms"].append(item["_t_set_svo_ms"])
        timing["producer_grab_ms"].append(item["_t_grab_ms"])
        timing["producer_retrieve_ms"].append(item["_t_retrieve_ms"])
        timing["producer_numpy_ms"].append(item["_t_numpy_ms"])
        timing["producer_total_ms"].append(item["_t_producer_total_ms"])
        timing["gpu_kernel_ms"].append(item["_t_gpu_kernel_ms"])
        timing["gpu_download_ms"].append(item["_t_gpu_download_ms"])
        timing["gpu_filter_ms"].append(item["_t_gpu_filter_ms"])
        t_render_start = time.perf_counter()
        ref_frame_id = item["ref_frame_id"]
        is_event_frame = item["is_event_frame"]
        xyz_trunc = item["xyz_trunc"]
        heatmap_colors = item["heatmap_colors"]
        pred_corners = item["pred_corners"]
        gt_corners = item["gt_corners"]

        geometries = []
        if 0 <= visualization_config["bbox_decay_frame_count"] <= gt_bbox_age:
            gt_bbox = None
        if 0 <= visualization_config["bbox_decay_frame_count"] <= pred_3d_bbox_age:
            pred_3d_bbox = None

        # Update bboxes when the producer found corners for this event frame.
        if pred_corners is not None:  # implies is_event_frame and in-bounds
            if not np.all(pred_corners == 0):
                pred_3d_bbox = o3d.geometry.OrientedBoundingBox.create_from_points(
                    o3d.utility.Vector3dVector(pred_corners)
                )
                pred_3d_bbox_age = 0
            else:
                pred_3d_bbox = None
                pred_3d_bbox_age = visualization_config["bbox_decay_frame_count"]
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

        # Tensor PointCloud: Tensor.from_numpy is near-zero-copy for contiguous float32 arrays.
        # heatmap_colors is already float32/255 (pre-divided in producer).
        t_vec3d_start = time.perf_counter()
        pcd_t = o3d.t.geometry.PointCloud()
        pcd_t.point["positions"] = o3d.core.Tensor.from_numpy(xyz_trunc)
        t_vec3d_pts = time.perf_counter()
        pcd_t.point["colors"] = o3d.core.Tensor.from_numpy(heatmap_colors)
        t_vec3d_done = time.perf_counter()
        timing["tensor_pts_ms"].append(1e3 * (t_vec3d_pts - t_vec3d_start))
        timing["tensor_colors_ms"].append(1e3 * (t_vec3d_done - t_vec3d_pts))

        # Legacy PointCloud only needed for interactive mode (o3d.visualization.draw).
        if args.interactive:
            pcd_heatmap = o3d.geometry.PointCloud()
            pcd_heatmap.points = o3d.utility.Vector3dVector(xyz_trunc.astype(np.float64))
            pcd_heatmap.colors = o3d.utility.Vector3dVector(heatmap_colors.astype(np.float64))
        geometries.append({"name": "ZED left", "geometry": origin, "material": pc_mat})
        geometries.append({"name": "Webcam", "geometry": webcam_origin, "material": pc_mat})
        if args.interactive:
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
            ref_frame = item["ref_frame"]
            ref_heatmap = item["ref_heatmap"]
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
            decay_enabled = visualization_config["bbox_decay_frame_count"] >= 0

            # -- Point Cloud (changes every frame) --
            # pcd_t is already a tensor PointCloud built above with Tensor.from_numpy.
            t_pc_scene_start = time.perf_counter()
            if "Point Cloud" in scene_has:
                renderer.scene.remove_geometry("Point Cloud")
            renderer.scene.add_geometry("Point Cloud", pcd_t, pc_mat, False)
            scene_has.add("Point Cloud")
            t_pc_scene_done = time.perf_counter()
            timing["pc_scene_ms"].append(1e3 * (t_pc_scene_done - t_pc_scene_start))

            # -- Pred Sphere --
            if args.show_pred_sphere and gt_sphere_diameter is not None:
                wants_pred_sphere = pred_3d_bbox is not None
                needs_update = wants_pred_sphere and (
                    is_event_frame or decay_enabled or "Pred Sphere" not in scene_has
                )
                if needs_update:
                    fade = (
                        1.0 - pred_3d_bbox_age / visualization_config["bbox_decay_frame_count"]
                        if decay_enabled
                        else 1.0
                    )
                    sc = pred_sphere_base_color.copy()
                    sc[3] *= fade
                    sphere_mat_pred_r.base_color = sc
                    pred_center = pred_3d_bbox.center
                    if pred_sphere_geom is None:
                        pred_sphere_geom = o3d.geometry.TriangleMesh.create_sphere(
                            radius=gt_sphere_diameter / 2.0
                        )
                        pred_sphere_geom.compute_vertex_normals()
                        pred_sphere_at = np.zeros(3)
                    pred_sphere_geom.translate(pred_center - pred_sphere_at)
                    pred_sphere_at[:] = pred_center
                    pred_sphere_geom.paint_uniform_color(sc[:3])
                    if "Pred Sphere" in scene_has:
                        renderer.scene.remove_geometry("Pred Sphere")
                    renderer.scene.add_geometry("Pred Sphere", pred_sphere_geom, sphere_mat_pred_r)
                    scene_has.add("Pred Sphere")
                elif not wants_pred_sphere and "Pred Sphere" in scene_has:
                    renderer.scene.remove_geometry("Pred Sphere")
                    scene_has.discard("Pred Sphere")

            # -- GT Sphere --
            if args.show_gt_sphere and gt_sphere_diameter is not None:
                wants_gt_sphere = gt_bbox is not None
                needs_update = wants_gt_sphere and (
                    is_event_frame or decay_enabled or "GT Sphere" not in scene_has
                )
                if needs_update:
                    fade = (
                        1.0 - gt_bbox_age / visualization_config["bbox_decay_frame_count"]
                        if decay_enabled
                        else 1.0
                    )
                    sc = gt_sphere_base_color.copy()
                    sc[3] *= fade
                    sphere_mat_gt_r.base_color = sc
                    gt_center = gt_bbox.center
                    if gt_sphere_geom is None:
                        gt_sphere_geom = o3d.geometry.TriangleMesh.create_sphere(
                            radius=gt_sphere_diameter / 2.0
                        )
                        gt_sphere_geom.compute_vertex_normals()
                        gt_sphere_at = np.zeros(3)
                    gt_sphere_geom.translate(gt_center - gt_sphere_at)
                    gt_sphere_at[:] = gt_center
                    gt_sphere_geom.paint_uniform_color(sc[:3])
                    if "GT Sphere" in scene_has:
                        renderer.scene.remove_geometry("GT Sphere")
                    renderer.scene.add_geometry("GT Sphere", gt_sphere_geom, sphere_mat_gt_r)
                    scene_has.add("GT Sphere")
                elif not wants_gt_sphere and "GT Sphere" in scene_has:
                    renderer.scene.remove_geometry("GT Sphere")
                    scene_has.discard("GT Sphere")

            # -- Pred BBox LineSet --
            if args.show_pred_bbox:
                wants_pred_bbox = pred_3d_bbox is not None
                needs_update = wants_pred_bbox and (
                    is_event_frame or decay_enabled or "Pred BBox" not in scene_has
                )
                if needs_update:
                    relative_age = (
                        pred_3d_bbox_age / visualization_config["bbox_decay_frame_count"]
                        if decay_enabled
                        else 0.0
                    )
                    base_color = (
                        relative_age * np.ones(4) + (1.0 - relative_age) * pred_bbox_base_color
                    )
                    base_color[3] = 1.0
                    pred_bbox_mat.base_color = base_color
                    line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(pred_3d_bbox)
                    line_set.paint_uniform_color(base_color[:3])
                    if "Pred BBox" in scene_has:
                        renderer.scene.remove_geometry("Pred BBox")
                    renderer.scene.add_geometry("Pred BBox", line_set, pred_bbox_mat)
                    scene_has.add("Pred BBox")
                elif not wants_pred_bbox and "Pred BBox" in scene_has:
                    renderer.scene.remove_geometry("Pred BBox")
                    scene_has.discard("Pred BBox")

            # -- GT BBox LineSet --
            if args.show_gt_bbox:
                wants_gt_bbox = gt_bbox is not None
                needs_update = wants_gt_bbox and (
                    is_event_frame or decay_enabled or "GT BBox" not in scene_has
                )
                if needs_update:
                    relative_age = (
                        gt_bbox_age / visualization_config["bbox_decay_frame_count"]
                        if decay_enabled
                        else 0.0
                    )
                    base_color = relative_age * np.ones(4) + (1.0 - relative_age) * gt_base_color
                    base_color[3] = 1.0
                    gt_bbox_mat.base_color = base_color
                    line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(gt_bbox)
                    line_set.paint_uniform_color(base_color[:3])
                    if "GT BBox" in scene_has:
                        renderer.scene.remove_geometry("GT BBox")
                    renderer.scene.add_geometry("GT BBox", line_set, gt_bbox_mat)
                    scene_has.add("GT BBox")
                elif not wants_gt_bbox and "GT BBox" in scene_has:
                    renderer.scene.remove_geometry("GT BBox")
                    scene_has.discard("GT BBox")

            # -- Camera setup and render --
            eye = visualization_config["eye"]
            if args.camera_motion:
                eye = compute_animated_eye(
                    frame_render_idx,
                    args.camera_motion,
                    args.camera_motion_amplitude,
                    args.camera_motion_period,
                    args.camera_motion_period_shift,
                    visualization_config["eye"],
                    visualization_config["lookat"],
                    visualization_config["up"],
                )
            t_geom_done = time.perf_counter()

            # Call setup_camera only after all geometry has been added!
            renderer.setup_camera(
                center=visualization_config["lookat"],
                eye=eye,
                up=visualization_config["up"],
                vertical_field_of_view=visualization_config["field_of_view"],
            )
            img = renderer.render_to_image()
            t_render_done = time.perf_counter()

            # -- Async JPEG write (off the rendering thread) --
            img_np = np.asarray(img)
            out_jpg = str(
                video_out_path
                / f"1_{args.scene_id:03d}_Movie2D{output_suffix}"
                / f"{ref_frame_id:06d}.jpg"
            )
            write_executor.submit(
                cv2.imwrite, out_jpg, img_np[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95]
            )
            timing["render_geom_ms"].append(1e3 * (t_geom_done - t_render_start))
            timing["render_o3d_ms"].append(1e3 * (t_render_done - t_geom_done))

        gt_bbox_age += 1
        pred_3d_bbox_age += 1
        frame_render_idx += 1
        pbar.update()

    pbar.close()
    producer_thread.join()
    if args.render_to_file:
        write_executor.shutdown(wait=True)

    # --- Timing summary ---
    if timing:
        import statistics

        cols = [
            ("queue_wait_ms", "queue.get() wait   "),
            ("producer_webcam_ms", "  webcam+heatmap    "),
            ("wc_read_ms", "    webcam.read()   "),
            ("hm_read_ms", "    heatmap.read()  "),
            ("set_svo_ms", "  set_svo_position  "),
            ("producer_grab_ms", "  zed.grab()        "),
            ("producer_retrieve_ms", "  zed.retrieve()    "),
            ("producer_numpy_ms", "  gpu pipeline      "),
            ("gpu_kernel_ms", "    kernel          "),
            ("gpu_download_ms", "    download        "),
            ("gpu_filter_ms", "    cpu filter      "),
            ("producer_total_ms", "producer total      "),
            ("tensor_pts_ms", "Tensor.from_np (pts)"),
            ("tensor_colors_ms", "Tensor.from_np (clr)"),
            ("pc_scene_ms", "  scene remove+add  "),
            ("render_geom_ms", "o3d geom build      "),
            ("render_o3d_ms", "o3d render_to_image "),
        ]
        print("\n--- Timing summary (ms) ---")
        print(f"{'Section':<24}  {'mean':>8}  {'median':>8}  {'max':>8}  {'n':>5}")
        print("-" * 60)
        for key, label in cols:
            vals = timing.get(key, [])
            if not vals:
                continue
            print(
                f"{label:<24}  {statistics.mean(vals):>8.1f}  "
                f"{statistics.median(vals):>8.1f}  {max(vals):>8.1f}  {len(vals):>5}"
            )


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
    parser.add_argument(
        "--camera_motion_period_shift",
        type=float,
        default=0.0,
        help="Shift the oscillation cycle by a fraction (default 0.0).",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Stop after processing this many frames (for timing/debugging).",
    )
    args = parser.parse_args()

    visualize_sound_sources(args)
