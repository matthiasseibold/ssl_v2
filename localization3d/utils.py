import numpy as np
import trimesh
import open3d as o3d
import matplotlib
import cv2
from skimage.color import rgb2hsv
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull, QhullError, HalfspaceIntersection


def get_scene_category(rec_id):
    if 1 <= rec_id <= 10:
        return "sawing"
    elif 11 <= rec_id <= 16:
        return "chiseling"
    elif 17 <= rec_id <= 22:
        return "drilling"
    else:
        return None


def project_points(
    points_3d: np.ndarray,
    camera_pose: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    """
    Projects 3D points to the 2D image plane using a 4x4 camera pose matrix.

    Args:
        points_3d (np.ndarray): A numpy array of 3D points in the world coordinate
                                system. Shape: (N, 3).
        camera_pose (np.ndarray): The 4x4 transformation matrix that maps points
                                  from the world frame to the camera frame (T_c_w).
        camera_matrix (np.ndarray): The 3x3 camera intrinsic matrix (K).
        dist_coeffs (np.ndarray): The camera distortion coefficients (k1,k2,p1,p2,k3).
                                  Shape: (5, 1) or (1, 5).

    Returns:
        np.ndarray: A numpy array of projected 2D points in pixel coordinates.
                    Shape: (N, 2). Returns None if projection fails.
    """
    # Ensure input arrays are of type float64, as cv2.projectPoints expects this.
    points_3d = np.asarray(points_3d, dtype=np.float64)
    camera_pose = np.asarray(camera_pose, dtype=np.float64)
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
    dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64)

    # 1. Decompose the 4x4 camera pose matrix
    # The rotation matrix (R) is the top-left 3x3 submatrix
    R = camera_pose[:3, :3]
    # The translation vector (t) is the first 3 elements of the last column
    t = camera_pose[:3, 3]

    # 2. Convert the rotation matrix to a rotation vector (rvec)
    # cv2.Rodrigues converts a rotation matrix to a 3x1 rotation vector (or vice-versa)
    rvec, _ = cv2.Rodrigues(R)

    # 3. Project the 3D points to the 2D image plane
    # cv2.projectPoints returns the projected 2D points and the Jacobian matrix
    points_2d, _ = cv2.projectPoints(
        objectPoints=points_3d,
        rvec=rvec,
        tvec=t,
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs,
    )

    # The output points_2d is an array of shape (N, 1, 2).
    # Reshape it to (N, 2) for easier use.
    if points_2d is not None:
        return points_2d.reshape(-1, 2)
    else:
        return None


def trimesh_iou_3d(
    box1: o3d.geometry.OrientedBoundingBox, box2: o3d.geometry.OrientedBoundingBox
) -> float:
    """
    Computes the exact intersection volume of two 3D oriented bounding boxes.

    This solution requires the 'trimesh' library.
    (pip install trimesh)

    Args:
        box1: o3d.geometry.OrientedBoundingBox
        box2: o3d.geometry.OrientedBoundingBox

    Returns:
        The intersection volume of the two bounding boxes.
    """

    # 1. Create o3d.geometry.TriangleMesh from o3d.geometry.OrientedBoundingBox
    # These are watertight, 8-vertex, 12-triangle meshes.
    mesh_box1 = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(box1)
    mesh_box2 = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(box2)

    # 2. Convert Open3D meshes to trimesh.Trimesh objects
    # trimesh requires vertices and faces
    tm_box1 = trimesh.Trimesh(
        vertices=np.asarray(mesh_box1.vertices), faces=np.asarray(mesh_box1.triangles)
    )
    tm_box2 = trimesh.Trimesh(
        vertices=np.asarray(mesh_box2.vertices), faces=np.asarray(mesh_box2.triangles)
    )

    # 3. Compute the boolean intersection using trimesh
    # The 'auto' engine will try to pick the best available (e.g., 'blender', 'scad').
    # This is the core operation.
    try:
        intersection_mesh = tm_box1.intersection(tm_box2, engine="manifold")
    except Exception as e:
        # Handle cases where the boolean operation might fail (e.g., degenerate intersections)
        print(f"Trimesh boolean intersection failed: {e}")  # Uncomment for debugging
        return 0.0

    # 4. Get the volume of the resulting intersection mesh
    # If the intersection is empty or non-volumetric (a plane, line, or point),
    # trimesh will correctly report the volume as 0.0.
    intersect_vol = intersection_mesh.volume

    # Handle potential numerical precision issues where volume is very close to zero
    if intersect_vol < 1e-10:
        return 0.0

    # Calculate the union volume
    union_vol = tm_box1.volume + tm_box2.volume - intersect_vol

    return intersect_vol / union_vol


def iou_3d(corners1: np.ndarray, corners2: np.ndarray) -> float:
    """
    Computes the Intersection over Union (IoU) of two 3D bounding boxes.

    The bounding boxes are defined by their 8 corners. They are not necessarily
    axis-aligned. The function relies on creating convex hulls for the boxes
    and finding the volume of their intersection.

    Args:
        corners1: A numpy array of shape (8, 3) representing the corners of the first bounding box.
        corners2: A numpy array of shape (8, 3) representing the corners of the second bounding box.

    Returns:
        The IoU of the two bounding boxes, a float value between 0.0 and 1.0.
        Returns 0.0 if the boxes are degenerate or do not intersect.
    """
    if not isinstance(corners1, np.ndarray) or corners1.shape != (8, 3):
        raise ValueError("corners1 must be a numpy array of shape (8, 3)")
    if not isinstance(corners2, np.ndarray) or corners2.shape != (8, 3):
        raise ValueError("corners2 must be a numpy array of shape (8, 3)")

    try:
        # Create convex hulls for both sets of corners
        hull1 = ConvexHull(corners1)
        hull2 = ConvexHull(corners2)

        # Volume of the first bounding box
        vol1 = hull1.volume
        # Volume of the second bounding box
        vol2 = hull2.volume

        # Combine the half-space equations from both hulls
        # A half-space is defined by the equation Ax + b <= 0
        # hull.equations is an array of shape (n_faces, 4) where each row is [A_x, A_y, A_z, b]
        halfspaces = np.vstack((hull1.equations, hull2.equations))

        # A feasible interior point for the half-space intersection is needed.
        # The centroid of the average of the two bounding box centroids is a good heuristic.
        interior_point = (corners1.mean(axis=0) + corners2.mean(axis=0)) / 2.0

        # Calculate the intersection of the half-spaces
        hs_intersection = HalfspaceIntersection(halfspaces, interior_point)

        # The intersection of two convex hulls is also a convex hull.
        # We can find its volume by forming a convex hull from its vertices.
        intersection_hull = ConvexHull(hs_intersection.intersections)
        intersection_vol = intersection_hull.volume

    except (QhullError, ValueError):
        # QhullError can be raised if the intersection is empty, degenerate (a line or point),
        # or if the input points are co-planar, which means they don't form a 3D volume.
        # In such cases, the intersection volume is zero.
        intersection_vol = 0.0
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return 0.0

    # Calculate the union volume
    union_vol = vol1 + vol2 - intersection_vol

    # Calculate the IoU
    if union_vol == 0:
        # This can happen if both boxes are degenerate (volume 0)
        return 0.0

    iou = intersection_vol / union_vol

    # Clamp the value to be between 0 and 1, handling potential floating point inaccuracies
    return np.clip(iou, 0.0, 1.0)


def find_weighted_cluster_centers(points, scores, eps, min_weight):
    """
    Uses sklearn's weighted DBSCAN to find high-score clusters
    and returns their score-weighted centers.

    These centers can be used as the 'M' candidate points for
    a fast, approximate box-fitting search.

    Args:
        points (np.ndarray): (N, 3) array of coordinates.
        scores (np.ndarray): (N,) array of scores (weights).
        eps (float): The DBSCAN neighborhood radius. This is a
                     crucial parameter to tune.
        min_weight (float): The minimum *sum of scores* within 'eps'
                            for a point to be considered a "core point".
                            (This is passed to sklearn's 'min_samples').

    Returns:
        tuple: (cluster_centers, labels)
            - cluster_centers (np.ndarray): (M, 3) array of M cluster centers.
            - labels (np.ndarray): (N,) array of cluster labels for each point
                                   (-1 is noise).
    """

    # 1. Ensure scores is a 1D array
    if scores.ndim > 1:
        scores = scores.flatten()

    # 2. Run weighted DBSCAN
    # We pass `scores` to `sample_weight`.
    # `min_samples` is now interpreted as `min_weight`.

    # n_jobs=-1 uses all available CPU cores for neighbor search
    db = DBSCAN(eps=eps, min_samples=min_weight, metric="minkowski", p=2, n_jobs=-1)

    labels = db.fit(points, sample_weight=scores).labels_

    # 3. Find unique cluster IDs (excluding noise -1)
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)

    num_clusters = len(unique_labels)

    if num_clusters == 0:
        print("Weighted DBSCAN found no clusters.")
        return np.array([]).reshape(0, 3), labels

    # 4. Calculate score-weighted center of mass for each cluster
    cluster_centers = np.zeros((num_clusters, 3))

    for i, cluster_id in enumerate(sorted(list(unique_labels))):

        # Get indices of all points in this cluster
        in_cluster_mask = labels == cluster_id

        cluster_points = points[in_cluster_mask]
        cluster_scores = scores[in_cluster_mask]

        total_score = np.sum(cluster_scores)

        if total_score > 0:
            # weighted_sum = sum(point[i] * score[i] for i in cluster)
            weighted_sum = np.sum(cluster_points * cluster_scores[:, np.newaxis], axis=0)

            # center = weighted_sum / sum(scores)
            center = weighted_sum / total_score
            cluster_centers[i] = center
        else:
            # Fallback: simple average (shouldn't happen if min_weight > 0)
            cluster_centers[i] = np.mean(cluster_points, axis=0)

    return cluster_centers, labels


def calculate_color_score(rgb_array: np.ndarray) -> np.ndarray:
    """
    Calculates a score (0.0 to 1.0) for RGB colors based on a specific
    color gradient (Pink=high, Blue=low).

    Args:
        rgb_array: An n-dimensional NumPy array with shape [..., 3]
                   containing RGB channels. Values can be 0-255 (int)
                   or 0.0-1.0 (float).

    Returns:
        An (n-1)-dimensional NumPy array with the calculated scores.
    """

    # --- 1. Normalize Input ---
    # Ensure array is float and normalized to 0.0-1.0 range
    if np.max(rgb_array) > 1.0 and np.issubdtype(rgb_array.dtype, np.integer):
        rgb_normalized = rgb_array.astype(np.float32) / 255.0
    else:
        rgb_normalized = rgb_array.astype(np.float32)

    # --- 2. Convert to HSV ---
    # hsv shape is also [..., 3]
    hsv = rgb2hsv(rgb_normalized)

    # --- 3. Extract Hue and Saturation ---
    # h, s, and v will have shape [...]
    h = hsv[..., 0]  # Hue (0.0 to 1.0)
    s = hsv[..., 1]  # Saturation (0.0 to 1.0)

    # --- 4. Calculate Hue-based Score ---
    # The scale in the image is Pink -> Red -> Yellow -> Green -> Cyan -> Blue.
    # This corresponds to a reversed hue circle, shifted to start at Pink/Magenta.
    # We shift the hue scale so that Pink (Magenta, hue=5/6 or ~0.833) is 0.
    HUE_SHIFT = 5.0 / 6.0  # Hue of Magenta/Pink

    # Shift and wrap the hue
    # (h - HUE_SHIFT) % 1.0
    # Pink (0.83) -> 0.0
    # Red (0.0) -> ~0.17
    # Blue (0.67) -> ~0.83
    h_shifted = (h - HUE_SHIFT) % 1.0

    # Invert the scale so Pink (0.0) becomes 1.0 and Blue (~0.83) becomes ~0.17
    hue_score = 1.0 - h_shifted

    # --- 5. Apply Saturation ---
    # The final score is gated by saturation.
    # Pure colors (S=1) get the full hue_score.
    # Grayscale colors (S=0) get a score of 0.
    final_score = hue_score * s

    return final_score


def get_distinct_colors_qualitative(N):
    """Generates N distinct RGB colors using a qualitative colormap."""
    # Using 'tab20' which works well up to N=20
    cmap = matplotlib.colormaps["tab20"]

    # Generate N evenly spaced numbers between 0 and 1
    colors = cmap(np.linspace(0, 1, N))

    # The result is a NumPy array of shape (N, 4) [R, G, B, Alpha]
    # We return the RGB components (first 3 columns)
    return colors[:, :3]
