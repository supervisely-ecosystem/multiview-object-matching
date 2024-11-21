import cv2
import numpy as np
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
import supervisely as sly
from typing import List
import src.globals as g
from copy import deepcopy


def bbox_from_array(points: np.array) -> sly.Rectangle:
    """
    Converts numpy array to Supervisely Rectangle object.
    """
    x_min = np.min(points[:, 0])
    y_min = np.min(points[:, 1])
    x_max = np.max(points[:, 0])
    y_max = np.max(points[:, 1])

    return sly.Rectangle(int(y_min), int(x_min), int(y_max), int(x_max))


def bbox_to_array(geometry: sly.Rectangle) -> np.array:
    """
    Converts Supervisely Rectangle object to numpy array.
    """
    return np.array(
        [
            [geometry.left, geometry.top],  # Top-left corner
            [geometry.right, geometry.top],  # Top-right corner
            [geometry.right, geometry.bottom],  # Bottom-right
            [geometry.left, geometry.bottom],  # Bottom-left
        ]
    ).astype(np.float32)


def apply_lightglue(
    image_paths: List[str],
    max_num_keypoints: int = 1024,
    resize=None,
    filter_threshold=0.3,
    device: str = "cpu",
):
    """
    Generator function to apply LightGlue to image paths
    """

    # * Get reference image path first
    reference_image_path = image_paths.pop(0)

    # * Initialize feature extractor and matcher
    extractor = (
        SuperPoint(max_num_keypoints=max_num_keypoints, model_dir=g.MODEL_DIR).eval().to(device)
    )
    matcher = (
        LightGlue(
            features="superpoint",
            # depth_confidence=-1,
            # width_confidence=-1,
            filter_threshold=filter_threshold,
            model_dir=g.MODEL_DIR,
        )
        .eval()
        .to(device)
    )
    g.api.task.set_output_text(g.task_id, "Application is started.")

    # * Load the reference image and extract features
    ref_image = load_image(reference_image_path).to(device)
    ref_features = extractor.extract(ref_image, resize=resize)

    for img_path in image_paths[:]:
        ref_features_copy = deepcopy(ref_features)
        img = load_image(img_path).to(device)
        img_features = extractor.extract(img, resize=resize)

        try:
            matches = matcher({"image0": ref_features_copy, "image1": img_features})
        except Exception as e:
            sly.logger.debug(f"Matching failed for image {img_path}: {e}")
            image_paths.remove(img_path)
            continue

        ref_features_copy, img_features, matches = [
            rbd(x) for x in [ref_features_copy, img_features, matches]
        ]

        ref_matched_pts = ref_features_copy["keypoints"][matches["matches"][..., 0]].cpu().numpy()
        img_matched_pts = img_features["keypoints"][matches["matches"][..., 1]].cpu().numpy()

        if len(ref_matched_pts) < 4 or len(img_matched_pts) < 4:
            sly.logger.warning(f"Not enough matches found for image {img_path}.")
            image_paths.remove(img_path)
            continue

        yield (ref_matched_pts, img_matched_pts)


# def apply_transform_to_bboxes(bbox_labels: List[sly.Label], ref_matched_pts, img_matched_pts):
#     # * Calculate the homography matrix between the reference and target image
#     H, _ = cv2.findHomography(ref_matched_pts, img_matched_pts)

#     for orig_label in bbox_labels:
#         geometry = orig_label.geometry
#         # * Get numpy array from bounding box points
#         box_points = bbox_to_array(geometry)
#         # * Transform original bounding boxes' points using cv2.PerspectiveTransform
#         transformed_pts = cv2.perspectiveTransform(np.array([box_points]), H)[0]
#         yield orig_label.clone(bbox_from_array(transformed_pts))


def transpose_bbox_with_keypoints(
    bbox_labels: List[sly.Label], ref_keypoints, img_keypoints, padding=5
):
    """
    Generator function to transpose bounding boxes to images using keypoints
    """
    for orig_label in bbox_labels:
        geometry = orig_label.geometry
        box_points = bbox_to_array(geometry)
        min_x, min_y = np.min(box_points, axis=0) - padding
        max_x, max_y = np.max(box_points, axis=0) + padding
        extended_bbox = np.array([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]])
        in_bbox_keypoints = [
            kp
            for kp in ref_keypoints
            if cv2.pointPolygonTest(extended_bbox.astype(np.float32), tuple(kp), False) >= 0
        ]
        if not in_bbox_keypoints:
            continue
        matched_idx = [
            np.where((ref_keypoints == kp).all(axis=1))[0][0] for kp in in_bbox_keypoints
        ]
        matched_pts_ref = np.array(in_bbox_keypoints)
        matched_pts_img = img_keypoints[matched_idx]
        new_box_points = []
        for box_point in box_points:
            distances = np.linalg.norm(matched_pts_ref - box_point, axis=1)
            nearest_idx = np.argmin(distances)
            offset = matched_pts_img[nearest_idx] - matched_pts_ref[nearest_idx]
            new_box_points.append(box_point + offset)
        yield orig_label.clone(bbox_from_array(np.array(new_box_points)))
