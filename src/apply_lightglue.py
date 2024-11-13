import cv2
import numpy as np
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
import supervisely as sly
from typing import List
import src.globals as g
from copy import deepcopy

def perspective_to_bbox(points):
    """
    Converts points from cv2.perspectiveTransform to bounding box coordinates.

    Parameters:
    points (np.ndarray): Transformed points from cv2.perspectiveTransform,
                         with shape (n, 1, 2) or (1, n, 2) where n is the number of points.

    Returns:
    tuple: Bounding box in (y_min, x_min, y_max, x_max) format.
    """
    x_min = np.min(points[:, 1])
    y_min = np.min(points[:, 0])
    x_max = np.max(points[:, 1])
    y_max = np.max(points[:, 0])

    return sly.Rectangle(int(y_min), int(x_min), int(y_max), int(x_max))


@sly.handle_exceptions
@sly.timeit
def apply_lightglue_bounding_box(image_paths: List[str], reference_bbox: sly.Rectangle, device: str = "cpu"):
    reference_image_path = image_paths.pop(0)

    # Initialize feature extractor and matcher
    extractor = SuperPoint(max_num_keypoints=1024, model_dir=g.MODEL_DIR).eval().to(device)
    matcher = LightGlue(features='superpoint', model_dir=g.MODEL_DIR).eval().to(device)

    # Load the reference image and extract features
    ref_image = load_image(reference_image_path).to(device)
    ref_features = extractor.extract(ref_image)

    # Define reference bounding box points
    ref_bbox_pts = np.array([
        [reference_bbox.top, reference_bbox.left],  # Top-left corner
        [reference_bbox.top, reference_bbox.right],  # Top-right corner
        [reference_bbox.bottom , reference_bbox.right],  # Bottom-right
        [reference_bbox.bottom , reference_bbox.left]  # Bottom-left
    ]).astype(np.float32)

    id_to_box = {}

    # Process each target image
    for img_path in image_paths:
        ref_features_copy = deepcopy(ref_features)
        # Load and process the current image
        img = load_image(img_path).to(device)
        img_features = extractor.extract(img)

        if (
            ref_features_copy["keypoints"].numel() == 0
            or ref_features_copy["descriptors"].numel() == 0
        ):
            sly.logger.warning(f"Reference image has no keypoints or descriptors, skipping.")
            continue

        if img_features["keypoints"].numel() == 0 or img_features["descriptors"].numel() == 0:
            sly.logger.warning(
                f"Target image {img_path} has no keypoints or descriptors, skipping."
            )
            continue

        if ref_features_copy["keypoints"].dim() == 2:
            ref_features_copy["keypoints"] = ref_features_copy["keypoints"].unsqueeze(0)
            ref_features_copy["descriptors"] = ref_features_copy["descriptors"].unsqueeze(0)

        if img_features["keypoints"].dim() == 2:
            img_features["keypoints"] = img_features["keypoints"].unsqueeze(0)
            img_features["descriptors"] = img_features["descriptors"].unsqueeze(0)

        try:
            matches = matcher({"image0": ref_features_copy, "image1": img_features})
        except Exception as e:
            sly.logger.warning(f"Matching failed for image {img_path}: {e}")
            continue
        ref_features_copy, img_features, matches = [
            rbd(x) for x in [ref_features_copy, img_features, matches]
        ]

        # Extract matched points
        ref_matched_pts = ref_features_copy["keypoints"][matches["matches"][..., 0]].cpu().numpy()
        img_matched_pts = img_features['keypoints'][matches['matches'][..., 1]].cpu().numpy()

        if len(ref_matched_pts) < 4 or len(img_matched_pts) < 4:
            sly.logger.warn(f"Not enough matches found for image {img_path}. Skipping.")
            continue

        # Calculate the homography matrix between the reference and target image
        H, status = cv2.findHomography(ref_matched_pts, img_matched_pts, cv2.RANSAC, 5.0)

        # Transform the bounding box using the homography
        transformed_pts = cv2.perspectiveTransform(np.array([ref_bbox_pts]), H)[0]
        img_id = g.CACHE.path_to_id[img_path]
        id_to_box[img_id] = perspective_to_bbox(transformed_pts)  # todo: fix?

    return id_to_box
