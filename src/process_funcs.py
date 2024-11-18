import cv2
import numpy as np
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
import supervisely as sly
from typing import List
import src.globals as g
from copy import deepcopy


def bbox_from_array(points):
    """
    Converts numpy array to Supervisely Rectangle object.
    """
    x_min = np.min(points[:, 0])
    y_min = np.min(points[:, 1])
    x_max = np.max(points[:, 0])
    y_max = np.max(points[:, 1])

    return sly.Rectangle(int(y_min), int(x_min), int(y_max), int(x_max))


@sly.handle_exceptions
@sly.timeit
def apply_lightglue_bounding_boxes(
    image_paths: List[str],
    ref_bbox_labels: List[sly.Label],
    max_num_keypoints: int = 1024,
    device: str = "cpu",
):
    # * Get reference image path first
    reference_image_path = image_paths.pop(0)

    # * Initialize feature extractor and matcher
    extractor = (
        SuperPoint(max_num_keypoints=max_num_keypoints, model_dir=g.MODEL_DIR).eval().to(device)
    )
    matcher = (
        LightGlue(
            features="superpoint",
            filter_threshold=0.5,
            model_dir=g.MODEL_DIR,
        )
        .eval()
        .to(device)
    )

    # * Load the reference image and extract features
    ref_image = load_image(reference_image_path).to(device)
    ref_features = extractor.extract(ref_image, resize=256)

    # * Define reference bounding box points for each bbox in reference_bboxes
    id_to_labels = {}

    for img_path in image_paths:
        ref_features_copy = deepcopy(ref_features)
        # * Load and extract the current image, resizing it to reduce processing times
        img = load_image(img_path).to(device)
        img_features = extractor.extract(img, resize=256)

        try:
            matches = matcher({"image0": ref_features_copy, "image1": img_features})
        except Exception as e:
            sly.logger.warning(f"Matching failed for image {img_path}: {e}")
            continue
        ref_features_copy, img_features, matches = [
            rbd(x) for x in [ref_features_copy, img_features, matches]
        ]

        # * Extract matched points
        ref_matched_pts = ref_features_copy["keypoints"][matches["matches"][..., 0]].cpu().numpy()
        img_matched_pts = img_features["keypoints"][matches["matches"][..., 1]].cpu().numpy()

        if len(ref_matched_pts) < 4 or len(img_matched_pts) < 4:
            sly.logger.warning(f"Not enough matches found for image {img_path}. Skipping.")
            continue

        # * Calculate the homography matrix between the reference and target image
        H, status = cv2.findHomography(ref_matched_pts, img_matched_pts, cv2.RANSAC, 5.0)

        # * Transform original bounding boxes' points using cv2.PerspectiveTransform
        result_labels = []
        for orig_label in ref_bbox_labels:
            geometry = orig_label.geometry
            geom_array = np.array(
                [
                    [geometry.left, geometry.top],  # Top-left corner
                    [geometry.right, geometry.top],  # Top-right corner
                    [geometry.right, geometry.bottom],  # Bottom-right
                    [geometry.left, geometry.bottom],  # Bottom-left
                ]
            ).astype(np.float32)
            transformed_pts = cv2.perspectiveTransform(np.array([geom_array]), H)[0]
            result_labels.append(orig_label.clone(bbox_from_array(transformed_pts)))

        # * Pass the result labels into a mapping
        img_id = g.CACHE.path_to_id[img_path]
        id_to_labels[img_id] = result_labels

    return id_to_labels
