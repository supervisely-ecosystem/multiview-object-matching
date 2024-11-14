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
    Converts numpy array to bounding box coordinates.
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
    max_num_keypoints=1024,
    device: str = "cpu",
):
    reference_image_path = image_paths.pop(0)

    # Initialize feature extractor and matcher
    extractor = (
        SuperPoint(max_num_keypoints=max_num_keypoints, model_dir=g.MODEL_DIR).eval().to(device)
    )
    matcher = LightGlue(features="superpoint", model_dir=g.MODEL_DIR).eval().to(device)

    # Load the reference image and extract features
    ref_image = load_image(reference_image_path).to(device)
    ref_features = extractor.extract(ref_image)  # todo: try resize

    # Define reference bounding box points for each bbox in reference_bboxes
    ref_bbox_points_list = [
        (
            bbox_label,
            np.array(
                [
                    [bbox_label.geometry.left, bbox_label.geometry.top],  # Top-left corner
                    [bbox_label.geometry.right, bbox_label.geometry.top],  # Top-right corner
                    [bbox_label.geometry.right, bbox_label.geometry.bottom],  # Bottom-right
                    [bbox_label.geometry.left, bbox_label.geometry.bottom],  # Bottom-left
                ]
            ).astype(np.float32),
        )
        for bbox_label in ref_bbox_labels
    ]

    id_to_labels = {}

    # Process each target image
    for img_path in image_paths:
        ref_features_copy = deepcopy(ref_features)
        # Load and process the current image
        img = load_image(img_path).to(device)
        img_features = extractor.extract(img)

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
        img_matched_pts = img_features["keypoints"][matches["matches"][..., 1]].cpu().numpy()

        if len(ref_matched_pts) < 4 or len(img_matched_pts) < 4:
            sly.logger.warning(f"Not enough matches found for image {img_path}. Skipping.")
            continue

        # Calculate the homography matrix between the reference and target image
        H, status = cv2.findHomography(ref_matched_pts, img_matched_pts, cv2.RANSAC, 5.0)

        # Transform each bounding box in reference_bboxes
        result_labels = []
        for orig_label, ref_bbox_pts in ref_bbox_points_list:
            transformed_pts = cv2.perspectiveTransform(np.array([ref_bbox_pts]), H)[0]
            result_labels.append(orig_label.clone(bbox_from_array(transformed_pts)))

        # Store transformed bounding boxes with the image ID
        img_id = g.CACHE.path_to_id[img_path]
        id_to_labels[img_id] = result_labels

    return id_to_labels
