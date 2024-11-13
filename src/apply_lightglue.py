import cv2
import numpy as np
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
import supervisely as sly
from typing import List
import src.globals as g

def perspective_to_bbox(points):
    """
    Converts points from cv2.perspectiveTransform to bounding box coordinates.
    
    Parameters:
    points (np.ndarray): Transformed points from cv2.perspectiveTransform, 
                         with shape (n, 1, 2) or (1, n, 2) where n is the number of points.
    
    Returns:
    tuple: Bounding box in (y_min, x_min, y_max, x_max) format.
    """
    # Reshape points if needed
    points = points.reshape(-1, 2)
    
    # Extract x and y coordinates
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    
    # Find bounding box coordinates
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    return sly.Rectangle(int(y_min), int(x_min), int(y_max), int(x_max))


@sly.handle_exceptions
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
        # Load and process the current image
        img = load_image(img_path).to(device)
        img_features = extractor.extract(img)
        
        # Match features between the reference and the current image
        try:
            matches = matcher({'image0': ref_features, 'image1': img_features})
        except Exception as e:
            sly.logger.warn(f"Error while matching images: {repr(e)}")
            continue
        ref_features, img_features, matches = [rbd(x) for x in [ref_features, img_features, matches]]
        
        # Extract matched points
        ref_matched_pts = ref_features['keypoints'][matches['matches'][..., 0]].cpu().numpy()
        img_matched_pts = img_features['keypoints'][matches['matches'][..., 1]].cpu().numpy()

        # Calculate the homography matrix between the reference and target image
        H, status = cv2.findHomography(ref_matched_pts, img_matched_pts, cv2.RANSAC, 5.0)
        
        # Transform the bounding box using the homography
        transformed_pts = cv2.perspectiveTransform(np.array([ref_bbox_pts]), H)[0]
        img_id = g.CACHE.path_to_id[img_path]
        id_to_box[img_id] = perspective_to_bbox(transformed_pts)

    return id_to_box