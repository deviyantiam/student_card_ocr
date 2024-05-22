from detectron2.engine import DefaultPredictor
from detectron2.structures import Instances
from detectron2.config import get_cfg
from typing import Tuple, Union
from pathlib import Path
import numpy as np 
import warnings
import cv2

warnings.filterwarnings("ignore")

CLASS_NAMES = ['bottom-left', 'bottom-right', 'top-left', 'top-right']
PADDING = 50
CONF_THRESHOLD = 0.85


def load_model(config_path: Union[str, Path]) -> DefaultPredictor:
    """Load Corner Detection Model for student ID

    Parameters
    ----------
    config_path : Union[str, Path], optional
        Path to detectron2.yaml file

    Returns
    -------
    DefaultPredictor
        Detectron2 DefaultPredictor Model

    Raises
    ------
    FileNotFoundError
        if config_path doesn't exist, raise FileNotFoundError
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file {config_path} doesn't exist!")
    cfg = get_cfg()
    cfg.merge_from_file(str(config_path))
    weight_path = config_path.parent / cfg.MODEL.WEIGHTS
    if not weight_path.exists():
        raise FileNotFoundError(f"Weight File Model {weight_path} from config {config_path.name} doesn't exists!")
    cfg.MODEL.WEIGHTS = str(weight_path)
    predictor = DefaultPredictor(cfg)
    return predictor

def get_corner_point(padding_img, output_instances: Instances) -> dict:

#     """Get student ID Corner point of instances"""

    edge_points = {}

    image_height, image_width, num_channels = padding_img.shape

    for idx, pred_class in enumerate(CLASS_NAMES):

        if idx in output_instances.pred_classes.tolist():

            pred_idx = output_instances.pred_classes.tolist().index(idx)
            pred_bbox = output_instances.pred_boxes.tensor[pred_idx].numpy()
            xmin, ymin, xmax, ymax = pred_bbox

            corners = np.array([[xmin, ymin], [xmax , ymin], [xmin, ymax], [xmax, ymax]])
            base_corners = np.array([[0, 0], [image_width, 0], [0, image_height], [image_width, image_height]])
            distances = np.sqrt(np.sum((corners [:, np.newaxis] -  base_corners) ** 2, axis=2))
            
            # Find the index of the nearest corner
            min_indices = np.unravel_index(np.argmin(distances), distances.shape)

            # Use the nearest corner index to get the nearest corner coordinates from corners
            nearest_corners = corners[[min_indices[0]]][0]
            edge_points[pred_class] = nearest_corners

    return edge_points       

def clean_output_instances(output_instances: Instances) -> Instances:
    """Clean Output Instances
    such as cleaning multiple predictions of the same corner 
    and remove predictions below the threshold"""
    clean_instances = Instances(output_instances._image_size)
    # Get duplicate corner prediction
    unique, counts = np.unique(output_instances.pred_classes, return_counts=True)
    duplicate_preds = unique[np.where(counts > 1)]
    idx_classes = []
    for pred_class in output_instances.pred_classes.unique():
        if pred_class in duplicate_preds:
            idx_scores = np.where(output_instances.pred_classes == pred_class)[0]
            idx_class = output_instances.scores[idx_scores].argmax().tolist()
        else:
            idx_class = output_instances.pred_classes.tolist().index(pred_class)
        idx_classes.append(idx_class)
    # Set Instances item
    clean_instances.set('pred_classes', output_instances.pred_classes[idx_classes])
    clean_instances.set('scores', output_instances.scores[idx_classes])
    clean_instances.set('pred_boxes', output_instances.pred_boxes[idx_classes])
    return clean_instances

def perspective_transform(image_rgb: np.array, edge_points: dict) -> np.array:
    """Create Transformation of student ID with size 300x500 px based on corner detection"""
    source_points = np.float32([edge_points['top-left'], edge_points['top-right'], 
                                edge_points['bottom-right'], edge_points['bottom-left']])
    widthA = np.sqrt(((edge_points['top-right'][0] - edge_points['top-left'][0]) ** 2) + ((edge_points['top-right'][1] - edge_points['top-left'][1]) ** 2))
    widthB = np.sqrt(((edge_points['bottom-right'][0] - edge_points['bottom-left'][0]) ** 2) + ((edge_points['bottom-right'][1] - edge_points['bottom-left'][1]) ** 2))
    X_MAX = max(int(widthA), int(widthB))

    heightA = np.sqrt(((edge_points['top-right'][0] - edge_points['bottom-right'][0]) ** 2) + ((edge_points['top-right'][1] - edge_points['bottom-right'][1]) ** 2))
    heightB = np.sqrt(((edge_points['top-left'][0] - edge_points['bottom-left'][0]) ** 2) + ((edge_points['top-left'][1] - edge_points['bottom-left'][1]) ** 2))
    Y_MAX = max(int(heightA), int(heightB))
    dest_points = np.float32([[0, 0], [X_MAX, 0], [X_MAX, Y_MAX], [0, Y_MAX]])
    M = cv2.getPerspectiveTransform(source_points, dest_points)
    dst = cv2.warpPerspective(image_rgb, M, (X_MAX, Y_MAX))
    return dst

def get_transformed_student(rgb_img: np.ndarray, predictor: DefaultPredictor) -> Tuple[bool, np.array]:
    """Run Corner detection model and return transformation of student ID with size 300x500 px

    Parameters
    ----------
    img_path : np.ndarray
        RGB Numpy Array Image

    Returns
    -------
    Tuple[bool, np.array]
        If 4 corner detected, return True and RGB Student Id Image transformed and False, None if Corner not detected
    """
    bgr_img = rgb_img[...,::-1]
    padding_img = cv2.copyMakeBorder(bgr_img, 
                                     top=PADDING, bottom=PADDING, left=PADDING, right=PADDING,
                                     borderType=cv2.BORDER_CONSTANT, value=(0,0,0))
    rgb_img = padding_img[...,::-1]
    output_instances = predictor(padding_img)['instances'].to('cpu')
    result_instances = clean_output_instances(output_instances)
    if len(result_instances.pred_classes) < 4:
        return False, bgr_img[...,::-1]
    else:
        edge_points = get_corner_point(padding_img, result_instances)
        transformed_img = perspective_transform(rgb_img, edge_points)
        return True, transformed_img
    
    
if __name__ == '__main__':
    img_path = r'C:\Users\deviy\Desktop\ocr_train\data\train\0.png'
    predictor = load_model(r'C:\Users\deviy\Desktop\ocr_train\student_card_model\corner_detection\config.yaml')
    rgb_img = cv2.imread(img_path)
    transformed = get_transformed_student(rgb_img, predictor)

    print(transformed)
