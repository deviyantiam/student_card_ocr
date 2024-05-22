import numpy as np 
import cv2
from student_id_ocr.corner_detection import get_transformed_student
from student_id_ocr.abstract_class import PreprocessingStudentCardOCR
from detectron2.engine import DefaultPredictor

class GrayscaleThresholdingPreprocessing(PreprocessingStudentCardOCR):
    """Standard Grayscaling and Thresholding Preprocessing"""

    def preprocessing(self, rgb_img: np.ndarray) -> np.ndarray:
        """
        Preprocess the input image by converting it to grayscale and applying thresholding.
        
        Parameters
        ----------
        rgb_img : np.ndarray
            Input RGB image as a numpy array
        
        Returns
        -------
        np.ndarray
            Preprocessed image for OCR
        """    
        # Convert the input RGB image to grayscale
        gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        
        # Apply thresholding
        _, threshed_img = cv2.threshold(gray_img, 180, 255, cv2.THRESH_TRUNC)
        
        return threshed_img

class CornerDetectionGrayscaleThresholdingPreprocessing(PreprocessingStudentCardOCR):
    """Using Detectron2 Corner Detection to align KTP Image and return GrayscaleThresholdingPreprocessing"""
    
    def __init__(self, corner_detection_model: DefaultPredictor) -> None:
        """
        Initialize the preprocessing module with the corner detection model.
        
        Parameters
        ----------
        corner_detection_model : DefaultPredictor
            Detectron2 Corner Detection model
        """
        self.CLASS_NAMES = ['bottom-left', 'bottom-right', 'top-left', 'top-right']
        self.PADDING = 50
        self.CONF_THRESHOLD = 0.85
        self.detection_model = corner_detection_model
    
    def preprocessing(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Preprocess the input image by running corner detection and converting it to grayscale.
        
        Parameters
        ----------
        img_rgb : np.ndarray
            Input RGB image as a numpy array
        
        Returns
        -------
        np.ndarray
            Preprocessed image for OCR
        """
        # Run corner detection for alignment
        flag, transformed_img = get_transformed_student(img_rgb, self.detection_model)
        
        # Convert the transformed image to grayscale
        gray_img = cv2.cvtColor(np.array(transformed_img), cv2.COLOR_RGB2GRAY)
        
        # Apply blurring for smoother thresholding
        blur = cv2.blur(gray_img, (2, 2))
        
        return blur
