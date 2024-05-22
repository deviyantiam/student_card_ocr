from student_id_ocr.preprocessing import (
    CornerDetectionGrayscaleThresholdingPreprocessing
)
from student_id_ocr.postprocessing import WordCoordPostprocessing
from student_id_ocr.abstract_class import (
    RunStudentCardOCR,
    PreprocessingStudentCardOCR,
    PostprocessingStudentCardOCR,
    StudentCardBase,
)
from student_id_ocr.student_card import StudentCardInformation, StudentCardObject
from student_id_ocr.corner_detection import load_model
from student_id_ocr.config_student_card import EASY_OCR_CONFIG
from typing import Union
from pathlib import Path
import pandas as pd
import numpy as np
import easyocr
import base64
import cv2


class StudentCardReader(RunStudentCardOCR):
    """Student Card OCR Using Trained EasyOCR"""

    def __init__(
        self,
        gpu: bool = False,
        model_path: Path = Path.home() / ".SC_OCR",
        preprocessing_class: PreprocessingStudentCardOCR = CornerDetectionGrayscaleThresholdingPreprocessing,
        postprocessing_class: PostprocessingStudentCardOCR = WordCoordPostprocessing
    ) -> None:
        """Initialization of Student Card OCR Reader

        Parameters
        ----------
        gpu : bool, optional
            If use GPU, by default True
        model_path : Path, optional
            Directory to model weight file and configuration, by default '~/.SC_OCR'
        preprocessing_class : PreprocessingOCR
            Preprocessing class to apply `preprocessing()` method, by default CornerDetectionGrayscaleThresholdingPreprocessing
        postprocessing_class : PostprocessingOCR
            Postprocessing class to apply `postprocessing()` method, by default WordCoordPostprocessing
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                'Model path not found'
            )
        if not issubclass(preprocessing_class, PreprocessingStudentCardOCR):
            raise TypeError(
                "Preprocessing class must be subclass of PreprocessingStudentCardOCR Abstract Class"
            )
        if not issubclass(postprocessing_class, PostprocessingStudentCardOCR):
            raise TypeError(
                "Postprocessing class must be subclass of PostprocessingStudentCardOCR Abstract Class"
            )
        self.OCR = easyocr.Reader(
            lang_list=["en"],
            gpu=gpu,
            recog_network=EASY_OCR_CONFIG["FILE_NAME"],
            model_storage_directory=str(
                (model_path / EASY_OCR_CONFIG["MODEL_PATH"]).resolve()
            ),
            user_network_directory=str(
                (model_path / EASY_OCR_CONFIG["USER_NETWORK_PATH"]).resolve()
            ),
        )
        # Preprocessing Requirement if using CornerDetectionGrayscaleThresholdingPreprocessing
        if preprocessing_class == CornerDetectionGrayscaleThresholdingPreprocessing:
            student_card_detection = load_model(model_path / EASY_OCR_CONFIG["PREPRO_CONFIG"])
            self._prepro = preprocessing_class(student_card_detection)
        else:
            self._prepro = preprocessing_class()
        
        # Postprocessing Requirement if using WordCoordPostprocessing
        if postprocessing_class == WordCoordPostprocessing:
            self._postpro = postprocessing_class(model_path)
        else:
            self._postpro = postprocessing_class()

    def run_ocr(self, img: Union[np.array, str, Path]) -> StudentCardBase:
        """Running OCR Text Detection using EasyOCR

        Parameters
        ----------
        img : Union[np.array, str, Path]
            Student Card Image to run OCR, can be RGB numpy array or path to image file

        Returns
        -------
        StudentCardBase
            Student Card Object that has already data atributes
        """
        # read image
        if not type(img) == np.ndarray and not isinstance(img, bytes):
            self.img_path = Path(img)
            if not self.img_path.exists():
                raise FileNotFoundError("Image path not found")
            self.img = cv2.imread(str(self.img_path.resolve()))[..., ::-1]
        elif isinstance(img, bytes):
            self.img_path = None
            im_bytes = base64.b64decode(img)
            im_arr = np.frombuffer(im_bytes, dtype=np.uint8)
            self.img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
        else:
            self.img_path = None
            self.img = img
        # Run preprocessing
        input_img = self._prepro.preprocessing(self.img)
        # Get input_img shape for postprocessing
        self._postpro.img_shape = input_img.shape
        # Run ReadText from EasyOCR & get word coordinate
        ocr_output = self.OCR.readtext(input_img, width_ths=1, contrast_ths=0.3)
        # Check if result is not empty
        if ocr_output:
            word_coord_df = self._get_word_coord(ocr_output)
            # Run post procesing on ocr_output
            student_form = self._postpro.postprocessing(word_coord_df)
        else:
            student_form = StudentCardInformation()
        # Return Student Card Object
        return StudentCardObject(student_form, self.img_path)

    def _get_word_coord(self, ocr_output: list) -> pd.DataFrame:
        """Extract Word and Coordinate from OCR Output

        Parameters
        ----------
        ocr_output : list
            output from EasyOCR

        Returns
        -------
        pd.DataFrame
            Dataframe of Word and Coordinate
        """
        data = []
        for line in ocr_output:
            p1, p3, p4, p2 = line[0]
            data.append([line[1], p1, p2, p3, p4])
        word_coord = pd.DataFrame(data, columns=["word", "p1", "p2", "p3", "p4"])
        return word_coord
