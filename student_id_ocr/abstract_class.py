from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Union
from pathlib import Path
import pandas as pd
import numpy as np


class StudentCardInfoBase:
    """Base class for storing student card information."""

    def __init__(self):
        """Initialize form fields for student card information."""
        pass


class StudentCardBase(ABC):
    """Abstract class for representing the result of student card OCR."""

    @abstractmethod
    def __repr__(self) -> str:
        """Return string representation of the student card object."""
        pass

    @abstractmethod
    def to_df(self) -> pd.DataFrame:
        """Convert the OCR student card object to a pandas DataFrame."""
        pass

    @abstractmethod
    def to_json(self) -> str:
        """Convert the OCR student card object to JSON format."""
        pass

    @property
    @abstractmethod
    def status(self) -> str:
        """Get the OCR prediction status: OK or NotOK."""
        pass


class PreprocessingStudentCardOCR(ABC):
    """
    Abstract class for preprocessing methodologies for student card OCR.
    Subclasses must implement the `preprocess` method.
    """

    @abstractmethod
    def preprocessing(self, rgb_img: np.ndarray) -> np.ndarray:
        """
        Preprocess the input image for OCR.
        :param rgb_img: Input RGB image as a numpy array.
        :return: Preprocessed image as a numpy array.
        """
        pass


class PostprocessingStudentCardOCR(ABC):
    """Abstract class for postprocessing methodologies for student card OCR."""

    @abstractmethod
    def postprocessing(self, ocr_output_df: pd.DataFrame) -> StudentCardInfoBase:
        """
        Postprocess the OCR output DataFrame to generate student card information.
        :param ocr_output_df: DataFrame containing OCR output.
        :return: StudentCardInfoBase object containing extracted information.
        """
        pass


class RunStudentCardOCR(ABC):
    """Abstract class for running OCR on student cards."""

    @abstractmethod
    def run_ocr(self, img: Union[np.array, str, Path]) -> StudentCardBase:
        """
        Run OCR on the input image.
        :param img: Input image as a numpy array, file path, or string.
        :return: StudentCardBase object representing OCR result.
        """
        pass
