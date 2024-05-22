from typing import Dict, Optional
from pathlib import Path
import pandas as pd
import json

from student_id_ocr.abstract_class import StudentCardBase, StudentCardInfoBase


class StudentCardInformation(StudentCardInfoBase):
    """Form Information for Student Card OCR"""
    
    def __init__(self) -> None:
        """Field for Student Card Information, Initialized with empty string"""
        self.name = ""
        self.member_id = ""
        self.phone_number = ""
        self.address = ""
        self.district = ""
        self.district_code = ""


class StudentCardObject(StudentCardBase):
    """Student Card Result Object from OCR"""
    
    def __init__(self, studentform: StudentCardInformation, filename: Optional[Path] = None) -> None:    
        """Initialization of Student Card Object Result from OCR 

        Parameters
        ----------
        studentform : StudentCardInformation
            StudentCardInformation object that filled by OCR Method
        filename : Optional[Path]
            Path to raw image file if input is image filepath, default is None
        """        
        self.filename = filename
        self.info = studentform
        self.asdict = self.info.__dict__
        self.field_list = list(self.asdict.keys())
        self.__status = 'NotOK' if '' in self.asdict.values() or ' ' in self.asdict.values() else 'OK'
               
    @property
    def status(self) -> str:
        return self.__status
    
    def to_dict(self) -> Dict:
        """Convert Student Card Object to python Dictionary

        Returns
        -------
        dict
            Dictionary of Student Card field as key and Student Card information as values
        """        
        return self.asdict  
        
    def to_df(self)->pd.DataFrame:
        """Convert Student Card Object to pandas DataFrame

        Returns
        -------
        pd.DataFrame
            Dataframe object with fields as column and filename as index
        """        
        index = [self.filename.name] if self.filename else [0]
        df = pd.DataFrame(self.asdict, index=index)
        df.index.name = 'filename'
        return df
    
    def to_json(self) -> str:
        """Convert Student Card Object to json dumps

        Returns
        -------
        str
            json dumps with indent=4
        """        
        return json.dumps(self.asdict, indent=4)
    
    def __repr__(self) -> str:
        output =f"Student Card file: {self.filename.name if self.filename else None}\nStatus: {self.status}\n\n"
        for field in self.field_list:
            output += f"{field}: {self.asdict[field]}\n"
        return output