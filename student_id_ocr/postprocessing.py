from student_id_ocr.abstract_class import PostprocessingStudentCardOCR
from student_id_ocr.student_card import StudentCardInformation, StudentCardInfoBase
from student_id_ocr.config_student_card import WORD_COORD_CONFIG
from typing import Dict, Iterable, Union
from pathlib import Path
import pandas as pd
import numpy as np
import jellyfish
import json
import re


class WordCoordPostprocessing(PostprocessingStudentCardOCR):
    """Postprocessing method for pd.DataFrame word coordinate from OCR Output"""

    def __init__(self, model_dir: Union[Path, str]) -> None:
        """Initialization of Word Coordinate PostProcessing

        Parameters
        ----------
        model_dir : Union[Path, str]
            Path to model directory to get config.json
        img_shape : Any[Tuple, None], optional
            Input Image Shape for grouping, by default None
        """
        CONFIG_PATH = model_dir / WORD_COORD_CONFIG["WORD_COOR"]
        if not CONFIG_PATH.exists():
            raise FileNotFoundError(
                f"""
        Config File for OCR Model doesn't exists in {CONFIG_PATH}
        """
            )
        with open(CONFIG_PATH, "r") as file:
            CONFIG_WORD = json.load(file)
        self.word_coord_cols = ["word", "p1", "p2", "p3", "p4"]
        self.form = StudentCardInformation()
        self.puncts = "!\"#$%&()*:;—,.'<=>?@[\]^_`{|}~“"
        self._input_img_shape = None
        self.split_y_pct = 0.45
        try:
            self.cols = CONFIG_WORD["col"]
            self.district_list =  [district["name"] for district in CONFIG_WORD["district"]]
            self.distric_mapping = CONFIG_WORD["district"]
            self.field_words = CONFIG_WORD["field_words"]
        except Exception as e:
            raise FileNotFoundError(
                "error getting values of config file"
            )
        
    @property
    def img_shape(self):
        return self._input_img_shape

    @img_shape.setter
    def img_shape(self, shape):
        self._input_img_shape = shape

    def postprocessing(self, ocr_output: pd.DataFrame) -> StudentCardInfoBase:
        """Postprocessing method by grouping bounding box and word to get KTP form object"""
        if not ocr_output.columns.isin(self.word_coord_cols).all():
            raise ValueError(f" ocr_output should contain columns: {self.cols}")
        # save ocr_output to object atributes
        self.ocr_output = ocr_output
        # Grouping the Words by binning the middle points to create one row
        self.ocr_output["middle_x"] = self.calculate_middle(0)
        self.ocr_output["middle_y"] = self.calculate_middle(1)
        self.ocr_output.sort_values(by=["middle_y", "middle_x"], inplace=True)
        self.ocr_output["bin_x"] = self.assign_bin_x()
        
        self.z_order()
        corrected_cols = self.get_similar_word(self.ocr_output["word"], self.field_words, 0.9)
        if len(corrected_cols) > 0:
            for key, value in corrected_cols.items():
                self.ocr_output.loc[self.ocr_output['word'] == key, 'field'] = value

        last_field = self.get_last_field([i for i in self.ocr_output["field"] if i!='' or not pd.isna(i)])
        self.ocr_output["field"] = self.ocr_output["field"].replace('', np.nan)
        self.ocr_output['prev_value'] = self.ocr_output['field'].shift(1)
        self.ocr_output['prev_value'] = self.ocr_output['prev_value'].ffill()
        self.ocr_output['prev_value'] = self.ocr_output['prev_value'].str.lower().str.replace(' ', '_')
        self.ocr_output.reset_index(drop=True, inplace=True)
        self.extract_info(last_field)
        self.add_district_info(var_similarity=0.9)
        return self.form

    def get_similar_word(
        self, word_result: Iterable[str], word_sim: Iterable[str], conf: float
    ) -> Dict:
        """Get similar word based on list of word dictionary
        Using jaro winkler similarity

        Parameters
        ----------
        word_result : Iterable
            Result of OCR
        word_sim : Iterable
            Word Dictionary
        conf : float
            Confidence (0 - 1)

        Returns
        -------
        Dict
            String with Replaced dictionary, if not found will throw nan.
        """
        corrected = {}
        for word in word_result:
            sim = [
                (form, jellyfish.jaro_winkler_similarity(word, form))
                for form in word_sim
            ]
            max_sim = max(sim, key=lambda x: x[1])
            if max_sim[1] >= conf:
                corrected[word] = max_sim[0]
            else:
                corrected[word] = np.nan

        return corrected

    def calculate_middle(self, axis: int) -> list:
        middles = []
        for _, rows in self.ocr_output.iterrows():
            middle = (
                int(rows.p1[axis])
                + int(rows.p2[axis])
                + int(rows.p3[axis])
                + int(rows.p4[axis])
            ) / 4
            middles.append(middle)
        return middles

    def z_order(self):
        if "bin_x" not in self.ocr_output:
            raise NotImplementedError(
                "bin_x not found in ocr_output, please run x axis grouping first"
            )
        iter_df = []
        for group in self.ocr_output["bin_x"].unique():
            group_df = (
                self.ocr_output.query(f"bin_x=={group}")
                .sort_values("middle_y")
                .reset_index(drop=True)
            )
            idx_done = []
            for idx, row in group_df.iterrows():
                if idx in idx_done:
                    continue
                if row["p1"][1] != row["p2"][1]:
                    top_y, bottom_y = row["p1"][1], row["p2"][1]
                else:
                    bottom_y = max(
                        [row["p1"][1], row["p2"][1], row["p3"][1], row["p4"][1]]
                    )
                    top_y = min(
                        [row["p1"][1], row["p2"][1], row["p3"][1], row["p4"][1]]
                    )
                row_df = group_df.loc[
                    (group_df["middle_y"] >= top_y) & (group_df["middle_y"] <= bottom_y)
                ].sort_values("middle_x")
                row_df = row_df[~row_df.index.isin(idx_done)]
                idx_done += row_df.index.to_list()
                row_df["bin_y"] = idx
                iter_df.append(row_df)

        self.ocr_output = pd.concat(iter_df, axis=0)

    def assign_bin_x(self) -> np.array:
        """Group KTP Text whether it's in the left side such as nama, alamat
        or in the right side like gol darah or date issued"""
        bin_x = np.where(
            self.ocr_output["p1"].str[0] > self.img_shape[1] * self.split_y_pct,
            1,  # if correct (right)
            0,  # else (left)
        )
        # Force first few lines to be left side
        bin_x[:4] = 0
        return bin_x
    
    def get_last_field(self, cleaned_df_list):
        for field in reversed(self.field_words):
            if field in cleaned_df_list:
                return field
        return None
    
    def extract_info(self, last_label):
        for index, row in self.ocr_output.iterrows():
            if row['field'] == 'Name' and self.form.name=='':
                self.form.name = ' '.join([row['word'] for i, row in self.ocr_output.iloc[index:index+4].iterrows() if pd.isna(row['field']) and row['prev_value'] == 'name'])
            elif row['field'] == 'Member ID' and self.form.member_id=='':
                self.form.member_id = ' '.join([row['word'] for i, row in self.ocr_output.iloc[index:index+2].iterrows() if pd.isna(row['field']) and row['prev_value'] == 'member_id'])
            elif row['field'] == 'Phone' and  self.form.phone_number=='':
                self.form.phone_number = ' '.join([row['word'] for i, row in self.ocr_output.iloc[index:index+2].iterrows() if pd.isna(row['field']) and row['prev_value'] == 'phone'])
            elif row['field'] == 'Address' and self.form.address=='':
                self.form.address = ' '.join([row['word'] for i, row in self.ocr_output.iloc[index:index+5].iterrows() if pd.isna(row['field']) and row['prev_value'] == 'address'])
            if row['field'] == last_label:
                break  # Stop iteration once last_label is found
         
    
    def get_district_code(self, district_name):
        for district in self.distric_mapping:
            if district["name"] == district_name:
                return district['code']
        return ''

    def add_district_info(self, var_similarity):
        corrected_district = np.nan
        temp_address = ''
        if self.form.address or self.form.address!='' :
            address_parts = self.form.address.split(', ')
            if len(address_parts)>1:
                last_address_part = address_parts[-1]
                if any(char.isdigit() for char in last_address_part): 
                    if len(address_parts[-2]) > 2:
                        corrected_district = self.get_similar_word(
                                    [address_parts[-2]], self.district_list, var_similarity
                                )[address_parts[-2]]
                        temp_address = address_parts[-2]
                    else:
                        corrected_district = self.get_similar_word(
                                    [address_parts[-3]], self.district_list, var_similarity
                                )[address_parts[-3]]
                        temp_address = address_parts[-3]
                self.form.district = (
                                corrected_district
                                if corrected_district is not np.nan
                                else temp_address
                            )
                self.form.district_code =  self.get_district_code(self.form.district)

if __name__ == '__main__':
    ocr_output = [([[137, 39], [618, 39], [618, 90], [137, 90]],
  'DAWNALE ACADEM OF',
  0.7931522201397618),
 ([[674, 69], [842, 69], [842, 99], [674, 99]],
  'Student ID Card',
  0.9997962604958386),
 ([[137, 88], [318, 88], [318, 136], [137, 136]],
  'MEDICINE',
  0.999964301355374),
 ([[454, 204], [524, 204], [524, 228], [454, 228]],
  'Name',
  0.9988865586884293),
 ([[453, 229], [641, 229], [641, 265], [453, 265]],
  'Emma Smith',
  0.9891406234196626),
 ([[454, 280], [578, 280], [578, 306], [454, 306]],
  'Member ID',
  0.9996052641237478),
 ([[452, 312], [612, 312], [612, 338], [452, 338]],
  '123-456-7890',
  0.9519679660149585),
 ([[451, 355], [526, 355], [526, 382], [451, 382]],
  'Phone',
  0.9999739303845567),
 ([[450, 386], [624, 386], [624, 414], [450, 414]],
  '+123-456-7890',
  0.7066969207315418),
 ([[452, 432], [548, 432], [548, 458], [452, 458]],
  'Address',
  0.9999934216890737),
 ([[450, 464], [858, 464], [858, 496], [450, 496]],
  '123 Anywhere St, Any City, ST 12345',
  0.7199318014784352)]
    
    def _get_word_coord(ocr_output: list) -> pd.DataFrame:
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
    
    model_path = r'C:\Users\deviy\Desktop\ocr_train\student_card_model'
    model_path = Path(model_path)
    postprocessing_class= WordCoordPostprocessing
    postpro = postprocessing_class(model_path)
    postpro.img_shape = (562, 925)

    word_coord_df = _get_word_coord(ocr_output)
                # Run post procesing on ocr_output
    student_form = postpro.postprocessing(word_coord_df)
    from student_id_ocr.student_card import  StudentCardObject
    img_path = r'C:\Users\deviy\Desktop\ocr_train\data\train\0.png'
    img_path= Path(img_path)
    print(StudentCardObject(student_form,img_path))
