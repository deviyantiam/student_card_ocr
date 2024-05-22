from dotenv import load_dotenv
from student_id_ocr.ocr_student_card import StudentCardReader

load_dotenv()


class OCRService(object):
    def __init__(self) -> None:
        self.easy_ocr_sc = StudentCardReader()

    def get_easy_ocr_sc(self, image) -> dict:
        result = self.easy_ocr_sc.run_ocr(image)
        result = result.to_dict()
        return result


