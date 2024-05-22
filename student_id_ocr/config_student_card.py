import os

from dotenv import load_dotenv

load_dotenv()

# word_coordinate_post
WORD_COORD_CONFIG = {
    "WORD_COOR": os.getenv("STUDENT_CARD_WORD_COOR_CONFIG", "config/config.json"),
}

# easy_ocr
EASY_OCR_CONFIG = {
    "FILE_NAME": os.getenv("STUDENT_CARD_EASY_OCR_FILENAME", "en_student_card"),
    "PREPRO_CONFIG": os.getenv("STUDENT_CARD_PREPRO_PATH", "corner_detection/config.yaml"),
    "MODEL_PATH": os.getenv("STUDENT_CARD_MODEL_PATH", "model"),
    "USER_NETWORK_PATH": os.getenv("STUDENT_CARD_USER_NETWORK_PATH", "user_network"),
}
