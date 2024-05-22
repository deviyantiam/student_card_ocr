import os

from dotenv import load_dotenv

load_dotenv()

APP_CONFIG = {
    "PORT": os.getenv("OCR_PORT", 5000),
    "HOST": os.getenv("OCR_HOST", "localhost"),
    "VERSION": os.getenv("OCR_VERSION", "0.1.0")
}
