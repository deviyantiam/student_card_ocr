import streamlit as st
from PIL import Image
import time
from pathlib import Path
import cv2
from student_id_ocr.ocr_student_card import StudentCardReader

TEMP_FILE = r'C:\Users\deviy\Desktop\ocr_student_service\demo_temp.jpg'
# Function to read image file and convert it to base64
def read_image(file):
    if file is not None:
        image = Image.open(file)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image.save(TEMP_FILE)
        rgb_img = cv2.imread(TEMP_FILE)[...,::-1]
    return rgb_img

# Main function to run OCR and display result
def run_ocr(image_base64):
    
    start = time.perf_counter()
    result = StudentCardReader().run_ocr(image_base64)
    result = result.to_dict()
    end = time.perf_counter()
    return result, start, end

# Streamlit app
def main():
    st.title('Dawnvale Academy of Medicine Student Card OCR')

    uploaded_file = st.file_uploader("Upload an image of the student card", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        image_base64 = read_image(uploaded_file)
        result_dict, start, end = run_ocr(image_base64)
        st.write(f'OCR Runtime: {end-start:.2f} seconds')
        st.subheader("OCR Result:")
        st.write(result_dict)

if __name__ == "__main__":
    main()