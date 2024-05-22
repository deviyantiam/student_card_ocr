# OCR for Dawnvale Academy of Medicine
## Table of Contents
* [About the Project](#about-the-project)
* [Getting Started](#getting-started)
    * [Prerequisites](#prerequisites)
    * [How to Use](#how-to-use)
    * [File Descriptions](#file-descriptions)
    * [Environment Variable](#environment-variable)
* [Author](#author)

<!-- About THE PROJECT -->
## [About The Project](#about-the-project)
This repository contains Python scripts for extracting information from dummy student cards using Optical Character Recognition (OCR). Our methods include leveraging our trained detectron2 model for corner detection and potential reorientation, followed by our trained OCR model application and post-processing for field extraction.
Plus, we've wrapped it all in a API, also a Streamlit interface

## [Getting Started](#getting-started)

### [Prerequisites](#prerequisites)
* **Python 3.9**
 and install requirements.txt

### [How to Use](#how-to-use)

1. Save your model folder in the home folder under the name ~/.SC_OCR. Your model folder should adhere to the following configuration:
```
.SC_OCR
├── config
│   ├── config.json                    <--- Master data for reference of similarity
├── model
│   ├── craft_mlt_25k.pth              <--- Text Detection Model (download from https://www.jaided.ai/easyocr/modelhub/)
│   └── en_student_card.pth            <--- OCR Trained Model Weight File
├── corner_detection                   
│   ├── config.yaml                    <--- Trained Detectron2 config
│   └── detectron2_studnt_id_corner_detection.pth    <--- Trained Model for corner detection
└── user_network
    ├── en_student_card.py             <--- OCR Model Architecture
    └── en_student_card.yaml           <--- OCR Model Config
```
2. run python app_main.py

### [File Descriptions](#file-descriptions)

1. **abstract_class.py** - Defines an abstract class for implementing OCR processing methods.
   
2. **config_student_card.py** - Configuration file specifying parameters for student card processing.

3. **corner_detection.py** - Utilizes corner detection algorithms to identify corners in images.

4. **preprocessing.py** - Applies corner detection and some grayscale before OCR processing.

5. **postprocessing.py** - Extracts text from the output of the OCR engine.

6. **student_card.py** - Contains templates and fields required for extracting information from student cards.

7. **ocr_student_card.py** - Orchestrates the entire OCR process to generate necessary information from student card images.

### [Environment Variable](#environment-variable)
The project is configured via environment variables, i.e. file `.env`

## [Author](#author)
Deviyanti AM [linkedin](https://linkedin.com/in/deviyanti-am)