import os
import cv2
import uuid
import pytz
import logging
import base64
import numpy as np
from datetime import datetime
import module.task_manager as task_manager
from tornado.web import Application
import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
from tornado.escape import json_decode
from concurrent.futures import ProcessPoolExecutor
from module.ocr_service import OCRService


from app_config import APP_CONFIG
from dotenv import load_dotenv
load_dotenv()

service_logger = logging.getLogger(
    name="ocr-service"
)



ocr = OCRService()

def from_string(base64_string):
    nparr = np.frombuffer(base64.b64decode(base64_string), np.uint8)
    return cv2.imdecode(nparr, flags=cv2.IMREAD_COLOR)


def create_response(
    req_handler,
    response_code=200,
    response_status="OK",
    reason="",
    additional_response={},
):
    try:
        req_handler.set_status(response_code, reason)

        response = {"code": response_code, "status": response_status, "message": reason}

        combined_response = {**response, **additional_response}

        req_handler.write(combined_response)
        req_handler.finish()
    except Exception as e:
        reason = "Error in creating response."
        service_logger.error(reason+' '+ str(e))

        return


class OCRProcessing(tornado.web.RequestHandler):
    def initialize(self, tm_conn):
        self.tm_conn = tm_conn
        self.executor = ProcessPoolExecutor()

    def post(self):
        request = json_decode(self.request.body)

        if "content" in request["image"]:
            base64_image = request["image"]["content"]
            image = from_string(base64_image)
        else:
            image = None
            service_logger.warning("No image found.")

        if "format" in request["image"]:
            image_format = request["image"]["format"]
        else:
            image_format = "jpg"

        service_logger.info(
            "Processing image ...",
            request_type=request["type"],
            image_format=image_format,
        )


            
        jakarta_timezone = pytz.timezone('Asia/Jakarta')
        begin_date = datetime.now(jakarta_timezone)
        try:
            task_id = str(uuid.uuid4())
            result = ocr.get_easy_ocr_sc(image)
            output = {}
            output["data"] = {}
            output["data"]["image"] = request["image"]
            output["data"]["result"] = result

        except Exception as e:
            reason = "Error while processing image."
            service_logger.error(reason+' '+ str(e))
            status = "NOK"
            additional_response = {"error": e}
            reason = "Processing image succeed."
            task_name = "student card ocr"
            end_date = datetime.now(jakarta_timezone)
            message = {'content': request["image"]["content"]}
            task_manager.update_task(
            conn = self.tm_conn,
            task_id = task_id,
            task_type = task_name,
            task_status = status,
            task_start_date = begin_date,
            task_end_date = end_date,
            messages=str(message),
            types="insert"
            )

            create_response(
                self,
                response_code=500,
                response_status="UNKNOWN",
                reason=reason,
                additional_response=additional_response,
            )
            return

        reason = "Processing image succeed."
        task_name = "student card ocr"
        end_date = datetime.now(jakarta_timezone)
        service_logger.info(reason, request_type=request["type"])
        status = 'OK'
        message = {'content': request["image"]["content"]}
        task_manager.update_task(
            conn = self.tm_conn,
            task_id = task_id,
            task_type = task_name,
            task_status = status,
            task_start_date = begin_date,
            task_end_date = end_date,
            messages=str(result + str(message)),
            types="insert"
        )

        create_response(
            self,
            response_code=200,
            response_status="OK",
            reason=reason,
            additional_response=output,
        )
        return


def make_app(tm_conn=None):
    settings = {"debug": True}
    return Application(
        [
            (r"/ocr-processing", OCRProcessing, dict(tm_conn=tm_conn)),
        ],
        **settings
    )


def main(
    tm_conn=None,
    PORT=None,
    HOST="localhost",
):
    service_logger.info("Starting OCR...")
    service_logger.debug("Current working directory: {}".format(os.getcwd()))
    service_logger.info("Host: {}, Port: {} ".format(HOST, PORT))
    app = make_app(tm_conn)
    app.listen(PORT, address=HOST)
    print("*** Application is ready! ***")
    service_logger.info("Application is ready to use.")
    tornado.ioloop.IOLoop.instance().start()


if __name__ == "__main__":
    load_dotenv()
    tm_conn = task_manager.create_connection(os.getcwd() + "/tasks.sqlite3")
    task_manager.create_table(tm_conn)
    main(tm_conn=tm_conn, PORT=APP_CONFIG["PORT"], HOST=APP_CONFIG["HOST"])
