#!/usr/bin/env python3

import io
import picamera2
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput, CircularOutput
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from wsgiref.simple_server import make_server
from ws4py.websocket import WebSocket
from ws4py.server.wsgirefserver import (
    WSGIServer,
    WebSocketWSGIHandler,
    WebSocketWSGIRequestHandler,
)
from ws4py.server.wsgiutils import WebSocketWSGIApplication
from threading import Thread, Condition
import time
import pytz
from datetime import datetime
import subprocess

import smtplib
from email.mime.text import MIMEText

import RPi.GPIO as GPIO

from pydrive.drive import GoogleDrive
from pydrive.auth import GoogleAuth
from google.cloud import storage

import requests

import ffmpeg

import os
from dotenv import load_dotenv

import logging
import sys

import sqlite3

import numpy as np

load_dotenv()

video_capture_endoder = H264Encoder()
video_capture_endoder.bitrate = 10000000
video_capture_output = CircularOutput()

startTime = time.time()

width_main = 1280
height_main = 720

width_lores = 720
height_lores = 405

last_motion_time = None

sender_email = "schilling.derek@gmail.com"
app_password = os.getenv('GOOGLE_APP_PASSWORD')
receiver_email = "schilling.derek@gmail.com"

# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')


gauth = GoogleAuth()
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)

google_drive_folder_id = "1Gut6eCG_p6WmLDRj3w3oHFOiqMHlXkFr"

PIR_PIN = 4
GPIO.setmode(GPIO.BCM)
GPIO.setup(PIR_PIN, GPIO.IN)

base_dir = "/root/birdcam"
video_dir = base_dir + "/static/videos/"
images_dir = base_dir + "/static/images"

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_h264_to_mp4(source_file_path, output_file_path):
    try:
        ffmpeg_path = '/usr/bin/ffmpeg'  # Adjust to the actual path of your ffmpeg executable
        (
            ffmpeg
            .input(source_file_path)
            .output(
                output_file_path,
                **{
                    'c': 'copy'  # Copy the video stream without re-encoding
                }
            )
            .run(cmd=ffmpeg_path)
        )
        logging.info(f"Conversion successful: {output_file_path}")
    except ffmpeg.Error as e:
        logging.error(f"Error occurred: {e.stderr.decode()}")

def upload_video(file_path, output_path):
    return None
    # try:
    #     convert_h264_to_mp4(file_path, output_path)
    #     logging.info(f"Conversion successful for {output_path}")

    #     logging.info("Uploading file...")
    #     f = drive.CreateFile({'parents': [{'id': google_drive_folder_id}], "title": str(os.path.basename(output_path))})
    #     f.SetContentFile(str(output_path))
    #     f.Upload()
    #     f = None
    #     logging.info("Upload Completed.")
    # except Exception as e:
    #     logging.info(f"Failed to upload video: {e}")

def start_video_upload(file_path, output_path):
    upload_thread = Thread(target=upload_video, args=(file_path, output_path))
    upload_thread.start()        

def show_time():
    """Return current time formatted for file names."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def send_email(subject, body, sender, receiver, password):
    def email_thread():
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = sender
        msg['To'] = receiver
        try:
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(sender, password)
                server.send_message(msg)
            logging.info("Email sent successfully!")
        except Exception as e:
            logging.info(f"Failed to send email: {e}")
    thread = Thread(target=email_thread)
    thread.start()

def get_video_duration(video_file):
    """Get the duration of the video in seconds using ffprobe."""
    command = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_file
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return float(result.stdout.strip())


class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()

    def read(self):
        with self.condition:
            self.condition.wait()
            return self.frame


class Camera:
    def __init__(self):
        self.picamera = picamera2.Picamera2()
        self.video_config = self.picamera.create_video_configuration(
            {"size": (width_main, height_main), "format": "RGB888"}, lores={"size": (width_lores, height_lores)}
        )
        self.picamera.configure(self.video_config)
        self.streaming_encoder = H264Encoder()
        self.streaming_encoder.bitrate = 2500000
        self.streaming_encoder.profile = "baseline"
        self.stream_out = StreamingOutput()
        self.stream_out_2 = FileOutput(self.stream_out)
        self.streaming_output = [self.stream_out_2]

        self.motion_detected = False  # Track if motion is currently detected
        self.email_allowed = True
        self.last_motion_detected_time = None  # Initialize to None
        self.is_recording = False  # Track if video recording is in progress
        self.bird_id = []  # Change to a list to hold multiple detections
        self.bird_score = []  # Change to a list to hold multiple detections
        self.last_capture_time = time.time()
        self.periodic_image_capture_delay = 20
        self.drive_image_id = None
        self.current_image_file = None
        self.current_video_file = None
        self.start_recording_time = None

        self.picamera.start()

    def perform_obj_detection_and_inference(self):
        logging.info(f"Processing frame at: {self.current_image_file}")
        try:
            url = "http://10.0.0.194:8080/process_image"
            data = {'file_id': self.drive_image_id}
            response = requests.post(url, json=data)
            logging.info("Frame processed")
            
            if response.status_code == 200:
                bird_results = response.json()
                self.bird_id, self.bird_score = zip(*bird_results) if bird_results else ([], [])
                logging.info(bird_results)
            else:
                logging.info(f"Error in response from server: {response.status_code}")
            
        except Exception as e:
            logging.info(f"Error sending image to server: {e}")

    def store_inference(self):
        if len(self.bird_id) > 0:
            print("storing inference data")
            conn = sqlite3.connect('/root/birdcam/db/bird_predictions.db')
            cursor = conn.cursor()

            # Create table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    bird_species TEXT,
                    confidence REAL
                )
            ''')

            data_to_insert = []
            utc_now = datetime.now(pytz.utc)
            timestamp = utc_now.strftime('%Y-%m-%d %H:%M:%S')
            for species, confidence in zip(self.bird_id, self.bird_score):
                data_to_insert.append((timestamp, species, confidence))

            # Insert multiple rows at once
            cursor.executemany('''
                INSERT INTO predictions (timestamp, bird_species, confidence)
                VALUES (?, ?, ?)
            ''', data_to_insert)

            conn.commit()
            conn.close()
        else:
            print("no data to store")

    def periodically_capture_and_process_frame(self):
        current_time = time.time()
        if current_time - self.last_capture_time > self.periodic_image_capture_delay:
            self.last_capture_time = current_time
            capture_process_frame_thread = Thread(target=self.capture_and_process_frame)
            capture_process_frame_thread.start()
            
    
    def start_recording(self):
        if not self.is_recording:
            logging.info("Starting video recording")
            basename = show_time()
            video_file_name = f"vid_{basename}.h264"
            self.current_video_file = os.path.join(video_dir, video_file_name)
            video_capture_output.fileoutput = self.current_video_file
            video_capture_output.start()
            self.is_recording = True
            self.start_recording_time = time.time()
            print("video started at: ", self.start_recording_time)

    def stop_recording(self):
        if self.is_recording:
            logging.info("Stopping video recording")
            video_capture_output.stop()
            if self.current_video_file:
                source_path = self.current_video_file
                output_path = source_path.replace('.h264', '.mp4')
                start_video_upload(source_path, output_path)
            self.is_recording = False

    def capture_and_process_frame(self):
        logging.info("Capturing frame from video stream")
        timestamp = show_time()
        self.current_image_file = f"{images_dir}/snap_{timestamp}.jpg"
        self.capture_image()
        self.upload_image_to_google_drive()
        self.perform_obj_detection_and_inference()
        self.store_inference()
        self.delete_image()

    def upload_image_to_gcs(self):
        logging.info("Uploading to GCS")
        storage_client = storage.Client(project="birdcam1")
        bucket = storage_client.bucket("bird_cam_media")
        blob = bucket.blob(str(os.path.basename(self.current_image_file)))
        blob.upload_from_filename(self.current_image_file)
        logging.info("Upload to GCS finished")

    def upload_image_to_google_drive(self):
        logging.info("Uploading file...")
        f = drive.CreateFile({'parents': [{'id': google_drive_folder_id}], "title": str(os.path.basename(self.current_image_file))})
        f.SetContentFile(str(self.current_image_file))
        f.Upload()
        self.drive_image_id = f['id']
        logging.info(self.drive_image_id)
        f = None
        logging.info("Upload Completed.")

    def capture_image(self):
        request = self.picamera.capture_request()
        request.save("main", self.current_image_file)
        request.release()

    def delete_image(self):
        if os.path.exists(self.current_image_file):
            os.remove(self.current_image_file)
        logging.info("Deleted image")


camera = Camera()





def stream():
    global last_motion_time
    prev = None
    mse = 0
    motion_detection_delay = 5
    last_motion_check = time.time()

    camera.picamera.start_encoder(
        camera.streaming_encoder, camera.streaming_output, name="lores"
    )
    camera.picamera.start_encoder(
        video_capture_endoder, video_capture_output, name="main"
    )

    try:
        WebSocketWSGIHandler.http_version = "1.1"
        websocketd = make_server(
            "",
            9000,
            server_class=WSGIServer,
            handler_class=WebSocketWSGIRequestHandler,
            app=WebSocketWSGIApplication(handler_cls=WebSocket),
        )
        websocketd.initialize_websockets_manager()
        websocketd_thread = Thread(target=websocketd.serve_forever)

        try:
            websocketd_thread.start()

            while True:
                current_time = time.time()
                current_hour = datetime.now().hour
                # Read from the StreamingOutput and broadcast via WebSocket
                frame_data = camera.stream_out.read()
                if frame_data:
                    # print("Sending frame of size:", len(frame_data))
                    websocketd.manager.broadcast(frame_data, binary=True)
                else:
                    print("No frame data received")

                if 7 <= current_hour < 21:
                    camera.periodically_capture_and_process_frame()

##############################################################################################################################################################

                                                                        # Motion Detection Handler

                # if current_time - last_motion_check > motion_detection_delay: #only check for motion every few seconds
                #     last_motion_check = current_time
                #     pir_motion_sensor = GPIO.input(PIR_PIN)
                #     cur = camera.picamera.capture_buffer("lores")
                #     if prev is not None:
                #         mse = np.square(np.subtract(cur, prev)).mean()
                    
                #     logging.info(f"PIR: {pir_motion_sensor}")
                #     logging.info(f"MSE: {mse}")

                    # if pir_motion_sensor and mse > 30: #Higher MSE is LESS sensitive
                    #     if camera.email_allowed:
                    #         # Motion is detected and email is allowed
                    #         if last_motion_time is None or (current_time - last_motion_time > 15):
                    #             send_email("Motion Detected", "Motion has been detected by your camera.", sender_email, receiver_email, app_password)
                    #             logging.info("Motion detected and email sent.")
                    #             last_motion_time = current_time  # Update the last motion time
                    #             camera.email_allowed = False  # Prevent further emails until condition resets
                    #             camera.start_recording()  # Start recording when motion is detected
                    #         # else:
                    #         #     logging.info("Motion detected but not eligible for email due to cooldown.")
                    #     # else:
                    #     #     logging.info("Motion detected but email not sent due to recent activity.")
                    #     camera.last_motion_detected_time = current_time
                    # else:
                    #     # No motion detected
                    #     if camera.last_motion_detected_time and (current_time - camera.last_motion_detected_time > 15) and not camera.email_allowed:
                    #         camera.email_allowed = True  # Re-enable sending emails after 15 seconds of no motion
                    #         logging.info("15 seconds of no motion passed, emails re-enabled.")
                    #         camera.last_motion_detected_time = current_time  # Reset to prevent message re-logging.infoing
                    #         camera.stop_recording()  # Stop recording when no motion is detected for 15 seconds

                    # prev = cur

        except KeyboardInterrupt:
            pass
        finally:
            websocketd.shutdown()
            camera.picamera.stop()
            camera.picamera.stop_encoder()
            raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass
    finally:
        camera.picamera.stop()
        camera.picamera.stop_encoder()


if __name__ == "__main__":
    stream()
