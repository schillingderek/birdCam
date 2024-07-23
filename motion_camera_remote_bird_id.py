# Based on https://github.com/IcyG1045/CM4Cam/tree/main

##############################################################################################################################################################

                                                                        # Imports

##############################################################################################################################################################

import picamera2 
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder, MJPEGEncoder, Quality
from picamera2.outputs import FileOutput, CircularOutput
import io

import RPi.GPIO as GPIO

from pydrive.drive import GoogleDrive
from pydrive.auth import GoogleAuth

import requests

import piexif

import subprocess
from flask import Flask, render_template, Response, jsonify, request, session, redirect, url_for
from flask_restful import Resource, Api, reqparse, abort
from PIL import Image
from datetime import datetime
from threading import Condition
import time
import os
import numpy as np
import threading
from dotenv import load_dotenv

from PIL import Image, ImageChops, ImageFilter

load_dotenv()

app = Flask(__name__, template_folder='template', static_url_path='/static')
app.secret_key = os.getenv('APP_SECRET_KEY')
api = Api(app)

encoder = H264Encoder()
output = CircularOutput()

import subprocess
import smtplib
from email.mime.text import MIMEText

##############################################################################################################################################################

                                                                        # Globals

##############################################################################################################################################################

# Global or session variable to hold the current recording file name
current_video_file = None

# Global set time for email time cooldown.
last_email_sent_time = 0

# Global Thread Lock
email_lock = threading.Lock()

# Global Last Motion Time
last_motion_time = None

# Global Email Routes
sender_email = "schilling.derek@gmail.com"
app_password = os.getenv('GOOGLE_APP_PASSWORD')
receiver_email = "schilling.derek@gmail.com"

# Global Login Credentials CHANGE THEM
users = {os.getenv('APP_LOGIN_USERNAME'): os.getenv('APP_LOGIN_PASSWORD')}

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

ROTATION = 270
WIDTH = 1200
HEIGHT = 720
rotation_header = bytes()
if ROTATION:
    WIDTH, HEIGHT = HEIGHT, WIDTH
    code = 6 if ROTATION == 90 else 8
    exif_bytes = piexif.dump({'0th': {piexif.ImageIFD.Orientation: code}})
    exif_len = len(exif_bytes) + 2
    rotation_header = bytes.fromhex('ffe1') + exif_len.to_bytes(2, 'big') + exif_bytes

##############################################################################################################################################################

                                                                        # H264 to MP4 Converter

##############################################################################################################################################################

def convert_h264_to_mp4(source_file_path, output_file_path):
    try:
        # Command to convert h264 to mp4
        command = [
            'ffmpeg', '-i', source_file_path,
            '-vf', 'transpose=2',  # 'transpose=2' rotates 270 degrees
            '-c:v', 'libx264',      # Use libx264 codec for encoding
            '-crf', '23',           # Constant Rate Factor for quality (0-51, where lower is better)
            '-preset', 'medium',    # Encoding speed (ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
            output_file_path
        ]         
        subprocess.run(command, check=True)
        print(f"Conversion successful: {output_file_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")

def upload_video(file_path, output_path):
    try:
        convert_h264_to_mp4(file_path, output_path)
        print(f"Conversion successful for {output_path}")

        print("Uploading file...")
        f = drive.CreateFile({'parents': [{'id': google_drive_folder_id}], "title": str(os.path.basename(output_path))})
        f.SetContentFile(str(output_path))
        f.Upload()
        f = None
        print("Upload Completed.")
    except Exception as e:
        print(f"Failed to upload video: {e}")

def start_video_upload(file_path, output_path):
    upload_thread = threading.Thread(target=upload_video, args=(file_path, output_path))
    upload_thread.start()        

def show_time():
    """Return current time formatted for file names."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

##############################################################################################################################################################

                                                                        # Email Handler

##############################################################################################################################################################

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
            print("Email sent successfully!")
        except Exception as e:
            print(f"Failed to send email: {e}")
    thread = threading.Thread(target=email_thread)
    thread.start()

##############################################################################################################################################################

                                                                        # Camera Handler

##############################################################################################################################################################

class Camera:
    def __init__(self):
        self.camera = picamera2.Picamera2()
        # self.lores_size = (640, 360)
        self.hires_size = (WIDTH,HEIGHT)
        self.video_config = self.camera.create_video_configuration(main={"size": self.hires_size, "format": "RGB888"})
        self.camera.configure(self.video_config)
        self.encoder = MJPEGEncoder()
        self.encoder.bitrate = 10000000
        self.streamOut = StreamingOutput()
        self.streamOut2 = FileOutput(self.streamOut)
        self.encoder.output = [self.streamOut2]

        self.camera.start_encoder(self.encoder)
        self.camera.start_recording(encoder, output)
        self.previous_image = None
        self.motion_detected = False  # Track if motion is currently detected
        self.email_allowed = True
        self.last_motion_detected_time = None  # Initialize to None
        self.is_recording = False  # Track if video recording is in progress
        self.bird_id = []  # Change to a list to hold multiple detections
        self.bird_score = []  # Change to a list to hold multiple detections
        self.last_capture_time = time.time()
        self.periodic_image_capture_delay = 20
        self.drive_image_id = None

        # Start motion detection thread
        self.motion_detection_thread = threading.Thread(target=self.motion_detection_loop)
        self.motion_detection_thread.start()

    def perform_obj_detection_and_inference(self):
            print("Processing frame at: ", self.file_output)
            try:
                # Send the captured image to the Flask app running on your MacBook
                url = "https://feed-the-birds-88.loca.lt/process_image"
                data = {'file_id': self.drive_image_id}
                response = requests.post(url, json=data)
                print("Frame processed")
                
                if response.status_code == 200:
                    bird_results = response.json()
                    self.bird_id, self.bird_score = zip(*bird_results) if bird_results else ([], [])
                    print(bird_results)
                else:
                    print(f"Error in response from server: {response.status_code}")
                
            except Exception as e:
                print(f"Error sending image to server: {e}")

    def periodically_capture_and_process_frame(self):
        current_time = time.time()
        if current_time - self.last_capture_time > self.periodic_image_capture_delay:
            capture_frame_thread = threading.Thread(target=self.capture_frame)
            capture_frame_thread.start()
            self.last_capture_time = current_time

    def get_frame(self):
        self.camera.start()
        with self.streamOut.condition:
            self.streamOut.condition.wait()
            frame_data = self.streamOut.frame
        return frame_data


##############################################################################################################################################################

                                                                        # Motion Detection Handler

    def motion_detection_loop(self):
        while True:
            frame_data = self.get_frame()
            image = Image.open(io.BytesIO(frame_data)).convert('L')  # Convert to grayscale
            image = image.filter(ImageFilter.GaussianBlur(radius=2))  # Apply Gaussian blur
            if self.previous_image is not None:
                self.detect_motion(self.previous_image, image)
            self.previous_image = image
            time.sleep(1)
    
    def detect_motion(self, prev_image, current_image):
        global last_motion_time, current_video_file
        current_time = time.time()
        diff = ImageChops.difference(prev_image, current_image)
        diff = diff.point(lambda x: x > 40 and 255)    #Adjust 40 to change sensitivity. Higher is less sensitive
        count = np.sum(np.array(diff) > 0)
        pir_motion_sensor = GPIO.input(PIR_PIN)
        image_motion_sensor = count > 500
        if self.is_recording:
            self.periodically_capture_and_process_frame()
        if image_motion_sensor and pir_motion_sensor:  # Sensitivity threshold for motion AND PIR motion sensor input
            if self.email_allowed:
                # Motion is detected and email is allowed
                if last_motion_time is None or (current_time - last_motion_time > 30):
                    send_email("Motion Detected", "Motion has been detected by your camera.", sender_email, receiver_email, app_password)
                    print("Motion detected and email sent.")
                    last_motion_time = current_time  # Update the last motion time
                    self.email_allowed = False  # Prevent further emails until condition resets
                    self.start_recording()  # Start recording when motion is detected
                # else:
                #     print("Motion detected but not eligible for email due to cooldown.")
            # else:
            #     print("Motion detected but email not sent due to recent activity.")
            self.last_motion_detected_time = current_time
        else:
            # No motion detected
            if self.last_motion_detected_time and (current_time - self.last_motion_detected_time > 30) and not self.email_allowed:
                self.email_allowed = True  # Re-enable sending emails after 30 seconds of no motion
                print("30 seconds of no motion passed, emails re-enabled.")
                self.last_motion_detected_time = current_time  # Reset to prevent message re-printing
                self.stop_recording()  # Stop recording when no motion is detected for 30 seconds

##############################################################################################################################################################

                                                                        # Video Recording Handler

    def start_recording(self):
        global current_video_file
        if not self.is_recording:
            print("Starting video recording")
            basename = show_time()
            parent_dir = video_dir
            current_video_file = f"vid_{basename}.h264"
            output.fileoutput = os.path.join(parent_dir, current_video_file)
            output.start()
            self.is_recording = True

    def stop_recording(self):
        global current_video_file
        if self.is_recording:
            print("Stopping video recording")
            output.stop()
            if current_video_file:
                source_path = os.path.join(video_dir, current_video_file)
                output_path = source_path.replace('.h264', '.mp4')
                start_video_upload(source_path, output_path)
            self.is_recording = False

##############################################################################################################################################################

                                                                        # Picture Snap Handler


    def capture_frame(self):
        print("Capturing frame from video stream")
        frame = self.streamOut.frame
        image = Image.open(io.BytesIO(frame))
        rotated_image = image.rotate(270, expand=True)  # Rotate the image by 270 degrees
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file_output = f"{images_dir}/snap_{timestamp}.jpg"
        rotated_image.save(self.file_output)
        self.uploadFile()
        self.perform_obj_detection_and_inference()

    def uploadFile(self):
        print("Uploading file...")
        f = drive.CreateFile({'parents': [{'id': google_drive_folder_id}], "title": str(os.path.basename(self.file_output))})
        f.SetContentFile(str(self.file_output))
        f.Upload()
        self.drive_image_id = f['id']
        print(self.drive_image_id)
        f = None
        print("Upload Completed.")

class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf[:2] + rotation_header + buf[2:]
            self.condition.notify_all()

camera = Camera()

def genFrames():
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


##############################################################################################################################################################

                                                                        # Login Redirector

##############################################################################################################################################################

class VideoFeed(Resource):
    def get(self):
        if 'username' not in session:
            return redirect(url_for('login'))  # Ensure this follows your app's login logic
        return Response(genFrames(), mimetype='multipart/x-mixed-replace; boundary=frame')

##############################################################################################################################################################

                                                                        # @App Routes

##############################################################################################################################################################

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

@app.route('/home', methods=['GET', 'POST'])
def home_func():
    """Video streaming home page."""
    return render_template("index.html")

@app.route('/info.html')
def info():
    """Info Pane"""
    if 'username' not in session:
        return redirect(url_for('login'))  # Redirect to login if not authenticated
    return render_template('info.html')

@app.route('/bird_info')
def bird_info():
    if 'username' not in session:
        return redirect(url_for('login'))
    bird_ids = camera.bird_id
    bird_scores = camera.bird_score
    bird_info_list = [{'bird_id': bird_id, 'bird_score': bird_score} for bird_id, bird_score in zip(bird_ids, bird_scores)]
    return jsonify(bird_info_list)

# @app.route('/snap.html')
# def snap():
#     """Snap Pane"""
#     print("Taking a photo")
#     camera.capture_frame()
#     camera.perform_obj_detection_and_inference()
#     return render_template('snap.html')

@app.route('/api/files')
def api_files():

    try:
        images = [img for img in os.listdir(images_dir) if img.endswith(('.jpg', '.jpeg', '.png'))]
        videos = [file for file in os.listdir(video_dir) if file.endswith('.mp4')]
        # print("Images found:", images)  # Debug print
        # print("Videos found:", videos)  # Debug print
        return jsonify({'images': images, 'videos': videos})
    except Exception as e:
        print("Error in api_files:", str(e))  # Debug print
        return jsonify({'error': str(e)})

@app.route('/delete-file/<filename>', methods=['DELETE'])
def delete_file(filename):
    # Determine if it's a video or picture based on the extension or another method
    if filename.endswith('.mp4') or filename.endswith('.mkv'):
        directory = video_dir
    else:
        directory = images_dir
    file_path = os.path.join(directory, filename)
    try:
        os.remove(file_path)
        return '', 204  # Successful deletion
    except Exception as e:
        return str(e), 500  # Internal server error

@app.route('/files')
def files():
    try:
        images = os.listdir(images_dir)
        videos = [file for file in os.listdir(video_dir) if file.endswith(('.mp4'))]  # Assuming video formats
        # Filtering out system files like .DS_Store which might be present in directories
        images = [img for img in images if img.endswith(('.jpg', '.jpeg', '.png'))]
        return render_template('files.html', images=images, videos=videos)
    except Exception as e:
        return str(e)  # For debugging purposes, show the exception in the browser

api.add_resource(VideoFeed, '/cam')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'username' in session:
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username] == password:
            session['username'] = username
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error="Invalid username or password")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('index'))  # Redirect to index which will force login due to session check

@app.before_request
def require_login():
    allowed_routes = ['login', 'static']  # Make sure the streaming endpoints are either correctly authenticated or exempted here.
    if request.endpoint not in allowed_routes and 'username' not in session:
        return redirect(url_for('login'))

##############################################################################################################################################################

                                                                        # Ip and Port Routing

##############################################################################################################################################################

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)