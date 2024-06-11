import time
import datetime
from datetime import date
from PIL import Image
from picamera2 import Picamera2
import cv2
import numpy as np
from pydrive.drive import GoogleDrive
from pydrive.auth import GoogleAuth
import os
import RPi.GPIO as GPIO
from flask import Flask, Response
from threading import Thread
from tflite_runtime.interpreter import Interpreter
from idBirdTfLite import check_for_bird

app = Flask(__name__)

PIR_PIN = 4
GPIO.setmode(GPIO.BCM)
GPIO.setup(PIR_PIN, GPIO.IN)

camera = Picamera2()

global video_size
video_size = (1920, 1080)
global still_size
still_size = (244, 244)
fps = 20.0
video_config = camera.create_video_configuration(
    main={"size": video_size, "format": "RGB"},
    controls={"FrameDurationLimits": (1000000 // fps, 1000000 // fps)}
)

font = cv2.FONT_HERSHEY_SIMPLEX
fourcc = cv2.VideoWriter_fourcc(*"H264")

global vid_dir
vid_dir = r"/home/schillingderek/SecurityCamera/output_vids"
global img_dir
img_dir = r"/home/schillingderek/SecurityCamera/images"
global motiondetection
motiondetection = 0

gauth = GoogleAuth()
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)

path_to_labels = "birds-label.txt"
path_to_model = "model.tflite"

prob_threshold = 0.4

global livefeed_active
livefeed_active = False

def motionvideo():
    global motionvideostart, frame, fps
    motionvideostart = 0
    now = datetime.datetime.now()
    filename = "video{}{}-{}.avi".format(date.today(), now.hour, now.minute)
    filepath = os.path.join(vid_dir, filename)
    print(filepath)
    
    out = cv2.VideoWriter(str(filepath), fourcc, fps, video_size)

    start_time = time.time()
    frame_count = 0
    frames_to_analyze = []
    num_frames_to_store = 5
    interval = int(fps * 30 / num_frames_to_store)  # even intervals over 30 seconds

    while time.time() - start_time < 30:
        frame_start_time = time.time()

        camera.configure(video_config)
        frame = camera.capture_array("main")
        if frame is not None:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
            if frame_count % interval == 0 and len(frames_to_analyze) < num_frames_to_store:
                frames_to_analyze.append(frame)
        else:
            print("No frame captured.")
        frame_count += 1
        
        # Ensure the loop runs at the correct frame rate
        time_elapsed = time.time() - frame_start_time
        time_to_wait = max(0, (1.0 / fps) - time_elapsed)
        time.sleep(time_to_wait)

    out.release()

    # Analyze the stored frames
    bird_labels = []
    for frame in frames_to_analyze:
        bird = check_for_bird(frame)
        if bird[0]:
            bird_labels.append(bird[0])

    if bird_labels:
        most_likely_bird = max(set(bird_labels), key=bird_labels.count)
        print(f"Most likely bird detected: {most_likely_bird}")
        rename_video_with_bird_name(filepath, most_likely_bird)
    else:
        print("No bird detected in the analyzed frames.")

    print("Uploading file...")
    f = drive.CreateFile({"title": str(os.path.basename(filepath))})
    f.SetContentFile(str(filepath))
    f.Upload()
    f = None
    print("Done")

    motionvideostart = 1

def rename_video_with_bird_name(filepath, bird_name):
    new_filepath = filepath.replace(".avi", f"_{bird_name}.avi")
    os.rename(filepath, new_filepath)

def livedetection():
    global frame, motiondetection, motionvideostart, livefeed_active
    motionvideostart = 1
    while livefeed_active or motiondetection:
        frame_start_time = time.time()

        camera.configure(video_config)
        frame = camera.capture_array("main")
        cv2.waitKey(1)
        pir_motion_sensor = GPIO.input(PIR_PIN)
        if pir_motion_sensor and motiondetection == 1 and motionvideostart == 1:
            t1 = Thread(target=motionvideo)
            t1.start()

        # Ensure the loop runs at the correct frame rate
        time_elapsed = time.time() - frame_start_time
        time_to_wait = max(0, (1.0 / fps) - time_elapsed)
        time.sleep(time_to_wait)

def webframes():
    print("Entering live feed")
    global frame
    while livefeed_active:
        try:
            ret, buffer = cv2.imencode(".jpg", frame)
            webframe = buffer.tobytes()
            yield (
                b"--webframe\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + webframe + b"\r\n"
            )
        except:
            break

@app.route("/")
def index():
    return "The system is running. For live feed, go to /livefeed"

@app.route("/start")
def startlive():
    global livefeed_active
    livefeed_active = True
    t = Thread(target=livedetection)
    t.start()
    return "System started"

@app.route("/stop")
def stoplive():
    global livefeed_active, motiondetection
    livefeed_active = False
    motiondetection = False
    return "System stopped"

@app.route("/livefeed")
def live_feed():
    global livefeed_active
    livefeed_active = True
    return Response(
        webframes(), mimetype="multipart/x-mixed-replace; boundary=webframe"
    )

@app.route("/disablemotion")
def disable():
    global motiondetection
    motiondetection = 0
    return "Motion detection turned off"

@app.route("/enablemotion")
def enable():
    global motiondetection
    motiondetection = 1
    return "Motion detection turned on"

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False)  # Runs the web server.
