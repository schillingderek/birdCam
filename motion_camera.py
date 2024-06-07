#!/usr/bin/env python
from PIL import Image
from picamera2 import Picamera2
import cv2
import numpy as np
import time
import datetime
from datetime import date
from pydrive.drive import GoogleDrive
from pydrive.auth import GoogleAuth
import os
import time
import RPi.GPIO as GPIO
from flask import Flask
from flask import Response
from threading import Thread
from tflite_runtime.interpreter import Interpreter

app = Flask(__name__)

PIR_PIN = 4
GPIO.setmode(GPIO.BCM)
GPIO.setup(PIR_PIN, GPIO.IN)

camera = Picamera2()
capture_config = camera.create_preview_configuration()
camera.start(capture_config)

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
path_to_model = "birds-model.tflite"

prob_threshold = 0.4


def motionvideo():
    global motionvideostart, frame
    motionvideostart = 0
    now = datetime.datetime.now()
    filename = "video{}{}-{}.avi".format(date.today(), now.hour, now.minute)
    filepath = os.path.join(vid_dir, filename)
    print(filepath)
    width = 640
    height = 480
    size = (width, height)
    fps = 20.0
    out = cv2.VideoWriter(str(filepath), fourcc, fps, size)

    num_frames = 20 * 10
    start_time = time.time()

    for x in range(num_frames):
        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            out.write(frame_rgb)
        else:
            print("No frame captured.")
        time.sleep(0.05)
    duration = time.time() - start_time
    print(f"Recording completed in {duration:.2f} seconds for {num_frames} frames.")

    out.release()

    print("Uploading file...")
    f = drive.CreateFile({"title": str(filename)})
    f.SetContentFile(str(filepath))
    f.Upload()
    f = None
    print("Done")

    motionvideostart = 1


def mse(frame1, frame2):
    h, w = frame1.shape
    diff = cv2.subtract(frame1, frame2)
    err = np.sum(diff**2)
    mse = err / (float(h * w))
    return mse


def livedetection():
    global frame, click, motiondetection, motionvideostart
    motionvideostart = 1
    framecount = 0
    click = 0
    nextframe = 0
    while True:
        global frame
        frame = camera.capture_array("main")
        cv2.waitKey(1)
        framecount += 1
        if framecount % 5 == 0:
            frame1mse = frame
            frame1gray = cv2.cvtColor(frame1mse, cv2.COLOR_BGR2GRAY)
            nextframe = framecount + 4

        if framecount == 50:
            framecount = 0

        if framecount == nextframe:
            frame2mse = frame
            frame2gray = cv2.cvtColor(frame2mse, cv2.COLOR_BGR2GRAY)
            error = mse(frame1gray, frame2gray)
            # print(error)
            pir_motion_sensor = GPIO.input(PIR_PIN)
            print(
                "motiondetection",
                motiondetection,
                "motionvideostart",
                motionvideostart,
                "cameramotiondetected",
                str(error >= 20.0),
                "pirmotionsensor",
                pir_motion_sensor,
            )
            if error >= 20.0 and pir_motion_sensor:
                cv2.putText(frame, "Motion detected", (50, 50), font, 1, (0, 255, 0), 2)
            if error >= 20.0 and pir_motion_sensor:
                print("checking to see if there is a bird")
                start_time = time.time()
                bird = check_for_bird()
                end_time = time.time()
                print(
                    f"Time taken to check for bird: {end_time - start_time:.2f} seconds"
                )
            # if error>=20.0 and pir_motion_sensor and motiondetection==1 and motionvideostart==1:
            #     t1 = Thread(target=motionvideo)
            #     t1.start()
        if click == 1:
            break


def check_for_bird():
    """is there a bird at the feeder?"""
    global frame
    labels = load_labels()
    interpreter = Interpreter(path_to_model)
    interpreter.allocate_tensors()
    _, height, width, _ = interpreter.get_input_details()[0]["shape"]

    resized_frame = cv2.resize(frame, (224, 224))
    image_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    results = classify_image(interpreter, image_pil)
    label_id, prob = results[0]
    print("bird: " + labels[label_id])
    print("prob: " + str(prob))

    now = datetime.datetime.now()

    filename = "image{}{}-{}-{}-{}.jpg".format(
        date.today(), now.hour, now.minute, labels[label_id], str(prob)
    )
    filepath = os.path.join(img_dir, filename)
    cv2.imwrite(filepath, resized_frame)
    print("Image saved successfully at:", filepath)

    if prob > prob_threshold:
        bird = labels[label_id]
        bird = bird[bird.find(",") + 1 :]
        prob_pct = str(round(prob * 100, 1)) + "%"
        return bird, prob_pct

    return None, None


def load_labels():
    """load labels for the ML model from the file specified"""
    with open(path_to_labels, "r") as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}


def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]["index"]
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image


def classify_image(interpreter, image, top_k=1):
    """return a sorted array of classification results"""
    set_input_tensor(interpreter, image)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    output = np.squeeze(interpreter.get_tensor(output_details["index"]))

    # if model is quantized (uint8 data), then dequantize the results
    if output_details["dtype"] == np.uint8:
        scale, zero_point = output_details["quantization"]
        output = scale * (output - zero_point)

    ordered = np.argpartition(-output, top_k)
    return [(i, output[i]) for i in ordered[:top_k]]


def webframes():
    print("Entering live feed")
    global frame
    while True:
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
    t = Thread(target=livedetection)
    t.start()
    return "System started"


@app.route("/stop")
def stoplive():
    global click
    click = 1
    return "System stopped"


@app.route("/livefeed")
def live_feed():
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
