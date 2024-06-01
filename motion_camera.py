#!/usr/bin/env python
from picamera2 import Picamera2
import cv2
import numpy as np
import time
import datetime
from datetime import date
from pydrive.drive import GoogleDrive
from pydrive.auth import GoogleAuth  
import os
import requests
from flask import Flask, render_template
from flask import Response
from threading import Thread

app = Flask(__name__)

camera = Picamera2()
capture_config = camera.create_preview_configuration()
camera.start(capture_config)

font = cv2.FONT_HERSHEY_SIMPLEX
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 

global vid_dir
vid_dir=r"/home/schillingderek/SecurityCamera/output_vids"
global motiondetection
motiondetection=0

gauth = GoogleAuth()
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)

def motionvideo():
    global motionvideostart, frame
    motionvideostart=0
    now=None
    now=datetime.datetime.now()
    filename=None
    filename="video "+str(date.today())+" "+str(now.hour)+":"+str(now.minute)+".mp4"
    filepath=None
    filepath=os.path.join(vid_dir,filename)
    print(filepath)
    out=None
    out = cv2.VideoWriter(str(filepath),fourcc, 20.0, (640, 480))

    for x in range(500):
        out.write(frame)
        cv2.waitKey(1)
    out.release()

    print("Uploading file...")
    f = drive.CreateFile({'title': str(filename)})
    f.SetContentFile(str(filepath))
    f.Upload()
    f=None
    print("Done")

    motionvideostart=1

def mse(frame1,frame2):
   h, w = frame1.shape
   diff = cv2.subtract(frame1, frame2)
   err = np.sum(diff**2)
   mse = err/(float(h*w))
   return mse

def livedetection():
    global frame, click, motiondetection, motionvideostart
    motionvideostart=1
    framecount=0
    click=0
    nextframe=0
    while True:
        global frame 
        frame=camera.capture_array("main")
        cv2.waitKey(1)
        framecount+=1
        if framecount%5==0:
            frame1mse=frame
            frame1gray=cv2.cvtColor(frame1mse,cv2.COLOR_BGR2GRAY)
            nextframe=framecount+4
            
        if framecount==50:
            framecount=0
            
        if framecount==nextframe:
            frame2mse=frame
            frame2gray=cv2.cvtColor(frame2mse,cv2.COLOR_BGR2GRAY)
            error = mse(frame1gray,frame2gray)
            print(error)
            print("motiondetection", motiondetection, "motionvideostart", motionvideostart)
            if error>=20.0:
                cv2.putText(frame, "Motion detected", (50,50), font, 1, (0,255,0), 2)
            if error>=20.0 and motiondetection==1 and motionvideostart==1:
                t1 = Thread(target=motionvideo)
                t1.start()
        if click==1:
            break

def webframes():
    print("Entering live feed")
    global frame
    while True:
        try:
            ret, buffer = cv2.imencode('.jpg', frame)
            webframe = buffer.tobytes()
            yield (b'--webframe\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + webframe + b'\r\n')
        except:
            break

@app.route('/')
def index():
    return "The system is running. For live feed, go to /livefeed"

@app.route('/start')
def startlive():
    t = Thread(target=livedetection)
    t.start()
    return "System started"

@app.route('/stop')
def stoplive():
    global click
    click=1
    return "System stopped"

@app.route('/livefeed')
def live_feed():
    return Response(webframes(), mimetype='multipart/x-mixed-replace; boundary=webframe')

@app.route('/disablemotion')
def disable():
    global motiondetection
    motiondetection=0
    return "Motion detection turned off"

@app.route('/enablemotion')
def enable():
    global motiondetection
    motiondetection=1
    return "Motion detection turned on"

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False) #Runs the web server.
