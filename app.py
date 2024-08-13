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

import subprocess
from flask import (
    Flask,
    render_template,
    Response,
    jsonify,
    request,
    session,
    redirect,
    url_for,
)
from flask_restful import Resource, Api, reqparse, abort
from datetime import datetime
from threading import Condition
import time
import os
import threading
from dotenv import load_dotenv

import logging
import sys

import sqlite3

from PIL import Image, ImageChops, ImageFilter

load_dotenv()

app = Flask(__name__, template_folder="template", static_url_path="/static")
app.secret_key = os.getenv("APP_SECRET_KEY")
api = Api(app)

# Global Login Credentials
users = {os.getenv("APP_LOGIN_USERNAME"): os.getenv("APP_LOGIN_PASSWORD")}

base_dir = "/root/birdcam"
video_dir = base_dir + "/static/videos/"
images_dir = base_dir + "/static/images"

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def get_predictions():
    conn = sqlite3.connect("/root/birdcam/db/bird_predictions.db")
    cursor = conn.cursor()

    # Fetch predictions from the last 60 seconds, only from the most recent INSERT
    cursor.execute(
        """
        SELECT timestamp, bird_species, confidence 
        FROM predictions 
        WHERE timestamp = (SELECT MAX(timestamp) FROM predictions) 
        AND timestamp >= datetime('now', '-360 seconds') 
        ORDER BY timestamp DESC
    """
    )
    predictions = cursor.fetchall()

    conn.close()
    return predictions


##############################################################################################################################################################

# Login Redirector

##############################################################################################################################################################


##############################################################################################################################################################

# @App Routes

##############################################################################################################################################################


@app.route("/")
def index():
    """Video streaming home page."""
    return render_template("index.html")


@app.route("/home", methods=["GET", "POST"])
def home_func():
    """Video streaming home page."""
    return render_template("index.html")


@app.route("/info.html")
def info():
    """Info Pane"""
    if "username" not in session:
        return redirect(url_for("login"))  # Redirect to login if not authenticated
    return render_template("info.html")


@app.route("/bird_info")
def bird_info():
    if "username" not in session:
        return redirect(url_for("login"))
    predictions = get_predictions()
    bird_ids = [prediction[1] for prediction in predictions]
    bird_scores = [prediction[2] for prediction in predictions]
    bird_info_list = [
        {"bird_id": bird_id, "bird_score": bird_score}
        for bird_id, bird_score in zip(bird_ids, bird_scores)
    ]
    return jsonify(bird_info_list)


@app.route("/api/files")
def api_files():

    try:
        images = [
            img
            for img in os.listdir(images_dir)
            if img.endswith((".jpg", ".jpeg", ".png"))
        ]
        videos = [file for file in os.listdir(video_dir) if file.endswith(".mp4")]
        # logging.info("Images found:", images)  # Debug logging.info
        # logging.info("Videos found:", videos)  # Debug logging.info
        return jsonify({"images": images, "videos": videos})
    except Exception as e:
        logging.info(f"Error in api_files: {str(e)}")  # Debug logging.info
        return jsonify({"error": str(e)})


@app.route("/delete-file/<filename>", methods=["DELETE"])
def delete_file(filename):
    # Determine if it's a video or picture based on the extension or another method
    if filename.endswith(".mp4") or filename.endswith(".mkv"):
        directory = video_dir
    else:
        directory = images_dir
    file_path = os.path.join(directory, filename)
    try:
        os.remove(file_path)
        return "", 204  # Successful deletion
    except Exception as e:
        return str(e), 500  # Internal server error


@app.route("/files")
def files():
    try:
        images = os.listdir(images_dir)
        videos = [
            file for file in os.listdir(video_dir) if file.endswith((".mp4"))
        ]  # Assuming video formats
        # Filtering out system files like .DS_Store which might be present in directories
        images = [img for img in images if img.endswith((".jpg", ".jpeg", ".png"))]
        return render_template("files.html", images=images, videos=videos)
    except Exception as e:
        return str(e)  # For debugging purposes, show the exception in the browser


@app.route("/login", methods=["GET", "POST"])
def login():
    if "username" in session:
        return redirect(url_for("index"))
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if username in users and users[username] == password:
            session["username"] = username
            return redirect(url_for("index"))
        else:
            return render_template("login.html", error="Invalid username or password")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(
        url_for("index")
    )  # Redirect to index which will force login due to session check


@app.before_request
def require_login():
    allowed_routes = [
        "login",
        "static",
    ]  # Make sure the streaming endpoints are either correctly authenticated or exempted here.
    if request.endpoint not in allowed_routes and "username" not in session:
        return redirect(url_for("login"))


##############################################################################################################################################################

# Ip and Port Routing

##############################################################################################################################################################

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8000)

    # Stream receive: ffplay udp://10.0.0.59:34235 -fflags nobuffer -flags low_delay -framedrop
    # Stream create: libcamera-vid -t 0 --inline -o udp://10.0.0.229:34235 --codec h264
