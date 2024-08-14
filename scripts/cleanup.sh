#!/bin/bash

# Define directories
IMAGE_DIR="/root/birdcam/static/images"
VIDEO_DIR="/root/birdcam/static/videos"

# Find and delete files older than 1 day in the image directory
find "$IMAGE_DIR" -type f -mtime +1 -exec rm {} \;

# Find and delete files older than 1 day in the video directory
find "$VIDEO_DIR" -type f -mtime +1 -exec rm {} \;

echo "Old files deleted."

