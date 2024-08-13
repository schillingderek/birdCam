#!/usr/bin/python3

from picamera2 import Picamera2
import time

# Initialize the Picamera2 instance
picam2 = Picamera2()

# Retrieve the still configuration (no call to a function)
still_config = picam2.create_still_configuration(main={"size": (1920, 1080)})
picam2.configure(still_config)

# Start the camera
picam2.start()

# Allow some time for the camera to adjust
time.sleep(2)

picam2.capture_file("testcapture.jpg", format="jpeg")

picam2.stop()


