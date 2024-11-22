#!/usr/bin/python3

from picamera2 import Picamera2
import time
import subprocess
from libcamera import controls

# Initialize the Picamera2 instance
picam2 = Picamera2()

# Retrieve the still configuration (no call to a function)
still_config = picam2.create_still_configuration()
picam2.configure(still_config)

# Start the camera
picam2.start()

# Allow some time for the camera to adjust
time.sleep(2)

for i in range(0, 21):
    lens_position = i * 0.05
    output_filename = f"image_focus_{lens_position:.2f}.jpg"  # Generate a filename with the lens position
    picam2.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": lens_position})
    picam2.capture_file(output_filename)
    print(f"Image captured successfully: {output_filename}")
    time.sleep(1)

picam2.stop()



