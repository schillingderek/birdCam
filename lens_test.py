#!/usr/bin/python3

from picamera2 import Picamera2
import time

# Initialize the Picamera2 instance
picam2 = Picamera2()

# Configure camera settings for still images
picam2.configure(picam2.still_configuration(main={"size": (1920, 1080)}))

# Start the camera
picam2.start()

# Allow some time for the camera to adjust
time.sleep(2)

# Set autofocus to manual
picam2.set_controls({"AfMode": "Manual"})

# Loop through lens positions from 0 to 1 in 0.05 increments
for lens_position in [i * 0.05 for i in range(21)]:  # Generates values from 0 to 1
    # Set the lens position for focus
    picam2.set_controls({"LensPosition": lens_position})
    
    # Capture a still image with higher quality
    picam2.capture_file(f"high_quality_image_{lens_position:.2f}.jpg", format="jpeg", quality=90)
    
    # Wait for 1 second before capturing the next image
    time.sleep(1)

# Stop the camera
picam2.stop()

print("Images captured at different lens positions.")
