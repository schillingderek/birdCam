#!/usr/bin/python3

from picamera2 import Picamera2
import time
import subprocess

# Initialize the Picamera2 instance
picam2 = Picamera2()

# Retrieve the still configuration (no call to a function)
still_config = picam2.create_still_configuration(main={"size": (1920, 1080)})
picam2.configure(still_config)

# Start the camera
picam2.start()

# Allow some time for the camera to adjust
time.sleep(2)

for i in range(0, 201):
    lens_position = i * 0.05
    output_filename = f"image_focus_{lens_position:.2f}.jpg"  # Generate a filename with the lens position
    
    command = [
        "libcamera-still",
        "-t", "2000",                         # Time to wait before capturing
        "--autofocus-mode", "manual",         # Set autofocus mode to manual
        "--lens-position", f"{lens_position}",# Set the lens position
        "--tuning-file", "/usr/share/libcamera/ipa/rpi/vc4/ov5647_af.json",  # Tuning file
        "-o", output_filename                 # Output file name
    ]

    # Run the command
    try:
        subprocess.run(command, check=True)
        print(f"Image captured successfully: {output_filename}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred with lens position {lens_position}: {e}")


picam2.stop()


