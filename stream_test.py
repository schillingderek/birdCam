#!/usr/bin/env python3

import io
import picamera2
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput, CircularOutput
import time
from libcamera import controls

from threading import Thread, Condition

width_main = 1920
height_main = 1080

width_lores = 720
height_lores = 405


class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()

    def read(self):
        with self.condition:
            self.condition.wait()
            return self.frame

class Camera:
    def __init__(self):
        self.picamera = picamera2.Picamera2()
        self.video_config = self.picamera.create_video_configuration(
            {"size": (width_main, height_main), "format": "RGB888"}, lores={"size": (width_lores, height_lores)}
        )
        self.picamera.configure(self.video_config)
        self.video_capture_endoder = H264Encoder(10000000)
        self.video_capture_output = CircularOutput(buffersize = 1)
        self.streaming_encoder = H264Encoder()
        self.streaming_encoder.bitrate = 2500000
        self.streaming_encoder.profile = "baseline"
        self.stream_out = StreamingOutput()
        self.stream_out_2 = FileOutput(self.stream_out)
        self.streaming_output = [self.stream_out_2]

        self.picamera.start()
        self.picamera.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": 5.0})

    def capture_image(self):
        still_config = self.picamera.create_still_configuration()
        self.picamera.switch_mode_and_capture_file(still_config, "test.jpg")

camera = Camera()


def stream():
    camera.picamera.start_encoder(
        camera.streaming_encoder, camera.streaming_output, name="lores"
    )
    camera.picamera.start_encoder(
        camera.video_capture_endoder, camera.video_capture_output, name="main"
    )

    time.sleep(3)
    camera.capture_image()


if __name__ == "__main__":
    stream()

