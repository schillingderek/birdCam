#!/usr/bin/env python3

import io
import picamera2
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput, CircularOutput
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from wsgiref.simple_server import make_server
from ws4py.websocket import WebSocket
from ws4py.server.wsgirefserver import (
    WSGIServer,
    WebSocketWSGIHandler,
    WebSocketWSGIRequestHandler,
)
from ws4py.server.wsgiutils import WebSocketWSGIApplication
from threading import Thread, Condition
import time
from PIL import Image
from datetime import datetime
import subprocess

video_capture_endoder = H264Encoder()
video_capture_output = CircularOutput()

startTime = time.time()

WIDTH = 1280
HEIGHT = 720


# class StreamingOutput(io.BufferedIOBase):
#     def __init__(self):
#         self.frame = None
#         self.condition = Condition()

#     def write(self, buf):
#         with self.condition:
#             self.frame = buf
#             self.condition.notify_all()

#     def read(self):
#         with self.condition:
#             self.condition.wait()
#             return self.frame
        
class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()
        self.buffer = io.BytesIO()  # Buffer to hold frame data

    def write(self, buf):
        with self.condition:
            self.buffer.write(buf)
            self.buffer.seek(0)
            self.frame = self.buffer.read()
            self.condition.notify_all()

    def read(self):
        with self.condition:
            self.condition.wait()
            return self.frame


class Camera:
    def __init__(self):
        self.picamera = picamera2.Picamera2()
        self.video_config = self.picamera.create_video_configuration(
            {"size": (WIDTH, HEIGHT)}, lores={"size": (640, 360)}
        )
        self.picamera.configure(self.video_config)
        self.streaming_encoder = H264Encoder()
        self.streaming_encoder.bitrate = 2500000
        self.streaming_encoder.profile = "baseline"
        self.stream_out = StreamingOutput()
        self.stream_out_2 = FileOutput(self.stream_out)
        self.streaming_output = [self.stream_out_2]

        self.picamera.start()


camera = Camera()

def convert_frame_to_image(frame_data):
    # Use ffmpeg to decode the H264 frame data to raw YUV format
    # Then convert raw YUV format to PIL image
    img = Image.fromarray(np.frombuffer(frame_data, np.uint8).reshape((HEIGHT, WIDTH, 3)), 'RGB')
    return img


def stream():

    camera.picamera.start_encoder(
        camera.streaming_encoder, camera.streaming_output, name="lores"
    )
    camera.picamera.start_encoder(
        video_capture_endoder, video_capture_output, name="main"
    )
    recording_complete = False
    is_recording = False
    try:
        WebSocketWSGIHandler.http_version = "1.1"
        websocketd = make_server(
            "",
            9000,
            server_class=WSGIServer,
            handler_class=WebSocketWSGIRequestHandler,
            app=WebSocketWSGIApplication(handler_cls=WebSocket),
        )
        websocketd.initialize_websockets_manager()
        websocketd_thread = Thread(target=websocketd.serve_forever)

        httpd = ThreadingHTTPServer(("", 8000), SimpleHTTPRequestHandler)
        httpd_thread = Thread(target=httpd.serve_forever)

        try:
            websocketd_thread.start()
            httpd_thread.start()

            while True:
                # Read from the StreamingOutput and broadcast via WebSocket
                frame_data = camera.stream_out.read()
                if frame_data:
                    # print("Sending frame of size:", len(frame_data))
                    websocketd.manager.broadcast(frame_data, binary=True)
                else:
                    print("No frame data received")

                if time.time() - startTime > 10 and not recording_complete:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    jpeg_image_path = f"/root/birdcam/streamingServer/stream_picamera_h264/images/snap_{timestamp}.jpg"
                    print("trying to capture frame to file")
                    img = convert_frame_to_image(frame_data)
                    img.save(jpeg_image_path, 'JPEG')
                    recording_complete = True

                # if time.time() - startTime > 10 and not recording_complete and not is_recording:
                #     print("starting to record")
                #     timestamp = timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                #     h264_file_path = f"/root/birdcam/streamingServer/stream_picamera_h264/images/snap_{timestamp}.h264"
                #     video_capture_output.fileoutput = h264_file_path
                #     video_capture_output.start()
                #     is_recording = True
                # elif time.time() - startTime > 20 and is_recording and not recording_complete:
                #     print("stopping recording")
                #     video_capture_output.stop()
                #     recording_complete = True
                #     is_recording = False
                #     jpeg_image_path = f"/root/birdcam/streamingServer/stream_picamera_h264/images/snap_{timestamp}.jpg"
                #     extract_frame_from_video(h264_file_path, jpeg_image_path)

        except KeyboardInterrupt:
            pass
        finally:
            websocketd.shutdown()
            httpd.shutdown()
            camera.picamera.stop()
            camera.picamera.stop_encoder()
            raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass
    finally:
        camera.picamera.stop()
        camera.picamera.stop_encoder()


if __name__ == "__main__":
    stream()
