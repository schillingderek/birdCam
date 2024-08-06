#!/usr/bin/env python3

import io
import picamera2
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput, CircularOutput
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from wsgiref.simple_server import make_server
from ws4py.websocket import WebSocket
from ws4py.server.wsgirefserver import WSGIServer, WebSocketWSGIHandler, WebSocketWSGIRequestHandler
from ws4py.server.wsgiutils import WebSocketWSGIApplication
from threading import Thread, Condition

videoCaptureEncoder = H264Encoder()
videoCaptureOutput = CircularOutput()

width = 1280
height = 720

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
        self.camera = picamera2.Picamera2()
        self.video_config = self.camera.create_video_configuration({'size': (width, height)})
        self.camera.configure(self.video_config)
        self.streamingEncoder = H264Encoder(bitrate=2500000, profile='baseline')
        self.streamingEncoder.bitrate = 5000000
        self.streamOut = StreamingOutput()
        self.streamOut2 = FileOutput(self.streamOut)
        self.streamingEncoder.output = [self.streamOut2]

        self.camera.start_encoder(self.streamingEncoder)
        self.camera.start_recording(videoCaptureEncoder, videoCaptureOutput)

camera = Camera()

def stream():
    camera.camera.start()

    try:
        WebSocketWSGIHandler.http_version = '1.1'
        websocketd = make_server('', 9000, server_class=WSGIServer,
                 handler_class=WebSocketWSGIRequestHandler,
                 app=WebSocketWSGIApplication(handler_cls=WebSocket))
        websocketd.initialize_websockets_manager()
        websocketd_thread = Thread(target=websocketd.serve_forever)

        httpd = ThreadingHTTPServer(('', 8000), SimpleHTTPRequestHandler)
        httpd_thread = Thread(target=httpd.serve_forever)

        try:
            websocketd_thread.start()
            httpd_thread.start()

            while True:
                # Read from the StreamingOutput and broadcast via WebSocket
                frame_data = camera.streamOut.read()
                if frame_data:
                    #print("Sending frame of size:", len(frame_data))
                    websocketd.manager.broadcast(frame_data, binary=True)
                else:
                    print("No frame data received")
        except KeyboardInterrupt:
            pass
        finally:
            websocketd.shutdown()
            httpd.shutdown()
            camera.camera.stop()
            camera.camera.stop_encoder()
            raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass
    finally:
        camera.camera.stop()
        camera.camera.stop_encoder()

if __name__ == "__main__":
    stream()

