import os
import time
import cv2
from screen import MyScreen
from warnings import filterwarnings

from pynput import keyboard
filterwarnings('ignore')


class DataRecorder:
    def __init__(self):
        self.video_record_fps = 20
        self.height = 800
        self.width = 1280
        self.myScreen = MyScreen((0, 0, self.width, self.height))
        self.root = '../data'

        self.counter = len(os.listdir(os.path.join(self.root, 'videos'))) if os.path.exists(os.path.join(self.root, 'videos')) else 0

        if not os.path.exists(self.root):
            os.makedirs(self.root)
        if not os.path.exists(os.path.join(self.root, "videos")):
            os.makedirs(os.path.join(self.root, "videos"))

        self.writer = None
        self.record_data = False

        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release)
        self.listener.start()

    def on_press(self, key):
        try:
            key = key.char
            if key == "b":
                if self.record_data is False:
                    self.writer = cv2.VideoWriter(os.path.join(self.root, "videos", '{}.mp4'.format(self.counter)), cv2.VideoWriter_fourcc(*'avc1'), self.video_record_fps, (self.width, self.height), True)
                    self.counter += 1
                self.record_data = True
            elif key == "p":
                if self.writer:
                    self.writer.release()
                    self.writer = None
                self.record_data = False
        except AttributeError:
            pass

    def on_release(self, key):
        pass

    def fetch_frame(self):
        frame = self.myScreen.get_frame()
        return frame

    # Start recording
    def record(self):
        while True:
            # Calculate loop_start_time for controlling FPS
            loop_start_time = time.time()

            # Fetch and save the current frame along with the key and mouse data
            frame = self.fetch_frame()
            if self.writer:
                self.writer.write(frame)
                print(loop_start_time)

            # Control the loop execution based on the specified video_record_fps
            while time.time() < loop_start_time + 1 / self.video_record_fps:
                pass


if __name__ == "__main__":
    data_recorder = DataRecorder()
    data_recorder.record()


