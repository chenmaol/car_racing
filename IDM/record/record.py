import os
import time
import cv2
import pandas as pd
from screen import MyScreen
from warnings import filterwarnings

from pynput import keyboard
filterwarnings('ignore')


class DataRecorder:
    def __init__(self):
        self.video_record_fps = 10
        self.myScreen = MyScreen((0, 0, 1280, 800))
        self.root = '../data'
        self.target_keys = ['w', 'a', 's', 'd', 'r']

        # create output dataframe
        self.columns_image = ['frame_name', 'record_time', 'seq']
        self.columns_key = ['key_name', 'start_time', 'end_time']

        self.df_image = pd.read_csv(os.path.join(self.root, 'images.csv')) if os.path.exists(os.path.join(self.root, 'images.csv')) else pd.DataFrame(columns=self.columns_image)
        self.df_key = pd.read_csv(os.path.join(self.root, 'keys.csv')) if os.path.exists(os.path.join(self.root, 'keys.csv')) else pd.DataFrame(columns=self.columns_key)

        self.seq_count = self.df_image['seq'].values[-1] if os.path.exists(os.path.join(self.root, 'images.csv')) else -1
        self.counter = len(os.listdir(os.path.join(self.root, 'images'))) if os.path.exists(os.path.join(self.root, 'images')) else 0

        if not os.path.exists(self.root):
            os.makedirs(self.root)
        if not os.path.exists(os.path.join(self.root, "images")):
            os.makedirs(os.path.join(self.root, "images"))

        self.key_buffer = {}
        for key in self.target_keys:
            self.key_buffer[key] = None
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
                    self.seq_count += 1
                self.record_data = True
            elif key == "p":
                self.record_data = False

            if key in self.target_keys and self.key_buffer[key] is None:
                self.key_buffer[key] = time.time()
        except AttributeError:
            pass

    def on_release(self, key):
        try:
            key = key.char
            if key in self.target_keys and self.key_buffer[key] is not None:
                self.save_key(key, self.key_buffer[key], time.time())
                self.key_buffer[key] = None
        except AttributeError:
            pass

    def fetch_frame(self):
        frame = self.myScreen.get_frame()
        return frame

    def save_image(self, frame):
        if self.record_data:
            self.counter += 1
            file_name = f"{self.counter}.jpg"

            # Set timestamp format
            timestamp = time.time()

            # Create a new row with the frame data, mouse data, and key data
            row_data = {
                "frame_name": file_name,
                "record_time": timestamp,
                "seq": self.seq_count
            }
            print(row_data)

            # Append the new row to the DataFrame and save it to the CSV file
            self.df_image = self.df_image.append(row_data, ignore_index=True)
            self.df_image.to_csv(os.path.join(self.root, "images.csv"), index=False)

            # Save the captured frame as an image
            cv2.imwrite(os.path.join(self.root, "images", file_name), frame)

    def save_key(self, key, start_time, end_time):
        if self.record_data:
            row_data = {
                "key_name": key,
                "start_time": start_time,
                "end_time": end_time
            }
            print(row_data)

            # Append the new row to the DataFrame and save it to the CSV file
            self.df_key = self.df_key.append(row_data, ignore_index=True)
            self.df_key.to_csv(os.path.join(self.root, "keys.csv"), index=False)

    # Start recording
    def record(self):
        while True:
            # Calculate loop_start_time for controlling FPS
            loop_start_time = time.time()

            # Fetch and save the current frame along with the key and mouse data
            frame = self.fetch_frame()
            self.save_image(frame)

            # Control the loop execution based on the specified video_record_fps
            while time.time() < loop_start_time + 1 / self.video_record_fps:
                pass


if __name__ == "__main__":
    data_recorder = DataRecorder()
    data_recorder.record()


