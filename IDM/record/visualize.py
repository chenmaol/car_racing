import cv2
import pandas as pd
from tkinter import Tk, Label, Scale, HORIZONTAL, Frame, Button, Canvas

from PIL import Image, ImageTk
import os
import json
import time
from tqdm import tqdm



# 显示图片的函数
def display_image(if_single_frame=False):
    global current_idx, is_playing, last_frame_time
    if if_single_frame:
        img = images[current_idx]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换颜色空间为RGB
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img
        return
    if not is_playing:
        return

    # 如果时间差小于期望的帧时间，则等待
    frame_duration = interval / fps
    while time.time() < last_frame_time + frame_duration:
        pass

    # 更新图像
    img = images[current_idx]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换颜色空间为RGB
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(img)
    image_label.config(image=img)
    image_label.image = img

    # 更新当前索引和时间
    current_idx = (current_idx + 1) % len(images)
    scale.set(current_idx)
    last_frame_time = time.time()

    # 继续播放下一帧
    root.after(1, display_image)


# 滑动条变化时调用的函数
def on_slide(event):
    global current_idx, is_playing
    last_playing = is_playing
    is_playing = False
    current_idx = int(scale.get())
    if not is_playing:
        display_image(True)
    is_playing = last_playing


# 播放/暂停按钮
def play_pause():
    global is_playing, last_frame_time
    is_playing = not is_playing
    if is_playing:
        # 如果是播放状态，则立即开始播放
        last_frame_time = time.time()
        display_image()
    else:
        # 如果是暂停状态，则停止播放
        pass


if __name__ == '__main__':
    fps = 10
    save_path = "../data"
    image_save_path = os.path.join(save_path, 'images')

    image_size = (128, 128)
    # 读取CSV文件
    df_image = pd.read_csv(os.path.join(save_path, "images.csv"))
    df_key = pd.read_csv(os.path.join(save_path, "keys.csv"))

    show_nums = 1000
    interval = 2

    # 创建主窗口
    root = Tk()
    root.title("Image Viewer")

    # 存储图片和对应的key状态
    images = []

    # show text
    target_keys = ['w', 'a', 's', 'd', 'r']
    last_row = None
    # 读取图片和key状态
    show_nums = min(len(df_image), show_nums)
    for index, row in tqdm(df_image.iterrows(), total=show_nums):
        if index >= show_nums:
            break
        if index % interval != 0:
            continue
        key_pressing_time = {}
        for key in target_keys:
            key_pressing_time[key] = 0.0
        frame_name = row['frame_name'].strip()
        timestamp = row['record_time']

        last_timestamp = timestamp - interval / fps if last_row is None else last_row['record_time']

        filtered_df = df_key[(df_key['start_time'] < timestamp) & (df_key['end_time'] > last_timestamp)]

        for _, key_row in filtered_df.iterrows():
            t_ratio = (min(timestamp, key_row['end_time']) - max(last_timestamp, key_row['start_time'])) / (timestamp - last_timestamp)
            key = key_row['key_name']
            key_pressing_time[key] = t_ratio
        image_path = os.path.join(image_save_path, frame_name)
        image = cv2.imread(image_path)
        image = cv2.resize(image, image_size)

        # draw
        x = 10
        y = 10
        for key in target_keys:
            cv2.putText(image, key, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
            cv2.rectangle(image, (x + 20, y), (x + 120, y + 20), (0, 0, 255), 3)
            cv2.rectangle(image, (x + 20, y), (x + 20 + int(100 * key_pressing_time[key]), y + 20), (0, 0, 255), -1)
            y += 50
        images.append(image)
        last_row = row

    # 画布大小
    canvas_size = image_size

    # 创建画布
    image_label = Label(root, width=canvas_size[0], height=canvas_size[1])
    image_label.pack()
    # 创建滑动条
    scale = Scale(root, from_=0, to=(show_nums - 1) // interval, orient=HORIZONTAL, command=on_slide)
    scale.pack()
    # 创建播放/暂停按钮
    play_pause_button = Button(root, text="Play", command=play_pause)
    play_pause_button.pack()

    # 初始化图像显示
    is_playing = False  # 播放状态
    last_frame_time = 0
    current_idx = 0
    display_image(True)

    # 运行Tkinter事件循环
    root.mainloop()