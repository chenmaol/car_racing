import tkinter

import cv2
import pandas as pd
from tkinter import Tk, Label, Scale, HORIZONTAL, Frame, Button, Canvas

from PIL import Image, ImageTk
import os
import json
import time
from tqdm import tqdm

from moviepy.editor import VideoFileClip
from tkinter import filedialog

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
    frame_duration = 1 / target_fps / 5
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


def mark_start():
    global is_mark_start, current_time_seq
    if is_mark_start:
        print('已经标记了start，请标记end后再点击')
    elif is_playing:
        print('请先暂停播放再标记')
    else:
        is_mark_start = True
        current_time_seq.append(current_idx)


def mark_end():
    global is_mark_start, current_time_seq, time_seqs
    if not is_mark_start:
        print('请标记start后再点击')
    elif is_playing:
        print('请先暂停播放再标记')
    else:
        is_mark_start = False
        current_time_seq.append(current_idx)
        time_seqs.append(current_time_seq.copy())
        current_time_seq = []
        # print(time_seqs)
        update_rectangles()


def merge_intervals(intervals):
    # 如果列表为空或只有一个区间，则无需合并
    if not intervals:
        return []

    # 按照每个区间的开始时间对列表进行排序
    intervals.sort(key=lambda x: x[0])

    # 初始化合并后的区间列表
    merged = [intervals[0]]

    for current_start, current_end in intervals[1:]:
        last_end = merged[-1][1]  # 获取最后一个合并区间的结束时间

        # 如果当前区间的开始时间小于或等于最后一个合并区间的结束时间，则合并区间
        if current_start <= last_end:
            merged[-1] = (merged[-1][0], max(last_end, current_end))
        else:
            # 否则，添加新的区间到合并后的列表
            merged.append((current_start, current_end))

    return merged


def update_rectangles():
    global canvas, time_seqs
    time_seqs = merge_intervals(time_seqs)
    print(time_seqs)
    canvas.delete("all")  # 清除画布上的所有内容
    width = canvas.winfo_width()
    height = canvas.winfo_height()
    # canvas.create_rectangle(0, 0, width, height, fill='green', outline='')
    # canvas.update()
    for start, end in time_seqs:
        start = max(0, min(1, start / frame_nums * interval))
        end = max(0, min(1, end / frame_nums * interval))
        # 计算矩形的起始和结束坐标
        x_start = start * width
        x_end = end * width
        # 绘制红色矩形
        canvas.create_rectangle(x_start, 0, x_end, height, fill='red', outline='')


def split_video_outside_intervals():
    global video_path, images

    del images
    # 加载原视频
    clip = VideoFileClip(video_path)

    # 按照结束时间对区间进行排序，以便从视频的开始到结束进行分割
    time_seqs.sort(key=lambda x: x[1])

    # 存储未在区间内的视频片段
    parts_to_keep = []

    # 初始化起始点
    start = 0.0

    for begin, end in time_seqs:
        # 如果区间开始之前有未覆盖的部分，添加一个片段
        if begin > start:
            parts_to_keep.append(clip.subclip(start / target_fps, begin / target_fps))
        # 更新起始点为区间的结束
        start = end

    # # 检查最后一个区间结束后是否有剩余部分
    # if start / target_fps < clip.duration:
    #     parts_to_keep.append(clip.subclip(start / target_fps))

    # 将所有非区间片段合并为一个视频
    for idx, sub_clip in enumerate(parts_to_keep):
        if os.path.exists(os.path.join(tgr_root, os.path.basename(video_path) + f'_{idx}.mp4')):
            continue
        # 写入文件
        output_path = os.path.join(tgr_root, os.path.basename(video_path) + f'_{idx}.mp4')
        sub_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

        # 释放资源
        sub_clip.close()
    clip.close()


if __name__ == '__main__':
    target_fps = 10
    src_root = "../data/raw_youtube_videos"
    tgr_root = "../data/processed_youtube_videos"
    videos = [os.path.join(src_root, x) for x in os.listdir(src_root)]
    image_size = (600, 400)

    # 创建主窗口
    root = Tk()
    root.title("Video Viewer")
    # root.withdraw()
    # 存储图片
    images = []

    file_types = (
        ("Video files", "*.mp4;*.avi"),
        ("All files", "*.*")
    )

    # 读取video
    videos = [os.path.join(src_root, x) for x in os.listdir(src_root)]
    for video in videos:
        if os.path.exists(video.replace(src_root, tgr_root) + '_0.mp4'):
            print(f'exist: {video}')
        else:
            print(video)
            break
    # video_path = filedialog.askopenfilename(initialdir=src_root, filetypes=file_types)
    video_path = video
    cap = cv2.VideoCapture(video_path)
    cnt = 0
    fps = cap.get(5)
    interval = round(fps / target_fps)
    frame_nums = int(cap.get(7))
    print(f"interval:{interval}, fps:{fps}, frame_nums:{frame_nums}")
    print(f'loading: {video_path}')

    if frame_nums > 150000:
        sub_num = frame_nums // 100000
        seq_len = frame_nums / sub_num
        time_seqs = []
        for i in range(1, sub_num):
            time_seqs.append([i * seq_len, i * seq_len + 1])
        time_seqs.append([frame_nums - 2, frame_nums - 1])
        tgr_root = src_root
        target_fps = fps
        split_video_outside_intervals()
        print('split done')
        exit(0)

    pbar = tqdm(total=frame_nums)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if cnt % interval == 0:
            pbar.update(interval)
            images.append(cv2.resize(frame, image_size))
        cnt += 1

    del pbar

    # 画布大小
    canvas_size = image_size

    # 创建画布
    image_label = Label(root, width=canvas_size[0], height=canvas_size[1])
    image_label.pack()

    # 创建矩形进度条
    canvas = tkinter.Canvas(root, width=600, height=20, bg="green")
    canvas.pack()
    # 创建滑动条
    scale = Scale(root, from_=0, to=len(images) - 1, orient=HORIZONTAL, command=on_slide, length=600)
    scale.pack()
    # 创建播放/暂停按钮
    play_pause_button = Button(root, text="Play", command=play_pause)
    play_pause_button.pack()
    # 创建标记按钮
    mark_start_button = Button(root, text="Mark Start Time", command=mark_start)
    mark_end_button = Button(root, text="Mark End Time", command=mark_end)
    mark_start_button.pack()
    mark_end_button.pack()
    # 创建保存按钮
    save_button = Button(root, text="Save", command=split_video_outside_intervals)
    save_button.pack()
    # 初始化图像显示
    is_playing = False  # 播放状态
    is_mark_start = False  # mark
    current_time_seq = []
    time_seqs = []
    last_frame_time = 0
    current_idx = 0
    display_image(True)

    # 运行Tkinter事件循环
    root.mainloop()