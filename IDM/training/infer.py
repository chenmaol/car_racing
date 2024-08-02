import argparse
import cv2
import os
import torch
import pandas as pd
import numpy as np
from model import Model
from collections import OrderedDict


def save_images_to_disk(images, output_dir, video_id, start_idx):
    image_paths = []
    for idx, image in enumerate(images):
        image_path = os.path.join(output_dir, f"{video_id}_{start_idx + idx}.png")
        cv2.imwrite(image_path, image)
        image_paths.append(image_path)
    return image_paths


def process_video(video_path, model, time_interval, sequence_length, output_dir):
    if not os.path.exists(video_path):
        print("Error: File not found.")
        exit(0)
    else:
        print("File found.")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit(0)
    else:
        print("Video file opened successfully.")
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)

    # sample frame by time_interval seconds
    frame_interval = int(fps * time_interval)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    # sampled_frames = frames[::frame_interval]  # 每隔 frame_interval 取一个帧
    sampled_frames = frames[:64]

    key_num = len(args.keys)

    predicted_labels = np.zeros((len(sampled_frames), key_num), dtype=bool)
    video_id = video_path.split('/')[-1].split('.')[0]  # 假设video_id是文件名

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    cur_label_loc = 0  # 现在predicted_labels标记到哪里了
    with (torch.no_grad()):
        for i in range(0, len(sampled_frames) - sequence_length + 1, sequence_length // 2):
            sequence = torch.tensor(np.array(sampled_frames[i:i + sequence_length])).unsqueeze(0).to(device)  # [b, t, h, w, c]
            predictions = model(sequence).cpu().numpy()  # [b, t, 4, 2]

            for j in range(key_num):
                if i == 0:
                    predicted_labels[cur_label_loc:cur_label_loc + sequence_length // 4, j] = np.argmax(
                        predictions[0, 0:sequence_length // 4, j], axis=1)
                    cur_label_loc += sequence_length // 4

                predicted_labels[cur_label_loc:cur_label_loc + sequence_length // 2, j] = np.argmax(
                    predictions[0, sequence_length // 4: sequence_length * 3 // 4, j],
                    axis=1)
                cur_label_loc = cur_label_loc + sequence_length // 2

                if i + sequence_length // 2 > len(sampled_frames) - sequence_length:
                    predicted_labels[cur_label_loc:cur_label_loc + sequence_length // 4, j] = np.argmax(
                        predictions[0, sequence_length * 3 //4:-1, j], axis=1)
                    cur_label_loc = cur_label_loc + sequence_length // 4
                    assert cur_label_loc == len(sampled_frames)
    # 保存图像到本地路径
    image_paths = save_images_to_disk(sampled_frames, output_dir, video_id, 0)
    predicted_labels = [labels for labels in predicted_labels]  # list, 里面每个元素都是一个含有四个元素的一维数组
    return image_paths, predicted_labels, video_id


def save_to_csv(image_paths, labels, video_ids, output_csv):
    assert len(labels) == len(image_paths) == len(video_ids)
    labels = np.array(labels)

    data = {
        'image_paths': image_paths,
        'video_id': video_ids,
    }

    # 添加labels到字典中
    for i, key in enumerate(args.keys):
        data[f'label_{key}'] = labels[i]

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)


def main(args):
    all_image_paths = []
    all_labels = []
    all_video_ids = []

    # 加载模型和预训练权重
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(config_file=args.model_config, batch_size=args.batch_size)
    # 加载模型权重
    state_dict = torch.load(args.weights, map_location=device)
    # model.load_state_dict(torch.load(args.weights, map_location=device))

    # 去掉 "module." 前缀
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # 去掉 "module." 前缀
        else:
            new_state_dict[k] = v

    # 加载修改后的 state_dict
    model.load_state_dict(new_state_dict)

    for video in args.input_videos:
        image_paths, labels, video_id = process_video(video, model, args.time_interval, args.sequence_length,
                                                      args.output_dir)
        all_image_paths.extend(image_paths)
        all_labels.extend(labels)
        all_video_ids.extend([video_id] * len(image_paths))

    save_to_csv(all_image_paths, all_labels, all_video_ids, args.output_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IDM inference script")
    parser.add_argument('--input_videos', type=str, nargs='+', default=['/vhome/liquanhao/workspace/car_racing/IDM/training/infer.mp4'], help='List of input videos')
    parser.add_argument('--time_interval', type=float, default=0.1, help='Time interval between frames in seconds')
    parser.add_argument('--sequence_length', type=int, default=64, help='the sequence_length to be processed')
    parser.add_argument('--output_csv', type=str, default='IDM_output.csv', help='Output CSV file to store results')
    parser.add_argument('--model_config', type=str, default='model_config.yaml', help='model config file')
    parser.add_argument('--batch_size', type=int, default=4, help='input batch size for training')
    parser.add_argument('--weights', default='weights/best_model_0.pth', type=str,
                        help='Path to the pretrained weights file')
    parser.add_argument('--output_dir', type=str, default='./IDM_output_images', help='Directory to save output images')
    parser.add_argument('--keys', type=str, default=['w', 's', 'a', 'd'], nargs='+', help='keys predicted')
    args = parser.parse_args()
    main(args)
