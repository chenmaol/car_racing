import argparse
import cv2
import os
import torch
import pandas as pd
import numpy as np
from model import Model
from collections import OrderedDict
import torchvision.transforms as transforms
from PIL import Image


def save_images_to_disk(images, output_dir, video_id):
    image_paths = []
    for idx, image in enumerate(images):
        image_path = os.path.join(output_dir, f"{video_id}_{idx}.jpg")
        cv2.imwrite(image_path, image)
        image_paths.append(image_path)
    return image_paths


def process_video(video_path, model, time_interval, sequence_length, output_dir, transform):
    if not os.path.exists(video_path):
        print(f"Error: File not found:{video_path}")
        # exit(0)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file:{video_path}")
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)

    # sample frame by time_interval seconds
    frame_interval = round(fps * time_interval)
    print(fps, frame_interval)
    cnt = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if cnt % frame_interval == 0:  # avoid store all image
            frames.append(frame)
        cnt += 1
    cap.release()
    frames_processed = []
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = transform(frame)
        frames_processed.append(frame)

    key_num = len(args.keys)

    predicted_labels = -np.ones((len(frames_processed), key_num), dtype=int)
    video_id = os.path.basename(video_path).split('.')[0]  # 假设video_id是文件名

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        for i in range(0, len(frames_processed) - sequence_length, sequence_length // 2):
            sequence = torch.stack(frames_processed[i:i + sequence_length], dim=0).permute(0, 2, 3, 1).unsqueeze(0).to(device)  # [1, t, h, w, c]
            probs = model(sequence).cpu().squeeze().numpy()  # [t, 4, 2]
            predictions = np.argmax(probs, axis=-1)     # [t, 4]
            if i == 0:
                predicted_labels[i:i + sequence_length // 4, :] = predictions[:sequence_length // 4, :]
            predicted_labels[i + sequence_length // 4:i + sequence_length * 3 // 4, :] = predictions[sequence_length // 4:sequence_length * 3 // 4, :]

        # if len(frames_processed) % sequence_length != 0:
        sequence = torch.stack(frames_processed[-sequence_length:], dim=0).permute(0, 2, 3, 1).unsqueeze(0).to(device)
        probs = model(sequence).cpu().squeeze().numpy()  # [t, 4, 2]
        predictions = np.argmax(probs, axis=-1)  # [t, 4]
        predicted_labels[-3 * sequence_length // 4:, :] = predictions[-3 * sequence_length // 4:, :]

    # 保存图像到本地路径
    image_paths = save_images_to_disk(frames, output_dir, video_id)
    predicted_labels = [labels for labels in predicted_labels]  # list, 里面每个元素都是一个含有四个元素的一维数组
    return image_paths, predicted_labels, video_id


def save_to_csv(image_paths, labels, video_ids, output_csv):
    assert len(labels) == len(image_paths) == len(video_ids)
    labels = np.array(labels)

    data = {
        'frame_name': image_paths,
        'seq': video_ids,
    }

    # 添加labels到字典中
    for i, key in enumerate(args.keys):
        data[f'{key}'] = labels[:, i]

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)


def main(args):
    all_image_paths = []
    all_labels = []
    all_video_ids = []

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    # 加载模型和预训练权重
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(config_file=args.model_config, batch_size=args.batch_size)
    model.to(device)
    model.eval()

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

    video_root = args.source_dir

    base_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])

    for video in os.listdir(video_root):
        print(video)
        image_paths, labels, video_id = process_video(os.path.join(video_root, video), model, args.time_interval, args.sequence_length,
                                                      args.output_dir, base_transform)
        all_image_paths.extend(image_paths)
        all_labels.extend(labels)
        all_video_ids.extend([video_id] * len(image_paths))

    save_to_csv(all_image_paths, all_labels, all_video_ids, args.output_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IDM inference script")
    parser.add_argument('--source_dir', type=str, default='../../FM/data/videos', help='List of input videos')
    parser.add_argument('--time_interval', type=float, default=0.1, help='Time interval between frames in seconds')
    parser.add_argument('--sequence_length', type=int, default=64, help='the sequence_length to be processed')
    parser.add_argument('--img_size', type=int, default=128, help='img size for input')
    parser.add_argument('--output_csv', type=str, default='../../FM/data/IDM_output.csv', help='Output CSV file to store results')
    parser.add_argument('--model_config', type=str, default='model_config.yaml', help='model config file')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size for training')
    parser.add_argument('--weights', default='../weights/best_model_2.pth', type=str,
                        help='Path to the pretrained weights file')
    parser.add_argument('--output_dir', type=str, default='../../FM/data/images', help='Directory to save output images')
    parser.add_argument('--keys', type=str, default=['w', 's', 'a', 'd'], nargs='+', help='keys predicted')
    args = parser.parse_args()
    main(args)
