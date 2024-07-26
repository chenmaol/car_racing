import os
import pandas as pd
from PIL import Image
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, DistributedSampler, WeightedRandomSampler
import torchvision.transforms as transforms
import torch.multiprocessing as mp
import torch.distributed as dist
from typing import List

class ConsistentTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, images):
        seed = random.randint(0, 2**32)
        transformed_images = []
        for image in images:
            random.seed(seed)
            transformed_images.append(self.transform(image))
        return transformed_images

class GameplayDataset(Dataset):
    """
    Dataset class for loading gameplay frames and their associated labels.

    Attributes:
        csv_file (str): Path to the CSV file containing frame paths and labels.
        data_dir (str): Directory where the frame images are stored.
        sequence_length (int): Number of frames in each sequence.
        base_transform (torchvision.transforms.Compose): Transformations applied to each frame.
        augment_transform (ConsistentTransform): Augmentation transformations applied consistently to all frames in a sequence.
    """
    def __init__(self, csv_file, data_dir, sequence_length=10):
        self.data_dir = data_dir
        self.df = pd.read_csv(csv_file)
        self.df['frame_name'] = self.df['frame_name'].apply(lambda x: x.replace('\\', '/'))
        self.sequence_length = sequence_length

        self.seq_groups = self.df.groupby('seq').filter(lambda x: len(x) >= sequence_length).groupby('seq')
        self.seq_indices = list(self.seq_groups.groups.keys())

        self.base_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

        self.augment_transform = ConsistentTransform(transforms.Compose([
            transforms.ColorJitter(hue=0.2, saturation=0.4, brightness=0.4, contrast=0.4),
            transforms.RandomAffine(degrees=2, scale=(0.98, 1.02), shear=2, translate=(0.02, 0.02))
        ]))

    def __getitem__(self, index):
        if index >= len(self.seq_indices):
            raise IndexError(f"Index {index} is out of range for dataset with length {len(self.seq_indices)}")
        seq_id = self.seq_indices[index]
        frames_df = self.seq_groups.get_group(seq_id).reset_index(drop=True)
        start_idx = random.randint(0, len(frames_df) - self.sequence_length)

        images, labels = [], []
        for i in range(start_idx, start_idx + self.sequence_length):
            row = frames_df.iloc[i]
            image_path = os.path.join(self.data_dir, row['frame_name'])
            image = Image.open(image_path).convert('RGB')
            images.append(image)
            labels.append(row[['w', 's', 'a', 'd']].values.astype(np.float32))

        images = [self.base_transform(image) for image in images]
        images = self.augment_transform(images)
        images = torch.stack(images).permute(0, 2, 3, 1)  # [t, h, w, c]

        labels = torch.tensor(np.array(labels), dtype=torch.float32)  # Convert list of numpy arrays to a single numpy array before tensor

        return images, labels

    def __len__(self):
        return len(self.seq_indices)

class SequenceSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        self.seq_indices = data_source.seq_indices

    def __iter__(self):
        return iter(self.seq_indices)

    def __len__(self):
        return len(self.seq_indices)

def get_class_weights(df, class_column_names):
    class_counts = np.zeros(len(class_column_names))
    for i, col in enumerate(class_column_names):
        class_counts[i] = df[col].sum()

    class_weights = 1.0 / class_counts
    return class_weights

def get_sample_weights(df, class_column_names, class_weights):
    sample_weights = np.zeros(len(df))
    for i, col in enumerate(class_column_names):
        sample_weights += df[col].values * class_weights[i]
    return sample_weights

def get_loaders(args):
    data_dir = args.data_dir
    csv_file = os.path.join(data_dir, 'labels_interval-1_dirty-5.0.csv')

    dataset = GameplayDataset(csv_file, data_dir, sequence_length=args.sequence_length)

    class_column_names = ['w', 's', 'a', 'd']
    class_weights = get_class_weights(dataset.df, class_column_names)
    sample_weights = get_sample_weights(dataset.df, class_column_names, class_weights)

    sample_weights = sample_weights / sample_weights.sum()
    
    # Create a weighted sampler based on the number of sequences, not the number of samples
    weighted_sampler = WeightedRandomSampler(weights=sample_weights[:len(dataset)], num_samples=len(dataset), replacement=True)

    train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=weighted_sampler, num_workers=args.num_workers, pin_memory=True)

    # Use DistributedSampler for validation
    val_sampler = DistributedSampler(dataset, num_replicas=args.world_size, rank=args.rank, shuffle=False)
    val_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=args.num_workers, pin_memory=True)

    return train_loader, val_loader

def init_process(rank, size, fn, backend='gloo'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)
    dist.destroy_process_group()

def train(rank, size):
    args = Args()
    args.rank = rank
    args.world_size = size
    train_loader, val_loader = get_loaders(args)

    for epoch in range(args.epochs):
        # Only set epoch for the DistributedSampler used in validation
        val_loader.sampler.set_epoch(epoch)
        for batch in train_loader:
            images, labels = batch
            # Ensure images are in [b, t, h, w, c] and labels are in [b, t, 4]
            print(f"Rank {rank}, Batch: images shape {images.shape}, labels shape {labels.shape}")

class Args:
    data_dir = '/home/swqa/shuai/data'
    batch_size = 32
    num_workers = 4
    sequence_length = 10
    img_height = 128
    img_width = 128
    epochs = 10

if __name__ == '__main__':
    size = 2
    processes = []
    mp.set_start_method('spawn')  # Use 'spawn' method for multiprocessing

    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, train))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

