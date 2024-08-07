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
from sklearn.model_selection import train_test_split


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
    def __init__(self, df, data_dir, sample_idx, img_size=128, sequence_length=10, pred_length=4, gap_length=0, keys=None, transform=None):
        if keys is None:
            self.keys = ['w', 's', 'a', 'd']
        else:
            self.keys = keys
        self.df = df
        self.data_dir = data_dir
        self.sample_idx = sample_idx
        self.sequence_length = sequence_length
        self.pred_length = pred_length
        self.gap_length = gap_length

        self.base_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        self.augment_transform = transform

    def __getitem__(self, index):
        df_index = self.sample_idx[index]

        rows = self.df.iloc[df_index:df_index+self.sequence_length]
        pred_rows = self.df.iloc[df_index+self.sequence_length+self.gap_length:df_index+self.sequence_length+self.gap_length+self.pred_length]
        images = [Image.open(os.path.join(self.data_dir, img)).convert('RGB') for img in rows['frame_name']]

        images = [self.base_transform(image) for image in images]

        if self.augment_transform:
            images = self.augment_transform(images)
        images = torch.stack(images).permute(0, 2, 3, 1)  # [t, h, w, c]

        labels = pred_rows[self.keys].values.astype(np.float32)
        labels = torch.tensor(np.array(labels), dtype=torch.float32)

        return images, labels

    def __len__(self):
        return len(self.sample_idx)


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


def filter_csv(df, seq_length, target_length, gap_length, stride=1):
    for index, row in df.iterrows():
        if index + seq_length + target_length + gap_length < len(df) and index % stride == 0 and row['seq'] == df.loc[index + seq_length + target_length + gap_length, 'seq']:
            df.loc[index, 'valid'] = True
        else:
            df.loc[index, 'valid'] = False

    filtered_df = df[df['valid'] == True]
    del filtered_df['valid']
    return filtered_df.index


def get_loaders(args):
    data_dir = args.data_dir
    csv_file = os.path.join(data_dir, args.label_file)
    img_size = args.img_size
    sequence_length = args.sequence_length
    pred_seq_length = args.pred_seq_length
    pred_gap_length = args.pred_gap_length
    data_stride = args.data_stride

    # filter valid index for training
    df = pd.read_csv(csv_file)
    df['frame_name'] = df['frame_name'].apply(lambda x: x.replace('\\', '/'))
    filter_idx = filter_csv(df, sequence_length, pred_seq_length, pred_gap_length, stride=data_stride)
    train_idx, val_idx = train_test_split(filter_idx, test_size=0.2)
    if int(os.environ["LOCAL_RANK"]) == 0:
        print(f'train data samples:{len(train_idx)}')
        print(f'val data samples:{len(val_idx)}')
        print(f"input shape:[{args.batch_size},{sequence_length},{img_size},{img_size},3]")
        print(f"output shape:[{args.batch_size},{pred_seq_length},4,2]")

    train_transform = ConsistentTransform(transforms.Compose([
            transforms.ColorJitter(hue=0.2, saturation=0.4, brightness=0.4, contrast=0.4),
            transforms.RandomAffine(degrees=2, scale=(0.98, 1.02), shear=2, translate=(0.02, 0.02))
        ]))

    train_dataset = GameplayDataset(df,
                                    data_dir,
                                    train_idx,
                                    transform=train_transform,
                                    img_size=img_size,
                                    sequence_length=sequence_length,
                                    pred_length=pred_seq_length,
                                    gap_length=pred_gap_length,
                                    keys=args.keys)
    val_dataset = GameplayDataset(df,
                                  data_dir,
                                  val_idx,
                                  img_size=img_size,
                                  sequence_length=sequence_length,
                                  pred_length=pred_seq_length,
                                  gap_length=pred_gap_length,
                                  keys=args.keys)

    # class_column_names = args.keys
    # class_weights = get_class_weights(dataset.df, class_column_names)
    # sample_weights = get_sample_weights(dataset.df, class_column_names, class_weights)
    #
    # sample_weights = sample_weights / sample_weights.sum()
    
    # Create a weighted sampler based on the number of sequences, not the number of samples
    # train_sampler = WeightedRandomSampler(weights=sample_weights[:len(dataset)], num_samples=len(dataset), replacement=True)
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    # Use DistributedSampler for validation
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    return train_loader, val_loader, train_sampler, val_sampler


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
    # size = 2
    # processes = []
    # mp.set_start_method('spawn')  # Use 'spawn' method for multiprocessing
    #
    # for rank in range(size):
    #     p = mp.Process(target=init_process, args=(rank, size, train))
    #     p.start()
    #     processes.append(p)
    #
    # for p in processes:
    #     p.join()
    sequence_length = 32
    pred_length = 4
    df = pd.read_csv('../data/labels_interval-1_dirty-5.0.csv')
    df['frame_name'] = df['frame_name'].apply(lambda x: x.replace('\\', '/'))
    filter_idx = filter_csv(df, sequence_length, pred_length, stride=pred_length)
    train_idx, val_idx = train_test_split(filter_idx, test_size=0.2)
    # dataset = GameplayDataset(df, '../data', train_idx, sequence_length=sequence_length)
    train_transform = ConsistentTransform(transforms.Compose([
        transforms.ColorJitter(hue=0.2, saturation=0.4, brightness=0.4, contrast=0.4),
        transforms.RandomAffine(degrees=2, scale=(0.98, 1.02), shear=2, translate=(0.02, 0.02))
    ]))
    dataset = GameplayDataset(df,
                              '../data',
                              train_idx,
                              transform=train_transform,
                              sequence_length=sequence_length,
                              pred_length=pred_length)

    dataloader = DataLoader(dataset, batch_size=8)

    for images, labels in dataloader:
        print(images.shape)
        print(labels.shape)
        a = 1


