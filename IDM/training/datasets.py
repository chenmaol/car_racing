import argparse


# class GameplayDataset(Dataset):
#     def __init__(self):
#         pass
#
#     def __getitem__(self, index):
#         images = ...
#         labels = ...
#         return images, labels
#
#     def __len__(self) -> int:
#         return ...


# def get_loaders(args):
#     # TODO: ...
#     train_dataset = GameplayDataset()
#     val_dataset = GameplayDataset()
#
#     train_dataloader = DataLoader(train_dataset, )
#     val_dataloader = DataLoader(val_dataset, )
#     return train_dataloader, val_dataloader

from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

def get_loaders(args):
    # 数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 下载并加载训练数据
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)

    # 下载并加载验证数据
    val_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    val_sampler = DistributedSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler)

    return train_dataloader, val_dataloader, train_sampler, val_sampler



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    args = parser.parse_args()
    train_dataset, val_dataset = get_loaders(args)



