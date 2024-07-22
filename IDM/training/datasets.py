from torch.utils.data import Dataset, DataLoader
import argparse


class GameplayDataset(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, index):
        images = ...
        labels = ...
        return images, labels

    def __len__(self) -> int:
        return ...


def get_loaders(args):
    # TODO: ...
    train_dataset = GameplayDataset()
    val_dataset = GameplayDataset()

    train_dataloader = DataLoader(train_dataset, )
    val_dataloader = DataLoader(val_dataset, )
    return train_dataloader, val_dataloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    args = parser.parse_args()
    train_dataset, val_dataset = get_loaders(args)



