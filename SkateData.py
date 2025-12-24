import torch
from torch.utils.data import Dataset

class SkateData(Dataset):
    def __init__(self, clips, labels, transform=None):
        self.clips = clips
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = self.clips[idx]
        if self.transform:
            clip = torch.stack([self.transform(frame) for frame in clip])
        label = self.labels[idx]

        return clip, label