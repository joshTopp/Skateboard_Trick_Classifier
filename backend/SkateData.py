import torch
from torchvision import transforms
from torch.utils.data import Dataset

class SkateData(Dataset):
    def __init__(self, clips, labels, transform=None, frame_size=224):
        self.clips = clips
        self.labels = labels
        self.transform = transform


    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = self.clips[idx]
        label = self.labels[idx]

        if self.transform:
            clip = torch.stack([self.transform(frame) for frame in clip])

        return clip, label