from torch import nn
from transformers import VivitImageProcessor, VivitForVideoClassification

class ViViTNet(nn.Module):
    def __init__(self, num_classes=3):
        super(ViViTNet, self).__init__()
        self.model = VivitForVideoClassification.from_pretrained('google/vivit-b-16x2-kinetics400')
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        batch, frames, color, height, width = x.shape
        x = x.view(batch * frames, color, height, width)
        logits = self.model(x)
        logits = logits.view(batch, frames, -1)
        full_logits = logits.mean(dim=1)
        return full_logits

