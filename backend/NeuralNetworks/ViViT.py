from torch import nn
from transformers import VivitImageProcessor, VivitForVideoClassification

class ViViTNet(nn.Module):
    def __init__(self, num_classes=3):
        super(ViViTNet, self).__init__()
        self.model = VivitForVideoClassification.from_pretrained('google/vivit-b-16x2-kinetics400')
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

    def forward(self, x):
        logits = self.model(x).logits
        full_logits = logits.mean(dim=1)
        return full_logits

