from torch import nn
from transformers import VivitForVideoClassification


class ViViTNet(nn.Module):
    def __init__(self, num_classes=3):
        super(ViViTNet, self).__init__()
        self.model = VivitForVideoClassification.from_pretrained('google/vivit-b-16x2-kinetics400')
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)
        self.classifier = self.model.classifier

        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.classifier.parameters():
            param.requires_grad = True

        for layer in self.model.vivit.encoder.layer[-2:]:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values).logits
