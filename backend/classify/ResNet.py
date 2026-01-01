from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


class Net(nn.Module):
    def __init__(self, num_classes=3):
        super(Net, self).__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        batch, frames, color, height, width = x.shape
        x = x.view(batch * frames, color, height, width)
        logits = self.model(x)
        logits = logits.view(batch, frames, -1)
        full_logits = logits.mean(dim=1)
        return full_logits

