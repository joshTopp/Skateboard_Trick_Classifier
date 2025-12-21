
#use this instead of creating a CNN by myself would need tons of data
#ImageNet/Kinetics


import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Precision, Accuracy, Recall, F1Score
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision import transforms

metric_precision = Precision(task="multiclass", num_classes=4, average=None)
metric_accuracy = Accuracy(task="multiclass", num_classes=4, average=None)
metric_recall = Recall(task="multiclass", num_classes=4, average=None)
metric_f1 = F1Score(task="multiclass", num_classes=4, average=None)


class Net(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(32, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ELU(),
            nn.Conv2d(256, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ELU(),
            nn.Conv2d(128, num_classes, kernel_size=(2,2), stride=(1,1), padding=(1,1)),
            nn.Flatten()
        )


net = Net(num_classes=4)

optimizer = optim.Adam(net.parameters(), lr=0.001)

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomAutocontrast(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset_train = ImageFolder("tricks_train", transform=train_transforms)

dataloader_train = DataLoader(dataset_train, batch_size=32)

dataloader_test = DataLoader(dataset_train, batch_size=32)

image, labels = next(iter(dataloader_train))
image = image.squeeze(0).permute(1, 2, 0)
plt.imshow(image)
plt.show()

criterion = nn.CrossEntropyLoss()
for epoch in range(5):
    net.train()
    epoch_loss = 0
    for image, labels in dataloader_train:
        outputs = net(image)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss = loss.item()
    epoch_loss /= running_loss / len(dataloader_train)
    print(f"Training loss: {epoch_loss}")

net.eval()
metric_precision.reset()
metric_accuracy.reset()
metric_recall.reset()
metric_f1.reset()

with torch.no_grad():
    for image, labels in dataloader_test:
        outputs = net(image)
        _, predicted = torch.max(outputs, 1)
        metric_precision.update(predicted, labels)
        metric_accuracy.update(predicted, labels)
        metric_recall.update(predicted, labels)
        metric_f1.update(predicted, labels)

test_accuracy = metric_accuracy.compute()
test_recall = metric_recall.compute()
test_precision = metric_precision.compute()
test_f1 = metric_f1.compute()

print(f"Precision: {test_precision * 100}%")
print(f"Accuracy: {test_accuracy * 100}%")
print(f"Recall: {test_recall * 100}%")
print(f"F1Score: {test_recall * 100}%")
