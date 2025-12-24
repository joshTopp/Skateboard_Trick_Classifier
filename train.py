import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Precision, Accuracy, Recall, F1Score
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.models import resnet18

metric_precision = Precision(task="multiclass", num_classes=4, average="macro")
metric_accuracy = Accuracy(task="multiclass", num_classes=4, average="macro")
metric_recall = Recall(task="multiclass", num_classes=4, average="macro")
metric_f1 = F1Score(task="multiclass", num_classes=4, average="macro")


model = resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 3)

optimizer = optim.Adam(model.parameters(), lr=0.001)

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomAutocontrast(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
    model.train()
    epoch_loss = 0
    for image, labels in dataloader_train:
        optimizer.zero_grad()
        outputs = model(image)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    epoch_loss += loss.item()
    epoch_loss /= len(dataloader_train)
    print(f"Training loss: {epoch_loss}")

model.eval()
metric_precision.reset()
metric_accuracy.reset()
metric_recall.reset()
metric_f1.reset()

with torch.no_grad():
    for image, labels in dataloader_test:
        outputs = model(image)
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
