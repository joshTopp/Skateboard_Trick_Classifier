import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Precision, Accuracy, Recall, F1Score
import torch.optim as optim
from torchvision import transforms
from backend.NeuralNetworks import neuralnet as net

from backend.SkateData import SkateData

metric_precision = Precision(task="multiclass", num_classes=3, average="macro")
metric_accuracy = Accuracy(task="multiclass", num_classes=3, average="macro")
metric_recall = Recall(task="multiclass", num_classes=3, average="macro")
metric_f1 = F1Score(task="multiclass", num_classes=3, average="macro")


class Train:
    def __init__(self, list_clips, list_labels):
        self.list_clips = list_clips
        self.list_labels = list_labels
        self.model = net.Net()



    def training(self, epochs=5):
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        train_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAutocontrast(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dataset_train = SkateData(self.list_clips, self.list_labels, transform=train_transforms)
        dataloader_train = DataLoader(dataset_train, batch_size=2, shuffle=True)

        image, labels = next(iter(dataloader_train))
        plt.imshow(image[0, 0].permute(1, 2, 0))
        plt.show()

        criterion = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            for image, labels in dataloader_train:
                optimizer.zero_grad()
                outputs = self.model(image)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(dataloader_train)
            print(f"Training loss: {epoch_loss}")
        torch.save(self.model.state_dict(), "skateboard_model.pt")


    def test(self):
        test_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.model.eval()
        dataset_test = SkateData(self.list_clips, self.list_labels, transform=test_transforms)
        dataloader_test = DataLoader(dataset_test, batch_size=2, shuffle=True)
        with torch.no_grad():
            for image, labels in dataloader_test:
                outputs = self.model(image)
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
        print(f"F1Score: {test_f1 * 100}%")


