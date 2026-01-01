import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Precision, Accuracy, Recall, F1Score, ConfusionMatrix
import torch.optim as optim
from torchvision import transforms
from backend.classify import ResNet as net

from backend.SkateData import SkateData

metric_precision = Precision(task="multiclass", num_classes=3, average=None)
metric_accuracy = Accuracy(task="multiclass", num_classes=3, average=None)
metric_recall = Recall(task="multiclass", num_classes=3, average=None)
metric_f1 = F1Score(task="multiclass", num_classes=3, average=None)
confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=3)

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
        torch.save(self.model.state_dict(), "resnet_model.pt")


    def test(self):
        test_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.model.eval()
        dataset_test = SkateData(self.list_clips, self.list_labels, transform=test_transforms)
        dataloader_test = DataLoader(dataset_test, batch_size=2, shuffle=True)

        metric_precision.reset()
        metric_accuracy.reset()
        metric_recall.reset()
        metric_f1.reset()

        with torch.no_grad():
            for image, labels in dataloader_test:
                outputs = self.model(image)
                _, predicted = torch.max(outputs, 1)
                metric_precision.update(predicted, labels)
                metric_accuracy.update(predicted, labels)
                metric_recall.update(predicted, labels)
                metric_f1.update(predicted, labels)
                confusion_matrix.update(predicted, labels)

        test_accuracy = metric_accuracy.compute()
        test_recall = metric_recall.compute()
        test_precision = metric_precision.compute()
        test_f1 = metric_f1.compute()

        print("Accuracy: ", test_accuracy)
        print("Recall: ", test_recall)
        print("Precision: ", test_precision)
        print("F1: ", test_f1)

        # Use after saving .pt weights
        # classes = ["kickflip", 'ollie', "pop shuv"]
        # x = np.arange(len(classes))
        # width = 0.2
        #
        # fig, ax = plt.subplots(figsize=(8, 5))
        #
        # ax.bar(x - 1.5*width, test_accuracy, width, label='Accuracy')
        # ax.bar(x - 0.5*width, test_precision, width, label="Precision")
        # ax.bar(x + 0.5*width, test_recall, width, label="Recall")
        # ax.bar(x + 1.5*width, test_f1, width, label="F1")
        # ax.set_xlabel("Classes")
        # ax.set_ylabel("Percentage")
        # ax.set_title("Metrics")
        # ax.set_xticklabels(classes)
        #
        # plt.tight_layout()
        # plt.savefig("metrics.png")
        #
        # fig, ax = confusion_matrix.plot()
        # fig.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')


