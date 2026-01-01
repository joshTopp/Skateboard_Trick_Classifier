import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Precision, Accuracy, Recall, F1Score
import torch.optim as optim
from torchvision import transforms
from backend.classify import ViViT as transformerNet
from transformers import VivitImageProcessor

from backend.SkateData import SkateData

metric_precision = Precision(task="multiclass", num_classes=3, average="macro")
metric_accuracy = Accuracy(task="multiclass", num_classes=3, average="macro")
metric_recall = Recall(task="multiclass", num_classes=3, average="macro")
metric_f1 = F1Score(task="multiclass", num_classes=3, average="macro")

def vivit_collate(batch):
    clips, labels = zip(*batch)
    return list(clips), torch.tensor(labels)

class TrainViViT:
    def __init__(self, list_clips, list_labels):
        self.list_clips = list_clips
        self.list_labels = list_labels
        self.model = transformerNet.ViViTNet()
        self.processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
        self.device = torch.device("cpu")


    def training(self, epochs=5):
        optimizer = optim.AdamW([
            {"params": self.model.classifier.parameters(), "lr": 3e-4},
            {"params": self.model.model.vivit.encoder.layer[-8:].parameters(), "lr": 1e-4},
        ],
        weight_decay=1e-4,)

        dataset_train = SkateData(self.list_clips, self.list_labels)
        dataloader_train = DataLoader(dataset_train, batch_size=2, shuffle=True, collate_fn=vivit_collate)


        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for clips, labels in dataloader_train:
                labels = labels.to(self.device)
                inputs = self.processor(clips, return_tensors="pt", do_resize=True, size=224, do_normalize=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                logits = self.model(pixel_values=inputs["pixel_values"])
                loss = criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(dataloader_train)
            print(f"Training loss: {epoch_loss}")
        torch.save(self.model.state_dict(), "vivit_model.pt")


    def test(self, clips, labels):

        dataset_test = SkateData(clips, labels)
        dataloader_test = DataLoader(dataset_test, batch_size=1, collate_fn=vivit_collate)

        metric_precision.reset()
        metric_accuracy.reset()
        metric_recall.reset()
        metric_f1.reset()

        self.model.eval()
        with torch.no_grad():
            for clips, labels in dataloader_test:
                inputs = self.processor(clips, return_tensors="pt", do_resize=True, size=224, do_normalize=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                logits = self.model(pixel_values=inputs["pixel_values"])
                predicted = torch.argmax(logits, dim=-1)

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