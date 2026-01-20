import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Precision, Accuracy, Recall, F1Score, ConfusionMatrix
import torch.optim as optim
from torchvision import transforms
from backend.classify import ViViT as transformerNet
from transformers import VivitImageProcessor

from backend.SkateData import SkateData

metric_precision = Precision(task="multiclass", num_classes=3, average="macro")
metric_accuracy = Accuracy(task="multiclass", num_classes=3, average="macro")
metric_recall = Recall(task="multiclass", num_classes=3, average="macro")
metric_f1 = F1Score(task="multiclass", num_classes=3, average="macro")
confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=3)

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


    def test(self, dict_frames, test_indices):
        self.model.load_state_dict(torch.load("vivit_model.pt"))

        self.model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for vid_index in test_indices:
                board_list = dict_frames[vid_index]["skateboard"]
                label = dict_frames[vid_index]["label"]
                label_id = {"kickflip": 0, "ollie": 1, "popshuv": 2}[label]

                video_clips_preds = []
                for begin_index in range(0, len(board_list), 24):
                    clips_frames = board_list[begin_index:begin_index + 32]
                    if len(clips_frames) < 32:
                        w, h = clips_frames[0].size
                        black_frame = Image.new("RGB", (w, h), (0, 0, 0))
                        clips_frames.extend([black_frame] * (32 - len(clips_frames)))

                    inputs = self.processor(clips_frames, return_tensors="pt", do_resize=True, size=224, do_normalize=True)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    logits = self.model(pixel_values=inputs["pixel_values"])
                    pred = torch.argmax(logits, dim=1).item()
                    video_clips_preds.append(pred)
                if len(video_clips_preds) == 0:
                    continue

                video_pred = max(set(video_clips_preds), key=video_clips_preds.count)

                all_preds.append(video_pred)
                all_labels.append(label_id)

        all_preds_tensor = torch.tensor(all_preds)
        all_labels_tensor = torch.tensor(all_labels)

        test_accuracy = metric_accuracy(all_preds_tensor, all_labels_tensor)
        test_precision = metric_precision(all_preds_tensor, all_labels_tensor)
        test_recall = metric_recall(all_preds_tensor, all_labels_tensor)
        test_f1 = metric_f1(all_preds_tensor, all_labels_tensor)
        test_cm = confusion_matrix(all_preds_tensor, all_labels_tensor)

        print("Accuracy: ", test_accuracy)
        print("Recall: ", test_recall)
        print("Precision: ", test_precision)
        print("F1: ", test_f1)
        print("cm: ", confusion_matrix.compute())

        # Use after saving .pt weights
        classes = ["kickflip", 'ollie', "pop shuv"]
        x = np.arange(len(classes))
        width = 0.2

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.bar(x - 1.5 * width, test_accuracy, width, label='Accuracy')
        ax.bar(x - 0.5 * width, test_precision, width, label="Precision")
        ax.bar(x + 0.5 * width, test_recall, width, label="Recall")
        ax.bar(x + 1.5 * width, test_f1, width, label="F1")
        ax.set_xlabel("Classes")
        ax.set_ylabel("Percentage")
        ax.set_title("Metrics")
        ax.set_xticks(x)
        ax.set_xticklabels(classes)

        plt.tight_layout()
        plt.savefig("metrics_vivit.png")

        fig, ax = confusion_matrix.plot()
        fig.savefig("confusion_matrix_vivit.png", dpi=300, bbox_inches='tight')