import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Precision, Accuracy, Recall, F1Score, ConfusionMatrix
import torch.optim as optim
from torchvision import transforms
from backend.classify import ResNet as net

from backend.SkateData import SkateData

metric_precision = Precision(task="multiclass", num_classes=3, average="macro")
metric_accuracy = Accuracy(task="multiclass", num_classes=3, average="macro")
metric_recall = Recall(task="multiclass", num_classes=3, average="macro")
metric_f1 = F1Score(task="multiclass", num_classes=3, average="macro")
confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=3)

class TrainCNN:
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


    def test(self, dict_frames, test_indices):
        test_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.model.load_state_dict(torch.load("resnet_model.pt"))
        self.model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for vid_idx in test_indices:
                board_list = dict_frames[vid_idx]["skateboard"]
                label = dict_frames[vid_idx]["label"]
                label_id = {"kickflip": 0, "ollie": 1, "popshuv": 2}[label]

                video_clip_preds = []

                for begin_index in range(0, len(board_list), 24):
                    clips_frames = board_list[begin_index: begin_index + 32]
                    frame_tensors = [test_transforms(f) for f in clips_frames]
                    if len(clips_frames) < 32:
                        c, h, w = clips_frames[0].shape
                        black_frame = torch.zeros((c,h,w), device=frame_tensors[0].device)
                        clips_frames.extend([black_frame] * (32 - len(clips_frames)))

                    clip_tensor = torch.stack(frame_tensors).unsqueeze(0)
                    outputs = self.model(clip_tensor)
                    pred = torch.argmax(outputs, dim=1).item()
                    video_clip_preds.append(pred)
                if len(video_clip_preds) == 0:
                    continue

                video_pred = max(set(video_clip_preds), key=video_clip_preds.count)

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

        ax.bar(x - 1.5*width, test_accuracy, width, label='Accuracy')
        ax.bar(x - 0.5*width, test_precision, width, label="Precision")
        ax.bar(x + 0.5*width, test_recall, width, label="Recall")
        ax.bar(x + 1.5*width, test_f1, width, label="F1")
        ax.set_xlabel("Classes")
        ax.set_ylabel("Percentage")
        ax.set_title("Metrics")
        ax.set_xticks(x)
        ax.set_xticklabels(classes)

        plt.tight_layout()
        plt.savefig("metrics.png")

        fig, ax = confusion_matrix.plot()
        fig.savefig("confusion_matrix_resnet.png", dpi=300, bbox_inches='tight')


