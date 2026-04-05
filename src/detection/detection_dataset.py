import os
import cv2

from torch.utils.data import Dataset


class DetectionDataset(Dataset):
    def __init__(self, path, mode, transform=None):
        self.path = os.path.join(path, "football_{}".format(mode))
        self.transform = transform

        img_dir = os.path.join(self.path, "images")
        label_dir = os.path.join(self.path, "labels")

        self.images = sorted([img for img in os.listdir(img_dir) if img.endswith(".jpg")])

        self.img_dir = img_dir
        self.label_dir = label_dir

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        label = image_name.replace(".jpg", ".txt")

        image_path = os.path.join(self.img_dir, image_name)
        label_path = os.path.join(self.label_dir, label)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = []

        with open(label_path, "r") as f:
            for line in f:
                label.append([float(x) for x in line.split()])

        if self.transform:
            image = self.transform(image)

        return image, label


