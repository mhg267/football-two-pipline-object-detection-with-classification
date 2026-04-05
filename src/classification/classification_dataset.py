import torch
import cv2
import os
import json
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import transforms


class ClassificationDataset(Dataset):
    def __init__(self, path, mode="train", transform=None):
        self.path = path

        if mode == "train":
            self.path = os.path.join(path, f"football_{mode}")
        elif mode == "val":
            self.path = os.path.join(path, f"football_{mode}")
        elif mode == "test":
            self.path = os.path.join(path, f"football_{mode}")
        else:
            raise ValueError("mode must be train or val or test")

        self.match_files = os.listdir(self.path)

        self.start_idx = 0
        self.end_idx = 0
        self.image_idx = {}
        self.all_annotations = []

        unsual_files = ['.DS_Store', 'images', 'labels']

        for folder in self.match_files:
            if folder not in unsual_files:

                folder_path = os.path.join(self.path, folder)
                json_file = os.path.join(folder_path, folder+".json")

                with open(json_file, "r") as f:
                    json_data = json.load(f)

                self.end_idx += len(json_data["images"])
                self.image_idx[folder] = [self.start_idx, self.end_idx-1]
                self.start_idx = self.end_idx



    def __len__(self):
        return self.end_idx

    def __getitem__(self, idx):
        for folder, index in self.image_idx.items():
            if index[0] <= idx <= index[1]:
                folder_name = folder
                idx = idx - index[0]
                break

        video_path = os.path.join(self.path, folder_name, folder_name+".mp4")
        json_file = os.path.join(self.path, folder_name, folder_name+".json")

        with open(json_file, "r") as f:
            all_annotations = json.load(f)["annotations"]

        all_image_id = {}
        for annotation in all_annotations:
            image_id = annotation["image_id"]
            if annotation['category_id'] == 4:
                if image_id not in all_image_id:
                    all_image_id[image_id] = []
                all_image_id[image_id].append(annotation)

        # Cut frame have idx
        video_capture = cv2.VideoCapture(video_path)
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = video_capture.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_capture.release()

        # Take annotations
        annotations = all_image_id[idx+1]
        bbox = [annotation["bbox"] for annotation in annotations]
        cropped_image = [image[int(ymin):int(ymin+h), int(xmin):int(xmin+w)] for [xmin, ymin, w, h] in bbox]
        jersey_num = [int(annotation["attributes"]["jersey_number"]) for annotation in annotations]
        jersey_color = [annotation["attributes"]["team_jersey_color"] for annotation in annotations]

        return cropped_image, jersey_num, jersey_color




