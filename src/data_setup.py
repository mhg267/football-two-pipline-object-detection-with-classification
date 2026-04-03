import os
import cv2
import json
import shutil

from torch.utils.data import Dataset


class FootballConverter(Dataset):
    def __init__(self, path, mode=None, transform=None):
        self.path = path
        self.transform = transform

        if mode == "train":
            self.path = os.path.join(path, "football_train")
        elif mode == "val":
            self.path = os.path.join(path, "football_val")
        elif mode == "test":
            self.path = os.path.join(path, "football_test")
        elif mode is None:
            print("None Mode")
        else:
            print("Wrong format!")

        img_output_folder = os.path.join(self.path, "images")
        label_output_folder = os.path.join(self.path, "labels")

        if os.path.exists(img_output_folder):
            shutil.rmtree(img_output_folder)
        os.makedirs(img_output_folder, exist_ok=True)
        if os.path.exists(label_output_folder):
            shutil.rmtree(label_output_folder)
        os.makedirs(label_output_folder, exist_ok=True)


        for folder_name in os.listdir(self.path):
            if folder_name in ["images", "labels"] or folder_name.startswith("."):
                continue

            video_path = os.path.join(self.path, folder_name, folder_name+".mp4")
            annotation_path = os.path.join(self.path, folder_name, folder_name+".json")

            video_capture = cv2.VideoCapture(video_path)
            with open(annotation_path, "r") as f:
                all_annotation = json.load(f)["annotations"]

            all_image_id = {}
            categories = [4, 3]
            for anno in all_annotation:
                image_id = anno["image_id"]
                if anno["category_id"] in categories:
                    if image_id not in all_image_id:
                        all_image_id[image_id] = []
                    all_image_id[image_id].append(anno)

            frame_id = 0

            while video_capture.isOpened():
                ret, frame = video_capture.read()
                if not ret:
                    break
                h, w, _ = frame.shape

                frame_id += 1

                # take frames
                img_name = f"{folder_name}_{frame_id}.jpg"
                img_path = os.path.join(img_output_folder, img_name)
                cv2.imwrite(img_path, frame)

                # take annotations
                current_data = all_image_id.get(frame_id, [])

                label_name = f"{folder_name}_{frame_id}.txt"
                label_path = os.path.join(label_output_folder, label_name)

                with open(label_path, "w") as f:
                    for item in current_data:
                        bbox = item["bbox"]
                        xmin, ymin, width, height = bbox
                        xcent = (xmin + width / 2) / w
                        ycent = (ymin + height / 2) / h
                        width /= w
                        height /= h

                        # 0: player, 1: ball
                        if item["category_id"] == 4:
                            class_id = 0
                        elif item["category_id"] == 3:
                            class_id = 1

                        f.write(f"{class_id} {xcent:.6f} {ycent:.6f} {width:.6f} {height:.6f}\n ")

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


if __name__ == '__main__':
    path = "/Computer Vision/Data/Dataset/Football"

    dataset = FootballConverter(path=path, mode="test", transform=None)
