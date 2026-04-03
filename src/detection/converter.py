import os
import json
import shutil
import cv2




class FootballConverter:
    def __init__(self, path, mode):
        self.path = os.path.join(path, mode)

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

                        f.write(f"{class_id} {xcent:.6f} {ycent:.6f} {width:.6f} {height:.6f}\n")