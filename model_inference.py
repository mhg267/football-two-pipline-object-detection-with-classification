import cv2
import torch
import argparse

from src.classification.efficientnetv2_custom import player_classifier
from src.classification.classification_dataset import ClassificationDataset

from torchvision.transforms import ToTensor, Normalize, Compose, Resize, ToPILImage



def parse_args():
    parser = argparse.ArgumentParser("Two-pipeline model inference")

    parser.add_argument('--detection_model', '-dm', type=str, required=True, help='Path to the trained object detection model')
    parser.add_argument('--classification_model', '-cm', type=str, required=True, help='Path to the trained classification model')
    parser.add_argument('--test_video', '-p', type=str, required=True, help='Path to the test video')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='Batch size')
    parser.add_argument('--img_size', '-imsz', type=int, default=1280, help='Image size')
    parser.add_argument('--conf_threshold', '-conf', type=float, default=0.3, help='Confidence threshold')

    args = parser.parse_args()

    return args

def object_detector():
    video = args.test_video
    detection_model = args.detection_model

    model = YOLO(detection_model)

    result = model.predict(source=video,
                           stream=True,
                           save=False,
                           conf=args.conf_threshold,
                           batch=args.batch_size,
                           imgsz=args.img_size
                           )

    for r in result:


def classifier():

    "cuda" if torch.cuda.is_available() else "cpu"

    jersey_team = {
        0: "team A",
        1: "team B"
    }

    jersey_number = {
        0: "Unknown",
        1: "Jersey number 1",
        2: "Jersey number 2",
        3: "Jersey number 3",
        4: "Jersey number 4",
        5: "Jersey number 5",
        6: "Jersey number 6",
        7: "Jersey number 7",
        8: "Jersey number 8",
        9: "Jersey number 9",
        10: "Jersey number 10",
        11: "Jersey number 11",
        12: "Jersey number 12",
        13: "Jersey number 13",
        14: "Jersey number 14",
        15: "Jersey number 15",
        16: "Jersey number 16",
        17: "Jersey number 17",
        18: "Jersey number 18",
        19: "Jersey number 19",
        20: "Jersey number 20"
    }

    # Load model
    model = player_classifier().to(device)
    trained_model = torch.load(args.classification_model)
    model.load_state_dict(trained_model)

    cap = cv2.VideoCapture(args.test_video)
    cap.release()

    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        model.eval()


















if __name__ == '__main__':
    args = parse_args()
