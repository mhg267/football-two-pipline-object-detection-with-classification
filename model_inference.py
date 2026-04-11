import cv2
import torch
import argparse

from src.classification.efficientnetv2_custom import player_classifier
from src.classification.classification_dataset import ClassificationDataset

from torchvision.transforms import ToTensor, Normalize, Compose, Resize, ToPILImage, transforms
from ultralytics import YOLO



def parse_args():
    parser = argparse.ArgumentParser("Two-pipeline model inference")

    parser.add_argument('--detection_model', '-dm', type=str, required=True, help='Path to the trained object detection model')
    parser.add_argument('--classification_model', '-cm', type=str, required=True, help='Path to the trained classification model')
    parser.add_argument('--test_video', '-p', type=str, required=True, help='Path to the test video')
    parser.add_argument('--output_video', type=str, default="result_video.mp4", help='Path to the output video')
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
        boxes = r.boxes
        detections = []

        for bbox in boxes:
            x1, y1, x2, y2 = map(int, bbox.xyxy[0])
            cls = int(bbox.cls[0])
            conf = float(bbox.conf[0])

            detections.append({
                'bbox' : [x1, y1, x2, y2],
                'cls': cls,
                'conf': conf
            })
        yield r.orig_img, detections

def crop_frame(frame, box):
    x1, y1, x2, y2 = box

    w, h = x2 - x1, y2 - y1

    x1_crop = int(x1)
    x2_crop = int(x2)
    y1_crop = int(y1 + 0.1 * h)
    y2_crop = int(y2 - 0.45 * h)

    cropped_frame = frame[y1_crop:y2_crop, x1_crop:x2_crop]

    return cropped_frame



def object_classifier():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    jersey_team = {0: "team A_white", 1: "team B_black"}
    jersey_number = {i: f"Jersey number {i+1}" for i in range(20)}
    jersey_visible_map= {0: "invisible", 1: "visible"}

    model = player_classifier().to(device)
    trained_model = torch.load(args.classification_model, map_location=device)
    model.load_state_dict(trained_model['model'])
    model.eval()

    transform = transforms.Compose([
        ToPILImage(),
        Resize((224, 224)),
        ToTensor(),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    for frame, detections in object_detector():
        all_images = []
        player_detections = []
        results = []

        for detection in detections:

            # Ball
            if detection['cls'] == 1:
                results.append({
                    'bbox': detection['bbox'],
                    'cls': detection['cls'],
                    'conf': detection['conf']
                })
                continue

            # Player
            crop = crop_frame(frame, detection['bbox'])
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = transform(crop)

            all_images.append(crop)
            player_detections.append(detection)

        if len(all_images) > 0:
            all_images = torch.stack(all_images).to(device)

            with torch.no_grad():
                visible, jersey_n_pred, jersey_c_pred = model(all_images)

                jersey_n = torch.argmax(jersey_n_pred, dim=1)
                jersey_c = torch.argmax(jersey_c_pred, dim=1)
                jersey_visible = torch.argmax(visible, dim=1)

            for det, n, c, vis in zip(player_detections, jersey_n, jersey_c, jersey_visible):
                n = n.item()
                c = c.item()
                vis = vis.item()

                results.append({
                    'bbox': det['bbox'],
                    'cls': det['cls'],
                    'conf': det['conf'],
                    'jersey_n_id': n if vis == 1 else None,
                    'jersey_c_id': c,
                    'jersey_visible_id': vis,
                    'jersey_number': jersey_number.get(n, str(n)) if vis == 1 else "Unknown",
                    'jersey_team': jersey_team.get(c, str(c)),
                    'status': jersey_visible_map.get(vis, str(vis))
                })

        yield frame, results




if __name__ == '__main__':
    args = parse_args()

    cap = cv2.VideoCapture(args.test_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output_video, fourcc, fps, (width, height))

    for frame, results in object_classifier():
        for result in results:
            x1, y1, x2, y2 = result['bbox']

            if result['cls'] == 0:
                team = result['jersey_team']
                number = result['jersey_number']
                status = result['status']

                if result['jersey_c_id'] == 0:
                    color = (255, 255, 255)
                else:
                    color = (0, 0, 0)

                if result['jersey_visible_id'] == 1:
                    label = f"{team} | {number} | {result['conf']:.2f} | status: {status}"
                else:
                    label = f"{team} | {result['conf']:.2f} | status: invisible"

            else:
                color = (0, 255, 255)
                label = f"Ball | {result['conf']:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        out.write(frame)

    out.release()
    print(f"Saved output video to: {args.output_video}")





