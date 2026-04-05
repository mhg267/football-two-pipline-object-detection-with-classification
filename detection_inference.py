import cv2
import argparse

from ultralytics import YOLO


def get_args():
    parser = argparse.ArgumentParser(description='detection inference')

    parser.add_argument('--video_path', '-p', type=str, required=True, help='video path')
    parser.add_argument('--best_model', '-m', type=str, required=True, help='model path')
    parser.add_argument('--conf', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    parser.add_argument('--img_size', '-imsz', type=int, default=640, help='image size')
    parser.add_argument('--output_path', type=str, default="output_result.mp4", help='output directory')


    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()

    model = YOLO(args.best_model)

    result = model.predict(source=args.video_path,
                           stream=True,
                           save=False,
                           conf=args.conf,
                           batch=args.batch_size,
                           imgsz=args.img_size
                           )

    cap = cv2.VideoCapture(args.video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_path, fourcc, fps, (width, height))

    for r in result:
        img = r.orig_img
        boxes = r.boxes

        for bbox in boxes:
            x1, y1, x2, y2 = map(int, bbox.xyxy[0])
            cls = int(bbox.cls[0])
            conf = float(bbox.conf[0])

            if cls == 0:
                color = (0, 0, 255)
                label = f"Player {conf:.2f}"
            else:
                color = (0, 255, 255)
                label = f"Ball {conf:.2f}"

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        out.write(img)

    out.release()