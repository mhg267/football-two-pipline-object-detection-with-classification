import torch
import argparse

from ultralytics import YOLO

def get_args():
    parser = argparse.ArgumentParser(description='detection model arguments')

    parser.add_argument('--epochs', '-e', type=int, default=200, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='batch size')
    parser.add_argument('--num_workers', '-w', type=int, default=8, help='number of workers')
    parser.add_argument('--img_sz', '-s', type=int, default=3840, help='image size')
    parser.add_argument('--lr', '-l', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--optimizer', '-o', type=str, default='AdamW', help='Optimizer of model')
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--name_folder', '-n', type=str, default='yolo26s_lr1e-3_opAdamW', help='name of model folder name with arguments')
    parser.add_argument('--data_path', '-p', type=str, required=True, help='data path')
    parser.add_argument('--model', '-m', type=str, default='yolo26s_model/yolo26s.pt', help='Original model')
    parser.add_argument('--last_model', '-lsm', type=str, help='Last model')

    args = parser.parse_args()

    return args


def main():
    args = get_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.last_model is not None:
        model = YOLO(args.last_model)
        resume_ret = True
    else:
        model = YOLO(args.model)
        resume_ret = False

    model.train(
        data=args.data_path,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.img_sz,
        device=device,
        project="yolo26s_trained",
        name=args.name_folder,
        save=True,
        exist_ok=True,
        optimizer=args.optimizer,
        lr0=args.lr,
        cos_lr=True,
        weight_decay=args.weight_decay,
        patience=20,
        workers=args.num_workers,
        seed=42,
        val=True,
        resume=resume_ret
    )


if __name__ == '__main__':
    main()