import argparse
import shutil
import torch.nn as nn
import torch
import os
import json

from src.classification.efficientnetv2_custom import player_classifier
from src.classification.classification_dataset import ClassificationDataset

from torchvision.transforms import Resize, ToTensor, ToPILImage, Normalize, Compose, ColorJitter, RandomAffine, InterpolationMode
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-3):
        self.patience = patience
        self.min_delta = min_delta
        self.best_acc = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_jersey_n_acc):
        if self.best_acc is None:
            self.best_acc = val_jersey_n_acc
            return

        if val_jersey_n_acc > self.best_acc - self.min_delta:
            self.best_acc = val_jersey_n_acc
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def collate_fn(batch):
    images, jersey_nums, jersey_colors = [], [], []

    for imgs, nums, colors in batch:
        images.extend(imgs)
        jersey_nums.extend(nums)
        jersey_colors.extend(colors)

    images = torch.stack(images, dim=0)
    jersey_nums = torch.tensor(jersey_nums, dtype=torch.long)
    jersey_colors = torch.tensor(jersey_colors, dtype=torch.long)

    return images, jersey_nums, jersey_colors

def plot_confusion_matrix(y_true, y_pred, num_classes=21):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-6)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap="Blues", ax=ax)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Normalized Confusion Matrix")

    return fig

def get_args():
    parser = argparse.ArgumentParser('classification model arguments')

    parser.add_argument('--data_path', '-p', type=str, required=True, help='path to dataset')
    parser.add_argument('--num_workers', '-nw', type=int, default=8, help='number of workers')
    parser.add_argument('--epochs', '-e', type=int, default=200, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=16, help='batch size')
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--weight_decay', '-wd', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--trained_dir', '-trd', type=str, default='efficientnetv2s_trained', help='trained folder')
    parser.add_argument('--checkpoint', '-cp', type=str, default=None, help='checkpoint path')
    parser.add_argument('--tensorboard_dir', '-td', type=str, default='tensorboard', help='tensorboard folder')

    args = parser.parse_args()

    return args

#################### ATTENTION: YOU SHOULD CHOOSE BATCH SIZE LOWER THAN NORMAL BECAUSE EACH FRAME USUALLY HAVE 10 IMAGES ####################


if __name__ == '__main__':
    args = get_args()

    print("---------------------------------------")
    print("Epochs: {}".format(args.epochs))
    print("Batch size: {}".format(args.batch_size))
    print("Number of workers: {}".format(args.num_workers))
    print("------------Hyperparameters------------")
    print("Learning rate: {}".format(args.learning_rate))
    print("Weight decay: {}".format(args.weight_decay))
    print("---------------------------------------")

    # Check cuda
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Initialize early stopping and tensorboard writer
    early_stopping = EarlyStopping(patience=7, min_delta=1e-3)

    if args.checkpoint is None:
        if os.path.exists(args.tensorboard_dir):
            shutil.rmtree(args.tensorboard_dir)

    if not os.path.exists(args.tensorboard_dir):
        os.makedirs(args.tensorboard_dir)

    tensorboard_writer = SummaryWriter(log_dir=args.tensorboard_dir)

    # Data transform and augmentation
    train_transforms = Compose([
        ToPILImage(),
        Resize((224, 224)),
        ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.01),
        RandomAffine(degrees=5,
                                translate=(0.05, 0.05),
                                scale=(0.9, 1.1),
                                shear=5,
                                interpolation=InterpolationMode.BILINEAR),
        ToTensor(),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]) # ImageNet mean and std
    ])

    # Load Dataset
    val_transforms = Compose([
        ToPILImage(),
        Resize((224, 224)),
        ToTensor(),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]   # ImageNet mean and std
        )
    ])

    train_dataset = ClassificationDataset(
        path=args.data_path,
        mode="train",
        transform=train_transforms
    )

    val_dataset = ClassificationDataset(
        path=args.data_path,
        mode="val",
        transform=val_transforms
    )

    # Dataloader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    if not os.path.exists(args.trained_dir):
        os.makedirs(args.trained_dir)

    ## Create weight to reduce overfiting ##
    counter = Counter()
    train_root = os.path.join(args.data_path, "football_train")

    for folder in sorted(os.listdir(train_root)):
        if folder in ['.DS_Store', 'images', 'labels']:
            continue

        json_file = os.path.join(train_root, folder, f"{folder}.json")
        with open(json_file, "r") as f:
            annotations = json.load(f)["annotations"]

        for ann in annotations:
            if ann["category_id"] != 4:
                continue

            if ann["attributes"]["number_visible"] == "visible":
                num = int(ann["attributes"]["jersey_number"])
            else:
                num = 0

            counter.update([num])

    num_classes = 21
    total = sum(counter.values())

    weight = [
        0.0 if counter.get(i, 0) == 0 else ((total / num_classes) / counter[i]) ** 0.8
        for i in range(num_classes)
    ]
    weight[0] *= 0.75
    weight[4] *=1.5
    weight[11] *=1.5
    weight[10] *=1.2
    weight[12] *= 1.5
    weight[13] *=1.5

    weight = torch.tensor(weight, dtype=torch.float, device=device)
    ########################################

    # Initialize model, optimizer, loss, scheduler
    model = player_classifier().to(device)
    criterion_n = nn.CrossEntropyLoss(weight=weight, label_smoothing=0.015)
    criterion_c = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=20,
        eta_min=1e-6
    )

    # Check checkpoint, we need load checkpoint before DataParallel
    if args.checkpoint is None:
        epoch_start = 0
        best_acc = 0
        if os.path.exists(f"{args.trained_dir}/last.pt"):
            os.remove(f"{args.trained_dir}/last.pt")
        if os.path.exists(f"{args.trained_dir}/best.pt"):
            os.remove(f"{args.trained_dir}/best.pt")

    else:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        epoch_start = checkpoint["epoch"]
        best_loss = checkpoint["best_loss"]
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        print("Loaded checkpoint from epoch {}".format(epoch_start))

    # Use multi-gpu
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model = model.to(device)

    # iter size
    train_iter_size = len(train_loader)
    val_iter_size = len(val_loader)

    for epoch in range(epoch_start, args.epochs):

        ### TRAINING
        model.train()

        train_jersey_n_outputs, train_jersey_n_labels = [], []
        train_jersey_c_outputs, train_jersey_c_labels = [], []
        train_loss_sum = 0.0
        train_sample_sum = 0

        train_progress_bar = tqdm(train_loader, colour="red", desc="Training")

        for iter, (images, jersey_numbers, jersey_colors) in enumerate(train_progress_bar):
            images, jersey_numbers, jersey_colors = images.to(device, non_blocking=True), jersey_numbers.to(device, non_blocking=True), jersey_colors.to(device, non_blocking=True)

            train_jersey_n_labels.extend(jersey_numbers.cpu().tolist())
            train_jersey_c_labels.extend(jersey_colors.cpu().tolist())

            ## Train forward

            # Train outputs
            jersey_n_output, jersey_c_output = model(images)

            jersey_n_pred = torch.argmax(jersey_n_output, dim=1)
            jersey_c_pred = torch.argmax(jersey_c_output, dim=1)

            train_jersey_n_outputs.extend(jersey_n_pred.cpu().tolist())
            train_jersey_c_outputs.extend(jersey_c_pred.cpu().tolist())

            # Loss
            jersey_n_loss = criterion_n(jersey_n_output, jersey_numbers)
            jersey_c_loss = criterion_c(jersey_c_output, jersey_colors)
            train_total_loss = jersey_n_loss + 0.05 * jersey_c_loss

            train_loss_sum += train_total_loss.item() * jersey_numbers.size(0)
            train_sample_sum += jersey_numbers.size(0)

            # Train Accuracy
            train_jersey_n_acc = accuracy_score(train_jersey_n_labels, train_jersey_n_outputs)
            train_jersey_c_acc = accuracy_score(train_jersey_c_labels, train_jersey_c_outputs)

            # Train backward
            optimizer.zero_grad()
            train_total_loss.backward()
            optimizer.step()

            # Print and tensorboard writer
            train_step = epoch * train_iter_size + iter

            train_progress_bar.set_description(
                f"Epoch {epoch + 1}/{args.epochs}"
            )

            train_progress_bar.set_postfix({
                "loss": f"{train_total_loss.item():.4f}",
                "n_acc": f"{train_jersey_n_acc:.4f}",
                "c_acc": f"{train_jersey_c_acc:.4f}"
            })

            tensorboard_writer.add_scalar("loss/train_iter", train_total_loss.item(), train_step)
            tensorboard_writer.add_scalar("jersey_n_acc/train_iter", train_jersey_n_acc, train_step)
            tensorboard_writer.add_scalar("jersey_c_acc/train_iter", train_jersey_c_acc, train_step)

        avg_train_loss = train_loss_sum / train_sample_sum

        tensorboard_writer.add_scalar("loss/train_epoch", avg_train_loss, epoch + 1)
        tensorboard_writer.add_scalar("jersey_n_acc/train_epoch", train_jersey_n_acc, epoch + 1)
        tensorboard_writer.add_scalar("jersey_c_acc/train_epoch", train_jersey_c_acc, epoch + 1)


        ### VALIDATION
        model.eval()

        val_jersey_n_outputs, val_jersey_n_labels = [], []
        val_jersey_c_outputs, val_jersey_c_labels = [], []
        val_loss_sum = 0.0
        val_sample_sum = 0

        val_progress_bar = tqdm(val_loader, desc="Validation")

        for iter, (images, jersey_numbers, jersey_colors) in enumerate(val_progress_bar):
            images, jersey_numbers, jersey_colors = images.to(device, non_blocking=True), jersey_numbers.to(device, non_blocking=True), jersey_colors.to(device, non_blocking=True)

            val_jersey_n_labels.extend(jersey_numbers.cpu().tolist())
            val_jersey_c_labels.extend(jersey_colors.cpu().tolist())

            with torch.no_grad():

                # Validation ouputs
                jersey_n_output, jersey_c_output = model(images)

                jersey_n_pred = torch.argmax(jersey_n_output, dim=1)
                jersey_c_pred = torch.argmax(jersey_c_output, dim=1)

                val_jersey_n_outputs.extend(jersey_n_pred.cpu().tolist())
                val_jersey_c_outputs.extend(jersey_c_pred.cpu().tolist())

                # Validation loss
                jersey_n_loss = criterion_n(jersey_n_output, jersey_numbers)
                jersey_c_loss = criterion_c(jersey_c_output, jersey_colors)
                val_total_loss = jersey_n_loss + 0.05 *jersey_c_loss

                val_loss_sum += val_total_loss.item() * jersey_numbers.size(0)
                val_sample_sum += jersey_numbers.size(0)

                # Validation accuracy
                val_jersey_n_acc = accuracy_score(val_jersey_n_labels, val_jersey_n_outputs)
                val_jersey_c_acc = accuracy_score(val_jersey_c_labels, val_jersey_c_outputs)

                # Print and tensorboard writer
                val_step = epoch * val_iter_size + iter

                val_progress_bar.set_description(f"Validation {epoch + 1}/{args.epochs}")

                val_progress_bar.set_postfix({
                    "loss": f"{val_total_loss.item():.4f}",
                    "n_acc": f"{val_jersey_n_acc:.4f}",
                    "c_acc": f"{val_jersey_c_acc:.4f}"
                })

                tensorboard_writer.add_scalar("loss/val_iter", val_total_loss.item(), val_step)
                tensorboard_writer.add_scalar("jersey_n_acc/val_iter", val_jersey_n_acc, val_step)
                tensorboard_writer.add_scalar("jersey_c_acc/val_iter", val_jersey_c_acc, val_step)

        avg_val_loss = val_loss_sum / val_sample_sum
        print(f"Average validation loss: {avg_val_loss:.4f}")

        tensorboard_writer.add_scalar("loss/val_epoch", avg_val_loss, epoch + 1)
        tensorboard_writer.add_scalar("jersey_n_acc/val_epoch", val_jersey_n_acc, epoch + 1)
        tensorboard_writer.add_scalar("jersey_c_acc/val_epoch", val_jersey_c_acc, epoch + 1)

        # scheduler step
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Current LR: {current_lr:.6f}")

        if best_acc < val_jersey_n_acc:
            best_acc = val_jersey_n_acc

            # Save best.pt
            checkpoint = {
                'epoch': epoch + 1,
                'model': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_acc': best_acc
            }

            torch.save(checkpoint, f"{args.trained_dir}/best.pt")

        # Save last.pt
        checkpoint = {
            'epoch': epoch + 1,
            'model': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }

        torch.save(checkpoint, f"{args.trained_dir}/last.pt")


        print("Classification report of jersey_n")
        print(classification_report(val_jersey_n_labels, val_jersey_n_outputs, zero_division=0, labels=list(range(21))))

        print("Classification report of jersey_c")
        print(classification_report(val_jersey_c_labels, val_jersey_c_outputs))

        fig = plot_confusion_matrix(
            val_jersey_n_labels,
            val_jersey_n_outputs,
            num_classes=21
        )

        tensorboard_writer.add_figure(
            "Confusion Matrix / Jersey Number",
            fig,
            global_step=epoch + 1
        )

        plt.close(fig)

        early_stopping(best_acc)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    tensorboard_writer.close()












