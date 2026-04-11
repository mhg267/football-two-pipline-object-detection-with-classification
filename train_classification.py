import argparse
import shutil
import torch.nn as nn
import torch
import os
import json

from src.classification.efficientnetv2_custom import player_classifier
from src.classification.classification_dataset import ClassificationDataset

from torchvision.transforms import Resize, ToTensor, ToPILImage, Normalize, Compose, ColorJitter, RandomAffine, InterpolationMode, RandomErasing, RandomPerspective
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

        if val_jersey_n_acc > self.best_acc + self.min_delta:
            self.best_acc = val_jersey_n_acc
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def collate_fn(batch):
    images, jersey_nums, jersey_colors, jersey_visibles = [], [], [], []
    for imgs, nums, colors, visibles in batch:
        images.extend(imgs)
        jersey_nums.extend(nums)
        jersey_colors.extend(colors)
        jersey_visibles.extend(visibles)

    images = torch.stack(images, dim=0)
    jersey_nums = torch.tensor(jersey_nums, dtype=torch.long)
    jersey_colors = torch.tensor(jersey_colors, dtype=torch.long)
    jersey_visibles = torch.tensor(jersey_visibles, dtype=torch.long)
    return images, jersey_nums, jersey_colors, jersey_visibles

def plot_confusion_matrix(y_true, y_pred, num_classes=20):
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
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight_decay', '-wd', type=float, default=2e-3, help='weight decay')
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
    print("Learning rate of 3 heads: {}".format(args.learning_rate))
    print("Learning rate of backbone: {}".format(1e-5))
    print("Weight decay: {}".format(args.weight_decay))
    print("---------------------------------------")

    # Check cuda
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Initialize early stopping and tensorboard writer
    early_stopping = EarlyStopping(patience=3, min_delta=1e-3)

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
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.01),
        RandomAffine(
            degrees=12,
            translate=(0.12, 0.12),
            scale=(0.8, 1.2),
            shear=8,
            interpolation=InterpolationMode.BILINEAR
        ),
        RandomPerspective(distortion_scale=0.2, p=0.3),
        ToTensor(),
        RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]   # ImageNet mean and std
        )
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

    ### Counter weight for visible and num ###
    train_root = os.path.join(args.data_path, "football_train")

    visible_counter = Counter()
    number_counter = Counter()

    for folder in sorted(os.listdir(train_root)):
        if folder in ['.DS_Store', 'images', 'labels']:
            continue

        json_path = os.path.join(train_root, folder, folder + ".json")

        with open(json_path, "r") as f:
            annotations = json.load(f)['annotations']

        for ann in annotations:
            if ann['category_id'] != 4:
                continue

            is_visible = 1 if ann["attributes"]["number_visible"] == "visible" else 0
            visible_counter.update([is_visible])

            if is_visible:
                jersey_num = int(ann["attributes"]["jersey_number"]) - 1
                number_counter.update([jersey_num])

    # visible weight
    total_visible = sum(visible_counter.values())
    visible_weight = torch.tensor(
        [((total_visible / 2) / visible_counter[i]) for i in range(2)],
        dtype=torch.float,
        device=device
    )
    visible_weight = visible_weight / visible_weight.mean()

    # jersey number weight
    num_classes = 20
    total_num = sum(number_counter.values())
    jersey_n_weight = torch.tensor(
        [((total_num / num_classes) / number_counter[i]) if number_counter[i] > 0 else 0.0
         for i in range(num_classes)],
        dtype=torch.float,
        device=device
    )

    valid_mask = jersey_n_weight > 0
    jersey_n_weight[valid_mask] = jersey_n_weight[valid_mask] / jersey_n_weight[valid_mask].mean()
    ###################################

    if not os.path.exists(args.trained_dir):
        os.makedirs(args.trained_dir)

    # Initialize model, optimizer, loss, scheduler
    model = player_classifier().to(device)
    criterion_n = nn.CrossEntropyLoss(weight=jersey_n_weight, label_smoothing=0.1)
    criterion_c = nn.CrossEntropyLoss()
    criterion_visible = nn.CrossEntropyLoss(weight=visible_weight)
    optimizer = torch.optim.AdamW([
        {'params': model.backbone.features.parameters(), 'lr': 1e-5},
        {'params': model.backbone.number_head.parameters(), 'lr': 1e-4},
        {'params': model.backbone.color_head.parameters(), 'lr': 1e-4},
        {'params': model.backbone.visible_head.parameters(), 'lr': 1e-4},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
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
        best_acc = checkpoint["best_acc"]
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
        train_visible_n_outputs, train_visible_n_labels = [], []
        train_loss_sum = 0.0
        train_sample_sum = 0

        train_progress_bar = tqdm(train_loader, colour="red", desc="Training")

        for iter, (images, jersey_numbers, jersey_colors, n_visible) in enumerate(train_progress_bar):
            images, jersey_numbers, jersey_colors = images.to(device, non_blocking=True), jersey_numbers.to(device, non_blocking=True), jersey_colors.to(device, non_blocking=True)
            n_visible = n_visible.to(device, non_blocking=True)

            train_visible_n_labels.extend(n_visible.cpu().tolist())
            train_jersey_c_labels.extend(jersey_colors.cpu().tolist())

            visible_mask = n_visible.bool()

            if visible_mask.any():
                train_jersey_n_labels.extend(jersey_numbers[visible_mask].cpu().tolist())

            ## Train forward

            # Train outputs
            visible, jersey_n_output, jersey_c_output = model(images)

            jersey_n_pred = torch.argmax(jersey_n_output, dim=1)
            jersey_c_pred = torch.argmax(jersey_c_output, dim=1)
            visible_pred = torch.argmax(visible, dim=1)

            if visible_mask.any():
                train_jersey_n_outputs.extend(jersey_n_pred[visible_mask].cpu().tolist())

            train_jersey_c_outputs.extend(jersey_c_pred.cpu().tolist())
            train_visible_n_outputs.extend(visible_pred.cpu().tolist())

            # Visible loss
            visible_loss = criterion_visible(visible, n_visible)

            # Loss
            if visible_mask.any():
                jersey_n_loss = criterion_n(jersey_n_output[visible_mask], jersey_numbers[visible_mask])
            else:
                jersey_n_loss = torch.tensor(0.0, device=device)

            jersey_c_loss = criterion_c(jersey_c_output, jersey_colors)
            train_total_loss = jersey_n_loss + 0.05 * jersey_c_loss + visible_loss

            train_loss_sum += train_total_loss.item() * jersey_numbers.size(0)
            train_sample_sum += jersey_numbers.size(0)

            # Train Accuracy
            train_jersey_n_acc = accuracy_score(train_jersey_n_labels, train_jersey_n_outputs)
            train_jersey_c_acc = accuracy_score(train_jersey_c_labels, train_jersey_c_outputs)
            train_visible_acc = accuracy_score(train_visible_n_labels, train_visible_n_outputs)

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
                "c_acc": f"{train_jersey_c_acc:.4f}",
                "vis_acc": f"{train_visible_acc:.4f}"
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
        val_visible_n_outputs, val_visible_n_labels = [], []
        val_loss_sum = 0.0
        val_sample_sum = 0

        val_progress_bar = tqdm(val_loader, desc="Validation")

        for iter, (images, jersey_numbers, jersey_colors, n_visible) in enumerate(val_progress_bar):
            images, jersey_numbers, jersey_colors = images.to(device, non_blocking=True), jersey_numbers.to(device, non_blocking=True), jersey_colors.to(device, non_blocking=True)
            n_visible = n_visible.to(device, non_blocking=True)

            val_visible_n_labels.extend(n_visible.cpu().tolist())
            val_jersey_c_labels.extend(jersey_colors.cpu().tolist())

            visible_mask = n_visible.bool()

            if visible_mask.any():
                val_jersey_n_labels.extend(jersey_numbers[visible_mask].cpu().tolist())

            with torch.no_grad():

                # Validation ouputs
                visible, jersey_n_output, jersey_c_output = model(images)

                jersey_n_pred = torch.argmax(jersey_n_output, dim=1)
                jersey_c_pred = torch.argmax(jersey_c_output, dim=1)
                visible_pred = torch.argmax(visible, dim=1)

                if visible_mask.any():
                    val_jersey_n_outputs.extend(jersey_n_pred[visible_mask].cpu().tolist())

                val_jersey_c_outputs.extend(jersey_c_pred.cpu().tolist())
                val_visible_n_outputs.extend(visible_pred.cpu().tolist())

                # Visible loss
                visible_loss = criterion_visible(visible, n_visible)

                # Validation loss
                if visible_mask.any():
                    jersey_n_loss = criterion_n(jersey_n_output[visible_mask], jersey_numbers[visible_mask])
                else:
                    jersey_n_loss = torch.tensor(0.0, device=device)

                jersey_c_loss = criterion_c(jersey_c_output, jersey_colors)
                val_total_loss = jersey_n_loss + 0.05 * jersey_c_loss + visible_loss

                val_loss_sum += val_total_loss.item() * jersey_numbers.size(0)
                val_sample_sum += jersey_numbers.size(0)

                # Validation accuracy
                val_jersey_n_acc = accuracy_score(val_jersey_n_labels, val_jersey_n_outputs)
                val_jersey_c_acc = accuracy_score(val_jersey_c_labels, val_jersey_c_outputs)
                val_visible_n_acc = accuracy_score(val_visible_n_labels, val_visible_n_outputs)

                # Print and tensorboard writer
                val_step = epoch * val_iter_size + iter

                val_progress_bar.set_description(f"Validation {epoch + 1}/{args.epochs}")

                val_progress_bar.set_postfix({
                    "loss": f"{val_total_loss.item():.4f}",
                    "n_acc": f"{val_jersey_n_acc:.4f}",
                    "c_acc": f"{val_jersey_c_acc:.4f}",
                    "vis_acc": f"{val_visible_n_acc:.4f}"
                })

                tensorboard_writer.add_scalar("loss/val_iter", val_total_loss.item(), val_step)
                tensorboard_writer.add_scalar("jersey_n_acc/val_iter", val_jersey_n_acc, val_step)
                tensorboard_writer.add_scalar("jersey_c_acc/val_iter", val_jersey_c_acc, val_step)

        avg_val_loss = val_loss_sum / val_sample_sum

        print("--------------------------------------------")
        print(f"Average validation loss: {avg_val_loss:.4f}")

        tensorboard_writer.add_scalar("loss/val_epoch", avg_val_loss, epoch + 1)
        tensorboard_writer.add_scalar("jersey_n_acc/val_epoch", val_jersey_n_acc, epoch + 1)
        tensorboard_writer.add_scalar("jersey_c_acc/val_epoch", val_jersey_c_acc, epoch + 1)

        # scheduler step
        scheduler.step()
        current_lr_head = optimizer.param_groups[1]['lr']
        current_lr_backbone = optimizer.param_groups[0]['lr']
        print("--------------------------------------------")
        print(f"Current LR of 3 heads: {current_lr_head:.6f}")
        print(f"Current LR of backbone: {current_lr_backbone:.6f}")
        print("--------------------------------------------")

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
            'scheduler': scheduler.state_dict(),
            'best_acc': best_acc
        }

        torch.save(checkpoint, f"{args.trained_dir}/last.pt")

        print("----------------------------------------------")
        print("Classification report of visible")
        print(classification_report(val_visible_n_labels, val_visible_n_outputs))
        print("----------------------------------------------")
        print("Classification report of jersey_n")
        print(classification_report(val_jersey_n_labels, val_jersey_n_outputs, zero_division=0, labels=list(range(20))))
        print("----------------------------------------------")
        print("Classification report of jersey_c")
        print(classification_report(val_jersey_c_labels, val_jersey_c_outputs))

        fig = plot_confusion_matrix(
            val_jersey_n_labels,
            val_jersey_n_outputs,
            num_classes=20
        )

        tensorboard_writer.add_figure(
            "Confusion Matrix / Jersey Number",
            fig,
            global_step=epoch + 1
        )

        plt.close(fig)

        early_stopping(val_jersey_n_acc)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    tensorboard_writer.close()












