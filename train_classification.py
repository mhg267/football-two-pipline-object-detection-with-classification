import argparse
import shutil

import torch.nn as nn
import torch
import os

from src.classification.efficientnetv2_custom import player_classifier
from src.classification.classification_dataset import ClassificationDataset

from torchvision.transforms import Resize, ToTensor, ToPILImage, Normalize, Compose, ColorJitter, RandomAffine, InterpolationMode
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report


class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-3):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
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

def get_args():
    parser = argparse.ArgumentParser('classification model arguments')

    parser.add_argument('--data_path', '-p', type=str, required=True, help='path to dataset')
    parser.add_argument('--num_workers', '-nw', type=int, default=12, help='number of workers')
    parser.add_argument('--epochs', '-e', type=int, default=200, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=24, help='batch size')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4, help='learning rate')
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
    early_stopping = EarlyStopping(patience=10, min_delta=1e-3)

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
        ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.01),
        RandomAffine(degrees=(-10, 10),
                                translate=(0.1, 0.1),
                                scale=(0.9, 1.1),
                                shear=(-10, 10),
                                interpolation=InterpolationMode.BILINEAR),
        ToTensor(),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])  # ImageNet mean and std
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

    # Initialize model, optimizer, loss, scheduler
    model = player_classifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    # Check checkpoint, we need load checkpoint before DataParallel
    if args.checkpoint is None:
        epoch_start = 0
        best_loss = float("inf")
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

        train_progress_bar = tqdm(train_loader)

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
            jersey_n_loss = criterion(jersey_n_output, jersey_numbers)
            jersey_c_loss = criterion(jersey_c_output, jersey_colors)
            train_total_loss = jersey_n_loss + jersey_c_loss

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

        val_progress_bar = tqdm(val_loader)

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
                jersey_n_loss = criterion(jersey_n_output, jersey_numbers)
                jersey_c_loss = criterion(jersey_c_output, jersey_colors)
                val_total_loss = jersey_n_loss + jersey_c_loss

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
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Current LR: {current_lr:.6f}")

        if best_loss > avg_val_loss:
            best_loss = avg_val_loss

            # Save best.pt
            checkpoint = {
                'epoch': epoch + 1,
                'model': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_loss': best_loss
            }

            torch.save(checkpoint, f"{args.trained_dir}/best.pt")

        # Save last.pt
        checkpoint = {
            'epoch': epoch + 1,
            'model': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_loss': best_loss
        }

        torch.save(checkpoint, f"{args.trained_dir}/last.pt")

        if (epoch + 1) % 5 == 0:
            print("Classification report of jersey_n")
            print(classification_report(val_jersey_n_labels, val_jersey_n_outputs))

            print("Classification report of jersey_c")
            print(classification_report(val_jersey_c_labels, val_jersey_c_outputs))

        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    tensorboard_writer.close()












