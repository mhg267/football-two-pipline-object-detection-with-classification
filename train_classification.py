import argparse
import torch.nn as nn
import torch
import os

from src.classification import efficientnetv2_custom
from src.classification import classification_dataset

from torchvision.transforms import Resize, ToTensor, RandomHorizontalFlip, ToPILImage, Normalize
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


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

def get_args():
    parser = argparse.ArgumentParser('classification model arguments')

    parser.add_argument('--epochs', '-e', type=int, default=200, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=16, help='batch size')
    parser.add_argument('--trained_dir', '-trd', type=str, default='efficientnetv2s_trained', help='trained folder')
    parser.add_argument('--checkpoint', '-cp', type=str, required=True, help='checkpoint path')


    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()

    print("---------------------------------------")
    print("Epochs: {}".format(args.epochs))
    print("Batch size: {}".format(args.batch_size))
    print("------------Hyperparameters------------")

    # Check cuda
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Initialize early stopping and tensorboard writer
    early_stopping = EarlyStopping(patience=20, min_delta=1e-3)
    tensorboard_writer = SummaryWriter()

    # Model
    model = efficientnetv2_custom().to(device)

    # Data transform and augmentation
    train_transforms = transforms.Compose([
        ToPILImage(),
        Resize((384, 384)),
        RandomHorizontalFlip(p=0.5),
        ToTensor(),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])  # ImageNet mean and std
    ])

    val_transforms = transforms.Compose([
        ToPILImage(),
        Resize((384, 384)),
        ToTensor(),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]   # ImageNet mean and std
        )
    ])



    if not os .path.exists(args.trained_dir):
        os.makedirs(args.trained_dir)

    # Check checkpoint
    if not os.path.exists(args.checkpoint):
        epoch_start = 0
    else:
        checkpoint = torch.load(args.checkpoint)












