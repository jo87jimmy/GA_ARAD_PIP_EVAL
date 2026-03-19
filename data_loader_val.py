import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from model_unet import ReconstructiveSubNetwork
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
import cv2
import glob
# === MVTec Dataset ===
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

class MVTecDataset(Dataset):
    def __init__(self, root, category="bottle", split="train", resize=256):
        self.root = root
        self.category = category
        self.split = split
        self.img_dir = os.path.join(root, category, split)
        self.gt_dir = os.path.join(root, category, "ground_truth")

        self.data = []
        self.labels = []
        self.masks = []

        for defect_type in sorted(os.listdir(self.img_dir)):
            img_folder = os.path.join(self.img_dir, defect_type)
            if not os.path.isdir(img_folder):
                continue
            img_files = sorted(os.listdir(img_folder))
            for f in img_files:
                img_path = os.path.join(img_folder, f)
                if defect_type == "normal":
                    self.data.append(img_path)
                    self.labels.append(0)
                    self.masks.append(None)
                else:
                    mask_path = os.path.join(self.gt_dir, defect_type, f.replace(".png","_mask.png"))
                    self.data.append(img_path)
                    self.labels.append(1)
                    self.masks.append(mask_path)

        self.transform = T.Compose([
            T.Resize((resize, resize)),
            T.ToTensor()
        ])
        self.mask_transform = T.Compose([
            T.Resize((resize, resize), interpolation=Image.NEAREST),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        mask_path = self.masks[idx]

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        if mask_path is None:
            mask = torch.zeros((1,img.shape[1], img.shape[2]))
        else:
            mask = Image.open(mask_path).convert("L")
            mask = self.mask_transform(mask)
            mask = (mask>0.5).float()

        return img, (torch.tensor(label, dtype=torch.long), mask)

# === train_loader & val_loader ===
train_dataset = MVTecDataset(root="./mvtec_ad", category="bottle", split="train", resize=256)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

val_dataset = MVTecDataset(root="./mvtec_ad", category="bottle", split="test", resize=256)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

print("Train samples:", len(train_dataset))
print("Val samples:", len(val_dataset))
