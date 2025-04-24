import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from generator import Generator
from discriminator import Discriminator
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import sys
sys.path.append('/content/drive/MyDrive/image_colorization')

# Define the base path
base_path = "/content/drive/MyDrive/image_colorization"

# Training data folders
gray_train_path = os.path.join(base_path, "train_gray")
color_train_path = os.path.join(base_path, "train")

# Testing data folders (if needed later)
gray_test_path = os.path.join(base_path, "test_gray")
color_test_path = os.path.join(base_path, "test")

# Dataset class
class ColorizationDataset(Dataset):
    def __init__(self, gray_folder, color_folder, transform_gray, transform_color):
        self.gray_paths = sorted(os.listdir(gray_folder))
        self.color_paths = sorted(os.listdir(color_folder))
        self.gray_folder = gray_folder
        self.color_folder = color_folder
        self.transform_gray = transform_gray
        self.transform_color = transform_color

    def __len__(self):
        return len(self.gray_paths)

    def __getitem__(self, idx):
        gray_img = Image.open(os.path.join(self.gray_folder, self.gray_paths[idx])).convert("L")
        color_img = Image.open(os.path.join(self.color_folder, self.color_paths[idx])).convert("RGB")
        gray_img = self.transform_gray(gray_img)
        color_img = self.transform_color(color_img)
        return gray_img, color_img

# Paths (set these before running)
gray_train_path = '/content/drive/MyDrive/image_colorization/train_gray'
color_train_path = '/content/drive/MyDrive/image_colorization/train'

# Transforms
transform_gray = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

transform_color = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Dataloader
dataset = ColorizationDataset(gray_train_path, color_train_path, transform_gray, transform_color)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = Generator().to(device)
D = Discriminator().to(device)

# Losses and optimizers
criterion_GAN = nn.BCELoss()
criterion_L1 = nn.L1Loss()
optimizer_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Learning rate schedulers
scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=10, gamma=0.1)
scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=10, gamma=0.1)

# Directories to save weights and outputs
save_dir = "model_checkpoints"
os.makedirs(save_dir, exist_ok=True)
image_dir = os.path.join(save_dir, "images")
os.makedirs(image_dir, exist_ok=True)

# Training loop
for epoch in range(80):
    total_loss_G = 0.0
    total_loss_D = 0.0

    for i, (gray, real) in enumerate(dataloader):
        gray, real = gray.to(device), real.to(device)

        # Train Discriminator
        optimizer_D.zero_grad()
        fake = G(gray)
        pred_real = D(gray, real)
        pred_fake = D(gray, fake.detach())

        loss_D = criterion_GAN(pred_real, torch.ones_like(pred_real)) + \
                 criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
        loss_D.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        pred_fake = D(gray, fake)
        loss_G = criterion_GAN(pred_fake, torch.ones_like(pred_fake)) + \
                 100 * criterion_L1(fake, real)
        loss_G.backward()
        optimizer_G.step()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)

        total_loss_G += loss_G.item()
        total_loss_D += loss_D.item()

        if i % 50 == 0:
            print(f"Epoch [{epoch+1}/100] Batch [{i}/{len(dataloader)}] Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")

    scheduler_G.step()
    scheduler_D.step()

    # Calculate average epoch loss
    avg_loss_G = total_loss_G / len(dataloader)
    avg_loss_D = total_loss_D / len(dataloader)
    print(f"Epoch [{epoch+1}/100] Completed | Avg Loss D: {avg_loss_D:.4f}, Avg Loss G: {avg_loss_G:.4f}")

    # Save model weights at every epoch
    torch.save(G.state_dict(), os.path.join(save_dir, f"generator_epoch_{epoch+1}.pth"))
    torch.save(D.state_dict(), os.path.join(save_dir, f"discriminator_epoch_{epoch+1}.pth"))

    # Save one example image from current epoch
    fake_images = G(gray).cpu().detach()
    save_image(fake_images[0], os.path.join(image_dir, f"epoch_{epoch+1}.png"))

print("✅ Training complete! All models and outputs saved.")
