import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from generator import Generator
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Set the paths to your Google Drive folders
test_gray_folder = "/content/drive/MyDrive/image_colorization/test_gray"
test_color_folder = "/content/drive/MyDrive/image_colorization/test"
output_folder = "/content/drive/MyDrive/image_colorization/outputs/samples"
os.makedirs(output_folder, exist_ok=True)

# Define transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

transform_color = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = Generator().to(device)
model_checkpoint_path = "/content/model_checkpoints/generator_epoch_77.pth"
G.load_state_dict(torch.load(model_checkpoint_path, map_location=device))
G.eval()

# Initialize metrics
ssim_scores = []
psnr_scores = []

# Function to save output image
def save_output_image(tensor, path):
    image = tensor.squeeze(0).cpu().detach()
    image = transforms.ToPILImage()(image)
    image.save(path)

# Process images
for file_name in os.listdir(test_gray_folder):
    gray_img = Image.open(os.path.join(test_gray_folder, file_name)).convert("L")
    gray_tensor = transform(gray_img).unsqueeze(0).to(device)

    with torch.no_grad():
        fake_color = G(gray_tensor)

    # Save generated image
    output_path = os.path.join(output_folder, file_name)
    save_output_image(fake_color, output_path)

    # Load corresponding real color image (ground-truth)
    real_color_img = Image.open(os.path.join(test_color_folder, file_name)).convert("RGB")
    real_color_tensor = transform_color(real_color_img)

    # Denormalize fake_color image
    fake_color_np = fake_color.squeeze(0).cpu().numpy()
    fake_color_np = np.transpose(fake_color_np, (1, 2, 0))  # CHW to HWC
    fake_color_np = (fake_color_np * 0.5 + 0.5) * 255.0
    fake_color_np = fake_color_np.astype(np.uint8)

    # Prepare real color image
    real_color_np = real_color_tensor.cpu().numpy()
    real_color_np = np.transpose(real_color_np, (1, 2, 0)) * 255.0
    real_color_np = real_color_np.astype(np.uint8)

    # Calculate SSIM and PSNR
    ssim_val = ssim(real_color_np, fake_color_np, channel_axis=-1)
    psnr_val = psnr(real_color_np, fake_color_np)

    ssim_scores.append(ssim_val)
    psnr_scores.append(psnr_val)

    # Display real vs. generated images side by side
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(real_color_np)
    axs[0].set_title('Ground Truth')
    axs[0].axis('off')

    axs[1].imshow(fake_color_np)
    axs[1].set_title('Generated')
    axs[1].axis('off')

    plt.suptitle(f"{file_name} - SSIM: {ssim_val:.4f}, PSNR: {psnr_val:.2f} dB", fontsize=10)

    # Save or show comparison
    comparison_output_path = os.path.join("/content/drive/MyDrive/image_colorization/outputs/comparisons", f"compare_{file_name}")
    os.makedirs("/content/drive/MyDrive/image_colorization/outputs/comparisons", exist_ok=True)
    plt.savefig(comparison_output_path, bbox_inches='tight')
    plt.close()

# Print average metrics
print("\n==== Average Metrics ====")
print(f"Average SSIM: {np.mean(ssim_scores):.4f}")
print(f"Average PSNR: {np.mean(psnr_scores):.2f} dB")
