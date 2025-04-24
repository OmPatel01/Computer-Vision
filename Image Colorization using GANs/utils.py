from torchvision.utils import save_image

def save_output_image(tensor, path):
    # Denormalize the image from [-1, 1] to [0, 1]
    denorm = (tensor + 1) / 2  # Normalize to the range [0, 1]
    save_image(denorm, path)
