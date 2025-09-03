import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

MEAN = torch.tensor((0.48145466, 0.4578275 , 0.40821073))
STD  = torch.tensor((0.26862954, 0.26130258, 0.27577711))


# -----------------------------------------------------------------------------------------------
def normalize(img_tensor):
    img_tensor = img_tensor.squeeze(0)
    mean = MEAN.to(img_tensor.device).view(-1, 1, 1)
    std = STD.to(img_tensor.device).view(-1, 1, 1)
    img_tensor = (img_tensor - mean) / std
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor


# -----------------------------------------------------------------------------------------------
def denormalize(img_tensor):
    img_tensor = img_tensor.squeeze(0)
    mean = MEAN.to(img_tensor.device).view(-1, 1, 1)
    std = STD.to(img_tensor.device).view(-1, 1, 1)
    img_tensor = (img_tensor * std) + mean
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor


# -----------------------------------------------------------------------------------------------
def display_img(orig_img_tensor, adv_img_tensor=None):

    def convert(img_tensor):
        img_tensor = img_tensor.squeeze(0).cpu()
        img_np = img_tensor.numpy().transpose(1, 2, 0)  # out: (H, W, C)
        img_np = np.clip(img_np, 0, 1)
        img_pil = Image.fromarray((img_np * 255).astype('uint8'))
        return img_pil

    if adv_img_tensor is not None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # plot the original image
        orig_pil = convert(orig_img_tensor)
        axes[0].imshow(orig_pil)
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        # plot the adversarial image
        adv_pil = convert(adv_img_tensor)
        axes[1].imshow(adv_pil)
        axes[1].set_title("Adversarial Image")
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()
    else:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

        # Plot the original image
        orig_pil = convert(orig_img_tensor)
        ax.imshow(orig_pil)
        ax.set_title("Original Image")
        ax.axis('off')

        plt.tight_layout()
        plt.show()


# -----------------------------------------------------------------------------------------------
def preprocess_img(img_path, img_size):
    raw_img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((img_size,img_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])
    img = transform(raw_img).unsqueeze(0)
    return img


# -----------------------------------------------------------------------------------------------
def num_of_parm(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 10 ** 6:.2f}M")


# -----------------------------------------------------------------------------------------------
def plot_loss(iters, loss, title):
    plt.figure(figsize=(8, 5))
    plt.plot(iters, loss, marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
