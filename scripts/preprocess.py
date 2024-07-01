from torchvision import transforms
import torch
h = 210
w = 210
transform1 = transforms.Compose([
        transforms.ConvertImageDtype(torch.float),
        transforms.Resize((h,w), antialias=True),  # (h,w)
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(90),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomCrop((h,w), pad_if_needed=True, padding_mode='edge'),    # (h,w)
    ])
transform2 = transforms.Compose([
        transforms.ConvertImageDtype(torch.float),
        transforms.Resize((h,w), antialias=True),  # (h,w)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomCrop((h,w), pad_if_needed=True, padding_mode='edge'),
    ])
