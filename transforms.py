import os
from typing import List, Union, Dict, Tuple

import PIL.Image as pil_img
import torch
import torch.nn as nn
import torchvision.transforms.functional as F

from torchvision.transforms import Compose, CenterCrop, \
    Normalize, RandomCrop, RandomHorizontalFlip


class ShortEdgeResize(nn.Module):
    """
    Resize an image to make the shrot edge equal to the specified size
    """
    def __init__(
        self,
        short_edge_size: int
    ) -> None:
        super(ShortEdgeResize, self).__init__()
        self.short_edge_size = short_edge_size
    
    def forward(
        self, 
        imgs: torch.Tensor
    ) -> torch.Tensor:
        """
        Args
        - `imgs`: `torch.Tensor`, shape `(..., h, w)`
        """
        org_h, org_w = imgs.height, imgs.width
        aspect_ratio = org_h / org_w

        if org_h > org_w:
            new_w = self.short_edge_size
            new_h = int(aspect_ratio * new_w)
        else:
            new_h = self.short_edge_size
            new_w = int(new_h / aspect_ratio)
        
        new_imgs = F.resize(imgs, (new_h, new_w))

        return new_imgs
    
class PCARandomNoise(nn.Module):
    def __init__(
        self, 
        eigenvectors: torch.Tensor,
        eigenvalues: torch.Tensor
    ) -> None:
        """
        Args
        - `eigenvectors`: `torch.Tensor`, shape `(3, 3)`, `(3, channel)`
        - `eigenvalues`: `torch.Tensor`, shape `(3, 1)`
        """
        super(PCARandomNoise, self).__init__()
        self.eigenvectors = eigenvectors
        self.eigenvalues = eigenvalues
    
    def forward(
        self, 
        imgs: torch.Tensor
    ) -> torch.Tensor:
        alpha = torch.normal(0, 0.01, (3, 1))
        noise = self.eigenvalues * alpha
        noise = torch.matmul(self.eigenvectors, noise)
        new_imgs = imgs + noise
        return new_imgs
    
class HorizontalFlip(nn.Module):
    def __init__(self) -> None:
        super(HorizontalFlip, self).__init__()
    
    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Args
        - `imgs`: `torch.Tensor`, shape `(n, c, h, w)`

        Returns
        - `new_imgs`: `torch.Tensor`, shape `(n * 2, c, h, w)`
        """
        imgs_flip = F.hflip(imgs)
        new_imgs = torch.concat([imgs, imgs_flip], dim = 0)
        return new_imgs

class CornerCenterCrop(nn.Module):
    def __init__(
        self, 
        crop_size: int
    ) -> None:
        super(CornerCenterCrop, self).__init__()
        self.crop_size = crop_size
    
    def forward(self, imgs) -> torch.Tensor:
        """
        Args
        - `imgs`: `torch.Tensor`, shape `(b, c, h, w)`

        Returns
        - `new_imgs`: `torch.Tensor`, shape `(5 * b, c, h, w)`
        """
        new_imgs = F.five_crop(imgs, self.crop_size)
        new_imgs = torch.concat(new_imgs, dim = 0)
        return new_imgs


def build_train_transform(
    normalize_channel_mean: torch.Tensor,
    eigenvectors: torch.Tensor,
    eigenvalues: torch.Tensor
) -> Compose:
    """
    Args
    - `normalize_channel_mean`: `torch.Tensor`, shape `(3, )`
    - `eigenvectors`: `torch.Tensor`, shape `(3, 3)`, `(3, channel)`
    - `eigenvalues`: `torch.Tensor`, shape `(3, 1)`
    """
    short_edge_size = 256
    center_crop_size = 256
    random_crop_size = 224
    horizontal_flip_prob = 0.5
    normalize_channel_std = torch.ones(3)

    short_edge_resize = ShortEdgeResize(short_edge_size)
    center_crop = CenterCrop(center_crop_size)
    normalize = Normalize(normalize_channel_mean, normalize_channel_std)
    random_crop = RandomCrop(random_crop_size)
    random_horizontal_flip = RandomHorizontalFlip(horizontal_flip_prob)
    pca_random_noise = PCARandomNoise(eigenvectors, eigenvalues)

    train_transform = Compose([
        short_edge_resize,
        center_crop,
        normalize,
        random_crop,
        random_horizontal_flip,
        pca_random_noise
    ])

    return train_transform

def build_test_loader(
    normalize_channel_mean: torch.Tensor,
) -> Compose:
    short_edge_size = 256
    corner_center_crop_size = 224
    normalize_channel_std = torch.ones(3)

    short_edge_resize = ShortEdgeResize(short_edge_size)
    horizontal_flip = HorizontalFlip()
    corner_center_crop = CornerCenterCrop(corner_center_crop_size)
    normalize = Normalize(normalize_channel_mean, normalize_channel_std)

    test_transform = Compose([
        short_edge_resize,
        horizontal_flip,
        corner_center_crop,
        normalize
    ])

    return test_transform