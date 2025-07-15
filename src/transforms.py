from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F

from torchvision.transforms import Compose, CenterCrop, Normalize, \
    RandomCrop, RandomHorizontalFlip


class ToTensor(nn.Module):
    """
    - `img`: `(h, w, c)`, 0 ~ 255, uint8 -> `(1, c, h, w)`, 0 ~ 1.0, float32
    """
    def __init__(self) -> None:
        super(ToTensor, self).__init__()
    
    def forward(
        self, 
        img: np.ndarray,
    ) -> Tuple[torch.Tensor, int]:
        img = np.transpose(img, (2, 0, 1))[np.newaxis, ...] / 255.0
        img = torch.as_tensor(img, dtype = torch.float32)
        return img

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
        imgs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args
        - `imgs`: `torch.Tensor`, shape `(n, c, h, w)`
        """
        org_h, org_w = imgs.shape[-2:]
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
        self.eigenvalues = eigenvalues.reshape(3, 1)
    
    def forward(
        self, 
        imgs: torch.Tensor,
    ) -> torch.Tensor:
        alpha = torch.normal(0, 0.01, (3, 1))
        noise = self.eigenvalues * alpha
        noise = torch.matmul(self.eigenvectors, noise)
        new_imgs = imgs + noise.view(-1, 3, 1, 1)
        return new_imgs
    
class StackHorizontalFlip(nn.Module):
    def __init__(self) -> None:
        super(StackHorizontalFlip, self).__init__()
    
    def forward(
        self, 
        imgs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args
        - `imgs`: `torch.Tensor`, shape `(n, c, h, w)`
        - `cat_ids`: `torch.Tensor`, shape `(n, )`

        Returns
        - `new_imgs`: `torch.Tensor`, shape `(n * 2, c, h, w)`
        - `new_cat_ids`: `torch.Tensor`, shape `(n * 2, )`
        """
        imgs_flip = F.hflip(imgs)
        new_imgs = torch.concat([imgs, imgs_flip], dim = 0)
        return new_imgs

class StackCornerCenterCrop(nn.Module):
    def __init__(
        self, 
        crop_size: int
    ) -> None:
        super(StackCornerCenterCrop, self).__init__()
        self.crop_size = crop_size
    
    def forward(
        self, 
        imgs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args
        - `imgs`: `torch.Tensor`, shape `(n, c, h, w)`
        - `cat_ids`: `torch.Tensor`, shape `(n, )`

        Returns
        - `new_imgs`: `torch.Tensor`, shape `(n * 5, c, h, w)`
        - `new_cat_ids`: `torch.Tensor`, shape `(n * 5, )`
        """
        new_imgs = F.five_crop(imgs, self.crop_size)
        new_imgs = torch.concat(new_imgs, dim = 0)
        return new_imgs


def build_train_transform(
    channel_mean: torch.Tensor,
    channel_std: torch.Tensor,
    eigenvectors: torch.Tensor,
    eigenvalues: torch.Tensor
) -> Compose:
    """
    Args
    - `channel_mean`: `torch.Tensor`, shape `(3, )`,
    - `channel_std`: `torch.Tensor`, shape `(3, )`,
    - `eigenvectors`: `torch.Tensor`, shape `(3, 3)`, `(3, channel)`
    - `eigenvalues`: `torch.Tensor`, shape `(3, 1)`
    """
    short_edge_size = 256
    center_crop_size = 256
    random_crop_size = 224
    horizontal_flip_prob = 0.5

    to_tensor = ToTensor()
    normalize = Normalize(channel_mean, channel_std)
    short_edge_resize = ShortEdgeResize(short_edge_size)
    center_crop = CenterCrop(center_crop_size)
    random_crop = RandomCrop(random_crop_size)
    random_horizontal_flip = RandomHorizontalFlip(horizontal_flip_prob)
    pca_random_noise = PCARandomNoise(eigenvectors, eigenvalues)

    train_transform = Compose([
        to_tensor,
        normalize,
        short_edge_resize,
        center_crop,
        random_crop,
        random_horizontal_flip,
        pca_random_noise
    ])

    return train_transform

def build_test_transfrom(
    channel_mean: torch.Tensor,
    channel_std: torch.Tensor
) -> Compose:
    short_edge_size = 256
    center_crop_size = 224

    to_tensor = ToTensor()
    normalize = Normalize(channel_mean, channel_std)
    short_edge_resize = ShortEdgeResize(short_edge_size)
    center_crop = CenterCrop(center_crop_size)

    test_transform = Compose([
        to_tensor,
        normalize,
        short_edge_resize,
        center_crop
    ])

    return test_transform

def build_test_transfrom_tta(
    channel_mean: torch.Tensor,
    channel_std: torch.Tensor
) -> Compose:
    short_edge_size = 256
    corner_center_crop_size = 224

    to_tensor = ToTensor()
    normalize = Normalize(channel_mean, channel_std)
    short_edge_resize = ShortEdgeResize(short_edge_size)
    horizontal_flip = StackHorizontalFlip()
    corner_center_crop = StackCornerCenterCrop(corner_center_crop_size)

    test_transform = Compose([
        to_tensor,
        normalize,
        short_edge_resize,
        horizontal_flip,
        corner_center_crop,
    ])

    return test_transform