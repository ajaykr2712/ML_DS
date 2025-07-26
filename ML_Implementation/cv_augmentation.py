"""
Computer Vision Data Augmentation Pipeline
Advanced data augmentation techniques for computer vision tasks
"""

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance, ImageFilter
from typing import List, Dict, Tuple, Optional, Callable, Union
import random
import logging
from dataclasses import dataclass
import json
from pathlib import Path

@dataclass
class AugmentationConfig:
    """Configuration for augmentation pipeline"""
    image_size: Tuple[int, int] = (224, 224)
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    geometric_prob: float = 0.5
    color_prob: float = 0.5
    noise_prob: float = 0.3
    blur_prob: float = 0.2
    cutout_prob: float = 0.3
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0

class GeometricAugmentations:
    """Geometric transformation augmentations"""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def get_geometric_transforms(self) -> A.Compose:
        """Get geometric augmentation pipeline"""
        transforms = [
            A.RandomRotate90(p=0.3),
            A.Rotate(limit=15, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.5
            ),
            A.OpticalDistortion(
                distort_limit=0.2,
                shift_limit=0.1,
                p=0.3
            ),
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.2,
                p=0.3
            ),
            A.ElasticTransform(
                alpha=1,
                sigma=50,
                alpha_affine=50,
                p=0.2
            ),
            A.Perspective(scale=(0.05, 0.1), p=0.3),
            A.RandomCrop(
                height=self.config.image_size[0],
                width=self.config.image_size[1],
                p=0.5
            )
        ]
        
        return A.Compose(transforms, p=self.config.geometric_prob)

class ColorAugmentations:
    """Color and brightness augmentations"""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def get_color_transforms(self) -> A.Compose:
        """Get color augmentation pipeline"""
        transforms = [
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.5
            ),
            A.ChannelShuffle(p=0.2),
            A.RGBShift(
                r_shift_limit=20,
                g_shift_limit=20,
                b_shift_limit=20,
                p=0.3
            ),
            A.RandomGamma(
                gamma_limit=(80, 120),
                p=0.3
            ),
            A.CLAHE(
                clip_limit=2,
                tile_grid_size=(8, 8),
                p=0.3
            ),
            A.ToGray(p=0.1),
            A.ToSepia(p=0.1)
        ]
        
        return A.Compose(transforms, p=self.config.color_prob)

class NoiseAugmentations:
    """Noise and quality degradation augmentations"""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def get_noise_transforms(self) -> A.Compose:
        """Get noise augmentation pipeline"""
        transforms = [
            A.GaussNoise(
                var_limit=(10.0, 50.0),
                p=0.4
            ),
            A.ISONoise(
                color_shift=(0.01, 0.05),
                intensity=(0.1, 0.5),
                p=0.3
            ),
            A.MultiplicativeNoise(
                multiplier=[0.9, 1.1],
                p=0.3
            ),
            A.ImageCompression(
                quality_lower=60,
                quality_upper=100,
                p=0.3
            ),
            A.Downscale(
                scale_min=0.5,
                scale_max=0.99,
                p=0.2
            ),
            A.Sharpen(
                alpha=(0.2, 0.5),
                lightness=(0.5, 1.0),
                p=0.3
            ),
            A.Emboss(
                alpha=(0.2, 0.5),
                strength=(0.2, 0.7),
                p=0.2
            )
        ]
        
        return A.Compose(transforms, p=self.config.noise_prob)

class BlurAugmentations:
    """Blur and smoothing augmentations"""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def get_blur_transforms(self) -> A.Compose:
        """Get blur augmentation pipeline"""
        transforms = [
            A.Blur(
                blur_limit=7,
                p=0.4
            ),
            A.MotionBlur(
                blur_limit=7,
                p=0.3
            ),
            A.MedianBlur(
                blur_limit=7,
                p=0.3
            ),
            A.GaussianBlur(
                blur_limit=7,
                p=0.4
            ),
            A.ZoomBlur(
                max_factor=1.1,
                p=0.2
            ),
            A.Defocus(
                radius=(3, 10),
                alias_blur=(0.1, 0.5),
                p=0.2
            )
        ]
        
        return A.Compose(transforms, p=self.config.blur_prob)

class CutoutAugmentations:
    """Cutout and masking augmentations"""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def get_cutout_transforms(self) -> A.Compose:
        """Get cutout augmentation pipeline"""
        transforms = [
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                min_holes=1,
                min_height=8,
                min_width=8,
                fill_value=0,
                p=0.5
            ),
            A.GridDropout(
                ratio=0.5,
                unit_size_min=None,
                unit_size_max=None,
                holes_number_x=None,
                holes_number_y=None,
                shift_x=0,
                shift_y=0,
                random_offset=False,
                fill_value=0,
                mask_fill_value=None,
                p=0.3
            ),
            A.MaskDropout(
                max_objects=1,
                image_fill_value=0,
                mask_fill_value=0,
                p=0.2
            )
        ]
        
        return A.Compose(transforms, p=self.config.cutout_prob)

class AdvancedAugmentations:
    """Advanced augmentation techniques"""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def mixup(self, batch_x: torch.Tensor, batch_y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Apply MixUp augmentation
        
        Args:
            batch_x: Input batch of images
            batch_y: Input batch of labels
            
        Returns:
            Mixed images, mixed labels, and lambda parameter
        """
        if self.config.mixup_alpha > 0:
            lam = np.random.beta(self.config.mixup_alpha, self.config.mixup_alpha)
        else:
            lam = 1
        
        batch_size = batch_x.size(0)
        index = torch.randperm(batch_size)
        
        mixed_x = lam * batch_x + (1 - lam) * batch_x[index, :]
        y_a, y_b = batch_y, batch_y[index]
        
        return mixed_x, (y_a, y_b), lam
    
    def cutmix(self, batch_x: torch.Tensor, batch_y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Apply CutMix augmentation
        
        Args:
            batch_x: Input batch of images
            batch_y: Input batch of labels
            
        Returns:
            Mixed images, mixed labels, and lambda parameter
        """
        if self.config.cutmix_alpha > 0:
            lam = np.random.beta(self.config.cutmix_alpha, self.config.cutmix_alpha)
        else:
            lam = 1
        
        batch_size = batch_x.size(0)
        index = torch.randperm(batch_size)
        
        # Generate random bounding box
        W, H = batch_x.size(3), batch_x.size(2)
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Apply cutmix
        batch_x[:, :, bby1:bby2, bbx1:bbx2] = batch_x[index, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda to match the exact area ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (batch_x.size()[-1] * batch_x.size()[-2]))
        
        y_a, y_b = batch_y, batch_y[index]
        return batch_x, (y_a, y_b), lam
    
    def random_erasing(self, img: torch.Tensor, probability: float = 0.5, 
                      area_ratio_range: Tuple[float, float] = (0.02, 0.33),
                      aspect_ratio_range: Tuple[float, float] = (0.3, 3.3)) -> torch.Tensor:
        """
        Apply Random Erasing augmentation
        
        Args:
            img: Input image tensor
            probability: Probability of applying random erasing
            area_ratio_range: Range of area ratio to erase
            aspect_ratio_range: Range of aspect ratio for erased area
            
        Returns:
            Augmented image tensor
        """
        if random.random() > probability:
            return img
        
        for _ in range(100):  # Max attempts
            area = img.size(1) * img.size(2)
            target_area = random.uniform(*area_ratio_range) * area
            aspect_ratio = random.uniform(*aspect_ratio_range)
            
            h = int(round(np.sqrt(target_area * aspect_ratio)))
            w = int(round(np.sqrt(target_area / aspect_ratio)))
            
            if w < img.size(2) and h < img.size(1):
                x1 = random.randint(0, img.size(1) - h)
                y1 = random.randint(0, img.size(2) - w)
                
                # Fill with random values
                img[:, x1:x1+h, y1:y1+w] = torch.randn(img.size(0), h, w)
                break
        
        return img

class AugmentationPipeline:
    """Complete augmentation pipeline manager"""
    
    def __init__(self, config: AugmentationConfig, mode: str = 'train'):
        """
        Initialize augmentation pipeline
        
        Args:
            config: Augmentation configuration
            mode: 'train' or 'val' mode
        """
        self.config = config
        self.mode = mode
        self.logger = logging.getLogger(__name__)
        
        # Initialize augmentation modules
        self.geometric_aug = GeometricAugmentations(config)
        self.color_aug = ColorAugmentations(config)
        self.noise_aug = NoiseAugmentations(config)
        self.blur_aug = BlurAugmentations(config)
        self.cutout_aug = CutoutAugmentations(config)
        self.advanced_aug = AdvancedAugmentations(config)
        
        # Build pipeline
        self.transform = self._build_pipeline()
    
    def _build_pipeline(self) -> A.Compose:
        """Build the complete augmentation pipeline"""
        transforms = []
        
        if self.mode == 'train':
            # Add all augmentations for training
            transforms.extend([
                A.Resize(
                    height=self.config.image_size[0],
                    width=self.config.image_size[1]
                ),
                A.OneOf([
                    self.geometric_aug.get_geometric_transforms(),
                    A.NoOp()
                ], p=1.0),
                A.OneOf([
                    self.color_aug.get_color_transforms(),
                    A.NoOp()
                ], p=1.0),
                A.OneOf([
                    self.noise_aug.get_noise_transforms(),
                    A.NoOp()
                ], p=1.0),
                A.OneOf([
                    self.blur_aug.get_blur_transforms(),
                    A.NoOp()
                ], p=1.0),
                A.OneOf([
                    self.cutout_aug.get_cutout_transforms(),
                    A.NoOp()
                ], p=1.0)
            ])
        else:
            # Only resize for validation
            transforms.append(
                A.Resize(
                    height=self.config.image_size[0],
                    width=self.config.image_size[1]
                )
            )
        
        # Add normalization and tensor conversion
        transforms.extend([
            A.Normalize(
                mean=self.config.mean,
                std=self.config.std
            ),
            ToTensorV2()
        ])
        
        return A.Compose(transforms)
    
    def __call__(self, image: Union[np.ndarray, Image.Image]) -> torch.Tensor:
        """Apply augmentation pipeline to image"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Apply transformations
        augmented = self.transform(image=image)
        return augmented['image']
    
    def get_test_time_augmentation(self, image: Union[np.ndarray, Image.Image], 
                                  n_augments: int = 5) -> List[torch.Tensor]:
        """
        Generate multiple augmented versions for test-time augmentation
        
        Args:
            image: Input image
            n_augments: Number of augmented versions to generate
            
        Returns:
            List of augmented image tensors
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        augmented_images = []
        
        # Original image (just resize and normalize)
        base_transform = A.Compose([
            A.Resize(
                height=self.config.image_size[0],
                width=self.config.image_size[1]
            ),
            A.Normalize(
                mean=self.config.mean,
                std=self.config.std
            ),
            ToTensorV2()
        ])
        
        augmented_images.append(base_transform(image=image)['image'])
        
        # Generate augmented versions
        tta_transform = A.Compose([
            A.Resize(
                height=self.config.image_size[0],
                width=self.config.image_size[1]
            ),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(
                mean=self.config.mean,
                std=self.config.std
            ),
            ToTensorV2()
        ])
        
        for _ in range(n_augments - 1):
            augmented = tta_transform(image=image)
            augmented_images.append(augmented['image'])
        
        return augmented_images
    
    def visualize_augmentations(self, image: Union[np.ndarray, Image.Image], 
                               n_examples: int = 8) -> List[np.ndarray]:
        """
        Generate examples of augmentations for visualization
        
        Args:
            image: Input image
            n_examples: Number of examples to generate
            
        Returns:
            List of augmented images as numpy arrays
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        examples = []
        
        for _ in range(n_examples):
            # Apply augmentations without normalization for visualization
            viz_transform = A.Compose([
                A.Resize(
                    height=self.config.image_size[0],
                    width=self.config.image_size[1]
                ),
                self.geometric_aug.get_geometric_transforms(),
                self.color_aug.get_color_transforms(),
                self.noise_aug.get_noise_transforms(),
                self.cutout_aug.get_cutout_transforms()
            ])
            
            augmented = viz_transform(image=image)
            examples.append(augmented['image'])
        
        return examples

# Example usage
if __name__ == "__main__":
    # Configuration
    config = AugmentationConfig(
        image_size=(224, 224),
        geometric_prob=0.7,
        color_prob=0.6,
        noise_prob=0.4,
        blur_prob=0.3,
        cutout_prob=0.4
    )
    
    # Create pipeline
    train_pipeline = AugmentationPipeline(config, mode='train')
    val_pipeline = AugmentationPipeline(config, mode='val')
    
    # Load and augment an image
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Apply training augmentations
    augmented_train = train_pipeline(image)
    print(f"Augmented training image shape: {augmented_train.shape}")
    
    # Apply validation augmentations
    augmented_val = val_pipeline(image)
    print(f"Augmented validation image shape: {augmented_val.shape}")
    
    # Test-time augmentation
    tta_images = train_pipeline.get_test_time_augmentation(image, n_augments=5)
    print(f"TTA generated {len(tta_images)} images")
    
    # Visualize augmentations
    examples = train_pipeline.visualize_augmentations(image, n_examples=4)
    print(f"Generated {len(examples)} visualization examples")
    
    print("Augmentation pipeline test completed successfully")
