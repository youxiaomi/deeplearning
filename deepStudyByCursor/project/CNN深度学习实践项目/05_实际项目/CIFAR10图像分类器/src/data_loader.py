"""
CIFAR-10 Data Loader Module
CIFAR-10 数据加载器模块

This module handles loading and preprocessing CIFAR-10 data.
Think of this as a chef preparing ingredients before cooking!
这个模块处理CIFAR-10数据的加载和预处理。
把它想象成厨师在烹饪前准备食材！
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

# CIFAR-10 class names (10 categories we want to recognize)
# CIFAR-10类别名称（我们想要识别的10个类别）
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

CIFAR10_CLASSES_CN = [
    '飞机', '汽车', '鸟', '猫', '鹿',
    '狗', '青蛙', '马', '船', '卡车'
]

class CIFAR10DataLoader:
    """
    CIFAR-10 Data Loader Class
    CIFAR-10数据加载器类
    
    This class is like a smart librarian that organizes and prepares
    images for our neural network to learn from.
    这个类就像一个聪明的图书管理员，为我们的神经网络
    整理和准备图像供学习使用。
    """
    
    def __init__(self, data_dir='./data', batch_size=32, validation_split=0.1):
        """
        Initialize the data loader
        初始化数据加载器
        
        Args:
            data_dir: Directory to store dataset (存储数据集的目录)
            batch_size: How many images to process at once (一次处理多少张图像)
            validation_split: Fraction of training data for validation (用于验证的训练数据比例)
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.validation_split = validation_split
        
        # Calculate mean and std for normalization
        # 计算用于归一化的均值和标准差
        # These values are pre-calculated for CIFAR-10
        # 这些值是为CIFAR-10预先计算的
        self.mean = [0.4914, 0.4822, 0.4465]
        self.std = [0.2023, 0.1994, 0.2010]
        
        self._setup_transforms()
        self._load_datasets()
    
    def _setup_transforms(self):
        """
        Set up image transformations
        设置图像变换
        
        Think of transforms as different ways to look at the same photo:
        把变换想象成看同一张照片的不同方式：
        - Rotation (旋转)
        - Flipping (翻转)
        - Color changes (颜色变化)
        """
        
        # Basic transform for testing (no augmentation)
        # 测试用的基本变换（无增强）
        self.basic_transform = transforms.Compose([
            transforms.ToTensor(),  # Convert to tensor (转换为张量)
            transforms.Normalize(self.mean, self.std)  # Normalize (归一化)
        ])
        
        # Training transform with data augmentation
        # 训练用的变换，包含数据增强
        # Data augmentation is like showing the same object from different angles
        # 数据增强就像从不同角度展示同一个物体
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),  # 50% chance to flip (50%概率翻转)
            transforms.RandomRotation(10),  # Rotate up to 10 degrees (最多旋转10度)
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
    
    def _load_datasets(self):
        """
        Load CIFAR-10 datasets
        加载CIFAR-10数据集
        """
        print("Loading CIFAR-10 dataset... (This may take a while for first time)")
        print("加载CIFAR-10数据集...（第一次可能需要一些时间）")
        
        # Load training data with augmentation
        # 加载带增强的训练数据
        train_dataset_full = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self.train_transform
        )
        
        # Load test data without augmentation
        # 加载不带增强的测试数据
        self.test_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self.basic_transform
        )
        
        # Split training data into train and validation
        # 将训练数据分割为训练集和验证集
        train_size = int((1 - self.validation_split) * len(train_dataset_full))
        val_size = len(train_dataset_full) - train_size
        
        self.train_dataset, self.val_dataset = random_split(
            train_dataset_full, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility (保证可重现性)
        )
        
        # Create validation dataset with basic transform
        # 创建使用基本变换的验证数据集
        val_dataset_basic = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=False,
            transform=self.basic_transform
        )
        
        # Get validation indices and create subset
        # 获取验证索引并创建子集
        val_indices = self.val_dataset.indices
        self.val_dataset = torch.utils.data.Subset(val_dataset_basic, val_indices)
        
        print(f"Dataset loaded successfully!")
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)}")
        print(f"Test samples: {len(self.test_dataset)}")
        print(f"数据集加载成功！")
        print(f"训练样本: {len(self.train_dataset)}")
        print(f"验证样本: {len(self.val_dataset)}")
        print(f"测试样本: {len(self.test_dataset)}")
    
    def get_data_loaders(self):
        """
        Get data loaders for training, validation, and testing
        获取训练、验证和测试的数据加载器
        
        Returns:
            tuple: (train_loader, val_loader, test_loader)
        """
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # Shuffle for better training (打乱以获得更好的训练效果)
            num_workers=2,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def visualize_samples(self, num_samples=16):
        """
        Visualize sample images from each class
        可视化每个类别的样本图像
        
        Args:
            num_samples: Number of samples to show (显示的样本数量)
        """
        # Get one batch of training data
        # 获取一批训练数据
        train_loader, _, _ = self.get_data_loaders()
        data_iter = iter(train_loader)
        images, labels = next(data_iter)
        
        # Denormalize images for visualization
        # 反归一化图像以便可视化
        def denormalize(tensor):
            for t, m, s in zip(tensor, self.mean, self.std):
                t.mul_(s).add_(m)
            return torch.clamp(tensor, 0, 1)
        
        # Create subplot
        # 创建子图
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        fig.suptitle('CIFAR-10 Sample Images / CIFAR-10样本图像', fontsize=16)
        
        for i in range(min(num_samples, len(images))):
            row, col = i // 4, i % 4
            
            # Denormalize and convert to numpy
            # 反归一化并转换为numpy
            img = denormalize(images[i].clone())
            img = img.permute(1, 2, 0).numpy()
            
            axes[row, col].imshow(img)
            axes[row, col].set_title(
                f'{CIFAR10_CLASSES[labels[i]]}\n{CIFAR10_CLASSES_CN[labels[i]]}',
                fontsize=10
            )
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def get_class_distribution(self):
        """
        Show class distribution in the dataset
        显示数据集中的类别分布
        """
        # Count samples per class in training set
        # 统计训练集中每个类别的样本数
        class_counts = torch.zeros(10)
        train_loader, _, _ = self.get_data_loaders()
        
        for _, labels in train_loader:
            for label in labels:
                class_counts[label] += 1
        
        # Create bar plot
        # 创建条形图
        plt.figure(figsize=(12, 6))
        x = range(10)
        plt.bar(x, class_counts.numpy())
        plt.xlabel('Classes / 类别')
        plt.ylabel('Number of Samples / 样本数量')
        plt.title('CIFAR-10 Class Distribution / CIFAR-10类别分布')
        plt.xticks(x, [f'{en}\n{cn}' for en, cn in zip(CIFAR10_CLASSES, CIFAR10_CLASSES_CN)])
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        print("Class distribution / 类别分布:")
        for i, (en, cn) in enumerate(zip(CIFAR10_CLASSES, CIFAR10_CLASSES_CN)):
            print(f"{en} ({cn}): {int(class_counts[i])} samples")


def test_data_loader():
    """
    Test function to verify data loader works correctly
    测试函数以验证数据加载器是否正常工作
    """
    print("Testing CIFAR-10 Data Loader...")
    print("测试CIFAR-10数据加载器...")
    
    # Create data loader
    # 创建数据加载器
    data_loader = CIFAR10DataLoader(batch_size=32)
    
    # Get data loaders
    # 获取数据加载器
    train_loader, val_loader, test_loader = data_loader.get_data_loaders()
    
    # Test one batch
    # 测试一个批次
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"批次形状: {images.shape}")
        print(f"标签形状: {labels.shape}")
        print(f"图像范围: [{images.min():.3f}, {images.max():.3f}]")
        break
    
    # Visualize samples
    # 可视化样本
    data_loader.visualize_samples()
    
    # Show class distribution
    # 显示类别分布
    data_loader.get_class_distribution()
    
    print("Data loader test completed successfully!")
    print("数据加载器测试成功完成！")


if __name__ == "__main__":
    test_data_loader() 