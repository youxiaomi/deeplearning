import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt

def get_mnist_dataloader(batch_size=64, train=True, normalize_range=(-1, 1), download=True):
    """
    获取MNIST数据加载器 / Get MNIST dataloader
    
    MNIST是一个手写数字数据集，包含60,000个训练样本和10,000个测试样本。
    每个图像是28x28像素的灰度图像，像素值范围为0-255。
    
    MNIST is a handwritten digit dataset with 60,000 training samples and 10,000 test samples.
    Each image is a 28x28 pixel grayscale image with pixel values ranging from 0-255.
    
    Args:
        batch_size: 批次大小 / Batch size
        train: 是否为训练集 / Whether to use training set
        normalize_range: 归一化范围 / Normalization range ((-1,1) for GAN, (0,1) for Autoencoder)
        download: 是否下载数据集 / Whether to download dataset
        
    Returns:
        dataloader: PyTorch数据加载器 / PyTorch dataloader
    """
    
    # 根据归一化范围设置变换 / Set transforms based on normalization range
    if normalize_range == (-1, 1):
        # GAN通常使用[-1, 1]范围 / GANs typically use [-1, 1] range
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # 将[0,1]映射到[-1,1] / Map [0,1] to [-1,1]
        ])
    elif normalize_range == (0, 1):
        # 自编码器通常使用[0, 1]范围 / Autoencoders typically use [0, 1] range
        transform = transforms.Compose([
            transforms.ToTensor()  # 自动将[0,255]映射到[0,1] / Automatically maps [0,255] to [0,1]
        ])
    else:
        raise ValueError("normalize_range must be (-1, 1) or (0, 1)")
    
    # 加载MNIST数据集 / Load MNIST dataset
    dataset = torchvision.datasets.MNIST(
        root='./data',
        train=train,
        transform=transform,
        download=download
    )
    
    # 创建数据加载器 / Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,  # 训练时打乱数据 / Shuffle data during training
        num_workers=2,
        pin_memory=True
    )
    
    return dataloader


def get_fashion_mnist_dataloader(batch_size=64, train=True, normalize_range=(-1, 1), download=True):
    """
    获取Fashion-MNIST数据加载器 / Get Fashion-MNIST dataloader
    
    Fashion-MNIST是MNIST的时尚版本，包含10种不同类型的服装和配饰。
    数据格式与MNIST相同，但内容更复杂，适合测试模型的生成能力。
    
    Fashion-MNIST is a fashion version of MNIST with 10 different types of clothing and accessories.
    The data format is the same as MNIST, but the content is more complex, suitable for testing model generation capabilities.
    
    Args:
        batch_size: 批次大小 / Batch size
        train: 是否为训练集 / Whether to use training set
        normalize_range: 归一化范围 / Normalization range
        download: 是否下载数据集 / Whether to download dataset
        
    Returns:
        dataloader: PyTorch数据加载器 / PyTorch dataloader
    """
    
    # 设置数据变换 / Set data transforms
    if normalize_range == (-1, 1):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    elif normalize_range == (0, 1):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
    else:
        raise ValueError("normalize_range must be (-1, 1) or (0, 1)")
    
    # 加载Fashion-MNIST数据集 / Load Fashion-MNIST dataset
    dataset = torchvision.datasets.FashionMNIST(
        root='./data',
        train=train,
        transform=transform,
        download=download
    )
    
    # 创建数据加载器 / Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=2,
        pin_memory=True
    )
    
    return dataloader


def get_cifar10_dataloader(batch_size=64, train=True, normalize_range=(-1, 1), download=True):
    """
    获取CIFAR-10数据加载器 / Get CIFAR-10 dataloader
    
    CIFAR-10是一个彩色图像数据集，包含10个类别的32x32像素图像。
    相比MNIST，CIFAR-10更具挑战性，因为它是彩色的且包含更复杂的场景。
    
    CIFAR-10 is a color image dataset with 32x32 pixel images in 10 categories.
    Compared to MNIST, CIFAR-10 is more challenging as it's in color and contains more complex scenes.
    
    Args:
        batch_size: 批次大小 / Batch size
        train: 是否为训练集 / Whether to use training set
        normalize_range: 归一化范围 / Normalization range
        download: 是否下载数据集 / Whether to download dataset
        
    Returns:
        dataloader: PyTorch数据加载器 / PyTorch dataloader
    """
    
    # 设置数据变换 / Set data transforms
    if normalize_range == (-1, 1):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 3通道归一化 / 3-channel normalization
        ])
    elif normalize_range == (0, 1):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
    else:
        raise ValueError("normalize_range must be (-1, 1) or (0, 1)")
    
    # 加载CIFAR-10数据集 / Load CIFAR-10 dataset
    dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=train,
        transform=transform,
        download=download
    )
    
    # 创建数据加载器 / Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=2,
        pin_memory=True
    )
    
    return dataloader


class CustomDataset(Dataset):
    """
    自定义数据集类 / Custom Dataset Class
    
    允许用户创建自己的数据集，用于特定的训练需求。
    Allows users to create their own dataset for specific training needs.
    """
    
    def __init__(self, data, labels=None, transform=None):
        """
        初始化自定义数据集
        Initialize custom dataset
        
        Args:
            data: 数据张量 / Data tensor
            labels: 标签张量 / Label tensor (可选 / optional)
            transform: 数据变换 / Data transforms (可选 / optional)
        """
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        """返回数据集大小 / Return dataset size"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """获取单个数据项 / Get single data item"""
        sample = self.data[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        if self.labels is not None:
            return sample, self.labels[idx]
        else:
            return sample


def denormalize_tensor(tensor, normalize_range=(-1, 1)):
    """
    反归一化张量用于可视化 / Denormalize tensor for visualization
    
    将归一化后的张量恢复到[0, 1]范围，以便正确显示图像。
    Restore normalized tensor to [0, 1] range for proper image display.
    
    Args:
        tensor: 归一化的张量 / Normalized tensor
        normalize_range: 原始归一化范围 / Original normalization range
        
    Returns:
        denormalized_tensor: 反归一化后的张量 / Denormalized tensor
    """
    if normalize_range == (-1, 1):
        # 从[-1, 1]恢复到[0, 1] / Restore from [-1, 1] to [0, 1]
        return (tensor + 1) / 2
    elif normalize_range == (0, 1):
        # 已经在[0, 1]范围内 / Already in [0, 1] range
        return tensor
    else:
        raise ValueError("normalize_range must be (-1, 1) or (0, 1)")


def visualize_batch(dataloader, num_samples=8, normalize_range=(-1, 1)):
    """
    可视化数据批次 / Visualize data batch
    
    显示数据加载器中的样本图像，用于验证数据加载是否正确。
    Display sample images from dataloader to verify data loading is correct.
    
    Args:
        dataloader: 数据加载器 / Dataloader
        num_samples: 显示的样本数量 / Number of samples to display
        normalize_range: 数据的归一化范围 / Normalization range of data
    """
    # 获取一个批次的数据 / Get one batch of data
    data_iter = iter(dataloader)
    batch = next(data_iter)
    
    if isinstance(batch, tuple):
        images, labels = batch
    else:
        images = batch
        labels = None
    
    # 反归一化图像 / Denormalize images
    images = denormalize_tensor(images, normalize_range)
    
    # 创建子图 / Create subplots
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(min(num_samples, len(images))):
        img = images[i]
        
        # 处理不同的图像格式 / Handle different image formats
        if img.shape[0] == 1:  # 灰度图像 / Grayscale image
            img = img.squeeze(0)
            axes[i].imshow(img, cmap='gray')
        elif img.shape[0] == 3:  # 彩色图像 / Color image
            img = img.permute(1, 2, 0)  # CHW -> HWC
            axes[i].imshow(img)
        
        axes[i].axis('off')
        
        # 如果有标签，显示标签 / If labels exist, display them
        if labels is not None:
            axes[i].set_title(f'Label: {labels[i].item()}')
    
    plt.tight_layout()
    plt.show()


def get_dataloader_info(dataloader):
    """
    获取数据加载器信息 / Get dataloader information
    
    显示数据加载器的基本统计信息。
    Display basic statistics of the dataloader.
    
    Args:
        dataloader: 数据加载器 / Dataloader
    """
    print("数据加载器信息 / Dataloader Information:")
    print(f"批次大小 / Batch size: {dataloader.batch_size}")
    print(f"数据集大小 / Dataset size: {len(dataloader.dataset)}")
    print(f"批次数量 / Number of batches: {len(dataloader)}")
    
    # 获取一个样本来检查数据形状 / Get one sample to check data shape
    data_iter = iter(dataloader)
    batch = next(data_iter)
    
    if isinstance(batch, tuple):
        images, labels = batch
        print(f"图像形状 / Image shape: {images.shape}")
        print(f"标签形状 / Label shape: {labels.shape}")
        print(f"图像数据类型 / Image dtype: {images.dtype}")
        print(f"图像值范围 / Image value range: [{images.min():.3f}, {images.max():.3f}]")
    else:
        images = batch
        print(f"图像形状 / Image shape: {images.shape}")
        print(f"图像数据类型 / Image dtype: {images.dtype}")
        print(f"图像值范围 / Image value range: [{images.min():.3f}, {images.max():.3f}]")


if __name__ == "__main__":
    # 测试数据加载器 / Test dataloaders
    print("测试MNIST数据加载器 / Testing MNIST Dataloader")
    
    # 测试MNIST / Test MNIST
    mnist_loader = get_mnist_dataloader(batch_size=32, train=True, normalize_range=(-1, 1))
    get_dataloader_info(mnist_loader)
    
    print("\n测试Fashion-MNIST数据加载器 / Testing Fashion-MNIST Dataloader")
    fashion_loader = get_fashion_mnist_dataloader(batch_size=32, train=True, normalize_range=(-1, 1))
    get_dataloader_info(fashion_loader)
    
    # 测试自定义数据集 / Test custom dataset
    print("\n测试自定义数据集 / Testing Custom Dataset")
    random_data = torch.randn(100, 1, 28, 28)
    random_labels = torch.randint(0, 10, (100,))
    custom_dataset = CustomDataset(random_data, random_labels)
    custom_loader = DataLoader(custom_dataset, batch_size=16, shuffle=True)
    get_dataloader_info(custom_loader) 