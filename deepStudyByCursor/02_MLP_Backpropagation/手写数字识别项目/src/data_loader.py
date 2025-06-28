"""
MNIST Data Loader
MNIST数据加载器

This module handles downloading, preprocessing, and loading the MNIST dataset.
本模块处理MNIST数据集的下载、预处理和加载。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch

class MNISTDataLoader:
    """
    MNIST Data Loader Class
    MNIST数据加载器类
    
    Handles data downloading, preprocessing, and batch loading.
    处理数据下载、预处理和批量加载。
    """
    
    def __init__(self, data_dir='../data', batch_size=64, validation_split=0.1):
        """
        Initialize the data loader
        初始化数据加载器
        
        Args:
            data_dir (str): Directory to store data (数据存储目录)
            batch_size (int): Batch size for training (训练批大小)
            validation_split (float): Fraction for validation set (验证集比例)
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.raw_dir = os.path.join(data_dir, 'raw')
        self.processed_dir = os.path.join(data_dir, 'processed')
        
        # Create directories
        # 创建目录
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Data transformations
        # 数据变换
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Convert to tensor (转换为张量)
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization (MNIST标准化)
        ])
    
    def download_data(self):
        """
        Download MNIST dataset
        下载MNIST数据集
        """
        print("Downloading MNIST dataset... (正在下载MNIST数据集...)")
        
        # Download training data (下载训练数据)
        # Check if data already exists (检查数据是否已存在)

      
        train_dataset =  datasets.MNIST(
            root=self.raw_dir,
            train=True,
            download=True,  # Only download if data doesn't exist (仅在数据不存在时下载)
            transform=self.transform
        )
        
        # Download test data (下载测试数据)
        test_dataset = datasets.MNIST(
            root=self.raw_dir,
            train=False,
            download=True,
            transform=self.transform
        )
        
        print("✅ MNIST dataset downloaded successfully! (MNIST数据集下载成功！)")
        return train_dataset, test_dataset
    
    def create_splits(self, train_dataset):
        """
        Create train/validation splits
        创建训练/验证分割
        
        Args:
            train_dataset: Original training dataset (原始训练数据集)
            
        Returns:
            train_subset, val_subset: Training and validation datasets
        """
        # Calculate split sizes (计算分割大小)
        total_size = len(train_dataset)
        val_size = int(total_size * self.validation_split)
        train_size = total_size - val_size
        
        print(f"Creating splits: (创建分割)")
        print(f"  Training size: {train_size} (训练集大小)")
        print(f"  Validation size: {val_size} (验证集大小)")
        
        # Create random split (创建随机分割)
        train_subset, val_subset = random_split(
            train_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility (可重现性)
        )
        
        return train_subset, val_subset
    
    def create_data_loaders(self):
        """
        Create PyTorch data loaders
        创建PyTorch数据加载器
        
        Returns:
            train_loader, val_loader, test_loader: Data loaders for each split
        """
        # Download data (下载数据)
        train_dataset, test_dataset = self.download_data()
        
        # Create train/validation splits (创建训练/验证分割)
        train_subset, val_subset = self.create_splits(train_dataset)
        
        # Create data loaders (创建数据加载器)
        train_loader = DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )
        
        print("✅ Data loaders created successfully! (数据加载器创建成功！)")
        return train_loader, val_loader, test_loader
    
    def prepare_numpy_data(self):
        """
        Prepare data in NumPy format for scratch implementation
        为从零实现准备NumPy格式的数据
        
        Returns:
            Dictionary with train/val/test data in NumPy format
        """
        print("Preparing NumPy data for scratch implementation...")
        print("为从零实现准备NumPy数据...")
        
        # Get PyTorch data loaders (获取PyTorch数据加载器)
        train_loader, val_loader, test_loader = self.create_data_loaders()
        
        def loader_to_numpy(loader):
            """Convert PyTorch DataLoader to NumPy arrays"""
            X_list, y_list = [], []
            for batch_X, batch_y in loader:
                # Flatten images: (batch_size, 1, 28, 28) -> (batch_size, 784)
                # 展平图片：(batch_size, 1, 28, 28) -> (batch_size, 784)
                batch_X_flat = batch_X.view(batch_X.size(0), -1)
                X_list.append(batch_X_flat.numpy())
                y_list.append(batch_y.numpy())
            
            X = np.vstack(X_list)
            y = np.hstack(y_list)
            return X, y
        
        # Convert all splits to NumPy (将所有分割转换为NumPy)
        X_train, y_train = loader_to_numpy(train_loader)
        X_val, y_val = loader_to_numpy(val_loader)
        X_test, y_test = loader_to_numpy(test_loader)
        
        # Create data dictionary (创建数据字典)
        data = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test
        }
        
        # Save processed data (保存处理后的数据)
        processed_file = os.path.join(self.processed_dir, 'mnist_numpy.pkl')
        with open(processed_file, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✅ NumPy data saved to: {processed_file}")
        print(f"✅ NumPy数据保存到: {processed_file}")
        print(f"Data shapes: (数据形状)")
        print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")
        print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")
        
        return data
    
    def load_numpy_data(self):
        """
        Load preprocessed NumPy data
        加载预处理的NumPy数据
        """
        processed_file = os.path.join(self.processed_dir, 'mnist_numpy.pkl')
        
        if not os.path.exists(processed_file):
            print("Processed data not found. Creating... (未找到处理后的数据，正在创建...)")
            return self.prepare_numpy_data()
        
        print(f"Loading preprocessed data from: {processed_file}")
        print(f"从以下位置加载预处理数据: {processed_file}")
        
        with open(processed_file, 'rb') as f:
            data = pickle.load(f)
        
        print("✅ Data loaded successfully! (数据加载成功！)")
        return data
    
    def visualize_samples(self, num_samples=10):
        """
        Visualize sample images from the dataset
        可视化数据集中的样本图片
        
        Args:
            num_samples (int): Number of samples to display (要显示的样本数量)
        """
        print(f"Visualizing {num_samples} random samples... (可视化{num_samples}个随机样本...)")
        
        # Load data (加载数据)
        data = self.load_numpy_data()
        X_train, y_train = data['X_train'], data['y_train']
        
        # Select random samples (选择随机样本)
        indices = np.random.choice(len(X_train), num_samples, replace=False)
        
        # Create subplot (创建子图)
        fig, axes = plt.subplots(2, 5, figsize=(12, 6))
        fig.suptitle('MNIST Sample Images (MNIST样本图片)', fontsize=16)
        
        for i, idx in enumerate(indices):
            row, col = i // 5, i % 5
            
            # Reshape flattened image back to 28x28
            # 将展平的图片重新变形为28x28
            image = X_train[idx].reshape(28, 28)
            label = y_train[idx]
            
            axes[row, col].imshow(image, cmap='gray')
            axes[row, col].set_title(f'Label: {label}')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        # Save visualization (保存可视化)
        plots_dir = os.path.join(self.data_dir, '..', 'results', 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        plt.savefig(os.path.join(plots_dir, 'sample_images.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualization saved to results/plots/sample_images.png")
        print("✅ 可视化保存到 results/plots/sample_images.png")
    
    def get_data_statistics(self):
        """
        Display dataset statistics
        显示数据集统计信息
        """
        print("=== MNIST Dataset Statistics (MNIST数据集统计) ===")
        
        data = self.load_numpy_data()
        
        for split_name, (X_key, y_key) in [
            ('Training', ('X_train', 'y_train')),
            ('Validation', ('X_val', 'y_val')),
            ('Test', ('X_test', 'y_test'))
        ]:
            X, y = data[X_key], data[y_key]
            
            print(f"\n{split_name} Set ({split_name.lower()}集):")
            print(f"  Shape: {X.shape} (形状)")
            print(f"  Data type: {X.dtype} (数据类型)")
            print(f"  Value range: [{X.min():.3f}, {X.max():.3f}] (数值范围)")
            print(f"  Labels: {len(np.unique(y))} classes (标签: {len(np.unique(y))}类)")
            
            # Class distribution (类别分布)
            unique, counts = np.unique(y, return_counts=True)
            print(f"  Class distribution (类别分布):")
            for label, count in zip(unique, counts):
                print(f"    Class {label}: {count} samples (类别 {label}: {count} 样本)")


def main():
    """
    Main function to demonstrate data loading
    演示数据加载的主函数
    """
    print("MNIST Data Loader Demo (MNIST数据加载器演示)")
    print("=" * 50)
    
    # Initialize data loader (初始化数据加载器)
    data_loader = MNISTDataLoader(
        data_dir='../data',
        batch_size=64,
        validation_split=0.1
    )
    
    # Prepare data (准备数据)
    data_loader.prepare_numpy_data()
    
    # Show statistics (显示统计信息)
    data_loader.get_data_statistics()
    
    # Visualize samples (可视化样本)
    data_loader.visualize_samples(num_samples=10)
    
    print("\n✅ Data loading demo completed! (数据加载演示完成！)")


if __name__ == "__main__":
    main() 