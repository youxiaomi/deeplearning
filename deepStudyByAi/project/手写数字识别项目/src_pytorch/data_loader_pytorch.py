import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import os

class MNISTDataLoaderPyTorch:
    """
    PyTorch DataLoader for MNIST Handwritten Digit Recognition.
    用于MNIST手写数字识别的PyTorch数据加载器。
    
    This class handles downloading, loading, and preprocessing the MNIST dataset,
    providing PyTorch DataLoaders for training, validation, and testing.
    该类负责下载、加载和预处理MNIST数据集，为训练、验证和测试提供PyTorch DataLoader。
    """
    def __init__(self, data_dir='./data', batch_size=64, validation_split=0.1):
        """
        Initializes the data loader.
        初始化数据加载器。

        Args:
            data_dir (str): Directory to store the MNIST dataset.
                            存储MNIST数据集的目录。
            batch_size (int): Number of samples per batch.
                              每个批次的样本数。
            validation_split (float): Fraction of the training data to be used for validation.
                                      用于验证的训练数据比例。
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.transform = transforms.Compose([
            transforms.ToTensor(), # Convert images to PyTorch Tensors (将图片转换为PyTorch张量)
            transforms.Normalize((0.1307,), (0.3081,)) # Normalize pixel values (标准化像素值)
        ])
        
        self._download_and_load_data()

    def _download_and_load_data(self):
        """
        Downloads and loads the MNIST dataset.
        下载并加载MNIST数据集。
        """
        # Create data directory if it doesn't exist (如果数据目录不存在则创建)
        os.makedirs(self.data_dir, exist_ok=True)

        # Download training and test datasets (下载训练和测试数据集)
        full_train_dataset = torchvision.datasets.MNIST(
            root=self.data_dir, train=True, download=True, transform=self.transform
        )
        self.test_dataset = torchvision.datasets.MNIST(
            root=self.data_dir, train=False, download=True, transform=self.transform
        )

        # Split training dataset into training and validation sets (将训练数据集分成训练和验证集)
        num_train = len(full_train_dataset)
        num_val = int(self.validation_split * num_train)
        num_train -= num_val # Adjust training set size (调整训练集大小)
        
        self.train_dataset, self.val_dataset = random_split(
            full_train_dataset, [num_train, num_val]
        )

    def get_train_loader(self):
        """
        Returns the DataLoader for the training set.
        返回训练集的DataLoader。
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def get_val_loader(self):
        """
        Returns the DataLoader for the validation set.
        返回验证集的DataLoader。
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def get_test_loader(self):
        """
        Returns the DataLoader for the test set.
        返回测试集的DataLoader。
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def get_num_features(self):
        """
        Returns the number of input features (pixels in MNIST images).
        返回输入特征的数量（MNIST图片中的像素）。
        MNIST images are 28x28 pixels = 784 features.
        """
        return 784
    
    def get_num_classes(self):
        """
        Returns the number of output classes (digits 0-9).
        返回输出类别的数量（数字0-9）。
        """
        return 10

if __name__ == '__main__':
    # Example Usage (示例用法)
    print("Testing MNISTDataLoaderPyTorch...")
    print("正在测试MNISTDataLoaderPyTorch...")
    
    # Initialize data loader (初始化数据加载器)
    data_loader = MNISTDataLoaderPyTorch(data_dir='../../data', batch_size=128, validation_split=0.1)
    
    train_loader = data_loader.get_train_loader()
    val_loader = data_loader.get_val_loader()
    test_loader = data_loader.get_test_loader()
    
    print(f"Number of training batches: {len(train_loader)}")
    print(f"训练批次数量: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"验证批次数量: {len(val_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
    print(f"测试批次数量: {len(test_loader)}")
    
    # Get a batch of data (获取一个数据批次)
    for images, labels in train_loader:
        print(f"Shape of image batch: {images.shape}")
        print(f"图片批次的形状: {images.shape}")
        print(f"Shape of label batch: {labels.shape}")
        print(f"标签批次的形状: {labels.shape}")
        break
        
    print(f"Number of input features: {data_loader.get_num_features()}")
    print(f"输入特征的数量: {data_loader.get_num_features()}")
    print(f"Number of classes: {data_loader.get_num_classes()}")
    print(f"类别数量: {data_loader.get_num_classes()}")
    
    print("MNISTDataLoaderPyTorch test completed.")
    print("MNISTDataLoaderPyTorch测试完成。") 