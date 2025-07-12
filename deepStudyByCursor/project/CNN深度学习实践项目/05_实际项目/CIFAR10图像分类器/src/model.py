"""
CNN Models for CIFAR-10 Classification
CIFAR-10分类的CNN模型

This module contains different CNN architectures for CIFAR-10 classification.
Think of these as different "brains" with varying complexity!
这个模块包含用于CIFAR-10分类的不同CNN架构。
把它们想象成具有不同复杂度的不同"大脑"！
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    Simple CNN Model - Great for beginners!
    简单CNN模型 - 非常适合初学者！
    
    This is like a basic camera that can recognize simple patterns.
    It has just a few layers but can still learn to identify objects.
    这就像一个可以识别简单模式的基本相机。
    它只有几层，但仍然可以学会识别物体。
    """
    
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # First convolutional block
        # 第一个卷积块
        # Think of this as the first "filter" that detects edges and simple shapes
        # 把这想象成检测边缘和简单形状的第一个"过滤器"
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 32x32x3 -> 32x32x32
        self.bn1 = nn.BatchNorm2d(32)  # Batch normalization for stability (批量归一化以保持稳定)
        self.pool1 = nn.MaxPool2d(2, 2)  # 32x32x32 -> 16x16x32
        
        # Second convolutional block
        # 第二个卷积块
        # This detects more complex patterns like curves and textures
        # 这检测更复杂的模式，如曲线和纹理
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 16x16x32 -> 16x16x64
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # 16x16x64 -> 8x8x64
        
        # Third convolutional block
        # 第三个卷积块
        # This detects even more complex patterns and object parts
        # 这检测更复杂的模式和物体部分
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 8x8x64 -> 8x8x128
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # 8x8x128 -> 4x4x128
        
        # Fully connected layers
        # 全连接层
        # These combine all the detected features to make the final decision
        # 这些结合所有检测到的特征来做出最终决定
        self.fc1 = nn.Linear(128 * 4 * 4, 256)  # Flatten and connect (展平并连接)
        self.dropout = nn.Dropout(0.5)  # Prevent overfitting (防止过拟合)
        self.fc2 = nn.Linear(256, num_classes)  # Final classification (最终分类)
        
    def forward(self, x):
        """
        Forward pass through the network
        网络的前向传播
        
        Args:
            x: Input images (batch_size, 3, 32, 32)
        
        Returns:
            Output logits (batch_size, num_classes)
        """
        # First block: Conv -> BatchNorm -> ReLU -> Pool
        # 第一块：卷积 -> 批量归一化 -> ReLU -> 池化
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # Second block
        # 第二块
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Third block
        # 第三块
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten for fully connected layers
        # 为全连接层展平
        x = x.view(x.size(0), -1)  # Flatten (展平)
        
        # Fully connected layers
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class ImprovedCNN(nn.Module):
    """
    Improved CNN Model - More sophisticated!
    改进的CNN模型 - 更复杂！
    
    This is like a professional camera with better lenses and more features.
    It can detect more subtle patterns and achieve better accuracy.
    这就像一个具有更好镜头和更多功能的专业相机。
    它可以检测更微妙的模式并获得更好的准确性。
    """
    
    def __init__(self, num_classes=10):
        super(ImprovedCNN, self).__init__()
        
        # First convolutional block with more filters
        # 第一个卷积块，有更多过滤器
        self.conv1a = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1b = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Second convolutional block
        # 第二个卷积块
        self.conv2a = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2b = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Third convolutional block
        # 第三个卷积块
        self.conv3a = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3b = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Fourth convolutional block
        # 第四个卷积块
        self.conv4a = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4b = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Fully connected layers with more neurons
        # 具有更多神经元的全连接层
        self.fc1 = nn.Linear(512 * 2 * 2, 1024)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        """Forward pass through the improved network"""
        # First block: Two conv layers
        # 第一块：两个卷积层
        x = F.relu(self.conv1a(x))
        x = self.pool1(F.relu(self.bn1(self.conv1b(x))))
        
        # Second block
        # 第二块
        x = F.relu(self.conv2a(x))
        x = self.pool2(F.relu(self.bn2(self.conv2b(x))))
        
        # Third block
        # 第三块
        x = F.relu(self.conv3a(x))
        x = self.pool3(F.relu(self.bn3(self.conv3b(x))))
        
        # Fourth block
        # 第四块
        x = F.relu(self.conv4a(x))
        x = self.pool4(F.relu(self.bn4(self.conv4b(x))))
        
        # Flatten and fully connected layers
        # 展平和全连接层
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x

class ResidualBlock(nn.Module):
    """
    Residual Block - Building block for ResNet
    残差块 - ResNet的构建块
    
    This is like a shortcut path that helps information flow better.
    Think of it as a highway with both main roads and express lanes.
    这就像一条帮助信息更好流动的捷径。
    把它想象成一条既有主干道又有快速车道的高速公路。
    """
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        # 捷径连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        """Forward pass with residual connection"""
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Add shortcut (residual connection)
        # 添加捷径（残差连接）
        out += self.shortcut(residual)
        out = F.relu(out)
        
        return out

class ResNetCNN(nn.Module):
    """
    ResNet-style CNN - Most advanced!
    ResNet风格的CNN - 最先进！
    
    This is like a super-intelligent camera system with multiple expert eyes
    working together and sharing information efficiently.
    这就像一个超级智能相机系统，有多个专家眼睛
    一起工作并有效地共享信息。
    """
    
    def __init__(self, num_classes=10):
        super(ResNetCNN, self).__init__()
        
        # Initial convolution
        # 初始卷积
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual blocks
        # 残差块
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Global average pooling and final classifier
        # 全局平均池化和最终分类器
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """Create a layer with multiple residual blocks"""
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through ResNet"""
        # Initial convolution
        # 初始卷积
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Residual layers
        # 残差层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global average pooling
        # 全局平均池化
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # Final classification
        # 最终分类
        x = self.fc(x)
        
        return x

def get_model(model_name='simple', num_classes=10):
    """
    Get a model by name
    根据名称获取模型
    
    Args:
        model_name: 'simple', 'improved', or 'resnet'
        num_classes: Number of output classes
    
    Returns:
        PyTorch model
    """
    if model_name.lower() == 'simple':
        return SimpleCNN(num_classes)
    elif model_name.lower() == 'improved':
        return ImprovedCNN(num_classes)
    elif model_name.lower() == 'resnet':
        return ResNetCNN(num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def count_parameters(model):
    """
    Count the number of trainable parameters in a model
    计算模型中可训练参数的数量
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_summary(model, input_size=(3, 32, 32)):
    """
    Print a summary of the model
    打印模型摘要
    
    Args:
        model: PyTorch model
        input_size: Input tensor size
    """
    print(f"Model: {model.__class__.__name__}")
    print(f"模型: {model.__class__.__name__}")
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"总参数: {count_parameters(model):,}")
    
    # Test with dummy input
    # 用虚拟输入测试
    dummy_input = torch.randn(1, *input_size)
    try:
        with torch.no_grad():
            output = model(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"输入形状: {dummy_input.shape}")
        print(f"输出形状: {output.shape}")
    except Exception as e:
        print(f"Error during forward pass: {e}")
        print(f"前向传播错误: {e}")

def test_models():
    """
    Test all model architectures
    测试所有模型架构
    """
    print("Testing CNN Models for CIFAR-10")
    print("测试CIFAR-10的CNN模型")
    print("=" * 50)
    
    models = ['simple', 'improved', 'resnet']
    
    for model_name in models:
        print(f"\n{model_name.upper()} CNN:")
        print(f"{model_name.upper()} CNN:")
        print("-" * 30)
        
        model = get_model(model_name)
        model_summary(model)
        
        # Test forward pass
        # 测试前向传播
        dummy_input = torch.randn(4, 3, 32, 32)  # Batch of 4 images
        try:
            with torch.no_grad():
                output = model(dummy_input)
            print(f"✓ Forward pass successful")
            print(f"✓ 前向传播成功")
            print(f"  Batch output shape: {output.shape}")
            print(f"  批次输出形状: {output.shape}")
        except Exception as e:
            print(f"✗ Forward pass failed: {e}")
            print(f"✗ 前向传播失败: {e}")

if __name__ == "__main__":
    test_models() 