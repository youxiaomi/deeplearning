import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicAutoencoder(nn.Module):
    """
    基础自编码器 / Basic Autoencoder
    
    自编码器是一种无监督学习模型，由编码器和解码器组成。
    编码器将输入数据压缩到低维潜在空间，解码器将潜在表示重构回原始数据。
    
    An autoencoder is an unsupervised learning model consisting of an encoder and decoder.
    The encoder compresses input data to a low-dimensional latent space, 
    and the decoder reconstructs the original data from the latent representation.
    """
    
    def __init__(self, input_dim=784, hidden_dim=128, latent_dim=32):
        """
        初始化自编码器
        Initialize the autoencoder
        
        Args:
            input_dim: 输入维度 / Input dimension (default: 784 for MNIST 28x28)
            hidden_dim: 隐藏层维度 / Hidden layer dimension
            latent_dim: 潜在空间维度 / Latent space dimension
        """
        super(BasicAutoencoder, self).__init__()
        
        # 编码器 / Encoder
        # 编码器负责将高维输入压缩到低维潜在空间
        # The encoder compresses high-dimensional input to low-dimensional latent space
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim)
        )
        
        # 解码器 / Decoder  
        # 解码器负责从潜在空间重构原始数据
        # The decoder reconstructs original data from latent space
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # 输出范围[0,1] / Output range [0,1]
        )
    
    def encode(self, x):
        """
        编码过程：将输入映射到潜在空间
        Encoding process: map input to latent space
        """
        return self.encoder(x)
    
    def decode(self, z):
        """
        解码过程：从潜在空间重构数据
        Decoding process: reconstruct data from latent space
        """
        return self.decoder(z)
    
    def forward(self, x):
        """
        前向传播：完整的编码-解码过程
        Forward pass: complete encode-decode process
        """
        # 将输入图像展平为向量 / Flatten input image to vector
        x = x.view(x.size(0), -1)
        
        # 编码 / Encode
        latent = self.encode(x)
        
        # 解码 / Decode
        reconstructed = self.decode(latent)
        
        return reconstructed, latent


class ConvAutoencoder(nn.Module):
    """
    卷积自编码器 / Convolutional Autoencoder
    
    使用卷积层处理图像数据，能够更好地保持空间结构信息。
    Uses convolutional layers to process image data, better preserving spatial structure.
    """
    
    def __init__(self, channels=1, latent_dim=128):
        """
        初始化卷积自编码器
        Initialize convolutional autoencoder
        
        Args:
            channels: 输入通道数 / Number of input channels (1 for grayscale, 3 for RGB)
            latent_dim: 潜在空间维度 / Latent space dimension
        """
        super(ConvAutoencoder, self).__init__()
        
        # 编码器 / Encoder
        self.encoder = nn.Sequential(
            # 第一层卷积: 28x28 -> 14x14
            # First conv layer: 28x28 -> 14x14
            nn.Conv2d(channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
            # 第二层卷积: 14x14 -> 7x7
            # Second conv layer: 14x14 -> 7x7
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
            # 第三层卷积: 7x7 -> 4x4
            # Third conv layer: 7x7 -> 4x4
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        
        # 全连接层映射到潜在空间 / Fully connected layer to latent space
        self.fc_encode = nn.Linear(64 * 4 * 4, latent_dim)
        
        # 从潜在空间映射回特征图 / Map from latent space back to feature maps
        self.fc_decode = nn.Linear(latent_dim, 64 * 4 * 4)
        
        # 解码器 / Decoder
        self.decoder = nn.Sequential(
            # 第一层反卷积: 4x4 -> 7x7
            # First deconv layer: 4x4 -> 7x7
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            
            # 第二层反卷积: 7x7 -> 14x14
            # Second deconv layer: 7x7 -> 14x14
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            
            # 第三层反卷积: 14x14 -> 28x28
            # Third deconv layer: 14x14 -> 28x28
            nn.ConvTranspose2d(16, channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        """编码过程 / Encoding process"""
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # 展平 / Flatten
        latent = self.fc_encode(x)
        return latent
    
    def decode(self, z):
        """解码过程 / Decoding process"""
        x = self.fc_decode(z)
        x = x.view(x.size(0), 64, 4, 4)  # 重塑为特征图 / Reshape to feature maps
        reconstructed = self.decoder(x)
        return reconstructed
    
    def forward(self, x):
        """前向传播 / Forward pass"""
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent


def autoencoder_loss(reconstructed, original):
    """
    自编码器损失函数 / Autoencoder loss function
    
    使用均方误差(MSE)或二元交叉熵损失来衡量重构质量。
    Uses Mean Squared Error (MSE) or Binary Cross Entropy loss to measure reconstruction quality.
    
    Args:
        reconstructed: 重构的数据 / Reconstructed data
        original: 原始数据 / Original data
    
    Returns:
        loss: 重构损失 / Reconstruction loss
    """
    # 二元交叉熵损失 / Binary Cross Entropy Loss
    bce_loss = F.binary_cross_entropy(reconstructed, original.view(original.size(0), -1), reduction='sum')
    
    # 均方误差损失 / Mean Squared Error Loss
    # mse_loss = F.mse_loss(reconstructed, original.view(original.size(0), -1), reduction='sum')
    
    return bce_loss


if __name__ == "__main__":
    # 测试基础自编码器 / Test basic autoencoder
    print("测试基础自编码器 / Testing Basic Autoencoder")
    model = BasicAutoencoder()
    test_input = torch.randn(32, 1, 28, 28)  # 批次大小32，1通道，28x28图像
    reconstructed, latent = model(test_input)
    print(f"输入形状 / Input shape: {test_input.shape}")
    print(f"潜在表示形状 / Latent shape: {latent.shape}")
    print(f"重构输出形状 / Reconstructed shape: {reconstructed.shape}")
    
    # 测试卷积自编码器 / Test convolutional autoencoder
    print("\n测试卷积自编码器 / Testing Convolutional Autoencoder")
    conv_model = ConvAutoencoder()
    reconstructed_conv, latent_conv = conv_model(test_input)
    print(f"输入形状 / Input shape: {test_input.shape}")
    print(f"潜在表示形状 / Latent shape: {latent_conv.shape}")
    print(f"重构输出形状 / Reconstructed shape: {reconstructed_conv.shape}") 