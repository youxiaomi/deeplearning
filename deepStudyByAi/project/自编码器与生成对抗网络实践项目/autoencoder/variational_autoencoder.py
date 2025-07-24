import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VariationalAutoencoder(nn.Module):
    """
    变分自编码器 / Variational Autoencoder (VAE)
    
    VAE是自编码器的概率版本，它不是直接学习潜在表示，而是学习潜在空间的概率分布。
    这使得VAE能够生成新的数据样本，而不仅仅是重构现有数据。
    
    VAE is a probabilistic version of autoencoder that learns probability distributions 
    in latent space rather than deterministic representations. This enables VAE to 
    generate new data samples, not just reconstruct existing data.
    """
    
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        """
        初始化变分自编码器
        Initialize Variational Autoencoder
        
        Args:
            input_dim: 输入维度 / Input dimension
            hidden_dim: 隐藏层维度 / Hidden layer dimension  
            latent_dim: 潜在空间维度 / Latent space dimension
        """
        super(VariationalAutoencoder, self).__init__()
        
        self.latent_dim = latent_dim
        
        # 编码器网络 / Encoder network
        # 编码器输出潜在分布的均值和方差参数
        # Encoder outputs mean and variance parameters of latent distribution
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 潜在分布参数 / Latent distribution parameters
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)      # 均值 / Mean
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # 对数方差 / Log variance
        
        # 解码器网络 / Decoder network
        # 解码器从潜在样本重构原始数据
        # Decoder reconstructs original data from latent samples
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # 输出范围[0,1] / Output range [0,1]
        )
    
    def encode(self, x):
        """
        编码过程：计算潜在分布的参数
        Encoding: compute parameters of latent distribution
        
        Args:
            x: 输入数据 / Input data
            
        Returns:
            mu: 潜在分布的均值 / Mean of latent distribution
            logvar: 潜在分布的对数方差 / Log variance of latent distribution
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        重参数化技巧 / Reparameterization trick
        
        为了使随机采样过程可微分，我们使用重参数化技巧：
        z = μ + σ * ε，其中ε ~ N(0,1)
        
        To make the stochastic sampling process differentiable, we use reparameterization:
        z = μ + σ * ε, where ε ~ N(0,1)
        
        Args:
            mu: 均值 / Mean
            logvar: 对数方差 / Log variance
            
        Returns:
            z: 从潜在分布采样的向量 / Sampled vector from latent distribution
        """
        if self.training:
            # 训练时进行随机采样 / Random sampling during training
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # 推理时直接使用均值 / Use mean directly during inference
            return mu
    
    def decode(self, z):
        """
        解码过程：从潜在表示重构数据
        Decoding: reconstruct data from latent representation
        """
        return self.decoder(z)
    
    def forward(self, x):
        """
        前向传播 / Forward pass
        
        Returns:
            reconstructed: 重构的数据 / Reconstructed data
            mu: 潜在分布均值 / Mean of latent distribution
            logvar: 潜在分布对数方差 / Log variance of latent distribution
        """
        # 将输入图像展平 / Flatten input image
        x = x.view(x.size(0), -1)
        
        # 编码得到分布参数 / Encode to get distribution parameters
        mu, logvar = self.encode(x)
        
        # 重参数化采样 / Reparameterized sampling
        z = self.reparameterize(mu, logvar)
        
        # 解码重构 / Decode and reconstruct
        reconstructed = self.decode(z)
        
        return reconstructed, mu, logvar
    
    def sample(self, num_samples, device='cpu'):
        """
        从先验分布采样生成新数据 / Generate new data by sampling from prior
        
        Args:
            num_samples: 采样数量 / Number of samples
            device: 设备 / Device
            
        Returns:
            generated: 生成的数据 / Generated data
        """
        with torch.no_grad():
            # 从标准正态分布采样 / Sample from standard normal distribution
            z = torch.randn(num_samples, self.latent_dim).to(device)
            
            # 解码生成数据 / Decode to generate data
            generated = self.decode(z)
            
        return generated


def vae_loss(reconstructed, original, mu, logvar, beta=1.0):
    """
    VAE损失函数 / VAE Loss Function
    
    VAE的损失包含两部分：
    1. 重构损失：衡量重构质量
    2. KL散度：正则化潜在空间，使其接近标准正态分布
    
    VAE loss consists of two parts:
    1. Reconstruction loss: measures reconstruction quality
    2. KL divergence: regularizes latent space to be close to standard normal
    
    Args:
        reconstructed: 重构数据 / Reconstructed data
        original: 原始数据 / Original data
        mu: 潜在分布均值 / Mean of latent distribution
        logvar: 潜在分布对数方差 / Log variance of latent distribution
        beta: KL散度权重 / Weight for KL divergence (β-VAE)
        
    Returns:
        total_loss: 总损失 / Total loss
        recon_loss: 重构损失 / Reconstruction loss
        kl_loss: KL散度损失 / KL divergence loss
    """
    # 重构损失 / Reconstruction loss
    recon_loss = F.binary_cross_entropy(
        reconstructed, 
        original.view(original.size(0), -1), 
        reduction='sum'
    )
    
    # KL散度损失 / KL divergence loss
    # KL(q(z|x) || p(z)) where p(z) = N(0,1)
    # 公式：-0.5 * sum(1 + log(σ²) - μ² - σ²)
    # Formula: -0.5 * sum(1 + log(σ²) - μ² - σ²)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # 总损失 / Total loss
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss


class ConvVariationalAutoencoder(nn.Module):
    """
    卷积变分自编码器 / Convolutional Variational Autoencoder
    
    结合卷积神经网络和变分自编码器，适用于图像数据的生成建模。
    Combines CNN with VAE for generative modeling of image data.
    """
    
    def __init__(self, channels=1, latent_dim=128):
        super(ConvVariationalAutoencoder, self).__init__()
        
        self.latent_dim = latent_dim
        
        # 编码器 / Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, 4, 2, 1),  # 28x28 -> 14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),       # 14x14 -> 7x7
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),      # 7x7 -> 3x3
            nn.ReLU(),
        )
        
        # 计算编码器输出的特征图大小 / Calculate encoder output feature map size
        self.encoder_output_size = 128 * 3 * 3
        
        # 潜在分布参数 / Latent distribution parameters
        self.fc_mu = nn.Linear(self.encoder_output_size, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_output_size, latent_dim)
        
        # 解码器输入映射 / Decoder input mapping
        self.fc_decode = nn.Linear(latent_dim, self.encoder_output_size)
        
        # 解码器 / Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 3x3 -> 7x7
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # 7x7 -> 15x15
            nn.ReLU(),
            nn.ConvTranspose2d(32, channels, 4, 2, 1),  # 15x15 -> 31x31
            nn.Sigmoid()
        )
        
        # 最终调整层以匹配28x28 / Final adjustment layer to match 28x28
        self.final_conv = nn.Conv2d(channels, channels, 4, 1, 1)  # 31x31 -> 28x28
    
    def encode(self, x):
        """编码过程 / Encoding process"""
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """重参数化技巧 / Reparameterization trick"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def decode(self, z):
        """解码过程 / Decoding process"""
        h = self.fc_decode(z)
        h = h.view(h.size(0), 128, 3, 3)
        h = self.decoder(h)
        # 调整到28x28 / Adjust to 28x28
        h = self.final_conv(h)
        return h
    
    def forward(self, x):
        """前向传播 / Forward pass"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar
    
    def sample(self, num_samples, device='cpu'):
        """生成新样本 / Generate new samples"""
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            generated = self.decode(z)
        return generated


if __name__ == "__main__":
    # 测试变分自编码器 / Test Variational Autoencoder
    print("测试变分自编码器 / Testing Variational Autoencoder")
    
    # 基础VAE测试 / Basic VAE test
    vae = VariationalAutoencoder()
    test_input = torch.randn(32, 1, 28, 28)
    reconstructed, mu, logvar = vae(test_input)
    
    print(f"输入形状 / Input shape: {test_input.shape}")
    print(f"重构形状 / Reconstructed shape: {reconstructed.shape}")
    print(f"均值形状 / Mean shape: {mu.shape}")
    print(f"对数方差形状 / Log variance shape: {logvar.shape}")
    
    # 计算损失 / Calculate loss
    total_loss, recon_loss, kl_loss = vae_loss(reconstructed, test_input, mu, logvar)
    print(f"总损失 / Total loss: {total_loss.item():.4f}")
    print(f"重构损失 / Reconstruction loss: {recon_loss.item():.4f}")
    print(f"KL散度损失 / KL divergence loss: {kl_loss.item():.4f}")
    
    # 生成新样本 / Generate new samples
    generated_samples = vae.sample(num_samples=10)
    print(f"生成样本形状 / Generated samples shape: {generated_samples.shape}")
    
    # 测试卷积VAE / Test Convolutional VAE
    print("\n测试卷积变分自编码器 / Testing Convolutional VAE")
    conv_vae = ConvVariationalAutoencoder()
    conv_reconstructed, conv_mu, conv_logvar = conv_vae(test_input)
    print(f"卷积VAE重构形状 / Conv VAE reconstructed shape: {conv_reconstructed.shape}") 