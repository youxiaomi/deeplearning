import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Generator(nn.Module):
    """
    生成器 / Generator
    
    生成器是GAN的一部分，它从随机噪声生成看起来真实的数据。
    生成器的目标是愚弄判别器，让它认为生成的数据是真实的。
    
    The generator is part of GAN that creates realistic-looking data from random noise.
    The generator's goal is to fool the discriminator into thinking generated data is real.
    """
    
    def __init__(self, noise_dim=100, hidden_dim=256, output_dim=784):
        """
        初始化生成器
        Initialize Generator
        
        Args:
            noise_dim: 输入噪声维度 / Input noise dimension
            hidden_dim: 隐藏层维度 / Hidden layer dimension
            output_dim: 输出数据维度 / Output data dimension (784 for MNIST 28x28)
        """
        super(Generator, self).__init__()
        
        self.noise_dim = noise_dim
        
        # 生成器网络 / Generator network
        # 从低维噪声逐步扩展到高维输出
        # Gradually expand from low-dimensional noise to high-dimensional output
        self.network = nn.Sequential(
            # 第一层：噪声到隐藏层 / First layer: noise to hidden
            nn.Linear(noise_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            # 第二层：隐藏层扩展 / Second layer: hidden expansion
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            # 第三层：进一步扩展 / Third layer: further expansion
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            # 输出层：生成最终数据 / Output layer: generate final data
            nn.Linear(hidden_dim * 4, output_dim),
            nn.Tanh()  # 输出范围[-1, 1] / Output range [-1, 1]
        )
    
    def forward(self, noise):
        """
        前向传播：从噪声生成数据
        Forward pass: generate data from noise
        
        Args:
            noise: 随机噪声张量 / Random noise tensor
            
        Returns:
            generated: 生成的数据 / Generated data
        """
        return self.network(noise)
    
    def generate_noise(self, batch_size, device='cpu'):
        """
        生成随机噪声 / Generate random noise
        
        Args:
            batch_size: 批次大小 / Batch size
            device: 设备 / Device
            
        Returns:
            noise: 随机噪声 / Random noise
        """
        return torch.randn(batch_size, self.noise_dim, device=device)


class Discriminator(nn.Module):
    """
    判别器 / Discriminator
    
    判别器是GAN的另一部分，它试图区分真实数据和生成的假数据。
    判别器的目标是准确识别哪些数据是真实的，哪些是生成的。
    
    The discriminator is the other part of GAN that tries to distinguish between 
    real data and generated fake data. Its goal is to accurately identify which 
    data is real and which is generated.
    """
    
    def __init__(self, input_dim=784, hidden_dim=256):
        """
        初始化判别器
        Initialize Discriminator
        
        Args:
            input_dim: 输入数据维度 / Input data dimension
            hidden_dim: 隐藏层维度 / Hidden layer dimension
        """
        super(Discriminator, self).__init__()
        
        # 判别器网络 / Discriminator network
        # 从高维输入逐步压缩到二分类输出
        # Gradually compress from high-dimensional input to binary classification output
        self.network = nn.Sequential(
            # 第一层：输入到隐藏层 / First layer: input to hidden
            nn.Linear(input_dim, hidden_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            # 第二层：压缩 / Second layer: compression
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            # 第三层：进一步压缩 / Third layer: further compression
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            # 输出层：二分类 / Output layer: binary classification
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 输出概率 / Output probability
        )
    
    def forward(self, data):
        """
        前向传播：判断数据真假
        Forward pass: judge if data is real or fake
        
        Args:
            data: 输入数据 / Input data
            
        Returns:
            probability: 数据为真实的概率 / Probability that data is real
        """
        # 将图像展平为向量 / Flatten image to vector
        if len(data.shape) > 2:
            data = data.view(data.size(0), -1)
        
        return self.network(data)


class BasicGAN:
    """
    基础生成对抗网络 / Basic Generative Adversarial Network
    
    GAN是一个博弈论框架，包含两个神经网络：生成器和判别器。
    它们在零和游戏中相互竞争：生成器试图生成假数据来欺骗判别器，
    而判别器试图区分真实数据和生成的假数据。
    
    GAN is a game-theoretic framework consisting of two neural networks: generator and discriminator.
    They compete in a zero-sum game: the generator tries to generate fake data to fool the discriminator,
    while the discriminator tries to distinguish between real and generated fake data.
    """
    
    def __init__(self, noise_dim=100, hidden_dim=256, data_dim=784, device='cpu'):
        """
        初始化GAN
        Initialize GAN
        
        Args:
            noise_dim: 噪声维度 / Noise dimension
            hidden_dim: 隐藏层维度 / Hidden layer dimension
            data_dim: 数据维度 / Data dimension
            device: 设备 / Device
        """
        self.device = device
        self.noise_dim = noise_dim
        
        # 初始化生成器和判别器 / Initialize generator and discriminator
        self.generator = Generator(noise_dim, hidden_dim, data_dim).to(device)
        self.discriminator = Discriminator(data_dim, hidden_dim).to(device)
        
        # 损失函数 / Loss function
        self.criterion = nn.BCELoss()
        
        # 优化器 / Optimizers
        self.g_optimizer = None
        self.d_optimizer = None
    
    def set_optimizers(self, g_lr=0.0002, d_lr=0.0002):
        """
        设置优化器 / Set optimizers
        
        Args:
            g_lr: 生成器学习率 / Generator learning rate
            d_lr: 判别器学习率 / Discriminator learning rate
        """
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=g_lr, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=d_lr, betas=(0.5, 0.999))
    
    def train_discriminator(self, real_data, batch_size):
        """
        训练判别器 / Train discriminator
        
        判别器的目标是最大化正确分类的概率：
        - 对真实数据输出接近1
        - 对生成数据输出接近0
        
        Discriminator's goal is to maximize correct classification probability:
        - Output close to 1 for real data
        - Output close to 0 for generated data
        
        Args:
            real_data: 真实数据 / Real data
            batch_size: 批次大小 / Batch size
            
        Returns:
            d_loss: 判别器损失 / Discriminator loss
        """
        self.d_optimizer.zero_grad()
        
        # 真实数据的损失 / Loss on real data
        real_labels = torch.ones(batch_size, 1, device=self.device)
        real_pred = self.discriminator(real_data)
        real_loss = self.criterion(real_pred, real_labels)
        
        # 生成数据的损失 / Loss on generated data
        noise = self.generator.generate_noise(batch_size, self.device)
        fake_data = self.generator(noise).detach()  # 不传播梯度到生成器 / Don't propagate gradients to generator
        fake_labels = torch.zeros(batch_size, 1, device=self.device)
        fake_pred = self.discriminator(fake_data)
        fake_loss = self.criterion(fake_pred, fake_labels)
        
        # 总损失 / Total loss
        d_loss = real_loss + fake_loss
        d_loss.backward()
        self.d_optimizer.step()
        
        return d_loss.item()
    
    def train_generator(self, batch_size):
        """
        训练生成器 / Train generator
        
        生成器的目标是最小化被判别器识别为假数据的概率：
        - 希望判别器对生成数据输出接近1（认为是真实的）
        
        Generator's goal is to minimize the probability of being identified as fake:
        - Want discriminator to output close to 1 for generated data (think it's real)
        
        Args:
            batch_size: 批次大小 / Batch size
            
        Returns:
            g_loss: 生成器损失 / Generator loss
        """
        self.g_optimizer.zero_grad()
        
        # 生成假数据 / Generate fake data
        noise = self.generator.generate_noise(batch_size, self.device)
        fake_data = self.generator(noise)
        
        # 生成器希望判别器认为假数据是真的 / Generator wants discriminator to think fake data is real
        real_labels = torch.ones(batch_size, 1, device=self.device)
        fake_pred = self.discriminator(fake_data)
        g_loss = self.criterion(fake_pred, real_labels)
        
        g_loss.backward()
        self.g_optimizer.step()
        
        return g_loss.item()
    
    def generate_samples(self, num_samples):
        """
        生成样本 / Generate samples
        
        Args:
            num_samples: 样本数量 / Number of samples
            
        Returns:
            samples: 生成的样本 / Generated samples
        """
        self.generator.eval()
        with torch.no_grad():
            noise = self.generator.generate_noise(num_samples, self.device)
            samples = self.generator(noise)
        self.generator.train()
        return samples


def gan_loss(discriminator_real_pred, discriminator_fake_pred, generator_fake_pred):
    """
    GAN损失函数 / GAN Loss Function
    
    计算生成对抗网络的损失，包括判别器损失和生成器损失。
    Calculate GAN losses including discriminator loss and generator loss.
    
    Args:
        discriminator_real_pred: 判别器对真实数据的预测 / Discriminator prediction on real data
        discriminator_fake_pred: 判别器对假数据的预测 / Discriminator prediction on fake data  
        generator_fake_pred: 生成器生成数据的判别器预测 / Discriminator prediction on generator's fake data
        
    Returns:
        d_loss: 判别器损失 / Discriminator loss
        g_loss: 生成器损失 / Generator loss
    """
    criterion = nn.BCELoss()
    
    # 判别器损失 / Discriminator loss
    real_labels = torch.ones_like(discriminator_real_pred)
    fake_labels = torch.zeros_like(discriminator_fake_pred)
    
    d_real_loss = criterion(discriminator_real_pred, real_labels)
    d_fake_loss = criterion(discriminator_fake_pred, fake_labels)
    d_loss = d_real_loss + d_fake_loss
    
    # 生成器损失 / Generator loss
    g_loss = criterion(generator_fake_pred, torch.ones_like(generator_fake_pred))
    
    return d_loss, g_loss


if __name__ == "__main__":
    # 测试基础GAN / Test Basic GAN
    print("测试基础GAN / Testing Basic GAN")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备 / Using device: {device}")
    
    # 初始化GAN / Initialize GAN
    gan = BasicGAN(noise_dim=100, hidden_dim=256, data_dim=784, device=device)
    gan.set_optimizers()
    
    # 测试生成器 / Test generator
    print("\n测试生成器 / Testing Generator")
    test_noise = torch.randn(32, 100, device=device)
    generated_data = gan.generator(test_noise)
    print(f"噪声形状 / Noise shape: {test_noise.shape}")
    print(f"生成数据形状 / Generated data shape: {generated_data.shape}")
    print(f"生成数据范围 / Generated data range: [{generated_data.min():.3f}, {generated_data.max():.3f}]")
    
    # 测试判别器 / Test discriminator
    print("\n测试判别器 / Testing Discriminator") 
    test_data = torch.randn(32, 784, device=device)
    discriminator_pred = gan.discriminator(test_data)
    print(f"输入数据形状 / Input data shape: {test_data.shape}")
    print(f"判别器预测形状 / Discriminator prediction shape: {discriminator_pred.shape}")
    print(f"判别器预测范围 / Discriminator prediction range: [{discriminator_pred.min():.3f}, {discriminator_pred.max():.3f}]")
    
    # 测试完整的训练步骤 / Test complete training step
    print("\n测试训练步骤 / Testing Training Step")
    batch_size = 32
    real_data = torch.randn(batch_size, 784, device=device)
    
    # 训练判别器 / Train discriminator
    d_loss = gan.train_discriminator(real_data, batch_size)
    print(f"判别器损失 / Discriminator loss: {d_loss:.4f}")
    
    # 训练生成器 / Train generator
    g_loss = gan.train_generator(batch_size)
    print(f"生成器损失 / Generator loss: {g_loss:.4f}")
    
    # 生成样本 / Generate samples
    samples = gan.generate_samples(10)
    print(f"生成样本形状 / Generated samples shape: {samples.shape}") 