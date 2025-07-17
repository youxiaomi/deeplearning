import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DCGenerator(nn.Module):
    """
    深度卷积生成器 / Deep Convolutional Generator (DCGAN)
    
    DCGAN使用转置卷积(反卷积)从低维噪声生成高质量图像。
    相比基础GAN，DCGAN能够更好地处理图像数据的空间结构。
    
    DCGAN uses transposed convolutions (deconvolutions) to generate high-quality images from low-dimensional noise.
    Compared to basic GAN, DCGAN better handles spatial structure of image data.
    """
    
    def __init__(self, noise_dim=100, channels=1, feature_map_size=64):
        """
        初始化深度卷积生成器
        Initialize Deep Convolutional Generator
        
        Args:
            noise_dim: 输入噪声维度 / Input noise dimension
            channels: 输出图像通道数 / Output image channels (1 for grayscale, 3 for RGB)
            feature_map_size: 特征图基础大小 / Base feature map size
        """
        super(DCGenerator, self).__init__()
        
        self.noise_dim = noise_dim
        self.feature_map_size = feature_map_size
        
        # 第一层：从噪声到特征图 / First layer: from noise to feature maps
        # 输入: (batch_size, noise_dim) -> 输出: (batch_size, feature_map_size*8, 4, 4)
        # Input: (batch_size, noise_dim) -> Output: (batch_size, feature_map_size*8, 4, 4)
        self.fc = nn.Sequential(
            nn.Linear(noise_dim, feature_map_size * 8 * 4 * 4),
            nn.BatchNorm1d(feature_map_size * 8 * 4 * 4),
            nn.ReLU()
        )
        
        # 转置卷积层 / Transposed convolution layers
        # 逐步上采样生成最终图像 / Gradually upsample to generate final image
        self.conv_transpose = nn.Sequential(
            # 第一层转置卷积: 4x4 -> 8x8
            # First transposed conv: 4x4 -> 8x8
            nn.ConvTranspose2d(feature_map_size * 8, feature_map_size * 4, 
                             kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_size * 4),
            nn.ReLU(),
            
            # 第二层转置卷积: 8x8 -> 16x16
            # Second transposed conv: 8x8 -> 16x16
            nn.ConvTranspose2d(feature_map_size * 4, feature_map_size * 2, 
                             kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.ReLU(),
            
            # 第三层转置卷积: 16x16 -> 32x32
            # Third transposed conv: 16x16 -> 32x32
            nn.ConvTranspose2d(feature_map_size * 2, feature_map_size, 
                             kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_size),
            nn.ReLU(),
            
            # 输出层: 32x32 -> 64x64，然后裁剪到28x28
            # Output layer: 32x32 -> 64x64, then crop to 28x28
            nn.ConvTranspose2d(feature_map_size, channels, 
                             kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()  # 输出范围[-1, 1] / Output range [-1, 1]
        )
        
        # 最终调整层以匹配MNIST 28x28大小 / Final adjustment layer to match MNIST 28x28 size
        self.final_conv = nn.Conv2d(channels, channels, kernel_size=37, stride=1, padding=0)
    
    def forward(self, noise):
        """
        前向传播：从噪声生成图像
        Forward pass: generate image from noise
        
        Args:
            noise: 随机噪声 / Random noise (batch_size, noise_dim)
            
        Returns:
            generated_image: 生成的图像 / Generated image (batch_size, channels, 28, 28)
        """
        # 噪声通过全连接层 / Pass noise through fully connected layer
        x = self.fc(noise)
        
        # 重塑为4D张量 / Reshape to 4D tensor
        x = x.view(x.size(0), self.feature_map_size * 8, 4, 4)
        
        # 通过转置卷积层 / Pass through transposed convolution layers
        x = self.conv_transpose(x)
        
        # 调整到28x28大小 / Adjust to 28x28 size
        # 从64x64裁剪到28x28 / Crop from 64x64 to 28x28
        x = x[:, :, 18:46, 18:46]  # 中心裁剪 / Center crop
        
        return x
    
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


class DCDiscriminator(nn.Module):
    """
    深度卷积判别器 / Deep Convolutional Discriminator
    
    使用卷积神经网络对图像进行分类，判断图像是真实的还是生成的。
    采用LeakyReLU激活函数和批归一化来提高训练稳定性。
    
    Uses convolutional neural network to classify images as real or generated.
    Uses LeakyReLU activation and batch normalization to improve training stability.
    """
    
    def __init__(self, channels=1, feature_map_size=64):
        """
        初始化深度卷积判别器
        Initialize Deep Convolutional Discriminator
        
        Args:
            channels: 输入图像通道数 / Input image channels
            feature_map_size: 特征图基础大小 / Base feature map size
        """
        super(DCDiscriminator, self).__init__()
        
        # 卷积层 / Convolutional layers
        # 逐步下采样并增加特征图数量 / Gradually downsample and increase feature maps
        self.conv_layers = nn.Sequential(
            # 第一层卷积: 28x28 -> 14x14
            # First conv layer: 28x28 -> 14x14
            nn.Conv2d(channels, feature_map_size, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 第二层卷积: 14x14 -> 7x7
            # Second conv layer: 14x14 -> 7x7
            nn.Conv2d(feature_map_size, feature_map_size * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 第三层卷积: 7x7 -> 3x3
            # Third conv layer: 7x7 -> 3x3
            nn.Conv2d(feature_map_size * 2, feature_map_size * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 第四层卷积: 3x3 -> 1x1
            # Fourth conv layer: 3x3 -> 1x1
            nn.Conv2d(feature_map_size * 4, 1, kernel_size=3, stride=1, padding=0, bias=False),
            nn.Sigmoid()  # 输出概率 / Output probability
        )
    
    def forward(self, image):
        """
        前向传播：判断图像真假
        Forward pass: judge if image is real or fake
        
        Args:
            image: 输入图像 / Input image (batch_size, channels, height, width)
            
        Returns:
            probability: 图像为真实的概率 / Probability that image is real
        """
        x = self.conv_layers(image)
        # 展平为一维输出 / Flatten to 1D output
        return x.view(x.size(0), -1)


class DCGAN:
    """
    深度卷积生成对抗网络 / Deep Convolutional Generative Adversarial Network
    
    DCGAN结合了卷积神经网络和生成对抗网络，专门用于图像生成。
    相比基础GAN，DCGAN在生成图像质量和训练稳定性方面都有显著提升。
    
    DCGAN combines CNNs with GANs specifically for image generation.
    Compared to basic GAN, DCGAN significantly improves generated image quality and training stability.
    """
    
    def __init__(self, noise_dim=100, channels=1, feature_map_size=64, device='cpu'):
        """
        初始化DCGAN
        Initialize DCGAN
        
        Args:
            noise_dim: 噪声维度 / Noise dimension
            channels: 图像通道数 / Image channels
            feature_map_size: 特征图基础大小 / Base feature map size
            device: 设备 / Device
        """
        self.device = device
        self.noise_dim = noise_dim
        
        # 初始化生成器和判别器 / Initialize generator and discriminator
        self.generator = DCGenerator(noise_dim, channels, feature_map_size).to(device)
        self.discriminator = DCDiscriminator(channels, feature_map_size).to(device)
        
        # 权重初始化 / Weight initialization
        self._initialize_weights()
        
        # 损失函数 / Loss function
        self.criterion = nn.BCELoss()
        
        # 优化器 / Optimizers
        self.g_optimizer = None
        self.d_optimizer = None
    
    def _initialize_weights(self):
        """
        初始化网络权重 / Initialize network weights
        
        DCGAN论文建议使用正态分布初始化权重，均值为0，标准差为0.02
        DCGAN paper suggests initializing weights from normal distribution with mean=0, std=0.02
        """
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
        
        self.generator.apply(init_func)
        self.discriminator.apply(init_func)
    
    def set_optimizers(self, g_lr=0.0002, d_lr=0.0002, beta1=0.5):
        """
        设置优化器 / Set optimizers
        
        Args:
            g_lr: 生成器学习率 / Generator learning rate
            d_lr: 判别器学习率 / Discriminator learning rate
            beta1: Adam优化器的beta1参数 / Beta1 parameter for Adam optimizer
        """
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=g_lr, betas=(beta1, 0.999))
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=d_lr, betas=(beta1, 0.999))
    
    def train_discriminator(self, real_images, batch_size):
        """
        训练判别器 / Train discriminator
        
        Args:
            real_images: 真实图像 / Real images
            batch_size: 批次大小 / Batch size
            
        Returns:
            d_loss: 判别器损失 / Discriminator loss
            d_real_acc: 判别器对真实图像的准确率 / Discriminator accuracy on real images
            d_fake_acc: 判别器对生成图像的准确率 / Discriminator accuracy on fake images
        """
        self.d_optimizer.zero_grad()
        
        # 真实图像的损失 / Loss on real images
        real_labels = torch.ones(batch_size, 1, device=self.device) * 0.9  # 标签平滑 / Label smoothing
        real_pred = self.discriminator(real_images)
        real_loss = self.criterion(real_pred, real_labels)
        
        # 生成图像的损失 / Loss on generated images
        noise = self.generator.generate_noise(batch_size, self.device)
        fake_images = self.generator(noise).detach()  # 不传播梯度到生成器 / Don't propagate gradients to generator
        fake_labels = torch.zeros(batch_size, 1, device=self.device)
        fake_pred = self.discriminator(fake_images)
        fake_loss = self.criterion(fake_pred, fake_labels)
        
        # 总损失 / Total loss
        d_loss = real_loss + fake_loss
        d_loss.backward()
        self.d_optimizer.step()
        
        # 计算准确率 / Calculate accuracy
        d_real_acc = ((real_pred > 0.5).float() == 1).float().mean()
        d_fake_acc = ((fake_pred < 0.5).float() == 1).float().mean()
        
        return d_loss.item(), d_real_acc.item(), d_fake_acc.item()
    
    def train_generator(self, batch_size):
        """
        训练生成器 / Train generator
        
        Args:
            batch_size: 批次大小 / Batch size
            
        Returns:
            g_loss: 生成器损失 / Generator loss
        """
        self.g_optimizer.zero_grad()
        
        # 生成假图像 / Generate fake images
        noise = self.generator.generate_noise(batch_size, self.device)
        fake_images = self.generator(noise)
        
        # 生成器希望判别器认为假图像是真的 / Generator wants discriminator to think fake images are real
        real_labels = torch.ones(batch_size, 1, device=self.device)
        fake_pred = self.discriminator(fake_images)
        g_loss = self.criterion(fake_pred, real_labels)
        
        g_loss.backward()
        self.g_optimizer.step()
        
        return g_loss.item()
    
    def generate_samples(self, num_samples):
        """
        生成样本图像 / Generate sample images
        
        Args:
            num_samples: 样本数量 / Number of samples
            
        Returns:
            samples: 生成的图像样本 / Generated image samples
        """
        self.generator.eval()
        with torch.no_grad():
            noise = self.generator.generate_noise(num_samples, self.device)
            samples = self.generator(noise)
        self.generator.train()
        return samples
    
    def save_models(self, generator_path, discriminator_path):
        """
        保存模型 / Save models
        
        Args:
            generator_path: 生成器保存路径 / Generator save path
            discriminator_path: 判别器保存路径 / Discriminator save path
        """
        torch.save(self.generator.state_dict(), generator_path)
        torch.save(self.discriminator.state_dict(), discriminator_path)
    
    def load_models(self, generator_path, discriminator_path):
        """
        加载模型 / Load models
        
        Args:
            generator_path: 生成器模型路径 / Generator model path
            discriminator_path: 判别器模型路径 / Discriminator model path
        """
        self.generator.load_state_dict(torch.load(generator_path, map_location=self.device))
        self.discriminator.load_state_dict(torch.load(discriminator_path, map_location=self.device))


def weights_init_normal(m):
    """
    DCGAN权重初始化函数 / DCGAN weight initialization function
    
    Args:
        m: 网络层 / Network layer
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


if __name__ == "__main__":
    # 测试DCGAN / Test DCGAN
    print("测试深度卷积GAN / Testing Deep Convolutional GAN")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备 / Using device: {device}")
    
    # 初始化DCGAN / Initialize DCGAN
    dcgan = DCGAN(noise_dim=100, channels=1, feature_map_size=64, device=device)
    dcgan.set_optimizers()
    
    # 测试生成器 / Test generator
    print("\n测试生成器 / Testing Generator")
    test_noise = torch.randn(16, 100, device=device)
    generated_images = dcgan.generator(test_noise)
    print(f"噪声形状 / Noise shape: {test_noise.shape}")
    print(f"生成图像形状 / Generated images shape: {generated_images.shape}")
    print(f"生成图像范围 / Generated images range: [{generated_images.min():.3f}, {generated_images.max():.3f}]")
    
    # 测试判别器 / Test discriminator
    print("\n测试判别器 / Testing Discriminator")
    test_images = torch.randn(16, 1, 28, 28, device=device)
    discriminator_pred = dcgan.discriminator(test_images)
    print(f"输入图像形状 / Input images shape: {test_images.shape}")
    print(f"判别器预测形状 / Discriminator prediction shape: {discriminator_pred.shape}")
    print(f"判别器预测范围 / Discriminator prediction range: [{discriminator_pred.min():.3f}, {discriminator_pred.max():.3f}]")
    
    # 测试完整的训练步骤 / Test complete training step
    print("\n测试训练步骤 / Testing Training Step")
    batch_size = 16
    real_images = torch.randn(batch_size, 1, 28, 28, device=device)
    
    # 训练判别器 / Train discriminator
    d_loss, d_real_acc, d_fake_acc = dcgan.train_discriminator(real_images, batch_size)
    print(f"判别器损失 / Discriminator loss: {d_loss:.4f}")
    print(f"判别器真实图像准确率 / Discriminator real accuracy: {d_real_acc:.4f}")
    print(f"判别器生成图像准确率 / Discriminator fake accuracy: {d_fake_acc:.4f}")
    
    # 训练生成器 / Train generator
    g_loss = dcgan.train_generator(batch_size)
    print(f"生成器损失 / Generator loss: {g_loss:.4f}")
    
    # 生成样本 / Generate samples
    samples = dcgan.generate_samples(8)
    print(f"生成样本形状 / Generated samples shape: {samples.shape}") 