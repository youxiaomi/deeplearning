import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.linalg import sqrtm
import torchvision.models as models
from torchvision import transforms

def calculate_reconstruction_metrics(original, reconstructed, normalize_range=(-1, 1)):
    """
    计算重构质量指标 / Calculate reconstruction quality metrics
    
    Args:
        original: 原始图像 / Original images
        reconstructed: 重构图像 / Reconstructed images
        normalize_range: 归一化范围 / Normalization range
        
    Returns:
        dict: 包含各种指标的字典 / Dictionary containing various metrics
    """
    # 反归一化到[0,1]范围 / Denormalize to [0,1] range
    if normalize_range == (-1, 1):
        original = (original + 1) / 2
        reconstructed = (reconstructed + 1) / 2
    
    # 确保在正确范围内 / Ensure in correct range
    original = torch.clamp(original, 0, 1)
    reconstructed = torch.clamp(reconstructed, 0, 1)
    
    # 展平图像 / Flatten images
    original_flat = original.view(original.size(0), -1)
    reconstructed_flat = reconstructed.view(reconstructed.size(0), -1)
    
    # 均方误差 / Mean Squared Error
    mse = F.mse_loss(reconstructed_flat, original_flat, reduction='mean')
    
    # 平均绝对误差 / Mean Absolute Error
    mae = F.l1_loss(reconstructed_flat, original_flat, reduction='mean')
    
    # 峰值信噪比 / Peak Signal-to-Noise Ratio
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    
    # 结构相似性指数 (简化版) / Structural Similarity Index (simplified)
    # 计算均值 / Calculate means
    mu1 = torch.mean(original_flat, dim=1)
    mu2 = torch.mean(reconstructed_flat, dim=1)
    
    # 计算方差 / Calculate variances
    var1 = torch.var(original_flat, dim=1)
    var2 = torch.var(reconstructed_flat, dim=1)
    
    # 计算协方差 / Calculate covariance
    covar = torch.mean((original_flat - mu1.unsqueeze(1)) * (reconstructed_flat - mu2.unsqueeze(1)), dim=1)
    
    # SSIM常数 / SSIM constants
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    
    # 计算SSIM / Calculate SSIM
    ssim = ((2 * mu1 * mu2 + c1) * (2 * covar + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (var1 + var2 + c2))
    ssim = torch.mean(ssim)
    
    return {
        'mse': mse.item(),
        'mae': mae.item(),
        'psnr': psnr.item(),
        'ssim': ssim.item()
    }


def calculate_fid_score(real_features, generated_features):
    """
    计算弗雷歇特起始距离 (FID) / Calculate Fréchet Inception Distance (FID)
    
    FID是评估生成图像质量的重要指标，通过比较真实图像和生成图像在特征空间中的分布来评估。
    FID is an important metric for evaluating generated image quality by comparing 
    the distributions of real and generated images in feature space.
    
    Args:
        real_features: 真实图像特征 / Real image features (N, feature_dim)
        generated_features: 生成图像特征 / Generated image features (N, feature_dim)
        
    Returns:
        fid: FID分数 / FID score
    """
    # 转换为numpy数组 / Convert to numpy arrays
    if isinstance(real_features, torch.Tensor):
        real_features = real_features.cpu().numpy()
    if isinstance(generated_features, torch.Tensor):
        generated_features = generated_features.cpu().numpy()
    
    # 计算均值 / Calculate means
    mu1 = np.mean(real_features, axis=0)
    mu2 = np.mean(generated_features, axis=0)
    
    # 计算协方差矩阵 / Calculate covariance matrices
    sigma1 = np.cov(real_features, rowvar=False)
    sigma2 = np.cov(generated_features, rowvar=False)
    
    # 计算均值差的平方 / Calculate squared difference of means
    diff = mu1 - mu2
    
    # 计算矩阵平方根 / Calculate matrix square root
    covmean = sqrtm(sigma1.dot(sigma2))
    
    # 处理数值不稳定性 / Handle numerical instability
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    # 计算FID / Calculate FID
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    
    return fid


class InceptionFeatureExtractor:
    """
    Inception特征提取器 / Inception Feature Extractor
    
    用于提取图像的Inception特征，通常用于计算FID分数。
    Used to extract Inception features from images, typically for FID score calculation.
    """
    
    def __init__(self, device='cpu'):
        """
        初始化特征提取器
        Initialize feature extractor
        
        Args:
            device: 计算设备 / Computing device
        """
        self.device = device
        
        # 加载预训练的Inception模型 / Load pretrained Inception model
        self.model = models.inception_v3(pretrained=True, transform_input=False)
        self.model.fc = nn.Identity()  # 移除最后的分类层 / Remove final classification layer
        self.model.eval()
        self.model.to(device)
        
        # 图像预处理 / Image preprocessing
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def extract_features(self, images):
        """
        提取特征 / Extract features
        
        Args:
            images: 输入图像 / Input images (B, C, H, W)
            
        Returns:
            features: 提取的特征 / Extracted features
        """
        with torch.no_grad():
            # 调整图像大小到299x299 / Resize images to 299x299
            if images.size(-1) != 299:
                images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
            
            # 如果是灰度图像，转换为RGB / If grayscale, convert to RGB
            if images.size(1) == 1:
                images = images.repeat(1, 3, 1, 1)
            
            # 归一化 / Normalize
            images = self.normalize(images)
            
            # 提取特征 / Extract features
            features = self.model(images)
            
        return features


def calculate_inception_score(generated_images, batch_size=50, splits=10):
    """
    计算Inception Score (IS) / Calculate Inception Score (IS)
    
    IS通过评估生成图像的质量和多样性来衡量生成模型的性能。
    IS measures generative model performance by evaluating quality and diversity of generated images.
    
    Args:
        generated_images: 生成的图像 / Generated images
        batch_size: 批次大小 / Batch size
        splits: 分割数量 / Number of splits
        
    Returns:
        is_mean: IS均值 / IS mean
        is_std: IS标准差 / IS standard deviation
    """
    # 加载预训练的Inception模型 / Load pretrained Inception model
    device = generated_images.device
    inception_model = models.inception_v3(pretrained=True, transform_input=False)
    inception_model.eval()
    inception_model.to(device)
    
    # 预处理 / Preprocessing
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def get_predictions(images):
        """获取预测概率 / Get prediction probabilities"""
        with torch.no_grad():
            # 调整图像大小 / Resize images
            if images.size(-1) != 299:
                images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
            
            # 转换为RGB / Convert to RGB
            if images.size(1) == 1:
                images = images.repeat(1, 3, 1, 1)
            
            # 归一化 / Normalize
            images = normalize(images)
            
            # 获取预测 / Get predictions
            pred = inception_model(images)
            pred = F.softmax(pred, dim=1)
            
        return pred
    
    # 分批处理 / Process in batches
    preds = []
    num_batches = len(generated_images) // batch_size + (1 if len(generated_images) % batch_size != 0 else 0)
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(generated_images))
        batch = generated_images[start_idx:end_idx]
        
        batch_preds = get_predictions(batch)
        preds.append(batch_preds.cpu().numpy())
    
    preds = np.concatenate(preds, axis=0)
    
    # 计算IS / Calculate IS
    scores = []
    for i in range(splits):
        part = preds[i * len(preds) // splits: (i + 1) * len(preds) // splits]
        
        # 计算边际分布 / Calculate marginal distribution
        py = np.mean(part, axis=0)
        
        # 计算KL散度 / Calculate KL divergence
        kl_div = part * (np.log(part + 1e-10) - np.log(py + 1e-10))
        kl_div = np.mean(np.sum(kl_div, axis=1))
        
        scores.append(np.exp(kl_div))
    
    return np.mean(scores), np.std(scores)


def evaluate_gan_quality(generator, discriminator, dataloader, device, num_samples=1000):
    """
    综合评估GAN质量 / Comprehensive GAN quality evaluation
    
    Args:
        generator: 生成器模型 / Generator model
        discriminator: 判别器模型 / Discriminator model
        dataloader: 真实数据加载器 / Real data loader
        device: 计算设备 / Computing device
        num_samples: 评估样本数量 / Number of samples for evaluation
        
    Returns:
        dict: 评估结果 / Evaluation results
    """
    generator.eval()
    discriminator.eval()
    
    # 生成样本 / Generate samples
    noise_dim = generator.noise_dim if hasattr(generator, 'noise_dim') else 100
    noise = torch.randn(num_samples, noise_dim, device=device)
    
    with torch.no_grad():
        generated_samples = generator(noise)
    
    # 获取真实样本 / Get real samples
    real_samples = []
    for batch in dataloader:
        if isinstance(batch, tuple):
            images, _ = batch
        else:
            images = batch
        real_samples.append(images)
        
        if len(real_samples) * images.size(0) >= num_samples:
            break
    
    real_samples = torch.cat(real_samples, dim=0)[:num_samples].to(device)
    
    # 评估判别器性能 / Evaluate discriminator performance
    with torch.no_grad():
        real_pred = discriminator(real_samples)
        fake_pred = discriminator(generated_samples)
        
        # 计算准确率 / Calculate accuracy
        real_acc = (real_pred > 0.5).float().mean()
        fake_acc = (fake_pred < 0.5).float().mean()
        overall_acc = (real_acc + fake_acc) / 2
    
    # 计算生成样本的多样性 / Calculate diversity of generated samples
    generated_flat = generated_samples.view(generated_samples.size(0), -1)
    diversity = torch.mean(torch.std(generated_flat, dim=0))
    
    results = {
        'discriminator_real_accuracy': real_acc.item(),
        'discriminator_fake_accuracy': fake_acc.item(),
        'discriminator_overall_accuracy': overall_acc.item(),
        'generated_sample_diversity': diversity.item(),
        'real_samples_mean': real_samples.mean().item(),
        'generated_samples_mean': generated_samples.mean().item(),
        'real_samples_std': real_samples.std().item(),
        'generated_samples_std': generated_samples.std().item()
    }
    
    generator.train()
    discriminator.train()
    
    return results


if __name__ == "__main__":
    # 测试评估指标 / Test evaluation metrics
    print("测试评估指标 / Testing evaluation metrics")
    
    # 创建测试数据 / Create test data
    original = torch.randn(32, 1, 28, 28)
    reconstructed = original + 0.1 * torch.randn_like(original)
    
    # 测试重构指标 / Test reconstruction metrics
    recon_metrics = calculate_reconstruction_metrics(original, reconstructed, normalize_range=(-1, 1))
    print("重构指标 / Reconstruction metrics:")
    for key, value in recon_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # 测试FID计算 / Test FID calculation
    real_features = torch.randn(100, 2048)
    fake_features = torch.randn(100, 2048)
    fid_score = calculate_fid_score(real_features, fake_features)
    print(f"\nFID分数 / FID score: {fid_score:.4f}")
    
    print("评估指标测试完成 / Evaluation metrics testing completed") 