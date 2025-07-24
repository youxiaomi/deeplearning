import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os

class Visualizer:
    """
    可视化工具类 / Visualizer Tool Class
    
    提供各种可视化功能，包括图像显示、损失曲线、潜在空间可视化等。
    Provides various visualization functions including image display, loss curves, latent space visualization, etc.
    """
    
    def __init__(self, save_dir='./results'):
        """
        初始化可视化器
        Initialize visualizer
        
        Args:
            save_dir: 保存结果的目录 / Directory to save results
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置matplotlib样式 / Set matplotlib style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def denormalize_tensor(self, tensor, normalize_range=(-1, 1)):
        """
        反归一化张量用于显示 / Denormalize tensor for display
        
        Args:
            tensor: 归一化的张量 / Normalized tensor
            normalize_range: 原始归一化范围 / Original normalization range
            
        Returns:
            反归一化的张量 / Denormalized tensor
        """
        tensor = tensor.clone()
        if normalize_range == (-1, 1):
            return (tensor + 1) / 2
        elif normalize_range == (0, 1):
            return tensor
        else:
            return torch.clamp(tensor, 0, 1)
    
    def show_images(self, images, titles=None, normalize_range=(-1, 1), 
                   rows=2, cols=4, figsize=(12, 6), save_name=None):
        """
        显示图像网格 / Display image grid
        
        用于展示生成的图像、重构的图像或原始图像。
        Used to display generated images, reconstructed images, or original images.
        
        Args:
            images: 图像张量 / Image tensor (batch_size, channels, height, width)
            titles: 图像标题列表 / List of image titles
            normalize_range: 图像的归一化范围 / Normalization range of images
            rows: 行数 / Number of rows
            cols: 列数 / Number of columns
            figsize: 图片大小 / Figure size
            save_name: 保存文件名 / Save filename
        """
        # 反归一化图像 / Denormalize images
        images = self.denormalize_tensor(images, normalize_range)
        
        # 转换为numpy数组 / Convert to numpy array
        if isinstance(images, torch.Tensor):
            images = images.detach().cpu().numpy()
        
        # 创建子图 / Create subplots
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows * cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        num_images = min(len(images), rows * cols)
        
        for i in range(num_images):
            img = images[i]
            
            # 处理不同的图像格式 / Handle different image formats
            if len(img.shape) == 3:
                if img.shape[0] == 1:  # 灰度图像 / Grayscale
                    img = img.squeeze(0)
                    axes[i].imshow(img, cmap='gray', vmin=0, vmax=1)
                elif img.shape[0] == 3:  # RGB图像 / RGB image
                    img = np.transpose(img, (1, 2, 0))
                    axes[i].imshow(img)
            else:  # 已经是2D图像 / Already 2D image
                axes[i].imshow(img, cmap='gray', vmin=0, vmax=1)
            
            axes[i].axis('off')
            
            # 添加标题 / Add title
            if titles and i < len(titles):
                axes[i].set_title(titles[i], fontsize=10)
        
        # 隐藏多余的子图 / Hide extra subplots
        for i in range(num_images, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # 保存图像 / Save image
        if save_name:
            save_path = os.path.join(self.save_dir, save_name)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图像已保存到 / Image saved to: {save_path}")
        
        plt.show()
    
    def compare_images(self, original, reconstructed, generated=None, 
                      normalize_range=(-1, 1), num_samples=8, save_name=None):
        """
        比较原始图像、重构图像和生成图像 / Compare original, reconstructed, and generated images
        
        用于评估自编码器的重构质量和生成器的生成质量。
        Used to evaluate autoencoder reconstruction quality and generator generation quality.
        
        Args:
            original: 原始图像 / Original images
            reconstructed: 重构图像 / Reconstructed images
            generated: 生成图像 / Generated images (可选 / optional)
            normalize_range: 归一化范围 / Normalization range
            num_samples: 显示的样本数量 / Number of samples to display
            save_name: 保存文件名 / Save filename
        """
        num_samples = min(num_samples, len(original))
        
        if generated is not None:
            rows = 3
            titles = ['原始 / Original', '重构 / Reconstructed', '生成 / Generated']
        else:
            rows = 2
            titles = ['原始 / Original', '重构 / Reconstructed']
        
        fig, axes = plt.subplots(rows, num_samples, figsize=(2*num_samples, 2*rows))
        
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        # 反归一化所有图像 / Denormalize all images
        original = self.denormalize_tensor(original[:num_samples], normalize_range)
        reconstructed = self.denormalize_tensor(reconstructed[:num_samples], normalize_range)
        if generated is not None:
            generated = self.denormalize_tensor(generated[:num_samples], normalize_range)
        
        # 转换为numpy / Convert to numpy
        original = original.detach().cpu().numpy()
        reconstructed = reconstructed.detach().cpu().numpy()
        if generated is not None:
            generated = generated.detach().cpu().numpy()
        
        for i in range(num_samples):
            # 显示原始图像 / Display original images
            img = original[i]
            if len(img.shape) == 3 and img.shape[0] == 1:
                img = img.squeeze(0)
            if len(img.shape) == 3 and img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))
            axes[0, i].imshow(img, cmap='gray' if len(img.shape) == 2 else None)
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_ylabel(titles[0], fontsize=12)
            
            # 显示重构图像 / Display reconstructed images
            img = reconstructed[i]
            if len(img.shape) == 3 and img.shape[0] == 1:
                img = img.squeeze(0)
            if len(img.shape) == 3 and img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))
            axes[1, i].imshow(img, cmap='gray' if len(img.shape) == 2 else None)
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_ylabel(titles[1], fontsize=12)
            
            # 显示生成图像 / Display generated images
            if generated is not None:
                img = generated[i]
                if len(img.shape) == 3 and img.shape[0] == 1:
                    img = img.squeeze(0)
                if len(img.shape) == 3 and img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
                axes[2, i].imshow(img, cmap='gray' if len(img.shape) == 2 else None)
                axes[2, i].axis('off')
                if i == 0:
                    axes[2, i].set_ylabel(titles[2], fontsize=12)
        
        plt.tight_layout()
        
        if save_name:
            save_path = os.path.join(self.save_dir, save_name)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"比较图像已保存到 / Comparison image saved to: {save_path}")
        
        plt.show()
    
    def plot_training_curves(self, losses, labels=None, title="训练损失曲线 / Training Loss Curves", 
                           xlabel="轮次 / Epoch", ylabel="损失 / Loss", save_name=None):
        """
        绘制训练损失曲线 / Plot training loss curves
        
        可视化训练过程中的损失变化，帮助分析模型训练状态。
        Visualize loss changes during training to help analyze model training status.
        
        Args:
            losses: 损失值列表或字典 / List or dict of loss values
            labels: 损失标签 / Loss labels
            title: 图表标题 / Chart title
            xlabel: X轴标签 / X-axis label
            ylabel: Y轴标签 / Y-axis label
            save_name: 保存文件名 / Save filename
        """
        plt.figure(figsize=(12, 6))
        
        if isinstance(losses, dict):
            for name, loss_values in losses.items():
                plt.plot(loss_values, label=name, linewidth=2)
        elif isinstance(losses, list) and isinstance(losses[0], list):
            # 多个损失序列 / Multiple loss sequences
            for i, loss_values in enumerate(losses):
                label = labels[i] if labels and i < len(labels) else f'损失 {i+1} / Loss {i+1}'
                plt.plot(loss_values, label=label, linewidth=2)
        else:
            # 单个损失序列 / Single loss sequence
            label = labels[0] if labels else '损失 / Loss'
            plt.plot(losses, label=label, linewidth=2)
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_name:
            save_path = os.path.join(self.save_dir, save_name)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"损失曲线已保存到 / Loss curve saved to: {save_path}")
        
        plt.show()
    
    def plot_latent_space(self, latent_vectors, labels=None, method='tsne', 
                         title="潜在空间可视化 / Latent Space Visualization", save_name=None):
        """
        可视化潜在空间 / Visualize latent space
        
        使用降维技术将高维潜在向量投影到2D空间进行可视化。
        Use dimensionality reduction techniques to project high-dimensional latent vectors to 2D space for visualization.
        
        Args:
            latent_vectors: 潜在向量 / Latent vectors (num_samples, latent_dim)
            labels: 数据标签 / Data labels (可选 / optional)
            method: 降维方法 / Dimensionality reduction method ('tsne' or 'pca')
            title: 图表标题 / Chart title
            save_name: 保存文件名 / Save filename
        """
        # 转换为numpy数组 / Convert to numpy array
        if isinstance(latent_vectors, torch.Tensor):
            latent_vectors = latent_vectors.detach().cpu().numpy()
        
        # 如果维度大于2，进行降维 / If dimension > 2, perform dimensionality reduction
        if latent_vectors.shape[1] > 2:
            if method == 'tsne':
                reducer = TSNE(n_components=2, random_state=42, perplexity=30)
                reduced_vectors = reducer.fit_transform(latent_vectors)
            elif method == 'pca':
                reducer = PCA(n_components=2, random_state=42)
                reduced_vectors = reducer.fit_transform(latent_vectors)
            else:
                raise ValueError("method must be 'tsne' or 'pca'")
        else:
            reduced_vectors = latent_vectors
        
        plt.figure(figsize=(10, 8))
        
        if labels is not None:
            # 如果有标签，按类别着色 / If labels exist, color by category
            if isinstance(labels, torch.Tensor):
                labels = labels.detach().cpu().numpy()
            
            unique_labels = np.unique(labels)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                plt.scatter(reduced_vectors[mask, 0], reduced_vectors[mask, 1], 
                          c=[colors[i]], label=f'类别 {label} / Class {label}', 
                          alpha=0.7, s=50)
            plt.legend()
        else:
            # 没有标签时，使用单一颜色 / Use single color when no labels
            plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], 
                       alpha=0.7, s=50, c='blue')
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel(f'{method.upper()} 组件 1 / {method.upper()} Component 1', fontsize=12)
        plt.ylabel(f'{method.upper()} 组件 2 / {method.upper()} Component 2', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        if save_name:
            save_path = os.path.join(self.save_dir, save_name)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"潜在空间可视化已保存到 / Latent space visualization saved to: {save_path}")
        
        plt.show()
    
    def plot_reconstruction_error(self, original, reconstructed, normalize_range=(-1, 1),
                                 title="重构误差分析 / Reconstruction Error Analysis", save_name=None):
        """
        分析重构误差 / Analyze reconstruction error
        
        计算并可视化原始图像和重构图像之间的误差。
        Calculate and visualize error between original and reconstructed images.
        
        Args:
            original: 原始图像 / Original images
            reconstructed: 重构图像 / Reconstructed images
            normalize_range: 归一化范围 / Normalization range
            title: 图表标题 / Chart title
            save_name: 保存文件名 / Save filename
        """
        # 反归一化 / Denormalize
        original = self.denormalize_tensor(original, normalize_range)
        reconstructed = self.denormalize_tensor(reconstructed, normalize_range)
        
        # 计算重构误差 / Calculate reconstruction error
        error = torch.abs(original - reconstructed)
        mse_per_image = torch.mean(error.view(error.size(0), -1), dim=1)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 误差分布直方图 / Error distribution histogram
        axes[0, 0].hist(mse_per_image.cpu().numpy(), bins=30, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('重构误差分布 / Reconstruction Error Distribution')
        axes[0, 0].set_xlabel('平均绝对误差 / Mean Absolute Error')
        axes[0, 0].set_ylabel('频次 / Frequency')
        
        # 误差随样本变化 / Error vs sample index
        axes[0, 1].plot(mse_per_image.cpu().numpy(), 'o-', alpha=0.7)
        axes[0, 1].set_title('重构误差变化 / Reconstruction Error Variation')
        axes[0, 1].set_xlabel('样本索引 / Sample Index')
        axes[0, 1].set_ylabel('平均绝对误差 / Mean Absolute Error')
        
        # 显示误差最大的图像 / Show image with largest error
        max_error_idx = torch.argmax(mse_per_image)
        worst_original = original[max_error_idx].squeeze()
        worst_reconstructed = reconstructed[max_error_idx].squeeze()
        worst_error = error[max_error_idx].squeeze()
        
        if len(worst_original.shape) == 3 and worst_original.shape[0] == 1:
            worst_original = worst_original.squeeze(0)
            worst_reconstructed = worst_reconstructed.squeeze(0)
            worst_error = worst_error.squeeze(0)
        
        # 转换为numpy / Convert to numpy
        worst_original = worst_original.cpu().numpy()
        worst_reconstructed = worst_reconstructed.cpu().numpy()
        worst_error = worst_error.cpu().numpy()
        
        axes[1, 0].imshow(worst_original, cmap='gray' if len(worst_original.shape) == 2 else None)
        axes[1, 0].set_title(f'最大误差原图 / Original (Error: {mse_per_image[max_error_idx]:.4f})')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(worst_error, cmap='hot')
        axes[1, 1].set_title('误差热图 / Error Heatmap')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_name:
            save_path = os.path.join(self.save_dir, save_name)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"重构误差分析已保存到 / Reconstruction error analysis saved to: {save_path}")
        
        plt.show()
        
        # 返回统计信息 / Return statistics
        return {
            'mean_error': torch.mean(mse_per_image).item(),
            'std_error': torch.std(mse_per_image).item(),
            'max_error': torch.max(mse_per_image).item(),
            'min_error': torch.min(mse_per_image).item()
        }


if __name__ == "__main__":
    # 测试可视化器 / Test visualizer
    print("测试可视化工具 / Testing Visualizer")
    
    visualizer = Visualizer('./test_results')
    
    # 创建测试数据 / Create test data
    test_images = torch.randn(8, 1, 28, 28)
    test_labels = torch.randint(0, 10, (8,))
    
    # 测试图像显示 / Test image display
    print("测试图像显示 / Testing image display")
    visualizer.show_images(test_images, save_name='test_images.png')
    
    # 测试损失曲线 / Test loss curves
    print("测试损失曲线 / Testing loss curves")
    fake_losses = {
        '生成器损失 / Generator Loss': np.random.exponential(2, 100),
        '判别器损失 / Discriminator Loss': np.random.exponential(1.5, 100)
    }
    visualizer.plot_training_curves(fake_losses, save_name='test_loss_curves.png')
    
    # 测试潜在空间可视化 / Test latent space visualization
    print("测试潜在空间可视化 / Testing latent space visualization")
    latent_vectors = torch.randn(100, 20)
    latent_labels = torch.randint(0, 5, (100,))
    visualizer.plot_latent_space(latent_vectors, latent_labels, 
                                method='pca', save_name='test_latent_space.png') 