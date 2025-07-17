import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import argparse

# 添加父目录到路径以导入模块 / Add parent directory to path for importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autoencoder.basic_autoencoder import BasicAutoencoder, ConvAutoencoder, autoencoder_loss
from autoencoder.variational_autoencoder import VariationalAutoencoder, ConvVariationalAutoencoder, vae_loss
from utils.data_loader import get_mnist_dataloader, get_fashion_mnist_dataloader
from utils.visualizer import Visualizer

class AutoencoderTrainer:
    """
    自编码器训练器 / Autoencoder Trainer
    
    负责训练各种类型的自编码器，包括基础自编码器、卷积自编码器和变分自编码器。
    Responsible for training various types of autoencoders including basic, convolutional, and variational autoencoders.
    """
    
    def __init__(self, model_type='basic', dataset='mnist', batch_size=128, 
                 learning_rate=1e-3, device='auto', save_dir='./results/autoencoder'):
        """
        初始化训练器
        Initialize trainer
        
        Args:
            model_type: 模型类型 / Model type ('basic', 'conv', 'vae', 'conv_vae')
            dataset: 数据集 / Dataset ('mnist', 'fashion_mnist')
            batch_size: 批次大小 / Batch size
            learning_rate: 学习率 / Learning rate
            device: 设备 / Device ('auto', 'cpu', 'cuda')
            save_dir: 保存目录 / Save directory
        """
        self.model_type = model_type
        self.dataset = dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.save_dir = save_dir
        
        # 设置设备 / Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"使用设备 / Using device: {self.device}")
        
        # 创建保存目录 / Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # 初始化可视化器 / Initialize visualizer
        self.visualizer = Visualizer(save_dir)
        
        # 加载数据 / Load data
        self._load_data()
        
        # 初始化模型 / Initialize model
        self._init_model()
        
        # 初始化优化器 / Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # 训练历史 / Training history
        self.train_losses = []
        self.val_losses = []
        
        if self.model_type in ['vae', 'conv_vae']:
            self.recon_losses = []
            self.kl_losses = []
    
    def _load_data(self):
        """加载数据集 / Load dataset"""
        print(f"加载{self.dataset}数据集 / Loading {self.dataset} dataset...")
        
        if self.dataset == 'mnist':
            self.train_loader = get_mnist_dataloader(
                batch_size=self.batch_size, 
                train=True, 
                normalize_range=(0, 1),  # 自编码器使用[0,1]范围 / Autoencoder uses [0,1] range
                download=True
            )
            self.val_loader = get_mnist_dataloader(
                batch_size=self.batch_size, 
                train=False, 
                normalize_range=(0, 1),
                download=True
            )
        elif self.dataset == 'fashion_mnist':
            self.train_loader = get_fashion_mnist_dataloader(
                batch_size=self.batch_size, 
                train=True, 
                normalize_range=(0, 1),
                download=True
            )
            self.val_loader = get_fashion_mnist_dataloader(
                batch_size=self.batch_size, 
                train=False, 
                normalize_range=(0, 1),
                download=True
            )
        else:
            raise ValueError(f"不支持的数据集 / Unsupported dataset: {self.dataset}")
        
        print(f"训练集大小 / Training set size: {len(self.train_loader.dataset)}")
        print(f"验证集大小 / Validation set size: {len(self.val_loader.dataset)}")
    
    def _init_model(self):
        """初始化模型 / Initialize model"""
        print(f"初始化{self.model_type}模型 / Initializing {self.model_type} model...")
        
        if self.model_type == 'basic':
            self.model = BasicAutoencoder(
                input_dim=784,
                hidden_dim=128,
                latent_dim=32
            ).to(self.device)
            
        elif self.model_type == 'conv':
            self.model = ConvAutoencoder(
                channels=1,
                latent_dim=128
            ).to(self.device)
            
        elif self.model_type == 'vae':
            self.model = VariationalAutoencoder(
                input_dim=784,
                hidden_dim=400,
                latent_dim=20
            ).to(self.device)
            
        elif self.model_type == 'conv_vae':
            self.model = ConvVariationalAutoencoder(
                channels=1,
                latent_dim=128
            ).to(self.device)
            
        else:
            raise ValueError(f"不支持的模型类型 / Unsupported model type: {self.model_type}")
        
        # 打印模型信息 / Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"模型参数总数 / Total parameters: {total_params:,}")
        print(f"可训练参数 / Trainable parameters: {trainable_params:,}")
    
    def _compute_loss(self, data):
        """
        计算损失 / Compute loss
        
        Args:
            data: 输入数据 / Input data
            
        Returns:
            loss: 损失值 / Loss value
            additional_info: 额外信息 / Additional information
        """
        if self.model_type in ['basic', 'conv']:
            # 基础自编码器损失 / Basic autoencoder loss
            reconstructed, latent = self.model(data)
            loss = autoencoder_loss(reconstructed, data)
            return loss, {'reconstructed': reconstructed, 'latent': latent}
            
        elif self.model_type in ['vae', 'conv_vae']:
            # 变分自编码器损失 / Variational autoencoder loss
            reconstructed, mu, logvar = self.model(data)
            total_loss, recon_loss, kl_loss = vae_loss(reconstructed, data, mu, logvar)
            return total_loss, {
                'reconstructed': reconstructed,
                'mu': mu,
                'logvar': logvar,
                'recon_loss': recon_loss,
                'kl_loss': kl_loss
            }
    
    def train_epoch(self):
        """训练一个轮次 / Train one epoch"""
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc="训练 / Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            if isinstance(batch, tuple):
                data, _ = batch  # 忽略标签 / Ignore labels
            else:
                data = batch
            
            data = data.to(self.device)
            
            # 前向传播 / Forward pass
            self.optimizer.zero_grad()
            loss, info = self._compute_loss(data)
            
            # 反向传播 / Backward pass
            loss.backward()
            self.optimizer.step()
            
            # 记录损失 / Record loss
            total_loss += loss.item()
            
            if self.model_type in ['vae', 'conv_vae']:
                total_recon_loss += info['recon_loss'].item()
                total_kl_loss += info['kl_loss'].item()
            
            # 更新进度条 / Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        
        if self.model_type in ['vae', 'conv_vae']:
            avg_recon_loss = total_recon_loss / len(self.train_loader)
            avg_kl_loss = total_kl_loss / len(self.train_loader)
            self.recon_losses.append(avg_recon_loss)
            self.kl_losses.append(avg_kl_loss)
            return avg_loss, avg_recon_loss, avg_kl_loss
        
        return avg_loss
    
    def validate(self):
        """验证模型 / Validate model"""
        self.model.eval()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                if isinstance(batch, tuple):
                    data, _ = batch
                else:
                    data = batch
                
                data = data.to(self.device)
                
                loss, info = self._compute_loss(data)
                total_loss += loss.item()
                
                if self.model_type in ['vae', 'conv_vae']:
                    total_recon_loss += info['recon_loss'].item()
                    total_kl_loss += info['kl_loss'].item()
        
        avg_loss = total_loss / len(self.val_loader)
        self.val_losses.append(avg_loss)
        
        if self.model_type in ['vae', 'conv_vae']:
            avg_recon_loss = total_recon_loss / len(self.val_loader)
            avg_kl_loss = total_kl_loss / len(self.val_loader)
            return avg_loss, avg_recon_loss, avg_kl_loss
        
        return avg_loss
    
    def train(self, epochs=50, save_interval=10, visualize_interval=5):
        """
        训练模型 / Train model
        
        Args:
            epochs: 训练轮次 / Training epochs
            save_interval: 保存模型间隔 / Model save interval
            visualize_interval: 可视化间隔 / Visualization interval
        """
        print(f"开始训练{epochs}个轮次 / Starting training for {epochs} epochs...")
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            print(f"\n轮次 {epoch+1}/{epochs} / Epoch {epoch+1}/{epochs}")
            
            # 训练 / Train
            train_result = self.train_epoch()
            
            # 验证 / Validate
            val_result = self.validate()
            
            # 打印结果 / Print results
            if self.model_type in ['vae', 'conv_vae']:
                train_loss, train_recon, train_kl = train_result
                val_loss, val_recon, val_kl = val_result
                print(f"训练损失 / Train Loss: {train_loss:.4f} (重构 / Recon: {train_recon:.4f}, KL: {train_kl:.4f})")
                print(f"验证损失 / Val Loss: {val_loss:.4f} (重构 / Recon: {val_recon:.4f}, KL: {val_kl:.4f})")
            else:
                train_loss = train_result
                val_loss = val_result
                print(f"训练损失 / Train Loss: {train_loss:.4f}")
                print(f"验证损失 / Val Loss: {val_loss:.4f}")
            
            # 保存最佳模型 / Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(f'best_{self.model_type}_model.pth')
            
            # 定期保存模型 / Periodically save model
            if (epoch + 1) % save_interval == 0:
                self.save_model(f'{self.model_type}_model_epoch_{epoch+1}.pth')
            
            # 定期可视化 / Periodically visualize
            if (epoch + 1) % visualize_interval == 0:
                self.visualize_results(epoch+1)
        
        # 训练完成后的最终可视化 / Final visualization after training
        self.plot_training_curves()
        self.visualize_results(epochs, final=True)
        
        print(f"训练完成！最佳验证损失 / Training completed! Best validation loss: {best_val_loss:.4f}")
    
    def visualize_results(self, epoch, final=False):
        """
        可视化结果 / Visualize results
        
        Args:
            epoch: 当前轮次 / Current epoch
            final: 是否为最终可视化 / Whether this is final visualization
        """
        self.model.eval()
        
        with torch.no_grad():
            # 获取验证数据 / Get validation data
            val_iter = iter(self.val_loader)
            batch = next(val_iter)
            
            if isinstance(batch, tuple):
                data, _ = batch
            else:
                data = batch
            
            data = data.to(self.device)
            
            # 重构图像 / Reconstruct images
            if self.model_type in ['basic', 'conv']:
                reconstructed, latent = self.model(data)
            else:  # VAE
                reconstructed, mu, logvar = self.model(data)
            
            # 比较原始和重构图像 / Compare original and reconstructed images
            save_name = f'reconstruction_epoch_{epoch}.png' if not final else 'final_reconstruction.png'
            self.visualizer.compare_images(
                data[:8], 
                reconstructed[:8], 
                normalize_range=(0, 1),
                save_name=save_name
            )
            
            # 如果是VAE，生成新样本 / If VAE, generate new samples
            if self.model_type in ['vae', 'conv_vae']:
                generated_samples = self.model.sample(16, self.device)
                save_name = f'generated_samples_epoch_{epoch}.png' if not final else 'final_generated_samples.png'
                self.visualizer.show_images(
                    generated_samples,
                    normalize_range=(0, 1),
                    rows=4,
                    cols=4,
                    save_name=save_name
                )
    
    def plot_training_curves(self):
        """绘制训练曲线 / Plot training curves"""
        if self.model_type in ['vae', 'conv_vae']:
            # VAE损失曲线 / VAE loss curves
            losses = {
                '训练总损失 / Train Total Loss': self.train_losses,
                '验证总损失 / Val Total Loss': self.val_losses,
                '重构损失 / Reconstruction Loss': self.recon_losses,
                'KL散度损失 / KL Divergence Loss': self.kl_losses
            }
        else:
            # 基础自编码器损失曲线 / Basic autoencoder loss curves
            losses = {
                '训练损失 / Train Loss': self.train_losses,
                '验证损失 / Val Loss': self.val_losses
            }
        
        self.visualizer.plot_training_curves(
            losses,
            title=f"{self.model_type.upper()} 训练损失曲线 / {self.model_type.upper()} Training Loss Curves",
            save_name='training_curves.png'
        )
    
    def save_model(self, filename):
        """
        保存模型 / Save model
        
        Args:
            filename: 文件名 / Filename
        """
        filepath = os.path.join(self.save_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_type': self.model_type,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'recon_losses': getattr(self, 'recon_losses', []),
            'kl_losses': getattr(self, 'kl_losses', [])
        }, filepath)
        print(f"模型已保存到 / Model saved to: {filepath}")


def main():
    """主函数 / Main function"""
    parser = argparse.ArgumentParser(description='训练自编码器 / Train Autoencoder')
    parser.add_argument('--model_type', type=str, default='basic', 
                       choices=['basic', 'conv', 'vae', 'conv_vae'],
                       help='模型类型 / Model type')
    parser.add_argument('--dataset', type=str, default='mnist',
                       choices=['mnist', 'fashion_mnist'],
                       help='数据集 / Dataset')
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮次 / Training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='批次大小 / Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='学习率 / Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='设备 / Device')
    parser.add_argument('--save_dir', type=str, default='./results/autoencoder',
                       help='保存目录 / Save directory')
    
    args = parser.parse_args()
    
    # 创建训练器 / Create trainer
    trainer = AutoencoderTrainer(
        model_type=args.model_type,
        dataset=args.dataset,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        save_dir=args.save_dir
    )
    
    # 开始训练 / Start training
    trainer.train(epochs=args.epochs)


if __name__ == "__main__":
    main() 