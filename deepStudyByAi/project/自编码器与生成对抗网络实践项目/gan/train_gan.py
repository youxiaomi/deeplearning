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

from gan.basic_gan import BasicGAN
from gan.dcgan import DCGAN
from utils.data_loader import get_mnist_dataloader, get_fashion_mnist_dataloader
from utils.visualizer import Visualizer

class GANTrainer:
    """
    GAN训练器 / GAN Trainer
    
    负责训练生成对抗网络，包括基础GAN和深度卷积GAN (DCGAN)。
    训练过程中需要平衡生成器和判别器的训练，避免模式崩溃。
    
    Responsible for training Generative Adversarial Networks including basic GAN and DCGAN.
    Training process requires balancing generator and discriminator training to avoid mode collapse.
    """
    
    def __init__(self, model_type='basic', dataset='mnist', batch_size=128, 
                 g_lr=0.0002, d_lr=0.0002, device='auto', save_dir='./results/gan'):
        """
        初始化GAN训练器
        Initialize GAN trainer
        
        Args:
            model_type: 模型类型 / Model type ('basic', 'dcgan')
            dataset: 数据集 / Dataset ('mnist', 'fashion_mnist')
            batch_size: 批次大小 / Batch size
            g_lr: 生成器学习率 / Generator learning rate
            d_lr: 判别器学习率 / Discriminator learning rate
            device: 设备 / Device ('auto', 'cpu', 'cuda')
            save_dir: 保存目录 / Save directory
        """
        self.model_type = model_type
        self.dataset = dataset
        self.batch_size = batch_size
        self.g_lr = g_lr
        self.d_lr = d_lr
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
        
        # 训练历史 / Training history
        self.g_losses = []  # 生成器损失 / Generator losses
        self.d_losses = []  # 判别器损失 / Discriminator losses
        self.d_real_accs = []  # 判别器对真实数据的准确率 / Discriminator accuracy on real data
        self.d_fake_accs = []  # 判别器对生成数据的准确率 / Discriminator accuracy on fake data
        
        # 固定噪声用于可视化 / Fixed noise for visualization
        self.fixed_noise = torch.randn(64, self.gan.noise_dim, device=self.device)
    
    def _load_data(self):
        """加载数据集 / Load dataset"""
        print(f"加载{self.dataset}数据集 / Loading {self.dataset} dataset...")
        
        if self.dataset == 'mnist':
            self.train_loader = get_mnist_dataloader(
                batch_size=self.batch_size, 
                train=True, 
                normalize_range=(-1, 1),  # GAN使用[-1,1]范围 / GAN uses [-1,1] range
                download=True
            )
        elif self.dataset == 'fashion_mnist':
            self.train_loader = get_fashion_mnist_dataloader(
                batch_size=self.batch_size, 
                train=True, 
                normalize_range=(-1, 1),
                download=True
            )
        else:
            raise ValueError(f"不支持的数据集 / Unsupported dataset: {self.dataset}")
        
        print(f"训练集大小 / Training set size: {len(self.train_loader.dataset)}")
    
    def _init_model(self):
        """初始化模型 / Initialize model"""
        print(f"初始化{self.model_type} GAN模型 / Initializing {self.model_type} GAN model...")
        
        if self.model_type == 'basic':
            self.gan = BasicGAN(
                noise_dim=100,
                hidden_dim=256,
                data_dim=784,
                device=self.device
            )
        elif self.model_type == 'dcgan':
            self.gan = DCGAN(
                noise_dim=100,
                channels=1,
                feature_map_size=64,
                device=self.device
            )
        else:
            raise ValueError(f"不支持的模型类型 / Unsupported model type: {self.model_type}")
        
        # 设置优化器 / Set optimizers
        self.gan.set_optimizers(g_lr=self.g_lr, d_lr=self.d_lr)
        
        # 打印模型信息 / Print model info
        g_params = sum(p.numel() for p in self.gan.generator.parameters())
        d_params = sum(p.numel() for p in self.gan.discriminator.parameters())
        print(f"生成器参数 / Generator parameters: {g_params:,}")
        print(f"判别器参数 / Discriminator parameters: {d_params:,}")
        print(f"总参数 / Total parameters: {g_params + d_params:,}")
    
    def train_epoch(self, epoch):
        """
        训练一个轮次 / Train one epoch
        
        Args:
            epoch: 当前轮次 / Current epoch
        """
        epoch_g_loss = 0
        epoch_d_loss = 0
        epoch_d_real_acc = 0
        epoch_d_fake_acc = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"轮次 {epoch} / Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            if isinstance(batch, tuple):
                real_data, _ = batch  # 忽略标签 / Ignore labels
            else:
                real_data = batch
            
            real_data = real_data.to(self.device)
            current_batch_size = real_data.size(0)
            
            # 训练判别器 / Train discriminator
            if self.model_type == 'basic':
                d_loss = self.gan.train_discriminator(real_data, current_batch_size)
                d_real_acc = d_fake_acc = 0  # 基础GAN没有返回准确率 / Basic GAN doesn't return accuracy
            else:  # DCGAN
                d_loss, d_real_acc, d_fake_acc = self.gan.train_discriminator(real_data, current_batch_size)
            
            # 训练生成器 / Train generator
            g_loss = self.gan.train_generator(current_batch_size)
            
            # 记录损失 / Record losses
            epoch_g_loss += g_loss
            epoch_d_loss += d_loss
            epoch_d_real_acc += d_real_acc
            epoch_d_fake_acc += d_fake_acc
            
            # 更新进度条 / Update progress bar
            progress_bar.set_postfix({
                'G_Loss': f'{g_loss:.4f}',
                'D_Loss': f'{d_loss:.4f}',
                'D_Real': f'{d_real_acc:.3f}',
                'D_Fake': f'{d_fake_acc:.3f}'
            })
        
        # 计算平均值 / Calculate averages
        num_batches = len(self.train_loader)
        avg_g_loss = epoch_g_loss / num_batches
        avg_d_loss = epoch_d_loss / num_batches
        avg_d_real_acc = epoch_d_real_acc / num_batches
        avg_d_fake_acc = epoch_d_fake_acc / num_batches
        
        # 保存到历史记录 / Save to history
        self.g_losses.append(avg_g_loss)
        self.d_losses.append(avg_d_loss)
        self.d_real_accs.append(avg_d_real_acc)
        self.d_fake_accs.append(avg_d_fake_acc)
        
        return avg_g_loss, avg_d_loss, avg_d_real_acc, avg_d_fake_acc
    
    def generate_and_save_samples(self, epoch, num_samples=64):
        """
        生成并保存样本 / Generate and save samples
        
        Args:
            epoch: 当前轮次 / Current epoch
            num_samples: 生成样本数量 / Number of samples to generate
        """
        self.gan.generator.eval()
        
        with torch.no_grad():
            # 使用固定噪声生成样本以便比较 / Use fixed noise to generate samples for comparison
            generated_samples = self.gan.generator(self.fixed_noise[:num_samples])
            
            # 如果是基础GAN，需要重塑为图像 / If basic GAN, need to reshape to images
            if self.model_type == 'basic':
                generated_samples = generated_samples.view(-1, 1, 28, 28)
            
            # 保存生成的样本 / Save generated samples
            self.visualizer.show_images(
                generated_samples,
                normalize_range=(-1, 1),
                rows=8,
                cols=8,
                save_name=f'generated_samples_epoch_{epoch}.png'
            )
        
        self.gan.generator.train()
    
    def evaluate_gan_quality(self, epoch):
        """
        评估GAN质量 / Evaluate GAN quality
        
        计算一些基本的质量指标，如判别器准确率平衡等。
        Calculate some basic quality metrics like discriminator accuracy balance.
        
        Args:
            epoch: 当前轮次 / Current epoch
        """
        self.gan.generator.eval()
        self.gan.discriminator.eval()
        
        total_real_score = 0
        total_fake_score = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in list(self.train_loader)[:10]:  # 只评估前10个批次 / Only evaluate first 10 batches
                if isinstance(batch, tuple):
                    real_data, _ = batch
                else:
                    real_data = batch
                
                real_data = real_data.to(self.device)
                batch_size = real_data.size(0)
                
                # 评估判别器对真实数据的评分 / Evaluate discriminator score on real data
                real_pred = self.gan.discriminator(real_data)
                total_real_score += real_pred.mean().item()
                
                # 生成假数据并评估 / Generate fake data and evaluate
                if self.model_type == 'basic':
                    noise = self.gan.generator.generate_noise(batch_size, self.device)
                    fake_data = self.gan.generator(noise)
                else:  # DCGAN
                    noise = self.gan.generator.generate_noise(batch_size, self.device)
                    fake_data = self.gan.generator(noise)
                
                fake_pred = self.gan.discriminator(fake_data)
                total_fake_score += fake_pred.mean().item()
                
                num_batches += 1
        
        avg_real_score = total_real_score / num_batches
        avg_fake_score = total_fake_score / num_batches
        
        print(f"轮次 {epoch} 质量评估 / Epoch {epoch} Quality Assessment:")
        print(f"  真实数据平均评分 / Average real score: {avg_real_score:.4f}")
        print(f"  生成数据平均评分 / Average fake score: {avg_fake_score:.4f}")
        print(f"  评分差异 / Score difference: {abs(avg_real_score - avg_fake_score):.4f}")
        
        self.gan.generator.train()
        self.gan.discriminator.train()
        
        return avg_real_score, avg_fake_score
    
    def train(self, epochs=100, save_interval=20, visualize_interval=10, evaluate_interval=10):
        """
        训练GAN / Train GAN
        
        Args:
            epochs: 训练轮次 / Training epochs
            save_interval: 保存模型间隔 / Model save interval
            visualize_interval: 可视化间隔 / Visualization interval
            evaluate_interval: 评估间隔 / Evaluation interval
        """
        print(f"开始训练{epochs}个轮次 / Starting training for {epochs} epochs...")
        print("=" * 60)
        
        for epoch in range(1, epochs + 1):
            # 训练一个轮次 / Train one epoch
            g_loss, d_loss, d_real_acc, d_fake_acc = self.train_epoch(epoch)
            
            # 打印训练结果 / Print training results
            print(f"\n轮次 {epoch}/{epochs} 完成 / Epoch {epoch}/{epochs} completed:")
            print(f"  生成器损失 / Generator Loss: {g_loss:.4f}")
            print(f"  判别器损失 / Discriminator Loss: {d_loss:.4f}")
            if self.model_type == 'dcgan':
                print(f"  判别器真实准确率 / Discriminator Real Acc: {d_real_acc:.4f}")
                print(f"  判别器假数据准确率 / Discriminator Fake Acc: {d_fake_acc:.4f}")
            
            # 检查训练稳定性 / Check training stability
            if epoch > 10:
                recent_g_losses = self.g_losses[-10:]
                recent_d_losses = self.d_losses[-10:]
                g_std = np.std(recent_g_losses)
                d_std = np.std(recent_d_losses)
                
                if g_std < 0.01 or d_std < 0.01:
                    print("  ⚠️  警告：损失变化很小，可能出现模式崩溃 / Warning: Small loss variation, possible mode collapse")
                
                if g_loss > 5.0 or d_loss > 5.0:
                    print("  ⚠️  警告：损失过大，训练可能不稳定 / Warning: Large loss, training may be unstable")
            
            # 定期生成样本 / Periodically generate samples
            if epoch % visualize_interval == 0:
                self.generate_and_save_samples(epoch)
            
            # 定期评估质量 / Periodically evaluate quality
            if epoch % evaluate_interval == 0:
                self.evaluate_gan_quality(epoch)
            
            # 定期保存模型 / Periodically save model
            if epoch % save_interval == 0:
                self.save_models(f'epoch_{epoch}')
            
            print("-" * 40)
        
        # 训练完成后的最终操作 / Final operations after training completion
        print("\n训练完成！/ Training completed!")
        
        # 最终可视化 / Final visualization
        self.generate_and_save_samples(epochs, num_samples=64)
        self.plot_training_curves()
        self.evaluate_gan_quality(epochs)
        
        # 保存最终模型 / Save final model
        self.save_models('final')
        
        print(f"所有结果已保存到 / All results saved to: {self.save_dir}")
    
    def plot_training_curves(self):
        """绘制训练曲线 / Plot training curves"""
        losses = {
            '生成器损失 / Generator Loss': self.g_losses,
            '判别器损失 / Discriminator Loss': self.d_losses
        }
        
        self.visualizer.plot_training_curves(
            losses,
            title=f"{self.model_type.upper()} GAN 训练损失曲线 / {self.model_type.upper()} GAN Training Loss Curves",
            save_name='training_curves.png'
        )
        
        # 如果有准确率数据，也绘制准确率曲线 / If accuracy data exists, also plot accuracy curves
        if self.model_type == 'dcgan' and len(self.d_real_accs) > 0:
            accuracies = {
                '判别器真实数据准确率 / Discriminator Real Accuracy': self.d_real_accs,
                '判别器生成数据准确率 / Discriminator Fake Accuracy': self.d_fake_accs
            }
            
            self.visualizer.plot_training_curves(
                accuracies,
                title="判别器准确率曲线 / Discriminator Accuracy Curves",
                ylabel="准确率 / Accuracy",
                save_name='accuracy_curves.png'
            )
    
    def save_models(self, suffix):
        """
        保存模型 / Save models
        
        Args:
            suffix: 文件名后缀 / Filename suffix
        """
        generator_path = os.path.join(self.save_dir, f'generator_{suffix}.pth')
        discriminator_path = os.path.join(self.save_dir, f'discriminator_{suffix}.pth')
        
        # 保存生成器 / Save generator
        torch.save({
            'model_state_dict': self.gan.generator.state_dict(),
            'optimizer_state_dict': self.gan.g_optimizer.state_dict(),
            'model_type': self.model_type,
            'noise_dim': self.gan.noise_dim
        }, generator_path)
        
        # 保存判别器 / Save discriminator
        torch.save({
            'model_state_dict': self.gan.discriminator.state_dict(),
            'optimizer_state_dict': self.gan.d_optimizer.state_dict(),
            'model_type': self.model_type
        }, discriminator_path)
        
        # 保存训练历史 / Save training history
        history_path = os.path.join(self.save_dir, f'training_history_{suffix}.pth')
        torch.save({
            'g_losses': self.g_losses,
            'd_losses': self.d_losses,
            'd_real_accs': self.d_real_accs,
            'd_fake_accs': self.d_fake_accs,
            'fixed_noise': self.fixed_noise
        }, history_path)
        
        print(f"模型已保存 / Models saved:")
        print(f"  生成器 / Generator: {generator_path}")
        print(f"  判别器 / Discriminator: {discriminator_path}")
        print(f"  训练历史 / Training history: {history_path}")


def main():
    """主函数 / Main function"""
    parser = argparse.ArgumentParser(description='训练生成对抗网络 / Train GAN')
    parser.add_argument('--model_type', type=str, default='basic',
                       choices=['basic', 'dcgan'],
                       help='模型类型 / Model type')
    parser.add_argument('--dataset', type=str, default='mnist',
                       choices=['mnist', 'fashion_mnist'],
                       help='数据集 / Dataset')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮次 / Training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='批次大小 / Batch size')
    parser.add_argument('--g_lr', type=float, default=0.0002,
                       help='生成器学习率 / Generator learning rate')
    parser.add_argument('--d_lr', type=float, default=0.0002,
                       help='判别器学习率 / Discriminator learning rate')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='设备 / Device')
    parser.add_argument('--save_dir', type=str, default='./results/gan',
                       help='保存目录 / Save directory')
    
    args = parser.parse_args()
    
    # 创建训练器 / Create trainer
    trainer = GANTrainer(
        model_type=args.model_type,
        dataset=args.dataset,
        batch_size=args.batch_size,
        g_lr=args.g_lr,
        d_lr=args.d_lr,
        device=args.device,
        save_dir=args.save_dir
    )
    
    # 开始训练 / Start training
    trainer.train(epochs=args.epochs)


if __name__ == "__main__":
    main() 