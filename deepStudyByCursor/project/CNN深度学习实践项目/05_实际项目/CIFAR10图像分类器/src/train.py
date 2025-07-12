"""
Training Script for CIFAR-10 CNN Models
CIFAR-10 CNN模型训练脚本

This script trains CNN models on CIFAR-10 dataset.
Think of this as the gym where your AI learns to recognize images!
这个脚本在CIFAR-10数据集上训练CNN模型。
把这想象成AI学习识别图像的健身房！
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import time
import argparse
import os
from tqdm import tqdm

# Import our custom modules
# 导入我们的自定义模块
from data_loader import CIFAR10DataLoader
from model import get_model
from utils import (
    EarlyStopping, TrainingLogger, calculate_accuracy,
    save_model, get_device, format_time
)

class CIFAR10Trainer:
    """
    CIFAR-10 Model Trainer
    CIFAR-10模型训练器
    
    This class handles the entire training process.
    Think of it as a personal trainer for your AI model!
    这个类处理整个训练过程。
    把它想象成AI模型的私人教练！
    """
    
    def __init__(self, model_name='simple', batch_size=32, learning_rate=0.001,
                 epochs=50, patience=10, data_dir='./data', save_dir='./models'):
        """
        Initialize the trainer
        初始化训练器
        
        Args:
            model_name: Type of model to train ('simple', 'improved', 'resnet')
            batch_size: Batch size for training (训练批次大小)
            learning_rate: Learning rate for optimizer (优化器学习率)
            epochs: Maximum number of epochs (最大epoch数)
            patience: Early stopping patience (早停耐心值)
            data_dir: Directory for dataset (数据集目录)
            save_dir: Directory to save models (保存模型的目录)
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.patience = patience
        self.data_dir = data_dir
        self.save_dir = save_dir
        
        # Get device (GPU if available, otherwise CPU)
        # 获取设备（如果可用则使用GPU，否则使用CPU）
        self.device = get_device()
        
        # Initialize components
        # 初始化组件
        self._setup_data()
        self._setup_model()
        self._setup_training()
        
    def _setup_data(self):
        """Set up data loaders"""
        print("Setting up data loaders...")
        print("设置数据加载器...")
        
        self.data_loader = CIFAR10DataLoader(
            data_dir=self.data_dir,
            batch_size=self.batch_size,
            validation_split=0.1
        )
        
        self.train_loader, self.val_loader, self.test_loader = self.data_loader.get_data_loaders()
        
        print(f"Training batches: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")
        print(f"Test batches: {len(self.test_loader)}")
        print(f"训练批次: {len(self.train_loader)}")
        print(f"验证批次: {len(self.val_loader)}")
        print(f"测试批次: {len(self.test_loader)}")
        
    def _setup_model(self):
        """Set up the model"""
        print(f"Setting up {self.model_name} model...")
        print(f"设置{self.model_name}模型...")
        
        self.model = get_model(self.model_name, num_classes=10)
        self.model.to(self.device)
        
        # Print model summary
        # 打印模型摘要
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"总参数: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
        
    def _setup_training(self):
        """Set up training components"""
        print("Setting up training components...")
        print("设置训练组件...")
        
        # Loss function
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4  # L2 regularization (L2正则化)
        )
        
        # Learning rate scheduler
        # 学习率调度器
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Early stopping
        # 早停
        self.early_stopping = EarlyStopping(
            patience=self.patience,
            min_delta=0.001,
            restore_best_weights=True
        )
        
        # Training logger
        # 训练记录器
        self.logger = TrainingLogger()
        
        # Create save directory
        # 创建保存目录
        os.makedirs(self.save_dir, exist_ok=True)
        
    def train_epoch(self):
        """
        Train for one epoch
        训练一个epoch
        
        Returns:
            tuple: (average_loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar
        # 进度条
        pbar = tqdm(self.train_loader, desc='Training', leave=False)
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Zero gradients
            # 清零梯度
            self.optimizer.zero_grad()
            
            # Forward pass
            # 前向传播
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            # 统计
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            # 更新进度条
            accuracy = 100.0 * correct / total
            pbar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{accuracy:.2f}%'
            })
        
        avg_loss = running_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self):
        """
        Validate for one epoch
        验证一个epoch
        
        Returns:
            tuple: (average_loss, accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation', leave=False)
            
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update progress bar
                # 更新进度条
                accuracy = 100.0 * correct / total
                pbar.set_postfix({
                    'Loss': f'{running_loss/(len(pbar.desc)+1):.4f}',
                    'Acc': f'{accuracy:.2f}%'
                })
        
        avg_loss = running_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def train(self):
        """
        Main training loop
        主训练循环
        """
        print(f"\nStarting training for {self.epochs} epochs...")
        print(f"开始训练{self.epochs}个epoch...")
        print("=" * 60)
        
        start_time = time.time()
        best_val_acc = 0.0
        
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            
            # Train one epoch
            # 训练一个epoch
            train_loss, train_acc = self.train_epoch()
            
            # Validate one epoch
            # 验证一个epoch
            val_loss, val_acc = self.validate_epoch()
            
            # Update learning rate scheduler
            # 更新学习率调度器
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log epoch results
            # 记录epoch结果
            self.logger.log_epoch(epoch, train_loss, val_loss, train_acc, val_acc, current_lr)
            
            # Calculate epoch time
            # 计算epoch时间
            epoch_time = time.time() - epoch_start_time
            
            # Print epoch summary
            # 打印epoch摘要
            print(f"Epoch [{epoch+1}/{self.epochs}] - {format_time(epoch_time)}")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {current_lr:.6f}")
            print(f"Epoch [{epoch+1}/{self.epochs}] - {format_time(epoch_time)}")
            print(f"训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.2f}%")
            print(f"验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.2f}%")
            print(f"学习率: {current_lr:.6f}")
            
            # Save best model
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_path = os.path.join(self.save_dir, f'best_{self.model_name}_model.pth')
                save_model(self.model, best_model_path, epoch, train_loss, val_loss, train_acc, val_acc)
                print(f"✓ New best model saved! Validation accuracy: {val_acc:.2f}%")
                print(f"✓ 保存了新的最佳模型！验证准确率: {val_acc:.2f}%")
            
            # Check early stopping
            # 检查早停
            if self.early_stopping(val_loss, self.model):
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                print(f"Best validation accuracy: {best_val_acc:.2f}%")
                print(f"在{epoch+1}个epoch后触发早停")
                print(f"最佳验证准确率: {best_val_acc:.2f}%")
                break
            
            print("-" * 60)
        
        # Training completed
        # 训练完成
        total_time = time.time() - start_time
        print(f"\nTraining completed in {format_time(total_time)}")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        print(f"训练在{format_time(total_time)}内完成")
        print(f"最佳验证准确率: {best_val_acc:.2f}%")
        
        # Save final model
        # 保存最终模型
        final_model_path = os.path.join(self.save_dir, f'final_{self.model_name}_model.pth')
        save_model(self.model, final_model_path, epoch, train_loss, val_loss, train_acc, val_acc)
        
        # Plot training history
        # 绘制训练历史
        history_plot_path = os.path.join(self.save_dir, f'{self.model_name}_training_history.png')
        self.logger.plot_training_history(save_path=history_plot_path)
        
        return best_val_acc
    
    def test(self, model_path=None):
        """
        Test the trained model
        测试训练好的模型
        
        Args:
            model_path: Path to saved model (保存模型的路径)
        """
        if model_path:
            # Load saved model
            # 加载保存的模型
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {model_path}")
            print(f"从{model_path}加载模型")
        
        print("\nTesting model...")
        print("测试模型...")
        
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc='Testing')
            
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                accuracy = 100.0 * correct / total
                pbar.set_postfix({'Acc': f'{accuracy:.2f}%'})
        
        test_accuracy = 100.0 * correct / total
        print(f"\nTest Accuracy: {test_accuracy:.2f}%")
        print(f"测试准确率: {test_accuracy:.2f}%")
        
        return test_accuracy

def main():
    """Main function to run training"""
    parser = argparse.ArgumentParser(description='Train CIFAR-10 CNN Model')
    parser.add_argument('--model', type=str, default='simple', 
                       choices=['simple', 'improved', 'resnet'],
                       help='Model architecture to use')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs to train')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Directory for dataset')
    parser.add_argument('--save-dir', type=str, default='./models',
                       help='Directory to save models')
    parser.add_argument('--test-only', action='store_true',
                       help='Only test the model (skip training)')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to saved model for testing')
    
    args = parser.parse_args()
    
    # Create trainer
    # 创建训练器
    trainer = CIFAR10Trainer(
        model_name=args.model,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        patience=args.patience,
        data_dir=args.data_dir,
        save_dir=args.save_dir
    )
    
    if args.test_only:
        # Test only
        # 仅测试
        if args.model_path:
            trainer.test(args.model_path)
        else:
            # Try to find best model
            # 尝试找到最佳模型
            best_model_path = os.path.join(args.save_dir, f'best_{args.model}_model.pth')
            if os.path.exists(best_model_path):
                trainer.test(best_model_path)
            else:
                print("No saved model found. Please provide --model-path or train first.")
                print("未找到保存的模型。请提供--model-path或先进行训练。")
    else:
        # Train and test
        # 训练和测试
        best_val_acc = trainer.train()
        
        # Test with best model
        # 用最佳模型测试
        best_model_path = os.path.join(args.save_dir, f'best_{args.model}_model.pth')
        if os.path.exists(best_model_path):
            trainer.test(best_model_path)

if __name__ == '__main__':
    main() 