"""
Utility Functions for CIFAR-10 Project
CIFAR-10项目的实用函数

This module contains helper functions for training, evaluation, and visualization.
Think of these as your toolkit - useful tools that make your work easier!
这个模块包含训练、评估和可视化的辅助函数。
把这些想象成你的工具包 - 让你的工作更容易的有用工具！
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support
import time
import os
from datetime import datetime

# CIFAR-10 class names
# CIFAR-10类别名称
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

CIFAR10_CLASSES_CN = [
    '飞机', '汽车', '鸟', '猫', '鹿',
    '狗', '青蛙', '马', '船', '卡车'
]

class EarlyStopping:
    """
    Early Stopping to prevent overfitting
    早停法防止过拟合
    
    This is like a smart coach who knows when to stop training
    to prevent the athlete from getting exhausted.
    这就像一个聪明的教练，知道何时停止训练
    以防止运动员过度疲劳。
    """
    
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        """
        Args:
            patience: How many epochs to wait after last improvement (改善后等待多少个epoch)
            min_delta: Minimum change to qualify as improvement (作为改善的最小变化)
            restore_best_weights: Whether to restore best weights (是否恢复最佳权重)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        """
        Check if training should stop
        检查是否应该停止训练
        
        Args:
            val_loss: Current validation loss (当前验证损失)
            model: Current model (当前模型)
        
        Returns:
            bool: True if training should stop (如果应该停止训练则返回True)
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        """Save the current best model weights"""
        self.best_weights = model.state_dict().copy()

class TrainingLogger:
    """
    Logger to track training progress
    记录器跟踪训练进度
    
    This is like a diary that records everything that happens during training.
    这就像一本记录训练期间发生的一切的日记。
    """
    
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        self.epochs = []
        
    def log_epoch(self, epoch, train_loss, val_loss, train_acc, val_acc, lr):
        """Log metrics for one epoch"""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)
        self.learning_rates.append(lr)
        
    def plot_training_history(self, save_path=None):
        """
        Plot training history
        绘制训练历史
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot losses
        # 绘制损失
        axes[0, 0].plot(self.epochs, self.train_losses, 'b-', label='Training Loss')
        axes[0, 0].plot(self.epochs, self.val_losses, 'r-', label='Validation Loss')
        axes[0, 0].set_title('Training and Validation Loss\n训练和验证损失')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot accuracies
        # 绘制准确率
        axes[0, 1].plot(self.epochs, self.train_accuracies, 'b-', label='Training Accuracy')
        axes[0, 1].plot(self.epochs, self.val_accuracies, 'r-', label='Validation Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy\n训练和验证准确率')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot learning rate
        # 绘制学习率
        axes[1, 0].plot(self.epochs, self.learning_rates, 'g-')
        axes[1, 0].set_title('Learning Rate Schedule\n学习率调度')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True)
        
        # Plot validation loss vs accuracy
        # 绘制验证损失与准确率
        axes[1, 1].scatter(self.val_losses, self.val_accuracies, alpha=0.6)
        axes[1, 1].set_title('Validation Loss vs Accuracy\n验证损失与准确率')
        axes[1, 1].set_xlabel('Validation Loss')
        axes[1, 1].set_ylabel('Validation Accuracy (%)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def calculate_accuracy(outputs, labels):
    """
    Calculate accuracy from model outputs and labels
    从模型输出和标签计算准确率
    
    Args:
        outputs: Model predictions (模型预测)
        labels: True labels (真实标签)
    
    Returns:
        float: Accuracy percentage (准确率百分比)
    """
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return 100.0 * correct / total

def save_model(model, filepath, epoch, train_loss, val_loss, train_acc, val_acc):
    """
    Save model checkpoint with metadata
    保存带有元数据的模型检查点
    
    Args:
        model: PyTorch model to save (要保存的PyTorch模型)
        filepath: Path to save the model (保存模型的路径)
        epoch: Current epoch (当前epoch)
        train_loss: Training loss (训练损失)
        val_loss: Validation loss (验证损失)
        train_acc: Training accuracy (训练准确率)
        val_acc: Validation accuracy (验证准确率)
    """
    # Create directory if it doesn't exist
    # 如果目录不存在则创建
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")
    print(f"模型已保存到 {filepath}")

def load_model(model, filepath):
    """
    Load model checkpoint
    加载模型检查点
    
    Args:
        model: PyTorch model to load weights into (要加载权重的PyTorch模型)
        filepath: Path to the saved model (保存模型的路径)
    
    Returns:
        dict: Checkpoint information (检查点信息)
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model loaded from {filepath}")
    print(f"Epoch: {checkpoint['epoch']}")
    print(f"Validation Accuracy: {checkpoint['val_accuracy']:.2f}%")
    print(f"模型已从 {filepath} 加载")
    print(f"Epoch: {checkpoint['epoch']}")
    print(f"验证准确率: {checkpoint['val_accuracy']:.2f}%")
    
    return checkpoint

def evaluate_model(model, test_loader, device, num_classes=10):
    """
    Comprehensive model evaluation
    全面的模型评估
    
    Args:
        model: Trained PyTorch model (训练好的PyTorch模型)
        test_loader: Test data loader (测试数据加载器)
        device: Device to run evaluation on (运行评估的设备)
        num_classes: Number of classes (类别数量)
    
    Returns:
        dict: Evaluation results (评估结果)
    """
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    print("Evaluating model...")
    print("评估模型...")
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    # 计算指标
    accuracy = 100.0 * sum(np.array(all_predictions) == np.array(all_labels)) / len(all_labels)
    avg_loss = total_loss / len(test_loader)
    
    # Calculate per-class metrics
    # 计算每类指标
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_predictions, average=None
    )
    
    # Create detailed classification report
    # 创建详细的分类报告
    class_report = classification_report(
        all_labels, all_predictions, 
        target_names=CIFAR10_CLASSES,
        output_dict=True
    )
    
    results = {
        'accuracy': accuracy,
        'loss': avg_loss,
        'predictions': all_predictions,
        'labels': all_labels,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'support': support,
        'classification_report': class_report
    }
    
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"测试准确率: {accuracy:.2f}%")
    print(f"测试损失: {avg_loss:.4f}")
    
    return results

def plot_confusion_matrix(labels, predictions, save_path=None):
    """
    Plot confusion matrix
    绘制混淆矩阵
    
    Args:
        labels: True labels (真实标签)
        predictions: Model predictions (模型预测)
        save_path: Path to save the plot (保存图片的路径)
    """
    cm = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CIFAR10_CLASSES,
                yticklabels=CIFAR10_CLASSES)
    plt.title('Confusion Matrix\n混淆矩阵', fontsize=16)
    plt.xlabel('Predicted Label\n预测标签', fontsize=12)
    plt.ylabel('True Label\n真实标签', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_class_performance(precision, recall, f1_score, support, save_path=None):
    """
    Plot per-class performance metrics
    绘制每类性能指标
    
    Args:
        precision: Precision scores per class (每类的精确率)
        recall: Recall scores per class (每类的召回率)
        f1_score: F1 scores per class (每类的F1分数)
        support: Support (number of samples) per class (每类的支持度)
        save_path: Path to save the plot (保存图片的路径)
    """
    x = np.arange(len(CIFAR10_CLASSES))
    width = 0.25
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot precision, recall, f1-score
    # 绘制精确率、召回率、F1分数
    ax1.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax1.bar(x, recall, width, label='Recall', alpha=0.8)
    ax1.bar(x + width, f1_score, width, label='F1-Score', alpha=0.8)
    
    ax1.set_xlabel('Classes / 类别')
    ax1.set_ylabel('Score')
    ax1.set_title('Per-Class Performance Metrics\n每类性能指标')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{en}\n{cn}' for en, cn in zip(CIFAR10_CLASSES, CIFAR10_CLASSES_CN)])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot support (number of samples)
    # 绘制支持度（样本数量）
    ax2.bar(x, support, alpha=0.8, color='orange')
    ax2.set_xlabel('Classes / 类别')
    ax2.set_ylabel('Number of Samples / 样本数量')
    ax2.set_title('Number of Test Samples per Class\n每类测试样本数量')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{en}\n{cn}' for en, cn in zip(CIFAR10_CLASSES, CIFAR10_CLASSES_CN)])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def visualize_predictions(model, test_loader, device, num_samples=16, save_path=None):
    """
    Visualize model predictions on test samples
    可视化模型对测试样本的预测
    
    Args:
        model: Trained model (训练好的模型)
        test_loader: Test data loader (测试数据加载器)
        device: Device to run inference on (运行推理的设备)
        num_samples: Number of samples to visualize (要可视化的样本数量)
        save_path: Path to save the plot (保存图片的路径)
    """
    model.eval()
    
    # Get one batch of test data
    # 获取一批测试数据
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    
    # Make predictions
    # 进行预测
    with torch.no_grad():
        images_gpu = images.to(device)
        outputs = model(images_gpu)
        _, predicted = torch.max(outputs, 1)
    
    # Denormalize images for visualization
    # 反归一化图像以便可视化
    mean = torch.tensor([0.4914, 0.4822, 0.4465])
    std = torch.tensor([0.2023, 0.1994, 0.2010])
    
    def denormalize(tensor):
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return torch.clamp(tensor, 0, 1)
    
    # Create subplot
    # 创建子图
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle('Model Predictions vs True Labels\n模型预测与真实标签', fontsize=16)
    
    for i in range(min(num_samples, len(images))):
        row, col = i // 4, i % 4
        
        # Denormalize and convert to numpy
        # 反归一化并转换为numpy
        img = denormalize(images[i].clone())
        img = img.permute(1, 2, 0).numpy()
        
        # Get prediction and label
        # 获取预测和标签
        pred_idx = predicted[i].item()
        true_idx = labels[i].item()
        
        # Set color based on correctness
        # 根据正确性设置颜色
        color = 'green' if pred_idx == true_idx else 'red'
        
        axes[row, col].imshow(img)
        axes[row, col].set_title(
            f'Pred: {CIFAR10_CLASSES[pred_idx]}\n'
            f'True: {CIFAR10_CLASSES[true_idx]}\n'
            f'预测: {CIFAR10_CLASSES_CN[pred_idx]}\n'
            f'真实: {CIFAR10_CLASSES_CN[true_idx]}',
            fontsize=8, color=color
        )
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def print_model_performance_summary(results):
    """
    Print a comprehensive performance summary
    打印全面的性能摘要
    
    Args:
        results: Evaluation results dictionary (评估结果字典)
    """
    print("\n" + "="*60)
    print("MODEL PERFORMANCE SUMMARY")
    print("模型性能摘要")
    print("="*60)
    
    print(f"Overall Accuracy: {results['accuracy']:.2f}%")
    print(f"Overall Loss: {results['loss']:.4f}")
    print(f"总体准确率: {results['accuracy']:.2f}%")
    print(f"总体损失: {results['loss']:.4f}")
    
    print("\nPer-Class Performance:")
    print("每类性能:")
    print("-" * 60)
    print(f"{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print(f"{'类别':<12} {'精确率':<10} {'召回率':<10} {'F1分数':<10} {'支持度':<10}")
    print("-" * 60)
    
    for i, (en, cn) in enumerate(zip(CIFAR10_CLASSES, CIFAR10_CLASSES_CN)):
        print(f"{en:<12} {results['precision'][i]:<10.3f} {results['recall'][i]:<10.3f} "
              f"{results['f1_score'][i]:<10.3f} {results['support'][i]:<10}")
    
    # Calculate macro averages
    # 计算宏平均
    macro_precision = np.mean(results['precision'])
    macro_recall = np.mean(results['recall'])
    macro_f1 = np.mean(results['f1_score'])
    
    print("-" * 60)
    print(f"{'Macro Avg':<12} {macro_precision:<10.3f} {macro_recall:<10.3f} {macro_f1:<10.3f}")
    print(f"{'宏平均':<12} {macro_precision:<10.3f} {macro_recall:<10.3f} {macro_f1:<10.3f}")
    print("="*60)

def get_device():
    """
    Get the best available device (GPU if available, otherwise CPU)
    获取最佳可用设备（如果可用则使用GPU，否则使用CPU）
    
    Returns:
        torch.device: Device to use for computation (用于计算的设备)
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
        print("使用CPU")
    
    return device

def format_time(seconds):
    """
    Format time in seconds to human readable format
    将秒数格式化为人类可读的格式
    
    Args:
        seconds: Time in seconds (秒数)
    
    Returns:
        str: Formatted time string (格式化的时间字符串)
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)}m {seconds:.1f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {seconds:.1f}s"

def test_utils():
    """
    Test utility functions
    测试实用函数
    """
    print("Testing utility functions...")
    print("测试实用函数...")
    
    # Test device detection
    # 测试设备检测
    device = get_device()
    
    # Test time formatting
    # 测试时间格式化
    test_times = [30, 90, 3661, 7323]
    for t in test_times:
        print(f"{t} seconds = {format_time(t)}")
    
    # Test early stopping
    # 测试早停
    early_stopping = EarlyStopping(patience=3)
    print(f"Early stopping patience: {early_stopping.patience}")
    print(f"早停耐心值: {early_stopping.patience}")
    
    # Test training logger
    # 测试训练记录器
    logger = TrainingLogger()
    for epoch in range(5):
        logger.log_epoch(epoch, 1.0-epoch*0.1, 1.2-epoch*0.1, 
                        50+epoch*10, 45+epoch*8, 0.001)
    
    print("All utility functions tested successfully!")
    print("所有实用函数测试成功！")

if __name__ == "__main__":
    test_utils() 