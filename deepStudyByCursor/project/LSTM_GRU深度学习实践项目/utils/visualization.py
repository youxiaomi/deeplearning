"""
可视化工具函数
Visualization Utility Functions

提供各种图表和可视化功能，帮助理解模型性能和数据分析。
Provides various charts and visualization functions to help understand model performance and data analysis.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体支持
# Set Chinese font support
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置图表样式
# Set chart style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def plot_loss_curve(train_losses: List[float], val_losses: Optional[List[float]] = None, 
                   title: str = "训练损失曲线 | Training Loss Curve", 
                   save_path: Optional[str] = None):
    """
    绘制训练损失曲线
    Plot training loss curve
    
    Args:
        train_losses: 训练损失列表 | List of training losses
        val_losses: 验证损失列表 | List of validation losses  
        title: 图表标题 | Chart title
        save_path: 保存路径 | Save path
    """
    plt.figure(figsize=(12, 6))
    
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='训练损失 | Training Loss', linewidth=2)
    
    if val_losses:
        plt.plot(epochs, val_losses, 'r-', label='验证损失 | Validation Loss', linewidth=2)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('轮次 | Epoch', fontsize=12)
    plt.ylabel('损失 | Loss', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # 添加最低点标记
    # Add minimum point marker
    min_idx = np.argmin(train_losses)
    plt.scatter(min_idx + 1, train_losses[min_idx], color='blue', s=100, zorder=5)
    plt.annotate(f'最低点: {train_losses[min_idx]:.4f}', 
                xy=(min_idx + 1, train_losses[min_idx]), 
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.3),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    if val_losses:
        val_min_idx = np.argmin(val_losses)
        plt.scatter(val_min_idx + 1, val_losses[val_min_idx], color='red', s=100, zorder=5)
        plt.annotate(f'最低点: {val_losses[val_min_idx]:.4f}', 
                    xy=(val_min_idx + 1, val_losses[val_min_idx]), 
                    xytext=(10, -20), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.3),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_predictions(actual: np.ndarray, predicted: np.ndarray, 
                    title: str = "预测结果对比 | Prediction Comparison",
                    x_label: str = "时间步 | Time Step",
                    y_label: str = "值 | Value",
                    save_path: Optional[str] = None):
    """
    绘制预测结果对比图
    Plot prediction comparison chart
    
    Args:
        actual: 实际值 | Actual values
        predicted: 预测值 | Predicted values
        title: 图表标题 | Chart title
        x_label: X轴标签 | X-axis label
        y_label: Y轴标签 | Y-axis label
        save_path: 保存路径 | Save path
    """
    plt.figure(figsize=(15, 8))
    
    x = range(len(actual))
    
    plt.plot(x, actual, 'b-', label='实际值 | Actual', alpha=0.8, linewidth=2)
    plt.plot(x, predicted, 'r-', label='预测值 | Predicted', alpha=0.8, linewidth=2)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # 计算并显示误差统计
    # Calculate and display error statistics
    mse = np.mean((actual - predicted) ** 2)
    mae = np.mean(np.abs(actual - predicted))
    
    # 在图上添加统计信息
    # Add statistics to the plot
    stats_text = f'MSE: {mse:.4f}\nMAE: {mae:.4f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         class_names: Optional[List[str]] = None,
                         title: str = "混淆矩阵 | Confusion Matrix",
                         save_path: Optional[str] = None):
    """
    绘制混淆矩阵
    Plot confusion matrix
    
    Args:
        y_true: 真实标签 | True labels
        y_pred: 预测标签 | Predicted labels
        class_names: 类别名称 | Class names
        title: 图表标题 | Chart title
        save_path: 保存路径 | Save path
    """
    from sklearn.metrics import confusion_matrix, classification_report
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    
    # 绘制热力图
    # Draw heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': '样本数量 | Sample Count'})
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('预测标签 | Predicted Label', fontsize=12)
    plt.ylabel('真实标签 | True Label', fontsize=12)
    
    # 计算准确率
    # Calculate accuracy
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    plt.text(0.5, -0.1, f'总体准确率 | Overall Accuracy: {accuracy:.3f}', 
             ha='center', transform=plt.gca().transAxes, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # 打印详细分类报告
    # Print detailed classification report
    print("详细分类报告 | Detailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))


def plot_attention_weights(attention_weights: np.ndarray, 
                          input_tokens: Optional[List[str]] = None,
                          title: str = "注意力权重可视化 | Attention Weights Visualization",
                          save_path: Optional[str] = None):
    """
    可视化注意力权重
    Visualize attention weights
    
    Args:
        attention_weights: 注意力权重矩阵 | Attention weight matrix
        input_tokens: 输入标记列表 | Input token list
        title: 图表标题 | Chart title
        save_path: 保存路径 | Save path
    """
    plt.figure(figsize=(12, 8))
    
    # 绘制注意力热力图
    # Draw attention heatmap
    sns.heatmap(attention_weights, cmap='Blues', cbar=True,
                xticklabels=input_tokens if input_tokens else range(attention_weights.shape[1]),
                yticklabels=range(attention_weights.shape[0]))
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('输入位置 | Input Position', fontsize=12)
    plt.ylabel('输出位置 | Output Position', fontsize=12)
    
    if input_tokens:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_model_architecture(model_type: str = "LSTM", save_path: Optional[str] = None):
    """
    绘制模型架构图
    Draw model architecture diagram
    
    Args:
        model_type: 模型类型 ("LSTM" 或 "GRU") | Model type ("LSTM" or "GRU")
        save_path: 保存路径 | Save path
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    if model_type.upper() == "LSTM":
        # LSTM架构图
        # LSTM architecture diagram
        ax.set_title('LSTM架构图 | LSTM Architecture Diagram', fontsize=16, fontweight='bold', pad=20)
        
        # 绘制LSTM单元
        # Draw LSTM cell
        cell_rect = patches.Rectangle((0.3, 0.3), 0.4, 0.4, 
                                     linewidth=2, edgecolor='black', facecolor='lightblue', alpha=0.7)
        ax.add_patch(cell_rect)
        ax.text(0.5, 0.5, 'LSTM\nCell', ha='center', va='center', fontsize=14, fontweight='bold')
        
        # 绘制三个门
        # Draw three gates
        gates = [
            {'name': '遗忘门\nForget Gate', 'pos': (0.15, 0.7), 'color': 'lightcoral'},
            {'name': '输入门\nInput Gate', 'pos': (0.5, 0.8), 'color': 'lightgreen'},
            {'name': '输出门\nOutput Gate', 'pos': (0.85, 0.7), 'color': 'lightyellow'}
        ]
        
        for gate in gates:
            gate_rect = patches.Rectangle((gate['pos'][0]-0.08, gate['pos'][1]-0.05), 0.16, 0.1,
                                        linewidth=1, edgecolor='black', facecolor=gate['color'], alpha=0.8)
            ax.add_patch(gate_rect)
            ax.text(gate['pos'][0], gate['pos'][1], gate['name'], ha='center', va='center', 
                   fontsize=10, fontweight='bold')
        
        # 绘制细胞状态线
        # Draw cell state line
        ax.arrow(0.1, 0.5, 0.8, 0, head_width=0.02, head_length=0.03, fc='red', ec='red', linewidth=3)
        ax.text(0.5, 0.53, '细胞状态 C_t | Cell State C_t', ha='center', va='bottom', 
               fontsize=12, fontweight='bold', color='red')
        
        # 绘制隐藏状态线
        # Draw hidden state line
        ax.arrow(0.5, 0.25, 0, -0.15, head_width=0.02, head_length=0.02, fc='blue', ec='blue', linewidth=2)
        ax.text(0.52, 0.15, '隐藏状态 h_t | Hidden State h_t', ha='left', va='center', 
               fontsize=12, fontweight='bold', color='blue')
    
    elif model_type.upper() == "GRU":
        # GRU架构图
        # GRU architecture diagram
        ax.set_title('GRU架构图 | GRU Architecture Diagram', fontsize=16, fontweight='bold', pad=20)
        
        # 绘制GRU单元
        # Draw GRU cell
        cell_rect = patches.Rectangle((0.3, 0.3), 0.4, 0.4, 
                                     linewidth=2, edgecolor='black', facecolor='lightblue', alpha=0.7)
        ax.add_patch(cell_rect)
        ax.text(0.5, 0.5, 'GRU\nCell', ha='center', va='center', fontsize=14, fontweight='bold')
        
        # 绘制两个门
        # Draw two gates
        gates = [
            {'name': '重置门\nReset Gate', 'pos': (0.25, 0.8), 'color': 'lightcoral'},
            {'name': '更新门\nUpdate Gate', 'pos': (0.75, 0.8), 'color': 'lightgreen'}
        ]
        
        for gate in gates:
            gate_rect = patches.Rectangle((gate['pos'][0]-0.08, gate['pos'][1]-0.05), 0.16, 0.1,
                                        linewidth=1, edgecolor='black', facecolor=gate['color'], alpha=0.8)
            ax.add_patch(gate_rect)
            ax.text(gate['pos'][0], gate['pos'][1], gate['name'], ha='center', va='center', 
                   fontsize=10, fontweight='bold')
        
        # 绘制隐藏状态线
        # Draw hidden state line
        ax.arrow(0.1, 0.5, 0.8, 0, head_width=0.02, head_length=0.03, fc='blue', ec='blue', linewidth=3)
        ax.text(0.5, 0.53, '隐藏状态 h_t | Hidden State h_t', ha='center', va='bottom', 
               fontsize=12, fontweight='bold', color='blue')
    
    # 绘制输入和输出箭头
    # Draw input and output arrows
    ax.arrow(0.5, 0.05, 0, 0.2, head_width=0.02, head_length=0.02, fc='green', ec='green', linewidth=2)
    ax.text(0.52, 0.05, '输入 x_t | Input x_t', ha='left', va='center', 
           fontsize=12, fontweight='bold', color='green')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_sequence_data(data: np.ndarray, title: str = "序列数据可视化 | Sequence Data Visualization",
                      labels: Optional[List[str]] = None, save_path: Optional[str] = None):
    """
    绘制序列数据
    Plot sequence data
    
    Args:
        data: 序列数据 [时间步, 特征] | Sequence data [time_steps, features]
        title: 图表标题 | Chart title
        labels: 特征标签 | Feature labels
        save_path: 保存路径 | Save path
    """
    plt.figure(figsize=(15, 8))
    
    if data.ndim == 1:
        plt.plot(data, linewidth=2, alpha=0.8)
        plt.ylabel('值 | Value', fontsize=12)
    else:
        for i in range(data.shape[1]):
            label = labels[i] if labels and i < len(labels) else f'特征 {i+1} | Feature {i+1}'
            plt.plot(data[:, i], label=label, linewidth=2, alpha=0.8)
        plt.legend(fontsize=11)
        plt.ylabel('值 | Value', fontsize=12)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('时间步 | Time Step', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_gradient_flow(gradients: Dict[str, float], title: str = "梯度流可视化 | Gradient Flow Visualization",
                      save_path: Optional[str] = None):
    """
    可视化梯度流
    Visualize gradient flow
    
    Args:
        gradients: 梯度字典 {层名: 梯度大小} | Gradient dictionary {layer_name: gradient_magnitude}
        title: 图表标题 | Chart title
        save_path: 保存路径 | Save path
    """
    plt.figure(figsize=(12, 6))
    
    layers = list(gradients.keys())
    grad_values = list(gradients.values())
    
    # 使用对数刻度显示梯度
    # Use log scale for gradients
    log_grads = [np.log10(abs(g) + 1e-8) for g in grad_values]
    
    bars = plt.bar(layers, log_grads, alpha=0.7, 
                  color=['red' if g < -5 else 'orange' if g < -3 else 'green' for g in log_grads])
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('网络层 | Network Layer', fontsize=12)
    plt.ylabel('梯度大小 (log10) | Gradient Magnitude (log10)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # 添加梯度爆炸/消失的阈值线
    # Add threshold lines for gradient explosion/vanishing
    plt.axhline(y=-5, color='red', linestyle='--', alpha=0.7, label='梯度消失阈值 | Vanishing Threshold')
    plt.axhline(y=2, color='red', linestyle='--', alpha=0.7, label='梯度爆炸阈值 | Exploding Threshold')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 在柱子上添加数值
    # Add values on bars
    for bar, value in zip(bars, grad_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{value:.2e}', ha='center', va='bottom', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_training_dashboard(metrics: Dict[str, List[float]], 
                            title: str = "训练仪表板 | Training Dashboard",
                            save_path: Optional[str] = None):
    """
    创建训练仪表板
    Create training dashboard
    
    Args:
        metrics: 指标字典 | Metrics dictionary
        title: 总标题 | Overall title
        save_path: 保存路径 | Save path
    """
    num_metrics = len(metrics)
    fig, axes = plt.subplots(2, (num_metrics + 1) // 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    if num_metrics == 1:
        axes = [axes]
    elif num_metrics <= 2:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i, (metric_name, values) in enumerate(metrics.items()):
        if i < len(axes):
            epochs = range(1, len(values) + 1)
            axes[i].plot(epochs, values, linewidth=2, marker='o', markersize=4)
            axes[i].set_title(f'{metric_name}', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('轮次 | Epoch')
            axes[i].set_ylabel('数值 | Value')
            axes[i].grid(True, alpha=0.3)
            
            # 标注最佳值
            # Mark best value
            if 'loss' in metric_name.lower() or 'error' in metric_name.lower():
                best_idx = np.argmin(values)
                best_value = values[best_idx]
                color = 'red'
            else:
                best_idx = np.argmax(values)
                best_value = values[best_idx]
                color = 'green'
            
            axes[i].scatter(best_idx + 1, best_value, color=color, s=100, zorder=5)
            axes[i].annotate(f'最佳: {best_value:.4f}', 
                           xy=(best_idx + 1, best_value), 
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3))
    
    # 隐藏多余的子图
    # Hide extra subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_lstm_vs_gru_comparison(lstm_results: Dict, gru_results: Dict,
                               title: str = "LSTM vs GRU 性能对比 | LSTM vs GRU Performance Comparison",
                               save_path: Optional[str] = None):
    """
    绘制LSTM和GRU的性能对比图
    Plot LSTM vs GRU performance comparison
    
    Args:
        lstm_results: LSTM结果字典 | LSTM results dictionary
        gru_results: GRU结果字典 | GRU results dictionary
        title: 图表标题 | Chart title
        save_path: 保存路径 | Save path
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 训练损失对比
    # Training loss comparison
    axes[0, 0].plot(lstm_results['train_losses'], label='LSTM', linewidth=2, alpha=0.8)
    axes[0, 0].plot(gru_results['train_losses'], label='GRU', linewidth=2, alpha=0.8)
    axes[0, 0].set_title('训练损失对比 | Training Loss Comparison')
    axes[0, 0].set_xlabel('轮次 | Epoch')
    axes[0, 0].set_ylabel('损失 | Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 验证损失对比
    # Validation loss comparison
    if 'val_losses' in lstm_results and 'val_losses' in gru_results:
        axes[0, 1].plot(lstm_results['val_losses'], label='LSTM', linewidth=2, alpha=0.8)
        axes[0, 1].plot(gru_results['val_losses'], label='GRU', linewidth=2, alpha=0.8)
        axes[0, 1].set_title('验证损失对比 | Validation Loss Comparison')
        axes[0, 1].set_xlabel('轮次 | Epoch')
        axes[0, 1].set_ylabel('损失 | Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # 性能指标对比
    # Performance metrics comparison
    if 'metrics' in lstm_results and 'metrics' in gru_results:
        metrics_names = list(lstm_results['metrics'].keys())
        lstm_values = list(lstm_results['metrics'].values())
        gru_values = list(gru_results['metrics'].values())
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, lstm_values, width, label='LSTM', alpha=0.8)
        axes[1, 0].bar(x + width/2, gru_values, width, label='GRU', alpha=0.8)
        axes[1, 0].set_title('性能指标对比 | Performance Metrics Comparison')
        axes[1, 0].set_xlabel('指标 | Metrics')
        axes[1, 0].set_ylabel('数值 | Value')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(metrics_names, rotation=45, ha='right')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 训练时间和参数数量对比
    # Training time and parameter count comparison
    comparison_data = ['训练时间 | Training Time', '参数数量 | Parameters', '内存使用 | Memory Usage']
    lstm_stats = [
        lstm_results.get('training_time', 0),
        lstm_results.get('parameters', 0),
        lstm_results.get('memory_usage', 0)
    ]
    gru_stats = [
        gru_results.get('training_time', 0),
        gru_results.get('parameters', 0),
        gru_results.get('memory_usage', 0)
    ]
    
    x = np.arange(len(comparison_data))
    axes[1, 1].bar(x - width/2, lstm_stats, width, label='LSTM', alpha=0.8)
    axes[1, 1].bar(x + width/2, gru_stats, width, label='GRU', alpha=0.8)
    axes[1, 1].set_title('资源使用对比 | Resource Usage Comparison')
    axes[1, 1].set_xlabel('类型 | Type')
    axes[1, 1].set_ylabel('相对值 | Relative Value')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(comparison_data, rotation=45, ha='right')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


# 设置默认颜色主题
# Set default color theme
COLORS = {
    'primary': '#3498db',
    'secondary': '#e74c3c', 
    'success': '#2ecc71',
    'warning': '#f39c12',
    'info': '#9b59b6',
    'light': '#ecf0f1',
    'dark': '#34495e'
}

def set_plot_style(style: str = 'seaborn'):
    """
    设置绘图样式
    Set plotting style
    
    Args:
        style: 样式名称 | Style name
    """
    if style == 'seaborn':
        sns.set_style("whitegrid")
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'font.size': 10,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'legend.fontsize': 11,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10
        })
    elif style == 'minimal':
        plt.style.use('default')
        plt.rcParams.update({
            'figure.figsize': (10, 6),
            'font.size': 9,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': True,
            'grid.alpha': 0.3
        })


# 初始化样式
# Initialize style
set_plot_style('seaborn') 