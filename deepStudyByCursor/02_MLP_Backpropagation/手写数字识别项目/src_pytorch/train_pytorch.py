import os
import sys
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

# Import our PyTorch specific modules
# 导入我们PyTorch特有的模块
from data_loader_pytorch import MNISTDataLoaderPyTorch
from model_pytorch import MLPClassifierPyTorch

def create_results_directory():
    """
    Create results directory structure
    创建结果目录结构
    """
    results_dir = '../results_pytorch' # New results directory for PyTorch
    subdirs = ['models', 'plots', 'logs']
    
    for subdir in [results_dir] + [os.path.join(results_dir, sub) for sub in subdirs]:
        os.makedirs(subdir, exist_ok=True)
    
    return results_dir

def train_mnist_classifier_pytorch(config):
    """
    Train MLP classifier on MNIST dataset using PyTorch and GPU (if available).
    在MNIST数据集上使用PyTorch和GPU（如果可用）训练MLP分类器。
    
    Args:
        config: Configuration dictionary with training parameters
    """
    print("🚀 Starting MNIST Handwritten Digit Recognition Training (PyTorch)")
    print("🚀 开始MNIST手写数字识别训练 (PyTorch)")
    print("=" * 70)
    
    # Set device (设置设备)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (正在使用设备: {device})")
    print("-" * 50)
    
    # Create results directory (创建结果目录)
    results_dir = create_results_directory()
    
    # Initialize data loader (初始化数据加载器)
    print("📁 Loading MNIST dataset... (正在加载MNIST数据集...)")
    data_loader = MNISTDataLoaderPyTorch(
        data_dir='../data',
        batch_size=config['batch_size'],
        validation_split=config['validation_split']
    )
    
    train_loader = data_loader.get_train_loader()
    val_loader = data_loader.get_val_loader()
    test_loader = data_loader.get_test_loader()
    
    print("✅ Data loaded successfully! (数据加载成功！)")
    print(f"Number of training batches: {len(train_loader)} (训练批次数量: {len(train_loader)})")
    print(f"Number of validation batches: {len(val_loader)} (验证批次数量: {len(val_loader)})")
    print(f"Number of test batches: {len(test_loader)} (测试批次数量: {len(test_loader)})")
    print(f"Input features: {data_loader.get_num_features()} (输入特征: {data_loader.get_num_features()})")
    print(f"Number of classes: {data_loader.get_num_classes()} (类别数: {data_loader.get_num_classes()})")
    print("-" * 50)
    
    # Create model (创建模型)
    print("🧠 Creating MLP model... (正在创建MLP模型...)")
    model = MLPClassifierPyTorch(
        layer_sizes=config['layer_sizes'],
        activations=config['activations']
    ).to(device) # Move model to device (将模型移动到设备)
    
    print(f"Model architecture: {model}")
    print(f"模型架构: {model}")
    print("-" * 50)
    
    # Define loss function and optimizer (定义损失函数和优化器)
    criterion = nn.CrossEntropyLoss() # Suitable for multi-class classification (适用于多分类)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Store training history (存储训练历史)
    history = {
        'train_loss': [], 'val_loss': [],
        'train_accuracy': [], 'val_accuracy': []
    }
    
    # Train model (训练模型)
    print("🏋️ Starting training... (开始训练...)")
    start_time = time.time()
    
    for epoch in range(1, config['epochs'] + 1):
        # Training phase (训练阶段)
        model.train() # Set model to training mode (设置模型为训练模式)
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device) # Move data to device (将数据移动到设备)
            
            optimizer.zero_grad() # Zero the gradients (梯度清零)
            output = model(data) # Forward pass (前向传播)
            loss = criterion(output, target) # Calculate loss (计算损失)
            loss.backward() # Backward pass (反向传播)
            optimizer.step() # Update weights (更新权重)
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()

        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = correct_train / total_train
        history['train_loss'].append(avg_train_loss)
        history['train_accuracy'].append(train_accuracy)
        
        # Validation phase (验证阶段)
        model.eval() # Set model to evaluation mode (设置模型为评估模式)
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad(): # Disable gradient calculation (禁用梯度计算)
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total_val += target.size(0)
                correct_val += (predicted == target).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct_val / total_val
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_accuracy)
        
        print(f"Epoch {epoch}/{config['epochs']} - "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
    
    training_time = time.time() - start_time
    print(f"⏱️ Training completed in {training_time:.2f} seconds")
    print(f"⏱️ 训练在 {training_time:.2f} 秒内完成")
    print("-" * 50)
    
    # Evaluate model on test set (在测试集上评估模型)
    print("📊 Evaluating model on test set... (正在测试集上评估模型...)")
    model.eval()
    test_loss = 0.0
    correct_test = 0
    total_test = 0
    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            test_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_test += target.size(0)
            correct_test += (predicted == target).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_true_labels.extend(target.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = correct_test / total_test
    
    print("📈 Final Results (最终结果):")
    print(f"  Test       - Loss: {avg_test_loss:.4f}, Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print("-" * 50)
    
    # Save model (保存模型)
    model_path = os.path.join(results_dir, 'models', 'mnist_mlp_pytorch.pth')
    torch.save(model.state_dict(), model_path) # Save only the model's state dictionary
    print(f"✅ Model saved to: {model_path}")
    print(f"✅ 模型保存到: {model_path}")
    
    # Plot and save training history (绘制并保存训练历史)
    plot_path = os.path.join(results_dir, 'plots', 'training_history_pytorch.png')
    plot_training_history(history, plot_path)
    
    # Generate detailed analysis (生成详细分析)
    generate_analysis_report_pytorch(model, data_loader, results_dir, config, training_time, 
                                     history, all_predictions, all_true_labels, device)
    
    # Visualize predictions (可视化预测)
    visualize_predictions_pytorch(model, test_loader, results_dir, device)
    
    return model, history

def plot_training_history(history, save_path):
    """
    Plot and save training history (loss and accuracy).
    绘制并保存训练历史（损失和准确率）。
    """
    plt.figure(figsize=(12, 5))

    # Plot Loss (绘制损失)
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs (每个Epoch的损失)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy (绘制准确率)
    plt.subplot(1, 2, 2)
    plt.plot(history['train_accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs (每个Epoch的准确率)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"✅ Training history plot saved to: {save_path}")
    print(f"✅ 训练历史图保存到: {save_path}")

def generate_analysis_report_pytorch(model, data_loader, results_dir, config, training_time, 
                                     history, all_predictions, all_true_labels, device):
    """
    Generate detailed analysis report for PyTorch model.
    为PyTorch模型生成详细分析报告。
    """
    print("📝 Generating analysis report... (正在生成分析报告...)")
    
    test_loader = data_loader.get_test_loader()
    train_loader = data_loader.get_train_loader()
    val_loader = data_loader.get_val_loader()

    # Calculate accuracies and losses (计算准确率和损失)
    model.eval() # Set model to evaluation mode
    
    def calculate_metrics(loader, criterion, device):
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        return total_loss / len(loader), correct / total

    criterion = nn.CrossEntropyLoss()
    train_loss, train_accuracy = calculate_metrics(train_loader, criterion, device)
    val_loss, val_accuracy = calculate_metrics(val_loader, criterion, device)
    test_loss, test_accuracy = calculate_metrics(test_loader, criterion, device)

    # Per-class accuracy (每个类别的准确率)
    cm = confusion_matrix(all_true_labels, all_predictions)
    class_accuracies = {}
    for i in range(10):
        true_positives = cm[i, i]
        total_for_class = np.sum(cm[i, :])
        if total_for_class > 0:
            class_accuracies[i] = true_positives / total_for_class
        else:
            class_accuracies[i] = 0.0
    
    # Generate report text (生成报告文本)
    report_path = os.path.join(results_dir, 'logs', 'training_report_pytorch.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("MNIST Handwritten Digit Recognition - PyTorch Training Report\n")
        f.write("MNIST手写数字识别 - PyTorch训练报告\n")
        f.write("=" * 60 + "\n\n")
        
        # Configuration (配置)
        f.write("Configuration (配置):\n")
        f.write("-" * 30 + "\n")
        f.write(f"Model Architecture: {config['layer_sizes']}\n")
        f.write(f"Activations: {config['activations']}\n")
        f.write(f"Epochs: {config['epochs']}\n")
        f.write(f"Batch Size: {config['batch_size']}\n")
        f.write(f"Learning Rate: {config['learning_rate']}\n")
        f.write(f"Validation Split: {config['validation_split']}\n")
        f.write(f"Training Time: {training_time:.2f} seconds\n")
        f.write(f"Device Used: {device}\n\n")
        
        # Dataset Information (数据集信息)
        f.write("Dataset Information (数据集信息):\n")
        f.write("-" * 30 + "\n")
        f.write(f"Training samples: {len(data_loader.train_dataset)}\n")
        f.write(f"Validation samples: {len(data_loader.val_dataset)}\n")
        f.write(f"Test samples: {len(data_loader.test_dataset)}\n")
        f.write(f"Input features: {data_loader.get_num_features()} (28x28 pixels)\n")
        f.write(f"Number of classes: {data_loader.get_num_classes()} (digits 0-9)\n\n")
        
        # Final Performance (最终性能)
        f.write("Final Performance (最终性能):\n")
        f.write("-" * 30 + "\n")
        f.write(f"Training Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)\n")
        f.write(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)\n")
        f.write(f"Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)\n\n")
        
        # Per-class accuracy (每个类别准确率)
        f.write("Per-Class Accuracy (每个类别准确率):\n")
        f.write("-" * 30 + "\n")
        for class_idx, accuracy in class_accuracies.items():
            f.write(f"Digit {class_idx}: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write("\n")
        
        # Training Progress (训练进度)
        f.write("Training Progress (训练进度):\n")
        f.write("-" * 30 + "\n")
        f.write(f"Best Training Accuracy: {max(history['train_accuracy']):.4f}\n")
        f.write(f"Best Validation Accuracy: {max(history['val_accuracy']):.4f}\n")
        f.write(f"Final Training Loss: {history['train_loss'][-1]:.4f}\n")
        f.write(f"Final Validation Loss: {history['val_loss'][-1]:.4f}\n")
    
    print(f"✅ Analysis report saved to: {report_path}")
    print(f"✅ 分析报告保存到: {report_path}")

def visualize_predictions_pytorch(model, test_loader, results_dir, device):
    """
    Visualize model predictions on test samples for PyTorch model.
    可视化PyTorch模型在测试样本上的预测。
    """
    print("🎨 Creating prediction visualizations... (正在创建预测可视化...)")
    
    model.eval() # Set model to evaluation mode
    
    # Get a batch of test data (获取一个批次的测试数据)
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    # Move images and labels to the same device as model
    # 将图片和标签移动到与模型相同的设备上
    images, labels = images.to(device), labels.to(device)

    # Make predictions (进行预测)
    with torch.no_grad():
        outputs = model(images)
        probabilities = F.softmax(outputs, dim=1) # Apply softmax to get probabilities
        _, predictions = torch.max(outputs, 1)

    # Select random samples for visualization (选择随机样本进行可视化)
    n_samples = 20
    if images.size(0) < n_samples:
        n_samples = images.size(0) # Adjust if batch size is smaller
    
    indices = np.random.choice(images.size(0), n_samples, replace=False)
    
    # Move data back to CPU for plotting (将数据移回CPU进行绘图)
    images_cpu = images.cpu().numpy()
    labels_cpu = labels.cpu().numpy()
    predictions_cpu = predictions.cpu().numpy()
    probabilities_cpu = probabilities.cpu().numpy()
    
    # Create visualization (创建可视化)
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    fig.suptitle('Model Predictions on Test Set (PyTorch) (模型在测试集上的预测)', fontsize=16, y=0.98)
    
    for i, idx in enumerate(indices):
        row, col = i // 5, i % 5
        
        # Reshape image (重塑图片)
        image = images_cpu[idx].reshape(28, 28)
        true_label = labels_cpu[idx]
        pred_label = predictions_cpu[idx]
        confidence = probabilities_cpu[idx][pred_label]
        
        # Plot image (绘制图片)
        axes[row, col].imshow(image, cmap='gray')
        
        # Set title with prediction info (设置带预测信息的标题)
        color = 'green' if pred_label == true_label else 'red'
        title = f'True: {true_label}, Pred: {pred_label}\nConf: {confidence:.3f}'
        axes[row, col].set_title(title, color=color, fontsize=10)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # Save visualization (保存可视化)
    pred_plot_path = os.path.join(results_dir, 'plots', 'predictions_visualization_pytorch.png')
    plt.savefig(pred_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create confusion matrix visualization (创建混淆矩阵可视化)
    create_confusion_matrix_pytorch(all_true_labels, all_predictions, results_dir)
    
    print(f"✅ Prediction visualizations saved to: {pred_plot_path}")
    print(f"✅ 预测可视化保存到: {pred_plot_path}")

def create_confusion_matrix_pytorch(true_labels, predictions, results_dir):
    """
    Create and save confusion matrix for PyTorch model.
    为PyTorch模型创建并保存混淆矩阵。
    """
    import seaborn as sns
    
    # Create confusion matrix (创建混淆矩阵)
    cm = confusion_matrix(true_labels, predictions)
    
    # Normalize confusion matrix (标准化混淆矩阵)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot confusion matrix (绘制混淆矩阵)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.title('Normalized Confusion Matrix (PyTorch) (标准化混淆矩阵)')
    plt.xlabel('Predicted Label (预测标签)')
    plt.ylabel('True Label (真实标签)')
    
    # Save confusion matrix (保存混淆矩阵)
    cm_path = os.path.join(results_dir, 'plots', 'confusion_matrix_pytorch.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Confusion matrix saved to: {cm_path}")
    print(f"✅ 混淆矩阵保存到: {cm_path}")

def main():
    """
    Main training function for PyTorch MLP.
    PyTorch MLP的主训练函数。
    """
    parser = argparse.ArgumentParser(description='Train MLP for MNIST digit recognition using PyTorch')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs') # Reduced epochs for faster demo
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_sizes', nargs='+', type=int, default=[256, 128], 
                        help='Hidden layer sizes')
    parser.add_argument('--validation_split', type=float, default=0.1, 
                        help='Validation split ratio')
    
    args = parser.parse_args()
    
    # Configuration (配置)
    config = {
        'layer_sizes': [784] + args.hidden_sizes + [10],  # Input: 784, Output: 10
        'activations': ['relu'] * len(args.hidden_sizes), # All hidden layers use ReLU
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'validation_split': args.validation_split
    }
    
    # Set random seed for reproducibility (设置随机种子以确保可重现性)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    try:
        # Train model (训练模型)
        model, history = train_mnist_classifier_pytorch(config)
        
        print("\n🎉 Training completed successfully! (训练成功完成！)")
        print("🎉 Check the results_pytorch directory for saved models and visualizations.")
        print("🎉 查看results_pytorch目录中保存的模型和可视化结果。")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        print(f"❌ 训练过程中出错: {e}")
        raise

if __name__ == "__main__":
    main() 