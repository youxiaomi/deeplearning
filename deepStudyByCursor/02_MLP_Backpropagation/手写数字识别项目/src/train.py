"""
Training Script for Handwritten Digit Recognition
手写数字识别训练脚本

This script trains an MLP from scratch on the MNIST dataset.
本脚本在MNIST数据集上从零训练MLP。
"""

import os
import sys
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from data_loader import MNISTDataLoader
from mlp_scratch import MLPClassifier

def create_results_directory():
    """
    Create results directory structure
    创建结果目录结构
    """
    results_dir = '../results'
    subdirs = ['models', 'plots', 'logs']
    
    for subdir in [results_dir] + [os.path.join(results_dir, sub) for sub in subdirs]:
        os.makedirs(subdir, exist_ok=True)
    
    return results_dir

def train_mnist_classifier(config):
    """
    Train MLP classifier on MNIST dataset
    在MNIST数据集上训练MLP分类器
    
    Args:
        config: Configuration dictionary with training parameters
    """
    print("🚀 Starting MNIST Handwritten Digit Recognition Training")
    print("🚀 开始MNIST手写数字识别训练")
    print("=" * 70)
    
    # Create results directory (创建结果目录)
    results_dir = create_results_directory()
    
    # Initialize data loader (初始化数据加载器)
    print("📁 Loading MNIST dataset... (正在加载MNIST数据集...)")
    data_loader = MNISTDataLoader(
        data_dir='../data',
        batch_size=config['batch_size'],
        validation_split=config['validation_split']
    )
    
    # Load preprocessed data (加载预处理的数据)
    data = data_loader.load_numpy_data()
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']
    
    print("✅ Data loaded successfully! (数据加载成功！)")
    print(f"Training set: {X_train.shape[0]} samples (训练集: {X_train.shape[0]} 样本)")
    print(f"Validation set: {X_val.shape[0]} samples (验证集: {X_val.shape[0]} 样本)")
    print(f"Test set: {X_test.shape[0]} samples (测试集: {X_test.shape[0]} 样本)")
    print(f"Input features: {X_train.shape[1]} (输入特征: {X_train.shape[1]})")
    print(f"Number of classes: {len(np.unique(y_train))} (类别数: {len(np.unique(y_train))})")
    print("-" * 50)
    
    # Create model (创建模型)
    print("🧠 Creating MLP model... (正在创建MLP模型...)")
    model = MLPClassifier(
        layer_sizes=config['layer_sizes'],
        activations=config['activations']
    )
    
    print(f"Model architecture: {config['layer_sizes']}")
    print(f"模型架构: {config['layer_sizes']}")
    print(f"Activation functions: {config['activations']}")
    print(f"激活函数: {config['activations']}")
    print("-" * 50)
    
    # Train model (训练模型)
    print("🏋️ Starting training... (开始训练...)")
    start_time = time.time()
    
    history = model.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        verbose=True
    )
    
    training_time = time.time() - start_time
    print(f"⏱️ Training completed in {training_time:.2f} seconds")
    print(f"⏱️ 训练在 {training_time:.2f} 秒内完成")
    print("-" * 50)
    
    # Evaluate model (评估模型)
    print("📊 Evaluating model... (正在评估模型...)")
    
    train_accuracy = model.compute_accuracy(X_train, y_train)
    val_accuracy = model.compute_accuracy(X_val, y_val)
    test_accuracy = model.compute_accuracy(X_test, y_test)
    
    # Compute final losses (计算最终损失)
    train_loss = model.compute_loss(X_train, model._to_onehot(y_train))
    val_loss = model.compute_loss(X_val, model._to_onehot(y_val))
    test_loss = model.compute_loss(X_test, model._to_onehot(y_test))
    
    print("📈 Final Results (最终结果):")
    print(f"  Training   - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"  Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    print(f"  Test       - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print("-" * 50)
    
    # Save model (保存模型)
    model_path = os.path.join(results_dir, 'models', 'mnist_mlp_scratch.pkl')
    model.save_model(model_path)
    
    # Plot and save training history (绘制并保存训练历史)
    plot_path = os.path.join(results_dir, 'plots', 'training_history.png')
    model.plot_training_history(save_path=plot_path)
    
    # Generate detailed analysis (生成详细分析)
    generate_analysis_report(model, data, results_dir, config, training_time)
    
    # Visualize predictions (可视化预测)
    visualize_predictions(model, X_test, y_test, results_dir)
    
    return model, history

def generate_analysis_report(model, data, results_dir, config, training_time):
    """
    Generate detailed analysis report
    生成详细分析报告
    """
    print("📝 Generating analysis report... (正在生成分析报告...)")
    
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']
    
    # Classification report for each class (每个类别的分类报告)
    test_predictions = model.predict(X_test)
    test_probabilities = model.predict_proba(X_test)
    
    # Per-class accuracy (每个类别的准确率)
    class_accuracies = {}
    for class_idx in range(10):
        mask = y_test == class_idx
        if np.sum(mask) > 0:
            class_acc = np.mean(test_predictions[mask] == class_idx)
            class_accuracies[class_idx] = class_acc
    
    # Generate report text (生成报告文本)
    report_path = os.path.join(results_dir, 'logs', 'training_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("MNIST Handwritten Digit Recognition - Training Report\n")
        f.write("MNIST手写数字识别 - 训练报告\n")
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
        f.write(f"Training Time: {training_time:.2f} seconds\n\n")
        
        # Dataset Information (数据集信息)
        f.write("Dataset Information (数据集信息):\n")
        f.write("-" * 30 + "\n")
        f.write(f"Training samples: {X_train.shape[0]}\n")
        f.write(f"Validation samples: {X_val.shape[0]}\n")
        f.write(f"Test samples: {X_test.shape[0]}\n")
        f.write(f"Input features: {X_train.shape[1]} (28x28 pixels)\n")
        f.write(f"Number of classes: 10 (digits 0-9)\n\n")
        
        # Final Performance (最终性能)
        train_acc = model.compute_accuracy(X_train, y_train)
        val_acc = model.compute_accuracy(X_val, y_val)
        test_acc = model.compute_accuracy(X_test, y_test)
        
        f.write("Final Performance (最终性能):\n")
        f.write("-" * 30 + "\n")
        f.write(f"Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)\n")
        f.write(f"Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)\n")
        f.write(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)\n\n")
        
        # Per-class accuracy (每个类别准确率)
        f.write("Per-Class Accuracy (每个类别准确率):\n")
        f.write("-" * 30 + "\n")
        for class_idx, accuracy in class_accuracies.items():
            f.write(f"Digit {class_idx}: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write("\n")
        
        # Training Progress (训练进度)
        f.write("Training Progress (训练进度):\n")
        f.write("-" * 30 + "\n")
        f.write(f"Best Training Accuracy: {max(model.train_accuracies):.4f}\n")
        f.write(f"Best Validation Accuracy: {max(model.val_accuracies):.4f}\n")
        f.write(f"Final Training Loss: {model.train_losses[-1]:.4f}\n")
        f.write(f"Final Validation Loss: {model.val_losses[-1]:.4f}\n")
    
    print(f"✅ Analysis report saved to: {report_path}")
    print(f"✅ 分析报告保存到: {report_path}")

def visualize_predictions(model, X_test, y_test, results_dir):
    """
    Visualize model predictions on test samples
    可视化模型在测试样本上的预测
    """
    print("🎨 Creating prediction visualizations... (正在创建预测可视化...)")
    
    # Select random test samples (选择随机测试样本)
    n_samples = 20
    indices = np.random.choice(len(X_test), n_samples, replace=False)
    
    # Make predictions (进行预测)
    predictions = model.predict(X_test[indices])
    probabilities = model.predict_proba(X_test[indices])
    true_labels = y_test[indices]
    
    # Create visualization (创建可视化)
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    fig.suptitle('Model Predictions on Test Set (模型在测试集上的预测)', fontsize=16, y=0.98)
    
    for i, idx in enumerate(indices):
        row, col = i // 5, i % 5
        
        # Reshape image (重塑图片)
        image = X_test[idx].reshape(28, 28)
        true_label = true_labels[i]
        pred_label = predictions[i]
        confidence = probabilities[i][pred_label]
        
        # Plot image (绘制图片)
        axes[row, col].imshow(image, cmap='gray')
        
        # Set title with prediction info (设置带预测信息的标题)
        color = 'green' if pred_label == true_label else 'red'
        title = f'True: {true_label}, Pred: {pred_label}\nConf: {confidence:.3f}'
        axes[row, col].set_title(title, color=color, fontsize=10)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # Save visualization (保存可视化)
    pred_plot_path = os.path.join(results_dir, 'plots', 'predictions_visualization.png')
    plt.savefig(pred_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create confusion matrix visualization (创建混淆矩阵可视化)
    create_confusion_matrix(model, X_test, y_test, results_dir)
    
    print(f"✅ Prediction visualizations saved to: {pred_plot_path}")
    print(f"✅ 预测可视化保存到: {pred_plot_path}")

def create_confusion_matrix(model, X_test, y_test, results_dir):
    """
    Create and save confusion matrix
    创建并保存混淆矩阵
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    # Get predictions (获取预测)
    predictions = model.predict(X_test)
    
    # Create confusion matrix (创建混淆矩阵)
    cm = confusion_matrix(y_test, predictions)
    
    # Normalize confusion matrix (标准化混淆矩阵)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot confusion matrix (绘制混淆矩阵)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.title('Normalized Confusion Matrix (标准化混淆矩阵)')
    plt.xlabel('Predicted Label (预测标签)')
    plt.ylabel('True Label (真实标签)')
    
    # Save confusion matrix (保存混淆矩阵)
    cm_path = os.path.join(results_dir, 'plots', 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Confusion matrix saved to: {cm_path}")
    print(f"✅ 混淆矩阵保存到: {cm_path}")

def main():
    """
    Main training function
    主训练函数
    """
    parser = argparse.ArgumentParser(description='Train MLP for MNIST digit recognition')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--hidden_sizes', nargs='+', type=int, default=[128, 64], 
                        help='Hidden layer sizes')
    parser.add_argument('--validation_split', type=float, default=0.1, 
                        help='Validation split ratio')
    
    args = parser.parse_args()
    
    # Configuration (配置)
    config = {
        'layer_sizes': [784] + args.hidden_sizes + [10],  # Input: 784, Output: 10
        'activations': ['relu'] * len(args.hidden_sizes),
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'validation_split': args.validation_split
    }
    
    # Set random seed for reproducibility (设置随机种子以确保可重现性)
    np.random.seed(42)
    
    try:
        # Train model (训练模型)
        model, history = train_mnist_classifier(config)
        
        print("\n🎉 Training completed successfully! (训练成功完成！)")
        print("🎉 Check the results directory for saved models and visualizations.")
        print("🎉 查看results目录中保存的模型和可视化结果。")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        print(f"❌ 训练过程中出错: {e}")
        raise

if __name__ == "__main__":
    main() 