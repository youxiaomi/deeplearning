"""
Testing and Evaluation Script for CIFAR-10 CNN Models
CIFAR-10 CNN模型测试和评估脚本

This script provides comprehensive evaluation of trained models.
Think of this as a detailed report card for your AI model!
这个脚本提供训练模型的全面评估。
把这想象成AI模型的详细成绩单！
"""

import torch
import argparse
import os
import numpy as np
from datetime import datetime

# Import our custom modules
# 导入我们的自定义模块
from data_loader import CIFAR10DataLoader
from model import get_model
from utils import (
    evaluate_model, plot_confusion_matrix, plot_class_performance,
    visualize_predictions, print_model_performance_summary,
    load_model, get_device
)

class CIFAR10Evaluator:
    """
    CIFAR-10 Model Evaluator
    CIFAR-10模型评估器
    
    This class provides comprehensive evaluation of trained models.
    Think of it as a thorough health check for your AI model!
    这个类提供训练模型的全面评估。
    把它想象成AI模型的全面健康检查！
    """
    
    def __init__(self, model_name='simple', batch_size=32, data_dir='./data'):
        """
        Initialize the evaluator
        初始化评估器
        
        Args:
            model_name: Type of model to evaluate ('simple', 'improved', 'resnet')
            batch_size: Batch size for evaluation (评估批次大小)
            data_dir: Directory for dataset (数据集目录)
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.data_dir = data_dir
        
        # Get device
        # 获取设备
        self.device = get_device()
        
        # Setup data and model
        # 设置数据和模型
        self._setup_data()
        self._setup_model()
        
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
        
    def _setup_model(self):
        """Set up the model"""
        print(f"Setting up {self.model_name} model...")
        print(f"设置{self.model_name}模型...")
        
        self.model = get_model(self.model_name, num_classes=10)
        self.model.to(self.device)
        
    def load_trained_model(self, model_path):
        """
        Load a trained model
        加载训练好的模型
        
        Args:
            model_path: Path to the saved model (保存模型的路径)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        checkpoint = load_model(self.model, model_path)
        return checkpoint
    
    def evaluate_comprehensive(self, model_path, save_dir='./results'):
        """
        Comprehensive evaluation of the model
        模型的全面评估
        
        Args:
            model_path: Path to the saved model (保存模型的路径)
            save_dir: Directory to save results (保存结果的目录)
        """
        print("\n" + "="*60)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("全面模型评估")
        print("="*60)
        
        # Load model
        # 加载模型
        checkpoint = self.load_trained_model(model_path)
        
        # Create results directory
        # 创建结果目录
        os.makedirs(save_dir, exist_ok=True)
        
        # Evaluate on test set
        # 在测试集上评估
        print("\nEvaluating on test set...")
        print("在测试集上评估...")
        results = evaluate_model(self.model, self.test_loader, self.device)
        
        # Print comprehensive summary
        # 打印全面摘要
        print_model_performance_summary(results)
        
        # Save results to file
        # 保存结果到文件
        self._save_results_to_file(results, checkpoint, save_dir)
        
        # Generate visualizations
        # 生成可视化
        self._generate_visualizations(results, save_dir)
        
        # Test on individual classes
        # 测试各个类别
        self._class_wise_analysis(results, save_dir)
        
        return results
    
    def _save_results_to_file(self, results, checkpoint, save_dir):
        """Save evaluation results to text file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(save_dir, f"evaluation_results_{timestamp}.txt")
        
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write("CIFAR-10 Model Evaluation Results\n")
            f.write("CIFAR-10模型评估结果\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"模型: {self.model_name}\n")
            f.write(f"评估日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Model training info
            # 模型训练信息
            f.write("Training Information:\n")
            f.write("训练信息:\n")
            f.write(f"Training Epoch: {checkpoint['epoch']}\n")
            f.write(f"Training Accuracy: {checkpoint['train_accuracy']:.2f}%\n")
            f.write(f"Validation Accuracy: {checkpoint['val_accuracy']:.2f}%\n")
            f.write(f"训练Epoch: {checkpoint['epoch']}\n")
            f.write(f"训练准确率: {checkpoint['train_accuracy']:.2f}%\n")
            f.write(f"验证准确率: {checkpoint['val_accuracy']:.2f}%\n\n")
            
            # Test results
            # 测试结果
            f.write("Test Results:\n")
            f.write("测试结果:\n")
            f.write(f"Test Accuracy: {results['accuracy']:.2f}%\n")
            f.write(f"Test Loss: {results['loss']:.4f}\n")
            f.write(f"测试准确率: {results['accuracy']:.2f}%\n")
            f.write(f"测试损失: {results['loss']:.4f}\n\n")
            
            # Per-class results
            # 每类结果
            f.write("Per-Class Results:\n")
            f.write("每类结果:\n")
            f.write(f"{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}\n")
            f.write("-"*60 + "\n")
            
            class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                          'dog', 'frog', 'horse', 'ship', 'truck']
            
            for i, class_name in enumerate(class_names):
                f.write(f"{class_name:<12} {results['precision'][i]:<10.3f} "
                       f"{results['recall'][i]:<10.3f} {results['f1_score'][i]:<10.3f} "
                       f"{results['support'][i]:<10}\n")
        
        print(f"Results saved to: {results_file}")
        print(f"结果已保存到: {results_file}")
    
    def _generate_visualizations(self, results, save_dir):
        """Generate and save visualization plots"""
        print("\nGenerating visualizations...")
        print("生成可视化...")
        
        # Confusion matrix
        # 混淆矩阵
        cm_path = os.path.join(save_dir, f"{self.model_name}_confusion_matrix.png")
        plot_confusion_matrix(results['labels'], results['predictions'], save_path=cm_path)
        
        # Class performance
        # 类别性能
        perf_path = os.path.join(save_dir, f"{self.model_name}_class_performance.png")
        plot_class_performance(
            results['precision'], results['recall'], 
            results['f1_score'], results['support'],
            save_path=perf_path
        )
        
        # Prediction visualizations
        # 预测可视化
        pred_path = os.path.join(save_dir, f"{self.model_name}_predictions.png")
        visualize_predictions(self.model, self.test_loader, self.device, 
                            num_samples=16, save_path=pred_path)
        
        print("Visualizations saved successfully!")
        print("可视化已成功保存！")
    
    def _class_wise_analysis(self, results, save_dir):
        """Perform detailed class-wise analysis"""
        print("\nPerforming class-wise analysis...")
        print("执行类别分析...")
        
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                      'dog', 'frog', 'horse', 'ship', 'truck']
        class_names_cn = ['飞机', '汽车', '鸟', '猫', '鹿',
                         '狗', '青蛙', '马', '船', '卡车']
        
        # Find best and worst performing classes
        # 找到表现最好和最差的类别
        f1_scores = results['f1_score']
        best_class_idx = np.argmax(f1_scores)
        worst_class_idx = np.argmin(f1_scores)
        
        print(f"\nBest performing class: {class_names[best_class_idx]} ({class_names_cn[best_class_idx]})")
        print(f"F1-Score: {f1_scores[best_class_idx]:.3f}")
        print(f"表现最好的类别: {class_names[best_class_idx]} ({class_names_cn[best_class_idx]})")
        print(f"F1分数: {f1_scores[best_class_idx]:.3f}")
        
        print(f"\nWorst performing class: {class_names[worst_class_idx]} ({class_names_cn[worst_class_idx]})")
        print(f"F1-Score: {f1_scores[worst_class_idx]:.3f}")
        print(f"表现最差的类别: {class_names[worst_class_idx]} ({class_names_cn[worst_class_idx]})")
        print(f"F1分数: {f1_scores[worst_class_idx]:.3f}")
        
        # Identify most confused classes
        # 识别最容易混淆的类别
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(results['labels'], results['predictions'])
        
        # Find most confused pairs (excluding diagonal)
        # 找到最容易混淆的对（排除对角线）
        cm_no_diag = cm.copy()
        np.fill_diagonal(cm_no_diag, 0)
        
        max_confusion_idx = np.unravel_index(np.argmax(cm_no_diag), cm_no_diag.shape)
        true_class, pred_class = max_confusion_idx
        
        print(f"\nMost confused classes:")
        print(f"True: {class_names[true_class]} ({class_names_cn[true_class]})")
        print(f"Predicted as: {class_names[pred_class]} ({class_names_cn[pred_class]})")
        print(f"Confusion count: {cm[true_class, pred_class]}")
        print(f"最容易混淆的类别:")
        print(f"真实: {class_names[true_class]} ({class_names_cn[true_class]})")
        print(f"预测为: {class_names[pred_class]} ({class_names_cn[pred_class]})")
        print(f"混淆数量: {cm[true_class, pred_class]}")
    
    def compare_models(self, model_paths, save_dir='./results'):
        """
        Compare multiple models
        比较多个模型
        
        Args:
            model_paths: List of paths to saved models (保存模型的路径列表)
            save_dir: Directory to save comparison results (保存比较结果的目录)
        """
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("模型比较")
        print("="*60)
        
        results_list = []
        model_names = []
        
        for model_path in model_paths:
            if not os.path.exists(model_path):
                print(f"Warning: Model file not found: {model_path}")
                print(f"警告: 未找到模型文件: {model_path}")
                continue
            
            # Extract model name from path
            # 从路径中提取模型名称
            model_name = os.path.basename(model_path).split('_')[1]
            model_names.append(model_name)
            
            # Load and evaluate model
            # 加载和评估模型
            print(f"\nEvaluating {model_name} model...")
            print(f"评估{model_name}模型...")
            
            # Update model architecture
            # 更新模型架构
            self.model_name = model_name
            self._setup_model()
            
            checkpoint = self.load_trained_model(model_path)
            results = evaluate_model(self.model, self.test_loader, self.device)
            results_list.append(results)
        
        # Create comparison summary
        # 创建比较摘要
        self._create_comparison_summary(model_names, results_list, save_dir)
    
    def _create_comparison_summary(self, model_names, results_list, save_dir):
        """Create a comparison summary of multiple models"""
        print("\nModel Comparison Summary:")
        print("模型比较摘要:")
        print("-" * 60)
        print(f"{'Model':<15} {'Accuracy':<10} {'Avg F1':<10} {'Best Class':<15} {'Worst Class':<15}")
        print("-" * 60)
        
        comparison_data = []
        
        for name, results in zip(model_names, results_list):
            accuracy = results['accuracy']
            avg_f1 = np.mean(results['f1_score'])
            
            class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                          'dog', 'frog', 'horse', 'ship', 'truck']
            
            best_class_idx = np.argmax(results['f1_score'])
            worst_class_idx = np.argmin(results['f1_score'])
            
            best_class = class_names[best_class_idx]
            worst_class = class_names[worst_class_idx]
            
            print(f"{name:<15} {accuracy:<10.2f} {avg_f1:<10.3f} {best_class:<15} {worst_class:<15}")
            
            comparison_data.append({
                'model': name,
                'accuracy': accuracy,
                'avg_f1': avg_f1,
                'best_class': best_class,
                'worst_class': worst_class
            })
        
        # Save comparison to file
        # 保存比较到文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comp_file = os.path.join(save_dir, f"model_comparison_{timestamp}.txt")
        
        with open(comp_file, 'w', encoding='utf-8') as f:
            f.write("Model Comparison Results\n")
            f.write("模型比较结果\n")
            f.write("="*50 + "\n\n")
            
            for data in comparison_data:
                f.write(f"Model: {data['model']}\n")
                f.write(f"Accuracy: {data['accuracy']:.2f}%\n")
                f.write(f"Average F1-Score: {data['avg_f1']:.3f}\n")
                f.write(f"Best Class: {data['best_class']}\n")
                f.write(f"Worst Class: {data['worst_class']}\n")
                f.write("-" * 30 + "\n")
        
        print(f"\nComparison results saved to: {comp_file}")
        print(f"比较结果已保存到: {comp_file}")

def main():
    """Main function to run evaluation"""
    parser = argparse.ArgumentParser(description='Evaluate CIFAR-10 CNN Model')
    parser.add_argument('--model', type=str, default='simple',
                       choices=['simple', 'improved', 'resnet'],
                       help='Model architecture to evaluate')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to saved model')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Directory for dataset')
    parser.add_argument('--save-dir', type=str, default='./results',
                       help='Directory to save results')
    parser.add_argument('--compare', nargs='+', default=None,
                       help='Paths to multiple models for comparison')
    
    args = parser.parse_args()
    
    # Create evaluator
    # 创建评估器
    evaluator = CIFAR10Evaluator(
        model_name=args.model,
        batch_size=args.batch_size,
        data_dir=args.data_dir
    )
    
    if args.compare:
        # Compare multiple models
        # 比较多个模型
        evaluator.compare_models(args.compare, args.save_dir)
    else:
        # Evaluate single model
        # 评估单个模型
        results = evaluator.evaluate_comprehensive(args.model_path, args.save_dir)
        
        print(f"\nEvaluation completed successfully!")
        print(f"Results saved to: {args.save_dir}")
        print(f"评估成功完成！")
        print(f"结果已保存到: {args.save_dir}")

if __name__ == '__main__':
    main() 