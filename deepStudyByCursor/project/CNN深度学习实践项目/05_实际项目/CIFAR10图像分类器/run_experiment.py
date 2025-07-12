#!/usr/bin/env python3
"""
Easy-to-use experiment runner for CIFAR-10 CNN project
CIFAR-10 CNN项目的易用实验运行器

This script makes it super easy to run experiments!
Just run this script and follow the interactive prompts.
这个脚本让运行实验变得超级简单！
只需运行这个脚本并按照交互式提示操作。
"""

import os
import sys
import argparse
from datetime import datetime

def print_banner():
    """Print a welcome banner"""
    print("=" * 60)
    print("🚀 CIFAR-10 CNN Image Classifier")
    print("🚀 CIFAR-10 CNN图像分类器")
    print("=" * 60)
    print("Welcome to your first deep learning project!")
    print("欢迎来到你的第一个深度学习项目！")
    print()

def print_model_info():
    """Print information about available models"""
    print("Available Models / 可用模型:")
    print("=" * 40)
    print("1. 'simple' - Simple CNN (初学者推荐)")
    print("   • 3 convolutional layers")
    print("   • ~500K parameters")
    print("   • Expected accuracy: 65-70%")
    print("   • Training time: ~30-45 minutes")
    print()
    print("2. 'improved' - Improved CNN (中级)")
    print("   • 8 convolutional layers")
    print("   • ~2M parameters")
    print("   • Expected accuracy: 80-85%")
    print("   • Training time: ~1-2 hours")
    print()
    print("3. 'resnet' - ResNet-style CNN (高级)")
    print("   • Residual connections")
    print("   • ~1.5M parameters")
    print("   • Expected accuracy: 85-90%")
    print("   • Training time: ~2-3 hours")
    print()

def get_user_choice():
    """Get user's choice for experiment"""
    print("What would you like to do? / 你想做什么？")
    print("1. Train a new model (训练新模型)")
    print("2. Test existing model (测试现有模型)")
    print("3. Quick demo (快速演示)")
    print("4. Compare models (比较模型)")
    print()
    
    while True:
        choice = input("Enter your choice (1-4): ").strip()
        if choice in ['1', '2', '3', '4']:
            return int(choice)
        print("Invalid choice. Please enter 1, 2, 3, or 4.")
        print("无效选择。请输入1、2、3或4。")

def get_model_choice():
    """Get user's model choice"""
    print_model_info()
    
    while True:
        model = input("Choose model (simple/improved/resnet): ").strip().lower()
        if model in ['simple', 'improved', 'resnet']:
            return model
        print("Invalid model. Please choose 'simple', 'improved', or 'resnet'.")
        print("无效模型。请选择'simple'、'improved'或'resnet'。")

def train_model():
    """Train a new model"""
    print("\n🏋️ Training Mode / 训练模式")
    print("=" * 40)
    
    model = get_model_choice()
    
    # Get training parameters
    print(f"\nTraining {model} model...")
    print(f"训练{model}模型...")
    
    # Ask for epochs
    while True:
        try:
            epochs = input("Number of epochs (default: 20): ").strip()
            if not epochs:
                epochs = 20
            else:
                epochs = int(epochs)
            break
        except ValueError:
            print("Please enter a valid number.")
            print("请输入有效数字。")
    
    # Ask for batch size
    while True:
        try:
            batch_size = input("Batch size (default: 32): ").strip()
            if not batch_size:
                batch_size = 32
            else:
                batch_size = int(batch_size)
            break
        except ValueError:
            print("Please enter a valid number.")
            print("请输入有效数字。")
    
    print(f"\nStarting training with:")
    print(f"Model: {model}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"开始训练，参数：")
    print(f"模型: {model}")
    print(f"Epochs: {epochs}")
    print(f"批次大小: {batch_size}")
    
    # Run training
    cmd = f"cd src && python train.py --model {model} --epochs {epochs} --batch-size {batch_size}"
    print(f"\nRunning: {cmd}")
    os.system(cmd)

def test_model():
    """Test an existing model"""
    print("\n🧪 Testing Mode / 测试模式")
    print("=" * 40)
    
    # List available models
    models_dir = "./models"
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
        if model_files:
            print("Available trained models:")
            print("可用的训练模型:")
            for i, model_file in enumerate(model_files, 1):
                print(f"{i}. {model_file}")
            
            while True:
                try:
                    choice = input(f"Choose model (1-{len(model_files)}): ").strip()
                    choice = int(choice) - 1
                    if 0 <= choice < len(model_files):
                        model_file = model_files[choice]
                        break
                    else:
                        print("Invalid choice.")
                        print("无效选择。")
                except ValueError:
                    print("Please enter a valid number.")
                    print("请输入有效数字。")
            
            # Extract model type from filename
            model_type = model_file.split('_')[1]
            model_path = os.path.join(models_dir, model_file)
            
            print(f"\nTesting {model_file}...")
            print(f"测试{model_file}...")
            
            # Run evaluation
            cmd = f"cd src && python test.py --model {model_type} --model-path ../{model_path}"
            print(f"\nRunning: {cmd}")
            os.system(cmd)
        else:
            print("No trained models found. Please train a model first.")
            print("未找到训练模型。请先训练一个模型。")
    else:
        print("Models directory not found. Please train a model first.")
        print("未找到模型目录。请先训练一个模型。")

def quick_demo():
    """Run a quick demo"""
    print("\n⚡ Quick Demo Mode / 快速演示模式")
    print("=" * 40)
    print("Running a quick demo with simple model for 5 epochs...")
    print("使用简单模型运行5个epoch的快速演示...")
    
    # Run quick training
    cmd = "cd src && python train.py --model simple --epochs 5 --batch-size 64"
    print(f"\nRunning: {cmd}")
    os.system(cmd)
    
    # Test the model
    print("\nTesting the demo model...")
    print("测试演示模型...")
    cmd = "cd src && python train.py --model simple --test-only"
    os.system(cmd)

def compare_models():
    """Compare multiple models"""
    print("\n📊 Model Comparison Mode / 模型比较模式")
    print("=" * 40)
    
    models_dir = "./models"
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth') and 'best_' in f]
        if len(model_files) >= 2:
            print("Available models for comparison:")
            print("可用于比较的模型:")
            for i, model_file in enumerate(model_files, 1):
                print(f"{i}. {model_file}")
            
            model_paths = [os.path.join(models_dir, f) for f in model_files]
            
            print(f"\nComparing {len(model_files)} models...")
            print(f"比较{len(model_files)}个模型...")
            
            # Run comparison
            paths_str = " ".join(model_paths)
            cmd = f"cd src && python test.py --compare {paths_str}"
            print(f"\nRunning comparison...")
            os.system(cmd)
        else:
            print("Need at least 2 trained models for comparison.")
            print("需要至少2个训练模型进行比较。")
    else:
        print("Models directory not found. Please train some models first.")
        print("未找到模型目录。请先训练一些模型。")

def check_requirements():
    """Check if required packages are installed"""
    print("Checking requirements...")
    print("检查依赖项...")
    
    try:
        import torch
        import torchvision
        import matplotlib
        import numpy
        import sklearn
        import seaborn
        import tqdm
        print("✓ All required packages are installed!")
        print("✓ 所有必需的包都已安装！")
        return True
    except ImportError as e:
        print(f"✗ Missing package: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        print("请安装依赖项: pip install -r requirements.txt")
        return False

def main():
    """Main function"""
    print_banner()
    
    # Check requirements
    if not check_requirements():
        return
    
    # Get user choice
    choice = get_user_choice()
    
    if choice == 1:
        train_model()
    elif choice == 2:
        test_model()
    elif choice == 3:
        quick_demo()
    elif choice == 4:
        compare_models()
    
    print("\n🎉 Experiment completed!")
    print("🎉 实验完成！")
    print("Check the 'models' and 'results' directories for outputs.")
    print("查看'models'和'results'目录获取输出结果。")

if __name__ == "__main__":
    # Add argument parsing for command line usage
    parser = argparse.ArgumentParser(description='CIFAR-10 CNN Experiment Runner')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'demo', 'compare'],
                       help='Mode to run: train, test, demo, or compare')
    parser.add_argument('--model', type=str, choices=['simple', 'improved', 'resnet'],
                       help='Model to use (for train/test mode)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs for training')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    
    args = parser.parse_args()
    
    if args.mode:
        # Command line mode
        print_banner()
        if not check_requirements():
            sys.exit(1)
        
        if args.mode == 'train':
            if not args.model:
                print("Please specify --model for training mode")
                sys.exit(1)
            cmd = f"cd src && python train.py --model {args.model} --epochs {args.epochs} --batch-size {args.batch_size}"
            os.system(cmd)
        elif args.mode == 'test':
            if not args.model:
                print("Please specify --model for testing mode")
                sys.exit(1)
            cmd = f"cd src && python train.py --model {args.model} --test-only"
            os.system(cmd)
        elif args.mode == 'demo':
            cmd = "cd src && python train.py --model simple --epochs 5 --batch-size 64"
            os.system(cmd)
        elif args.mode == 'compare':
            compare_models()
    else:
        # Interactive mode
        main() 