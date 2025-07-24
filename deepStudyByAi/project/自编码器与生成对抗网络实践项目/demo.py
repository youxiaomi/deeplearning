#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
自编码器与生成对抗网络演示脚本
Autoencoder and GAN Demo Script

这个脚本演示如何使用项目中的各种模型进行训练和推理。
This script demonstrates how to use various models in the project for training and inference.
"""

import os
import sys
import torch
import argparse
from pathlib import Path

# 添加项目路径 / Add project path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from autoencoder.basic_autoencoder import BasicAutoencoder
from autoencoder.variational_autoencoder import VariationalAutoencoder
from gan.basic_gan import BasicGAN
from gan.dcgan import DCGAN
from utils.data_loader import get_mnist_dataloader
from utils.visualizer import Visualizer

def demo_basic_autoencoder():
    """演示基础自编码器 / Demo basic autoencoder"""
    print("=" * 60)
    print("基础自编码器演示 / Basic Autoencoder Demo")
    print("=" * 60)
    
    # 设置设备 / Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备 / Using device: {device}")
    
    # 创建模型 / Create model
    model = BasicAutoencoder(input_dim=784, hidden_dim=128, latent_dim=32).to(device)
    print(f"模型参数数量 / Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 加载数据 / Load data
    dataloader = get_mnist_dataloader(batch_size=64, train=False, normalize_range=(0, 1))
    
    # 获取一批数据 / Get one batch of data
    data_iter = iter(dataloader)
    batch = next(data_iter)
    images, labels = batch
    images = images.to(device)
    
    # 前向传播 / Forward pass
    model.eval()
    with torch.no_grad():
        reconstructed, latent = model(images)
    
    print(f"输入图像形状 / Input shape: {images.shape}")
    print(f"潜在表示形状 / Latent shape: {latent.shape}")
    print(f"重构图像形状 / Reconstructed shape: {reconstructed.shape}")
    
    # 可视化结果 / Visualize results
    visualizer = Visualizer('./demo_results')
    visualizer.compare_images(
        images[:8], 
        reconstructed[:8].view(-1, 1, 28, 28),
        normalize_range=(0, 1),
        save_name='basic_autoencoder_demo.png'
    )
    
    print("基础自编码器演示完成 / Basic autoencoder demo completed")
    print("结果保存到 ./demo_results/basic_autoencoder_demo.png")

def demo_vae():
    """演示变分自编码器 / Demo Variational Autoencoder"""
    print("\n" + "=" * 60)
    print("变分自编码器演示 / Variational Autoencoder Demo")
    print("=" * 60)
    
    # 设置设备 / Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备 / Using device: {device}")
    
    # 创建模型 / Create model
    model = VariationalAutoencoder(input_dim=784, hidden_dim=400, latent_dim=20).to(device)
    print(f"模型参数数量 / Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 加载数据 / Load data
    dataloader = get_mnist_dataloader(batch_size=64, train=False, normalize_range=(0, 1))
    
    # 获取一批数据 / Get one batch of data
    data_iter = iter(dataloader)
    batch = next(data_iter)
    images, labels = batch
    images = images.to(device)
    
    # 前向传播 / Forward pass
    model.eval()
    with torch.no_grad():
        reconstructed, mu, logvar = model(images)
        
        # 生成新样本 / Generate new samples
        generated_samples = model.sample(16, device)
    
    print(f"输入图像形状 / Input shape: {images.shape}")
    print(f"均值形状 / Mean shape: {mu.shape}")
    print(f"对数方差形状 / Log variance shape: {logvar.shape}")
    print(f"重构图像形状 / Reconstructed shape: {reconstructed.shape}")
    print(f"生成样本形状 / Generated samples shape: {generated_samples.shape}")
    
    # 可视化结果 / Visualize results
    visualizer = Visualizer('./demo_results')
    
    # 重构比较 / Reconstruction comparison
    visualizer.compare_images(
        images[:8], 
        reconstructed[:8].view(-1, 1, 28, 28),
        normalize_range=(0, 1),
        save_name='vae_reconstruction_demo.png'
    )
    
    # 生成样本 / Generated samples
    visualizer.show_images(
        generated_samples.view(-1, 1, 28, 28),
        normalize_range=(0, 1),
        rows=4,
        cols=4,
        save_name='vae_generated_demo.png'
    )
    
    print("变分自编码器演示完成 / VAE demo completed")
    print("结果保存到 ./demo_results/vae_*.png")

def demo_basic_gan():
    """演示基础GAN / Demo Basic GAN"""
    print("\n" + "=" * 60)
    print("基础GAN演示 / Basic GAN Demo")
    print("=" * 60)
    
    # 设置设备 / Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备 / Using device: {device}")
    
    # 创建模型 / Create model
    gan = BasicGAN(noise_dim=100, hidden_dim=256, data_dim=784, device=device)
    gan.set_optimizers()
    
    g_params = sum(p.numel() for p in gan.generator.parameters())
    d_params = sum(p.numel() for p in gan.discriminator.parameters())
    print(f"生成器参数 / Generator parameters: {g_params:,}")
    print(f"判别器参数 / Discriminator parameters: {d_params:,}")
    
    # 生成样本 / Generate samples
    print("生成随机样本 / Generating random samples...")
    samples = gan.generate_samples(16)
    samples = samples.view(-1, 1, 28, 28)  # 重塑为图像 / Reshape to images
    
    print(f"生成样本形状 / Generated samples shape: {samples.shape}")
    
    # 可视化结果 / Visualize results
    visualizer = Visualizer('./demo_results')
    visualizer.show_images(
        samples,
        normalize_range=(-1, 1),
        rows=4,
        cols=4,
        save_name='basic_gan_demo.png'
    )
    
    print("基础GAN演示完成 / Basic GAN demo completed")
    print("结果保存到 ./demo_results/basic_gan_demo.png")
    print("注意：未训练的GAN生成的图像质量较差 / Note: Untrained GAN generates poor quality images")

def demo_dcgan():
    """演示DCGAN / Demo DCGAN"""
    print("\n" + "=" * 60)
    print("深度卷积GAN演示 / DCGAN Demo")
    print("=" * 60)
    
    # 设置设备 / Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备 / Using device: {device}")
    
    # 创建模型 / Create model
    dcgan = DCGAN(noise_dim=100, channels=1, feature_map_size=64, device=device)
    dcgan.set_optimizers()
    
    g_params = sum(p.numel() for p in dcgan.generator.parameters())
    d_params = sum(p.numel() for p in dcgan.discriminator.parameters())
    print(f"生成器参数 / Generator parameters: {g_params:,}")
    print(f"判别器参数 / Discriminator parameters: {d_params:,}")
    
    # 生成样本 / Generate samples
    print("生成随机样本 / Generating random samples...")
    samples = dcgan.generate_samples(16)
    
    print(f"生成样本形状 / Generated samples shape: {samples.shape}")
    
    # 可视化结果 / Visualize results
    visualizer = Visualizer('./demo_results')
    visualizer.show_images(
        samples,
        normalize_range=(-1, 1),
        rows=4,
        cols=4,
        save_name='dcgan_demo.png'
    )
    
    print("DCGAN演示完成 / DCGAN demo completed")
    print("结果保存到 ./demo_results/dcgan_demo.png")
    print("注意：未训练的DCGAN生成的图像质量较差 / Note: Untrained DCGAN generates poor quality images")

def show_training_instructions():
    """显示训练说明 / Show training instructions"""
    print("\n" + "=" * 60)
    print("训练说明 / Training Instructions")
    print("=" * 60)
    
    print("要训练模型，请使用以下命令 / To train models, use the following commands:")
    print()
    print("1. 训练基础自编码器 / Train basic autoencoder:")
    print("   python autoencoder/train_autoencoder.py --model_type basic --epochs 50")
    print()
    print("2. 训练变分自编码器 / Train VAE:")
    print("   python autoencoder/train_autoencoder.py --model_type vae --epochs 50")
    print()
    print("3. 训练基础GAN / Train basic GAN:")
    print("   python gan/train_gan.py --model_type basic --epochs 100")
    print()
    print("4. 训练DCGAN:")
    print("   python gan/train_gan.py --model_type dcgan --epochs 100")
    print()
    print("训练结果将保存在 ./results/ 目录下")
    print("Training results will be saved in ./results/ directory")
    print()
    print("更多选项请查看 --help")
    print("For more options, see --help")

def main():
    """主函数 / Main function"""
    parser = argparse.ArgumentParser(description='自编码器与GAN演示 / Autoencoder and GAN Demo')
    parser.add_argument('--model', type=str, default='all',
                       choices=['all', 'autoencoder', 'vae', 'gan', 'dcgan'],
                       help='要演示的模型 / Model to demonstrate')
    
    args = parser.parse_args()
    
    print("自编码器与生成对抗网络演示")
    print("Autoencoder and Generative Adversarial Network Demo")
    print("=" * 60)
    print("此演示将展示项目中各种模型的基本功能")
    print("This demo showcases basic functionality of various models in the project")
    
    # 创建结果目录 / Create results directory
    os.makedirs('./demo_results', exist_ok=True)
    
    try:
        if args.model == 'all':
            demo_basic_autoencoder()
            demo_vae()
            demo_basic_gan()
            demo_dcgan()
            show_training_instructions()
        elif args.model == 'autoencoder':
            demo_basic_autoencoder()
        elif args.model == 'vae':
            demo_vae()
        elif args.model == 'gan':
            demo_basic_gan()
        elif args.model == 'dcgan':
            demo_dcgan()
        
        print("\n" + "=" * 60)
        print("演示完成！/ Demo completed!")
        print("所有结果保存在 ./demo_results/ 目录下")
        print("All results saved in ./demo_results/ directory")
        print("=" * 60)
    
    except Exception as e:
        print(f"\n演示过程中出现错误 / Error during demo: {e}")
        print("请确保已安装所有必要的依赖包")
        print("Please ensure all required dependencies are installed")
        print("运行: pip install -r requirements.txt")
        print("Run: pip install -r requirements.txt")

if __name__ == "__main__":
    main() 