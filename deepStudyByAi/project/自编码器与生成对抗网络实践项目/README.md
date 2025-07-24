# 自编码器与生成对抗网络实践项目
# Autoencoder and Generative Adversarial Network Practice Project

本项目包含自编码器(Autoencoder)和生成对抗网络(GAN)的完整实现，使用PyTorch框架。
This project contains complete implementations of Autoencoder and Generative Adversarial Network (GAN) using PyTorch framework.

## 项目结构 / Project Structure

```
自编码器与生成对抗网络实践项目/
├── README.md                    # 项目说明
├── requirements.txt             # 依赖包
├── autoencoder/                 # 自编码器相关代码
│   ├── basic_autoencoder.py     # 基础自编码器
│   ├── variational_autoencoder.py  # 变分自编码器(VAE)
│   ├── train_autoencoder.py     # 训练脚本
│   └── test_autoencoder.py      # 测试脚本
├── gan/                         # GAN相关代码
│   ├── basic_gan.py             # 基础GAN
│   ├── dcgan.py                 # 深度卷积GAN
│   ├── train_gan.py             # 训练脚本
│   └── test_gan.py              # 测试脚本
├── utils/                       # 工具函数
│   ├── data_loader.py           # 数据加载器
│   ├── visualizer.py            # 可视化工具
│   └── metrics.py               # 评估指标
└── results/                     # 结果保存目录
    ├── autoencoder/             # 自编码器结果
    └── gan/                     # GAN结果
```

## 主要功能 / Main Features

### 自编码器 / Autoencoder
- **基础自编码器**: 实现数据压缩和重构
- **变分自编码器(VAE)**: 实现数据生成和潜在空间学习

### 生成对抗网络 / GAN
- **基础GAN**: 实现简单的数据生成
- **DCGAN**: 使用卷积层的深度生成对抗网络

## 使用方法 / Usage

1. 安装依赖 / Install dependencies:
```bash
pip install -r requirements.txt
```

2. 训练自编码器 / Train autoencoder:
```bash
python autoencoder/train_autoencoder.py
```

3. 训练GAN / Train GAN:
```bash
python gan/train_gan.py
```

## 数据集 / Dataset
项目使用MNIST手写数字数据集进行训练和测试。
The project uses MNIST handwritten digit dataset for training and testing. 