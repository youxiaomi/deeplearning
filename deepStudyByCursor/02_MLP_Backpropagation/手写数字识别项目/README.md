# Handwritten Digit Recognition Project

手写数字识别项目

## Project Overview

项目概述

This project demonstrates handwritten digit recognition using Multi-Layer Perceptron (MLP) with backpropagation algorithm. We'll build a neural network from scratch and then compare it with PyTorch implementation.

本项目展示了使用多层感知机（MLP）和反向传播算法进行手写数字识别。我们将从零开始构建神经网络，然后与PyTorch实现进行比较。

## Dataset

数据集

We use the MNIST dataset which contains:
- **Training set**: 60,000 images (训练集：60,000张图片)
- **Validation set**: 10,000 images (验证集：10,000张图片) 
- **Test set**: 10,000 images (测试集：10,000张图片)
- **Image size**: 28×28 pixels (图片尺寸：28×28像素)
- **Classes**: 10 digits (0-9) (类别：10个数字0-9)

## Project Structure

项目结构

```
手写数字识别项目/
├── README.md                    # Project documentation (项目文档)
├── data/                        # Data directory (数据目录)
│   ├── raw/                     # Raw MNIST data (原始MNIST数据)
│   └── processed/               # Processed data (处理后数据)
├── src/                         # Source code (源代码)
│   ├── data_loader.py          # Data loading utilities (数据加载工具)
│   ├── mlp_scratch.py          # MLP from scratch (从零实现MLP)
│   ├── mlp_pytorch.py          # MLP using PyTorch (使用PyTorch的MLP)
│   ├── train.py                # Training script (训练脚本)
│   ├── evaluate.py             # Evaluation script (评估脚本)
│   └── utils.py                # Utility functions (工具函数)
├── notebooks/                   # Jupyter notebooks (Jupyter笔记本)
│   ├── 01_data_exploration.ipynb      # Data exploration (数据探索)
│   ├── 02_mlp_from_scratch.ipynb      # MLP implementation (MLP实现)
│   └── 03_pytorch_comparison.ipynb    # PyTorch comparison (PyTorch对比)
├── models/                      # Saved models (保存的模型)
├── results/                     # Training results (训练结果)
│   ├── plots/                   # Training plots (训练图表)
│   └── logs/                    # Training logs (训练日志)
└── requirements.txt             # Dependencies (依赖项)
```

## Learning Objectives

学习目标

1. **Mathematical Foundation** (数学基础)
   - Understand forward propagation (理解前向传播)
   - Master backpropagation algorithm (掌握反向传播算法)
   - Learn gradient descent optimization (学习梯度下降优化)

2. **Implementation Skills** (实现技能)
   - Build neural network from scratch (从零构建神经网络)
   - Use matrix operations efficiently (高效使用矩阵运算)
   - Handle data preprocessing (处理数据预处理)

3. **Practical Experience** (实践经验)
   - Train and validate models (训练和验证模型)
   - Analyze training dynamics (分析训练动态)
   - Compare different implementations (比较不同实现)

## Getting Started

开始使用

1. **Install Dependencies** (安装依赖)
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Data** (下载数据)
   ```bash
   python src/data_loader.py
   ```

3. **Train Model** (训练模型)
   ```bash
   # Train from scratch implementation
   python src/train.py --model scratch
   
   # Train PyTorch implementation
   python src/train.py --model pytorch
   ```

4. **Evaluate Model** (评估模型)
   ```bash
   python src/evaluate.py --model_path models/best_model.pkl
   ```

## Key Concepts Covered

涵盖的关键概念

### 1. Neural Network Architecture

神经网络架构

- **Input Layer**: 784 neurons (28×28 flattened) (输入层：784个神经元)
- **Hidden Layer 1**: 128 neurons with ReLU activation (隐藏层1：128个神经元，ReLU激活)
- **Hidden Layer 2**: 64 neurons with ReLU activation (隐藏层2：64个神经元，ReLU激活)
- **Output Layer**: 10 neurons with Softmax activation (输出层：10个神经元Softmax激活)

### 2. Mathematical Components

数学组件

- **Activation Functions**: ReLU, Sigmoid, Softmax (激活函数)
- **Loss Function**: Cross-entropy loss (损失函数：交叉熵损失)
- **Optimization**: Stochastic Gradient Descent (优化：随机梯度下降)
- **Regularization**: L2 regularization, Dropout (正则化：L2正则化，Dropout)

### 3. Training Process

训练过程

- **Forward Pass**: Compute predictions (前向传播：计算预测)
- **Loss Calculation**: Measure prediction error (损失计算：测量预测误差)
- **Backward Pass**: Compute gradients (反向传播：计算梯度)
- **Parameter Update**: Apply gradient descent (参数更新：应用梯度下降)

## Expected Results

预期结果

- **Training Accuracy**: ~98% (训练准确率：约98%)
- **Validation Accuracy**: ~97% (验证准确率：约97%)
- **Test Accuracy**: ~96% (测试准确率：约96%)
- **Training Time**: ~10 minutes on CPU (训练时间：CPU约10分钟)

## Next Steps

下一步

After completing this project, you can:

完成此项目后，您可以：

1. **Experiment with architectures** (实验不同架构)
   - Try different layer sizes (尝试不同层大小)
   - Add more hidden layers (添加更多隐藏层)
   - Use different activation functions (使用不同激活函数)

2. **Improve performance** (提升性能)
   - Implement batch normalization (实现批量归一化)
   - Add learning rate scheduling (添加学习率调度)
   - Use advanced optimizers (使用高级优化器)

3. **Extend to other datasets** (扩展到其他数据集)
   - Fashion-MNIST
   - CIFAR-10 (requires CNN architecture)
   - Custom digit datasets

## Mathematical Notation Reference

数学符号参考

For detailed mathematical notation used in this project, refer to:
`../01_Perceptron/数学符号详解与读音.md`

有关此项目中使用的详细数学符号，请参考：
`../01_Perceptron/数学符号详解与读音.md`

## Support

支持

If you encounter any issues or have questions:

如果您遇到任何问题或有疑问：

1. Check the mathematical foundations in `../多层感知机与反向传播.md`
2. Review the quiz questions in `../quiz.md`
3. Study the notation guide in `../01_Perceptron/数学符号详解与读音.md`

---

**Happy Learning! 快乐学习！** 🚀 