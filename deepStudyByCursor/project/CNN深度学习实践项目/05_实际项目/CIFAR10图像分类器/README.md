# CIFAR-10 Image Classifier Project
# CIFAR-10图像分类器项目

🚀 **Welcome to your first deep learning project!** 🚀  
🚀 **欢迎来到你的第一个深度学习项目！** 🚀

This project teaches you how to build, train, and evaluate Convolutional Neural Networks (CNNs) for image classification using the CIFAR-10 dataset.

这个项目教你如何使用CIFAR-10数据集构建、训练和评估用于图像分类的卷积神经网络（CNN）。

## 🎯 What You'll Learn / 你将学到什么

- **Image Classification Basics** / **图像分类基础**
  - Understanding pixels and images as data
  - How computers "see" images
  - 理解像素和图像作为数据
  - 计算机如何"看"图像

- **Convolutional Neural Networks (CNNs)** / **卷积神经网络（CNN）**
  - What are convolutions and why they work for images
  - Building CNN architectures from scratch
  - 什么是卷积以及为什么它们适用于图像
  - 从零开始构建CNN架构

- **Deep Learning Pipeline** / **深度学习流水线**
  - Data loading and preprocessing
  - Model training and validation
  - Performance evaluation and visualization
  - 数据加载和预处理
  - 模型训练和验证
  - 性能评估和可视化

## 📁 Project Structure / 项目结构

```
CIFAR10图像分类器/
├── src/                     # Source code / 源代码
│   ├── data_loader.py      # Data loading and preprocessing / 数据加载和预处理
│   ├── model.py            # CNN model definitions / CNN模型定义
│   ├── train.py            # Training script / 训练脚本
│   ├── test.py             # Testing and evaluation / 测试和评估
│   └── utils.py            # Utility functions / 实用函数
├── data/                   # Dataset storage (auto-created) / 数据集存储（自动创建）
├── models/                 # Saved models (auto-created) / 保存的模型（自动创建）
├── results/                # Evaluation results (auto-created) / 评估结果（自动创建）
├── notebooks/              # Jupyter notebooks for exploration / 用于探索的Jupyter笔记本
├── requirements.txt        # Python dependencies / Python依赖项
├── run_experiment.py       # Easy experiment runner / 简易实验运行器
└── README.md              # This file / 这个文件
```

## 🚀 Quick Start / 快速开始

### Option 1: Interactive Mode (Recommended for Beginners) / 选项1：交互模式（推荐给初学者）

```bash
# 1. Install dependencies / 安装依赖项
pip install -r requirements.txt

# 2. Run the interactive experiment runner / 运行交互式实验运行器
python run_experiment.py
```

The interactive runner will guide you through:
- Choosing a model (simple/improved/resnet)
- Setting training parameters
- Running experiments
- Viewing results

交互式运行器将指导你完成：
- 选择模型（simple/improved/resnet）
- 设置训练参数
- 运行实验
- 查看结果

### Option 2: Command Line Mode / 选项2：命令行模式

```bash
# Quick demo (5 epochs) / 快速演示（5个epoch）
python run_experiment.py --mode demo

# Train a specific model / 训练特定模型
python run_experiment.py --mode train --model simple --epochs 20

# Test a trained model / 测试训练好的模型
python run_experiment.py --mode test --model simple
```

### Option 3: Manual Mode (For Advanced Users) / 选项3：手动模式（适合高级用户）

```bash
cd src

# Train a model / 训练模型
python train.py --model simple --epochs 20 --batch-size 32

# Test a model / 测试模型
python test.py --model simple --model-path ../models/best_simple_model.pth

# Compare multiple models / 比较多个模型
python test.py --compare ../models/best_simple_model.pth ../models/best_improved_model.pth
```

## 🤖 Available Models / 可用模型

### 1. Simple CNN (初学者推荐)
- **Architecture**: 3 convolutional layers + 2 fully connected layers
- **Parameters**: ~500K
- **Expected Accuracy**: 65-70%
- **Training Time**: ~30-45 minutes on CPU
- **Best for**: Learning CNN basics

### 2. Improved CNN (中级)
- **Architecture**: 8 convolutional layers + 3 fully connected layers
- **Parameters**: ~2M
- **Expected Accuracy**: 80-85%
- **Training Time**: ~1-2 hours on CPU
- **Best for**: Understanding deeper networks

### 3. ResNet-style CNN (高级)
- **Architecture**: Residual connections + global average pooling
- **Parameters**: ~1.5M
- **Expected Accuracy**: 85-90%
- **Training Time**: ~2-3 hours on CPU
- **Best for**: Learning advanced techniques

## 📊 CIFAR-10 Dataset / CIFAR-10数据集

The CIFAR-10 dataset contains 60,000 32x32 color images in 10 classes:

CIFAR-10数据集包含60,000张32x32彩色图像，分为10个类别：

1. **Airplane** (飞机) - 6,000 images
2. **Automobile** (汽车) - 6,000 images  
3. **Bird** (鸟) - 6,000 images
4. **Cat** (猫) - 6,000 images
5. **Deer** (鹿) - 6,000 images
6. **Dog** (狗) - 6,000 images
7. **Frog** (青蛙) - 6,000 images
8. **Horse** (马) - 6,000 images
9. **Ship** (船) - 6,000 images
10. **Truck** (卡车) - 6,000 images

- **Training**: 50,000 images / 训练：50,000张图像
- **Testing**: 10,000 images / 测试：10,000张图像

## 🎯 Understanding the Results / 理解结果

### Accuracy Metrics / 准确率指标

- **Training Accuracy**: How well the model performs on training data
- **Validation Accuracy**: How well the model performs on unseen validation data
- **Test Accuracy**: Final performance on the test set

- **训练准确率**: 模型在训练数据上的表现
- **验证准确率**: 模型在未见过的验证数据上的表现
- **测试准确率**: 在测试集上的最终性能

### What Good Results Look Like / 好结果应该是什么样的

- **Training and validation curves should be smooth** / **训练和验证曲线应该平滑**
- **Validation accuracy should be close to training accuracy** / **验证准确率应该接近训练准确率**
- **No severe overfitting (huge gap between train/val accuracy)** / **没有严重过拟合（训练/验证准确率之间的巨大差距）**

### Expected Performance / 预期性能

| Model | Accuracy Range | Training Time |
|-------|---------------|---------------|
| Simple CNN | 65-70% | 30-45 min |
| Improved CNN | 80-85% | 1-2 hours |
| ResNet CNN | 85-90% | 2-3 hours |

## 🔧 Troubleshooting / 故障排除

### Common Issues / 常见问题

1. **"No module named 'torch'"** 
   - Solution: `pip install -r requirements.txt`
   - 解决方案：`pip install -r requirements.txt`

2. **CUDA out of memory**
   - Solution: Reduce batch size (try 16 or 8)
   - 解决方案：减少批次大小（尝试16或8）

3. **Training is very slow**
   - This is normal on CPU. Consider using Google Colab for GPU access
   - 在CPU上这是正常的。考虑使用Google Colab获取GPU访问

4. **Accuracy is not improving**
   - Try training for more epochs
   - Check if learning rate is too high/low
   - 尝试训练更多epoch
   - 检查学习率是否过高/过低

### Performance Tips / 性能提示

- **Use GPU if available** / **如果可用，使用GPU**
- **Increase batch size for faster training** / **增加批次大小以加快训练**
- **Use data augmentation for better generalization** / **使用数据增强以获得更好的泛化**

## 📚 Learning Resources / 学习资源

### Concepts to Understand / 需要理解的概念

1. **Convolution Operation** / **卷积运算**
   - Think of it as a sliding window that detects patterns
   - 把它想象成检测模式的滑动窗口

2. **Pooling** / **池化**
   - Reduces image size while keeping important information
   - 在保持重要信息的同时减少图像大小

3. **Activation Functions** / **激活函数**
   - ReLU: Introduces non-linearity (like a switch)
   - ReLU：引入非线性（像开关一样）

4. **Backpropagation** / **反向传播**
   - How the network learns from mistakes
   - 网络如何从错误中学习

### Recommended Reading / 推荐阅读

- **Deep Learning Book** by Ian Goodfellow (Chapter 9: Convolutional Networks)
- **CS231n Stanford Course** (Convolutional Neural Networks)
- **PyTorch Tutorials** (official documentation)

## 🎨 Visualization and Analysis / 可视化和分析

The project automatically generates several visualizations:

项目自动生成几种可视化：

1. **Training History Plots** / **训练历史图表**
   - Loss and accuracy curves over epochs
   - 损失和准确率随epoch变化的曲线

2. **Confusion Matrix** / **混淆矩阵**
   - Shows which classes are confused with each other
   - 显示哪些类别容易相互混淆

3. **Per-Class Performance** / **每类性能**
   - Precision, recall, and F1-score for each class
   - 每类的精确率、召回率和F1分数

4. **Sample Predictions** / **样本预测**
   - Visual examples of correct and incorrect predictions
   - 正确和错误预测的可视化示例

## 🚀 Next Steps / 下一步

After completing this project, you can:

完成这个项目后，你可以：

1. **Try Different Architectures** / **尝试不同架构**
   - Implement VGG, ResNet, or DenseNet
   - 实现VGG、ResNet或DenseNet

2. **Experiment with Other Datasets** / **尝试其他数据集**
   - CIFAR-100, ImageNet, or custom datasets
   - CIFAR-100、ImageNet或自定义数据集

3. **Add Advanced Techniques** / **添加高级技术**
   - Transfer learning, data augmentation, ensemble methods
   - 迁移学习、数据增强、集成方法

4. **Deploy Your Model** / **部署你的模型**
   - Create a web app or mobile app
   - 创建网络应用或移动应用

## 🤝 Contributing / 贡献

Feel free to:
- Report bugs / 报告错误
- Suggest improvements / 建议改进
- Add new features / 添加新功能
- Share your results / 分享你的结果

## 📄 License / 许可证

This project is for educational purposes. Feel free to use and modify!

这个项目用于教育目的。请随意使用和修改！

---

## 🎉 Congratulations! / 恭喜！

You've just set up your first deep learning project! Remember:

你刚刚设置了你的第一个深度学习项目！记住：

- **Start simple** - Begin with the simple CNN model / **从简单开始** - 从简单的CNN模型开始
- **Be patient** - Training takes time, especially on CPU / **要有耐心** - 训练需要时间，特别是在CPU上
- **Experiment** - Try different parameters and see what happens / **实验** - 尝试不同参数，看看会发生什么
- **Learn from mistakes** - Low accuracy? That's part of learning! / **从错误中学习** - 准确率低？这是学习的一部分！

Happy learning! 🎓  
学习愉快！🎓 