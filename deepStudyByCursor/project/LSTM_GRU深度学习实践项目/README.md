# LSTM与GRU深度学习实践项目
# LSTM and GRU Deep Learning Practice Projects

## 项目简介 | Project Overview

本项目是基于PyTorch框架的LSTM和GRU实践教学项目，通过多个具体案例帮助初学者深入理解和掌握长短期记忆网络和门控循环单元的工作原理及应用。

This project is a PyTorch-based LSTM and GRU practice teaching project that helps beginners deeply understand and master the working principles and applications of Long Short-Term Memory networks and Gated Recurrent Units through multiple specific cases.

## 学习目标 | Learning Objectives

### 理论理解 | Theoretical Understanding
- 深入理解LSTM和GRU的网络结构和数学原理
- Deep understanding of LSTM and GRU network structures and mathematical principles
- 掌握序列建模和时间序列预测的核心概念
- Master core concepts of sequence modeling and time series prediction
- 理解梯度消失问题及其解决方案
- Understand gradient vanishing problem and its solutions

### 实践技能 | Practical Skills
- 使用PyTorch实现LSTM和GRU网络
- Implement LSTM and GRU networks using PyTorch
- 处理序列数据的预处理和特征工程
- Handle sequence data preprocessing and feature engineering
- 掌握模型训练、验证和优化技巧
- Master model training, validation and optimization techniques

## 项目结构 | Project Structure

```
LSTM_GRU深度学习实践项目/
├── README.md                           # 项目说明文档
├── requirements.txt                    # 依赖包列表
├── utils/                             # 工具函数
│   ├── __init__.py
│   ├── data_utils.py                  # 数据处理工具
│   ├── model_utils.py                 # 模型工具
│   └── visualization.py              # 可视化工具
├── 01_基础理论实现/                    # Basic Theory Implementation
│   ├── lstm_from_scratch.py           # 从零实现LSTM
│   ├── gru_from_scratch.py            # 从零实现GRU
│   └── theory_comparison.py           # 理论对比分析
├── 02_文本情感分析/                    # Text Sentiment Analysis
│   ├── data/                          # 数据文件夹
│   ├── sentiment_lstm.py              # LSTM情感分析
│   ├── sentiment_gru.py               # GRU情感分析
│   └── model_comparison.py            # 模型对比
├── 03_股票价格预测/                    # Stock Price Prediction
│   ├── data/                          # 股票数据
│   ├── stock_lstm.py                  # LSTM股票预测
│   ├── stock_gru.py                   # GRU股票预测
│   └── performance_analysis.py        # 性能分析
├── 04_文本生成/                       # Text Generation
│   ├── data/                          # 文本数据
│   ├── text_lstm.py                   # LSTM文本生成
│   ├── text_gru.py                    # GRU文本生成
│   └── creative_writing.py            # 创意写作
├── 05_序列分类/                       # Sequence Classification
│   ├── data/                          # 序列数据
│   ├── sequence_lstm.py               # LSTM序列分类
│   ├── sequence_gru.py                # GRU序列分类
│   └── classification_metrics.py      # 分类指标
└── 06_综合对比分析/                   # Comprehensive Comparison
    ├── model_comparison.py            # 模型全面对比
    ├── performance_benchmark.py       # 性能基准测试
    └── visualization_dashboard.py     # 可视化仪表板
```

## 实践案例介绍 | Practice Cases Introduction

### 1. 基础理论实现 | Basic Theory Implementation
- 从数学公式出发，手动实现LSTM和GRU的前向传播和反向传播
- Implement forward and backward propagation of LSTM and GRU manually from mathematical formulas
- 深入理解门控机制和记忆单元的工作原理
- Deeply understand the working principles of gating mechanisms and memory cells

### 2. 文本情感分析 | Text Sentiment Analysis
- 使用电影评论数据进行情感分类
- Perform sentiment classification using movie review data
- 对比LSTM和GRU在自然语言处理任务中的表现
- Compare LSTM and GRU performance in natural language processing tasks

### 3. 股票价格预测 | Stock Price Prediction
- 基于历史股价数据预测未来走势
- Predict future trends based on historical stock price data
- 学习时间序列预测的实际应用
- Learn practical applications of time series prediction

### 4. 文本生成 | Text Generation
- 训练模型生成连贯的文本内容
- Train models to generate coherent text content
- 探索序列到序列生成的奥秘
- Explore the mysteries of sequence-to-sequence generation

### 5. 序列分类 | Sequence Classification
- 对时间序列数据进行分类任务
- Perform classification tasks on time series data
- 理解序列特征提取和模式识别
- Understand sequence feature extraction and pattern recognition

### 6. 综合对比分析 | Comprehensive Comparison
- 全面对比LSTM和GRU的性能差异
- Comprehensively compare performance differences between LSTM and GRU
- 分析不同任务场景下的最佳选择
- Analyze optimal choices for different task scenarios

## 环境配置 | Environment Setup

### 系统要求 | System Requirements
- Python 3.8+
- PyTorch 1.10+
- CUDA支持（可选，用于GPU加速）
- CUDA support (optional, for GPU acceleration)

### 安装依赖 | Install Dependencies
```bash
pip install -r requirements.txt
```

## 使用方法 | Usage

### 快速开始 | Quick Start
```bash
# 进入项目目录
cd LSTM_GRU深度学习实践项目

# 运行基础理论实现
python 01_基础理论实现/lstm_from_scratch.py

# 运行情感分析案例
python 02_文本情感分析/sentiment_lstm.py
```

### 学习路径建议 | Recommended Learning Path
1. 首先阅读 `../05_LSTM_GRU/长短期记忆网络与门控循环单元.md` 理论基础
2. 运行 `01_基础理论实现/` 中的代码理解核心算法
3. 按顺序完成各个实践案例
4. 最后进行综合对比分析

## 项目特色 | Project Features

### 渐进式学习 | Progressive Learning
- 从基础理论到实际应用的完整学习路径
- Complete learning path from basic theory to practical applications
- 每个案例都有详细的代码注释和说明
- Each case has detailed code comments and explanations

### 多样化应用 | Diverse Applications
- 涵盖自然语言处理、时间序列预测等多个领域
- Covers multiple domains including NLP and time series prediction
- 真实数据集和实际问题场景
- Real datasets and practical problem scenarios

### 深入对比分析 | In-depth Comparative Analysis
- LSTM vs GRU 全面性能对比
- Comprehensive performance comparison between LSTM and GRU
- 不同超参数设置的影响分析
- Analysis of different hyperparameter settings' impact

### 可视化展示 | Visualization Display
- 丰富的图表和可视化分析
- Rich charts and visualization analysis
- 模型训练过程的实时监控
- Real-time monitoring of model training process

## 注意事项 | Notes

### 数据准备 | Data Preparation
- 某些案例需要下载额外的数据集
- Some cases require downloading additional datasets
- 具体数据获取方法请参考各个子项目的说明
- Please refer to individual sub-project instructions for specific data acquisition methods

### 计算资源 | Computing Resources
- 建议使用GPU加速训练过程
- GPU acceleration is recommended for training process
- 可根据实际硬件配置调整batch_size等参数
- Adjust parameters like batch_size according to actual hardware configuration

## 贡献与反馈 | Contribution and Feedback

欢迎提出改进建议和问题反馈！
Welcome to provide improvement suggestions and feedback!

---

**开始你的LSTM/GRU学习之旅吧！**
**Start your LSTM/GRU learning journey!** 