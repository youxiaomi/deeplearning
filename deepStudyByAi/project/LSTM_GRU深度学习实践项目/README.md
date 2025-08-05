# LSTM and GRU Deep Learning Practice Projects
# LSTM与GRU深度学习实践项目

## Project Overview | 项目简介

This project is a PyTorch-based LSTM and GRU practice teaching project that helps beginners deeply understand and master the working principles and applications of Long Short-Term Memory networks and Gated Recurrent Units through multiple specific cases.

本项目是基于PyTorch框架的LSTM和GRU实践教学项目，通过多个具体案例帮助初学者深入理解和掌握长短期记忆网络和门控循环单元的工作原理及应用。

## Learning Objectives | 学习目标

### Theoretical Understanding | 理论理解
- Deep understanding of LSTM and GRU network structures and mathematical principles
- 深入理解LSTM和GRU的网络结构和数学原理
- Master core concepts of sequence modeling and time series prediction
- 掌握序列建模和时间序列预测的核心概念
- Understand gradient vanishing problem and its solutions
- 理解梯度消失问题及其解决方案

### Practical Skills | 实践技能
- Implement LSTM and GRU networks using PyTorch
- 使用PyTorch实现LSTM和GRU网络
- Handle sequence data preprocessing and feature engineering
- 处理序列数据的预处理和特征工程
- Master model training, validation and optimization techniques
- 掌握模型训练、验证和优化技巧

## Project Structure | 项目结构

```
LSTM_GRU深度学习实践项目/
├── README.md                           # Project documentation
├── requirements.txt                    # List of dependencies
├── utils/                             # Utility functions
│   ├── __init__.py
│   ├── data_utils.py                  # Data processing utilities
│   ├── model_utils.py                 # Model utilities
│   └── visualization.py              # Visualization tools
├── 01_Basic Theory Implementation/    # 基础理论实现
│   ├── lstm_from_scratch.py           # Implement LSTM from scratch
│   ├── gru_from_scratch.py            # Implement GRU from scratch
│   └── theory_comparison.py           # Theoretical comparison and analysis
├── 02_Text Sentiment Analysis/        # 文本情感分析
│   ├── data/                          # Data folder
│   ├── sentiment_lstm.py              # LSTM sentiment analysis
│   ├── sentiment_gru.py               # GRU sentiment analysis
│   └── model_comparison.py            # Model comparison
├── 03_Stock Price Prediction/         # 股票价格预测
│   ├── data/                          # Stock data
│   ├── stock_lstm.py                  # LSTM stock prediction
│   ├── stock_gru.py                   # GRU stock prediction
│   └── performance_analysis.py        # Performance analysis
├── 04_Text Generation/                # 文本生成
│   ├── data/                          # Text data
│   ├── text_lstm.py                   # LSTM text generation
│   ├── text_gru.py                    # GRU text generation
│   └── creative_writing.py            # Creative writing
├── 05_Sequence Classification/        # 序列分类
│   ├── data/                          # Sequence data
│   ├── sequence_lstm.py               # LSTM sequence classification
│   ├── sequence_gru.py                # GRU sequence classification
│   └── classification_metrics.py      # Classification metrics
└── 06_Comprehensive Comparison/       # 综合对比分析
    ├── model_comparison.py            # Comprehensive model comparison
    ├── performance_benchmark.py       # Performance benchmark testing
    └── visualization_dashboard.py     # Visualization dashboard
```

## Practice Cases Introduction | 实践案例介绍

### 1. Basic Theory Implementation | 基础理论实现
- Implement forward and backward propagation of LSTM and GRU manually from mathematical formulas
- 从数学公式出发，手动实现LSTM和GRU的前向传播和反向传播
- Deeply understand the working principles of gating mechanisms and memory cells
- 深入理解门控机制和记忆单元的工作原理

### 2. Text Sentiment Analysis | 文本情感分析
- Perform sentiment classification using movie review data
- 使用电影评论数据进行情感分类
- Compare LSTM and GRU performance in natural language processing tasks
- 对比LSTM和GRU在自然语言处理任务中的表现

### 3. Stock Price Prediction | 股票价格预测
- Predict future trends based on historical stock price data
- 基于历史股价数据预测未来走势
- Learn practical applications of time series prediction
- 学习时间序列预测的实际应用

### 4. Text Generation | 文本生成
- Train models to generate coherent text content
- 训练模型生成连贯的文本内容
- Explore the mysteries of sequence-to-sequence generation
- 探索序列到序列生成的奥秘

### 5. Sequence Classification | 序列分类
- Perform classification tasks on time series data
- 对时间序列数据进行分类任务
- Understand sequence feature extraction and pattern recognition
- 理解序列特征提取和模式识别

### 6. Comprehensive Comparison | 综合对比分析
- Comprehensively compare performance differences between LSTM and GRU
- 全面对比LSTM和GRU的性能差异
- Analyze optimal choices for different task scenarios
- 分析不同任务场景下的最佳选择

## Environment Setup | 环境配置

### System Requirements | 系统要求
- Python 3.8+
- PyTorch 1.10+
- CUDA support (optional, for GPU acceleration)
- CUDA支持（可选，用于GPU加速）

### Install Dependencies | 安装依赖
```bash
pip install -r requirements.txt
```

## Usage | 使用方法

### Quick Start | 快速开始
```bash
# 进入项目目录
cd LSTM_GRU深度学习实践项目

# 运行基础理论实现
python 01_基础理论实现/lstm_from_scratch.py

# 运行情感分析案例
python 02_文本情感分析/sentiment_lstm.py
```

### Recommended Learning Path | 学习路径建议
1. 首先阅读 `../05_LSTM_GRU/长短期记忆网络与门控循环单元.md` 理论基础
2. 运行 `01_基础理论实现/` 中的代码理解核心算法
3. 按顺序完成各个实践案例
4. 最后进行综合对比分析

## Project Features | 项目特色

### Progressive Learning | 渐进式学习
- Complete learning path from basic theory to practical applications
- 从基础理论到实际应用的完整学习路径
- Each case has detailed code comments and explanations
- 每个案例都有详细的代码注释和说明

### Diverse Applications | 多样化应用
- Covers multiple domains including NLP and time series prediction
- 涵盖自然语言处理、时间序列预测等多个领域
- Real datasets and practical problem scenarios
- 真实数据集和实际问题场景

### In-depth Comparative Analysis | 深入对比分析
- Comprehensive performance comparison between LSTM and GRU
- LSTM vs GRU 全面性能对比
- Analysis of different hyperparameter settings' impact
- 不同超参数设置的影响分析

### Visualization Display | 可视化展示
- Rich charts and visualization analysis
- 丰富的图表和可视化分析
- Real-time monitoring of model training process
- 模型训练过程的实时监控

## Notes | 注意事项

### Data Preparation | 数据准备
- Some cases require downloading additional datasets
- 某些案例需要下载额外的数据集
- Please refer to individual sub-project instructions for specific data acquisition methods
- 具体数据获取方法请参考各个子项目的说明

### Computing Resources | 计算资源
- GPU acceleration is recommended for training process
- 建议使用GPU加速训练过程
- Adjust parameters like batch_size according to actual hardware configuration
- 可根据实际硬件配置调整batch_size等参数

## Contribution and Feedback | 贡献与反馈

Welcome to provide improvement suggestions and feedback!
欢迎提出改进建议和问题反馈！

---

**开始你的LSTM/GRU学习之旅吧！**
**Start your LSTM/GRU learning journey!**