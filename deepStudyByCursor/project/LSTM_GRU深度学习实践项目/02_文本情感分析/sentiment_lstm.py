"""
使用LSTM进行文本情感分析 - 理解自然语言处理中的序列建模
Text Sentiment Analysis using LSTM - Understanding Sequence Modeling in NLP

这个项目演示如何使用LSTM来理解文本的情感倾向，就像教计算机读懂人类的喜怒哀乐。
This project demonstrates how to use LSTM to understand text sentiment, 
like teaching computers to read human emotions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re
import jieba
from collections import Counter
import sys
import os

# 添加上级目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_utils import set_random_seed, TextProcessor


class SentimentDataset(Dataset):
    """
    情感分析数据集
    Sentiment Analysis Dataset
    
    这个类就像一个智能的文本整理员，把原始文本转换成模型能理解的数字序列。
    This class is like a smart text organizer that converts raw text into numerical sequences 
    that models can understand.
    """
    
    def __init__(self, texts, labels, text_processor, max_length=100):
        """
        初始化数据集
        Initialize dataset
        
        Args:
            texts: 文本列表 | List of texts
            labels: 标签列表 | List of labels  
            text_processor: 文本处理器 | Text processor
            max_length: 最大序列长度 | Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.text_processor = text_processor
        self.max_length = max_length
        
        # 将文本转换为序列
        # Convert texts to sequences
        self.sequences = []
        for text in texts:
            sequence = self.text_processor.text_to_sequence(text)
            self.sequences.append(sequence)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # 填充或截断序列到固定长度
        # Pad or truncate sequence to fixed length
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]
        else:
            sequence = sequence + [0] * (self.max_length - len(sequence))
        
        return torch.LongTensor(sequence), torch.LongTensor([label])


class SentimentLSTM(nn.Module):
    """
    基于LSTM的情感分析模型
    LSTM-based Sentiment Analysis Model
    
    这个模型就像一个理解文字情感的专家，通过学习单词的顺序和组合来判断情感。
    This model is like an expert in understanding text emotions, 
    learning from word sequences and combinations to judge sentiment.
    """
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, 
                 n_layers=2, dropout=0.3, bidirectional=True):
        """
        初始化LSTM模型
        Initialize LSTM model
        
        Args:
            vocab_size: 词汇表大小 | Vocabulary size
            embed_dim: 词嵌入维度 | Word embedding dimension
            hidden_dim: 隐藏层维度 | Hidden layer dimension
            output_dim: 输出维度 | Output dimension
            n_layers: LSTM层数 | Number of LSTM layers
            dropout: Dropout比率 | Dropout ratio
            bidirectional: 是否双向 | Whether bidirectional
        """
        super(SentimentLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        
        # 词嵌入层 - 将单词ID转换为向量表示
        # Word embedding layer - convert word IDs to vector representations
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # LSTM层 - 核心的序列处理层
        # LSTM layer - core sequence processing layer
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers, 
                           dropout=dropout, batch_first=True, 
                           bidirectional=bidirectional)
        
        # 计算最终的隐藏维度
        # Calculate final hidden dimension
        final_hidden_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # 分类器 - 将LSTM输出转换为情感分类
        # Classifier - convert LSTM output to sentiment classification
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(final_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 初始化权重
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x):
        """
        前向传播
        Forward propagation
        
        Args:
            x: 输入序列 [batch_size, seq_length] | Input sequence
            
        Returns:
            情感分类输出 | Sentiment classification output
        """
        # 词嵌入
        # Word embedding
        embedded = self.embedding(x)  # [batch_size, seq_length, embed_dim]
        
        # LSTM处理
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # 使用最后一个时间步的输出（对于双向LSTM，连接两个方向）
        # Use output from last time step (for bidirectional LSTM, concatenate both directions)
        if self.bidirectional:
            # 双向LSTM的最后隐藏状态
            # Final hidden state of bidirectional LSTM
            final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            final_hidden = hidden[-1]
        
        # 分类
        # Classification
        output = self.classifier(final_hidden)
        
        return output


def create_sample_data():
    """
    创建示例情感分析数据
    Create sample sentiment analysis data
    
    在实际项目中，你可以使用真实的数据集如IMDb电影评论数据集。
    In real projects, you can use real datasets like IMDb movie review dataset.
    """
    print("📝 创建示例情感分析数据 | Creating Sample Sentiment Analysis Data")
    
    # 正面情感的文本示例
    # Positive sentiment text examples
    positive_texts = [
        "这部电影太棒了，演员表演得很好",
        "我非常喜欢这个产品，质量很不错",
        "今天心情很好，天气也很棒",
        "这家餐厅的食物很美味，服务也很周到",
        "这本书写得很精彩，情节引人入胜",
        "这个软件用起来很方便，功能很实用",
        "演唱会太精彩了，歌手唱得很棒",
        "这个游戏很有趣，画面也很精美",
        "老师讲课很生动，学到了很多知识",
        "这次旅行很愉快，风景很美丽"
    ]
    
    # 负面情感的文本示例
    # Negative sentiment text examples
    negative_texts = [
        "这部电影很无聊，剧情拖沓",
        "产品质量很差，不推荐购买",
        "今天心情很糟糕，什么都不顺利",
        "这家餐厅的服务很差，食物也不好吃",
        "这本书写得很乏味，看不下去",
        "这个软件bugs很多，体验很差",
        "演唱会很失望，音响效果很差",
        "这个游戏很无聊，操作也很复杂",
        "课程内容很枯燥，听不懂",
        "这次旅行很糟糕，酒店条件很差"
    ]
    
    # 扩展数据 - 生成更多样本
    # Expand data - generate more samples
    extended_positive = positive_texts * 10  # 重复10次
    extended_negative = negative_texts * 10
    
    # 组合文本和标签
    # Combine texts and labels
    texts = extended_positive + extended_negative
    labels = [1] * len(extended_positive) + [0] * len(extended_negative)  # 1=正面, 0=负面
    
    print(f"总样本数: {len(texts)} (正面: {len(extended_positive)}, 负面: {len(extended_negative)})")
    print(f"Total samples: {len(texts)} (Positive: {len(extended_positive)}, Negative: {len(extended_negative)})")
    
    return texts, labels


def train_sentiment_model():
    """
    训练情感分析模型
    Train sentiment analysis model
    """
    print("\n🚀 开始训练LSTM情感分析模型 | Starting LSTM Sentiment Analysis Model Training")
    print("=" * 70)
    
    set_random_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device} | Using device: {device}")
    
    # 创建数据
    # Create data
    texts, labels = create_sample_data()
    
    # 划分训练集和测试集
    # Split train and test sets
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"训练集大小: {len(train_texts)} | Training set size: {len(train_texts)}")
    print(f"测试集大小: {len(test_texts)} | Test set size: {len(test_texts)}")
    
    # 构建词汇表
    # Build vocabulary
    text_processor = TextProcessor(max_vocab_size=5000, min_freq=1)
    text_processor.build_vocab(train_texts, language='chinese')
    
    print(f"词汇表大小: {text_processor.vocab_size} | Vocabulary size: {text_processor.vocab_size}")
    
    # 创建数据集和数据加载器
    # Create datasets and data loaders
    train_dataset = SentimentDataset(train_texts, train_labels, text_processor, max_length=50)
    test_dataset = SentimentDataset(test_texts, test_labels, text_processor, max_length=50)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # 创建模型
    # Create model
    model = SentimentLSTM(
        vocab_size=text_processor.vocab_size,
        embed_dim=128,
        hidden_dim=64,
        output_dim=2,  # 二分类：正面和负面 | Binary classification: positive and negative
        n_layers=2,
        dropout=0.3,
        bidirectional=True
    ).to(device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数和优化器
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    
    # 训练循环
    # Training loop
    num_epochs = 30
    train_losses = []
    train_accuracies = []
    
    print("\n开始训练... | Starting training...")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.squeeze().to(device)
            
            # 前向传播
            # Forward propagation
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # 反向传播
            # Backward propagation
            loss.backward()
            
            # 梯度裁剪
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 统计
            # Statistics
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct_predictions += (pred == target).sum().item()
            total_predictions += target.size(0)
        
        # 计算训练指标
        # Calculate training metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions
        
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)
        
        # 学习率调度
        # Learning rate scheduling
        scheduler.step()
        
        if (epoch + 1) % 5 == 0:
            print(f'轮次 {epoch+1}/{num_epochs}:')
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'  训练损失: {avg_loss:.4f} | Training Loss: {avg_loss:.4f}')
            print(f'  训练准确率: {accuracy:.4f} | Training Accuracy: {accuracy:.4f}')
            print(f'  学习率: {scheduler.get_last_lr()[0]:.6f} | Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
    
    # 测试模型
    # Test model
    model.eval()
    test_predictions = []
    test_targets = []
    test_loss = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.squeeze().to(device)
            output = model(data)
            
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            
            test_predictions.extend(pred.cpu().numpy())
            test_targets.extend(target.cpu().numpy())
    
    # 计算测试指标
    # Calculate test metrics
    test_accuracy = accuracy_score(test_targets, test_predictions)
    avg_test_loss = test_loss / len(test_loader)
    
    print(f"\n📊 最终测试结果 | Final Test Results:")
    print(f"测试损失: {avg_test_loss:.4f} | Test Loss: {avg_test_loss:.4f}")
    print(f"测试准确率: {test_accuracy:.4f} | Test Accuracy: {test_accuracy:.4f}")
    
    # 详细分类报告
    # Detailed classification report
    print(f"\n📈 详细分类报告 | Detailed Classification Report:")
    print(classification_report(test_targets, test_predictions, 
                              target_names=['负面 | Negative', '正面 | Positive']))
    
    # 可视化结果
    # Visualize results
    visualize_results(train_losses, train_accuracies, test_targets, test_predictions)
    
    return model, text_processor


def visualize_results(train_losses, train_accuracies, test_targets, test_predictions):
    """
    可视化训练结果
    Visualize training results
    """
    plt.figure(figsize=(15, 10))
    
    # 训练损失曲线
    # Training loss curve
    plt.subplot(2, 3, 1)
    plt.plot(train_losses)
    plt.title('训练损失曲线 | Training Loss Curve')
    plt.xlabel('轮次 | Epoch')
    plt.ylabel('损失 | Loss')
    plt.grid(True)
    
    # 训练准确率曲线
    # Training accuracy curve
    plt.subplot(2, 3, 2)
    plt.plot(train_accuracies)
    plt.title('训练准确率曲线 | Training Accuracy Curve')
    plt.xlabel('轮次 | Epoch')
    plt.ylabel('准确率 | Accuracy')
    plt.grid(True)
    
    # 混淆矩阵
    # Confusion matrix
    plt.subplot(2, 3, 3)
    cm = confusion_matrix(test_targets, test_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['负面 | Negative', '正面 | Positive'],
                yticklabels=['负面 | Negative', '正面 | Positive'])
    plt.title('混淆矩阵 | Confusion Matrix')
    plt.ylabel('真实标签 | True Label')
    plt.xlabel('预测标签 | Predicted Label')
    
    # 预测分布
    # Prediction distribution
    plt.subplot(2, 3, 4)
    labels = ['负面 | Negative', '正面 | Positive']
    true_counts = [test_targets.count(0), test_targets.count(1)]
    pred_counts = [test_predictions.count(0), test_predictions.count(1)]
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.bar(x - width/2, true_counts, width, label='真实 | True', alpha=0.7)
    plt.bar(x + width/2, pred_counts, width, label='预测 | Predicted', alpha=0.7)
    
    plt.xlabel('类别 | Category')
    plt.ylabel('数量 | Count')
    plt.title('预测分布对比 | Prediction Distribution Comparison')
    plt.xticks(x, labels)
    plt.legend()
    
    # 样本预测置信度分布
    # Sample prediction confidence distribution
    plt.subplot(2, 3, 5)
    correct_mask = np.array(test_targets) == np.array(test_predictions)
    plt.hist([np.array(test_predictions)[correct_mask], 
              np.array(test_predictions)[~correct_mask]], 
             bins=2, alpha=0.7, label=['正确 | Correct', '错误 | Incorrect'])
    plt.xlabel('预测类别 | Predicted Category')
    plt.ylabel('样本数 | Sample Count')
    plt.title('预测正确性分布 | Prediction Correctness Distribution')
    plt.legend()
    
    # 模型性能总结
    # Model performance summary
    plt.subplot(2, 3, 6)
    metrics = ['准确率 | Accuracy', '精确率 | Precision', '召回率 | Recall', 'F1分数 | F1-Score']
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    values = [
        accuracy_score(test_targets, test_predictions),
        precision_score(test_targets, test_predictions, average='weighted'),
        recall_score(test_targets, test_predictions, average='weighted'),
        f1_score(test_targets, test_predictions, average='weighted')
    ]
    
    bars = plt.bar(metrics, values, color=['skyblue', 'lightgreen', 'lightcoral', 'lightyellow'])
    plt.title('模型性能指标 | Model Performance Metrics')
    plt.ylabel('分数 | Score')
    plt.ylim(0, 1)
    
    # 在柱子上添加数值
    # Add values on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('sentiment_lstm_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def test_model_predictions(model, text_processor, device):
    """
    测试模型的实际预测效果
    Test model's actual prediction performance
    """
    print("\n🔮 测试模型预测效果 | Testing Model Prediction Performance")
    print("=" * 60)
    
    # 测试样本
    # Test samples
    test_samples = [
        "这部电影真的很棒，我强烈推荐",
        "产品质量太差了，完全不值得购买",
        "今天天气很好，心情也很愉快",
        "服务态度很差，让人很不满意",
        "这本书写得很精彩，内容很有趣",
        "软件经常崩溃，体验很糟糕"
    ]
    
    model.eval()
    
    with torch.no_grad():
        for text in test_samples:
            # 文本预处理
            # Text preprocessing
            sequence = text_processor.text_to_sequence(text)
            
            # 填充到固定长度
            # Pad to fixed length
            max_length = 50
            if len(sequence) > max_length:
                sequence = sequence[:max_length]
            else:
                sequence = sequence + [0] * (max_length - len(sequence))
            
            # 转换为张量
            # Convert to tensor
            input_tensor = torch.LongTensor(sequence).unsqueeze(0).to(device)
            
            # 预测
            # Predict
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = output.argmax(dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            
            sentiment = "正面 | Positive" if predicted_class == 1 else "负面 | Negative"
            
            print(f"文本: {text}")
            print(f"Text: {text}")
            print(f"预测情感: {sentiment}")
            print(f"Predicted sentiment: {sentiment}")
            print(f"置信度: {confidence:.4f}")
            print(f"Confidence: {confidence:.4f}")
            print("-" * 40)


if __name__ == "__main__":
    print("🎭 LSTM文本情感分析项目 | LSTM Text Sentiment Analysis Project")
    print("=" * 70)
    
    # 训练模型
    # Train model
    model, text_processor = train_sentiment_model()
    
    # 测试预测
    # Test predictions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_model_predictions(model, text_processor, device)
    
    print("\n🎉 情感分析项目完成！| Sentiment Analysis Project Completed!")
    print("📚 通过这个项目，你学会了：")
    print("📚 Through this project, you learned:")
    print("1. 如何将文本转换为LSTM可以处理的序列")
    print("   How to convert text into sequences that LSTM can process")
    print("2. 双向LSTM在文本分类中的应用")
    print("   Application of bidirectional LSTM in text classification")
    print("3. 词嵌入和序列建模的结合")
    print("   Combination of word embeddings and sequence modeling")
    print("4. 情感分析的完整流程和评估方法")
    print("   Complete process and evaluation methods for sentiment analysis") 