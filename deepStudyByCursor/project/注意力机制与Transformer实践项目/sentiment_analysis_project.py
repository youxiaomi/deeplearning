"""
基于Transformer的情感分析项目
Sentiment Analysis Project based on Transformer

这个项目演示如何使用Transformer进行中文情感分析
This project demonstrates how to use Transformer for Chinese sentiment analysis
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import jieba
import pickle
import os
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from attention_mechanism import TransformerEncoder, create_padding_mask

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class SentimentDataset(Dataset):
    """
    情感分析数据集
    Sentiment Analysis Dataset
    """
    
    def __init__(self, texts, labels, vocab, max_len=128):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 分词
        # Tokenization
        tokens = list(jieba.cut(text))
        
        # 转换为ID
        # Convert to IDs
        token_ids = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        
        # 截断或填充
        # Truncate or pad
        if len(token_ids) > self.max_len:
            token_ids = token_ids[:self.max_len]
        else:
            token_ids = token_ids + [self.vocab['<PAD>']] * (self.max_len - len(token_ids))
        
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


class Vocabulary:
    """
    词汇表类
    Vocabulary class
    """
    
    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.token2id = {}
        self.id2token = {}
        self.counter = Counter()
        
    def build_vocab(self, texts):
        """
        构建词汇表
        Build vocabulary
        """
        # 统计词频
        # Count word frequency
        for text in texts:
            tokens = list(jieba.cut(text))
            self.counter.update(tokens)
        
        # 添加特殊词汇
        # Add special tokens
        self.token2id = {
            '<PAD>': 0,
            '<UNK>': 1,
        }
        
        # 添加高频词汇
        # Add high frequency tokens
        idx = 2
        for token, freq in self.counter.most_common():
            if freq >= self.min_freq:
                self.token2id[token] = idx
                idx += 1
        
        # 构建反向映射
        # Build reverse mapping
        self.id2token = {v: k for k, v in self.token2id.items()}
        
    def __len__(self):
        return len(self.token2id)
    
    def get_vocab_dict(self):
        return self.token2id


class SentimentTransformer(nn.Module):
    """
    基于Transformer的情感分析模型
    Transformer-based Sentiment Analysis Model
    """
    
    def __init__(self, vocab_size, d_model=256, num_heads=8, num_layers=4, 
                 d_ff=1024, max_len=128, num_classes=2, dropout=0.1):
        super(SentimentTransformer, self).__init__()
        
        # Transformer编码器
        # Transformer encoder
        self.transformer = TransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout
        )
        
        # 分类层
        # Classification layer
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, x, mask=None):
        """
        前向传播
        Forward pass
        
        Args:
            x: 输入token序列 [batch_size, seq_len]
            mask: 注意力掩码
        
        Returns:
            logits: 分类得分 [batch_size, num_classes]
        """
        # Transformer编码
        # Transformer encoding
        encoded = self.transformer(x, mask)  # [batch_size, seq_len, d_model]
        
        # 使用平均池化获取句子表示
        # Use average pooling to get sentence representation
        if mask is not None:
            # 计算有效长度
            # Calculate valid lengths
            lengths = mask.sum(dim=-1, keepdim=True).float()  # [batch_size, 1, 1]
            pooled = (encoded * mask.unsqueeze(-1).float()).sum(dim=1) / lengths.squeeze(-1)
        else:
            pooled = encoded.mean(dim=1)  # [batch_size, d_model]
        
        # 分类
        # Classification
        logits = self.classifier(pooled)  # [batch_size, num_classes]
        
        return logits


def create_sample_data():
    """
    创建示例数据
    Create sample data
    """
    # 正面评论示例
    # Positive review examples
    positive_texts = [
        "这个产品真的很好用，强烈推荐！",
        "服务态度很好，物流也很快",
        "质量不错，价格合理，很满意",
        "包装精美，商品完好无损",
        "客服回复及时，解决问题很专业",
        "这家店铺信誉很好，值得信赖",
        "产品功能齐全，使用简单方便",
        "发货速度很快，收到货品很惊喜",
        "性价比很高，下次还会再买",
        "朋友推荐的，确实不错"
    ]
    
    # 负面评论示例  
    # Negative review examples
    negative_texts = [
        "产品质量太差了，不推荐购买",
        "客服态度恶劣，解决问题不积极",
        "物流太慢了，等了很久才收到",
        "商品与描述不符，很失望",
        "包装破损，商品有问题",
        "价格偏高，性价比不好",
        "使用体验很差，功能不完善",
        "售后服务不好，问题得不到解决",
        "商品有瑕疵，不值这个价格",
        "店铺信誉有问题，不建议购买"
    ]
    
    # 合并数据
    # Combine data
    texts = positive_texts + negative_texts
    labels = [1] * len(positive_texts) + [0] * len(negative_texts)  # 1为正面，0为负面
    
    return texts, labels


def train_model(model, train_loader, val_loader, num_epochs=10, lr=1e-4, device='cpu'):
    """
    训练模型
    Train model
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_accuracies = []
    
    model.to(device)
    
    for epoch in range(num_epochs):
        # 训练阶段
        # Training phase
        model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # 创建掩码
            # Create mask
            mask = create_padding_mask(data, pad_token=0)
            
            optimizer.zero_grad()
            output = model(data, mask)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # 验证阶段
        # Validation phase
        model.eval()
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                mask = create_padding_mask(data, pad_token=0)
                
                output = model(data, mask)
                pred = output.argmax(dim=1)
                
                val_predictions.extend(pred.cpu().numpy())
                val_targets.extend(target.cpu().numpy())
        
        val_acc = accuracy_score(val_targets, val_predictions)
        val_accuracies.append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    return train_losses, val_accuracies


def plot_training_history(train_losses, val_accuracies):
    """
    绘制训练历史
    Plot training history
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 损失曲线
    # Loss curve
    ax1.plot(train_losses)
    ax1.set_title('训练损失 (Training Loss)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # 准确率曲线
    # Accuracy curve
    ax2.plot(val_accuracies)
    ax2.set_title('验证准确率 (Validation Accuracy)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


def predict_sentiment(model, text, vocab, device='cpu', max_len=128):
    """
    预测单个文本的情感
    Predict sentiment for a single text
    """
    model.eval()
    
    # 分词和编码
    # Tokenization and encoding
    tokens = list(jieba.cut(text))
    token_ids = [vocab.get(token, vocab['<UNK>']) for token in tokens]
    
    # 填充或截断
    # Pad or truncate
    if len(token_ids) > max_len:
        token_ids = token_ids[:max_len]
    else:
        token_ids = token_ids + [vocab['<PAD>']] * (max_len - len(token_ids))
    
    # 转换为张量
    # Convert to tensor
    x = torch.tensor([token_ids], dtype=torch.long).to(device)
    mask = create_padding_mask(x, pad_token=0)
    
    # 预测
    # Predict
    with torch.no_grad():
        output = model(x, mask)
        prob = F.softmax(output, dim=1)
        pred = output.argmax(dim=1).item()
        confidence = prob.max().item()
    
    sentiment = "正面 (Positive)" if pred == 1 else "负面 (Negative)"
    return sentiment, confidence


def main():
    """
    主函数
    Main function
    """
    print("=== 基于Transformer的情感分析项目 ===")
    print("=== Transformer-based Sentiment Analysis Project ===\n")
    
    # 设置设备
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备 (Using device): {device}\n")
    
    # 1. 准备数据
    # Prepare data
    print("1. 准备数据 (Preparing data)...")
    texts, labels = create_sample_data()
    
    # 构建词汇表
    # Build vocabulary
    vocab_builder = Vocabulary(min_freq=1)
    vocab_builder.build_vocab(texts)
    vocab = vocab_builder.get_vocab_dict()
    vocab_size = len(vocab)
    
    print(f"词汇表大小 (Vocabulary size): {vocab_size}")
    print(f"数据样本数量 (Number of samples): {len(texts)}\n")
    
    # 2. 创建数据集和数据加载器
    # Create dataset and data loader
    print("2. 创建数据集 (Creating datasets)...")
    
    # 简单划分训练和验证集
    # Simple train-validation split
    train_texts, val_texts = texts[:16], texts[16:]
    train_labels, val_labels = labels[:16], labels[16:]
    
    train_dataset = SentimentDataset(train_texts, train_labels, vocab)
    val_dataset = SentimentDataset(val_texts, val_labels, vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    print(f"训练集大小 (Training set size): {len(train_dataset)}")
    print(f"验证集大小 (Validation set size): {len(val_dataset)}\n")
    
    # 3. 创建模型
    # Create model
    print("3. 创建模型 (Creating model)...")
    model = SentimentTransformer(
        vocab_size=vocab_size,
        d_model=128,  # 减小模型以适应小数据集
        num_heads=4,
        num_layers=2,
        d_ff=256,
        max_len=128,
        num_classes=2,
        dropout=0.1
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数数量 (Model parameters): {total_params:,}\n")
    
    # 4. 训练模型
    # Train model
    print("4. 训练模型 (Training model)...")
    train_losses, val_accuracies = train_model(
        model, train_loader, val_loader, 
        num_epochs=20, lr=1e-3, device=device
    )
    
    # 5. 绘制训练历史
    # Plot training history
    print("\n5. 可视化训练结果 (Visualizing training results)...")
    try:
        plot_training_history(train_losses, val_accuracies)
    except Exception as e:
        print(f"绘图错误: {e}")
    
    # 6. 测试模型
    # Test model
    print("\n6. 测试模型 (Testing model)...")
    
    test_texts = [
        "这个手机拍照效果很棒，电池续航也不错",
        "送货太慢了，而且包装也很差",
        "客服态度很好，及时解决了我的问题",
        "产品质量有问题，用了一天就坏了"
    ]
    
    for text in test_texts:
        sentiment, confidence = predict_sentiment(model, text, vocab, device)
        print(f"文本: {text}")
        print(f"预测: {sentiment} (置信度: {confidence:.3f})\n")
    
    print("=== 项目演示完成 (Project demonstration complete) ===")


if __name__ == "__main__":
    main()
            ' 