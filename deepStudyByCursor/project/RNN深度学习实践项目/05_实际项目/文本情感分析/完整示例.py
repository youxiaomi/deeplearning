#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete Sentiment Analysis Example with RNN
使用RNN的完整情感分析示例
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re

# Set random seed / 设置随机种子
torch.manual_seed(42)

class TextPreprocessor:
    """Text preprocessing utilities / 文本预处理工具"""
    
    def __init__(self, max_vocab_size=1000, max_length=20):
        self.max_vocab_size = max_vocab_size
        self.max_length = max_length
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.vocab_size = 0
        
    def clean_text(self, text):
        """Clean text / 清理文本"""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return ' '.join(text.split())
    
    def build_vocabulary(self, texts):
        """Build vocabulary / 构建词汇表"""
        word_counts = Counter()
        for text in texts:
            tokens = self.clean_text(text).split()
            word_counts.update(tokens)
        
        # Add most frequent words / 添加最频繁的词
        most_common = word_counts.most_common(self.max_vocab_size - 2)
        for word, _ in most_common:
            self.word_to_idx[word] = len(self.word_to_idx)
        
        self.vocab_size = len(self.word_to_idx)
        print(f"Vocabulary size: {self.vocab_size}")
    
    def text_to_sequence(self, text):
        """Convert text to sequence / 将文本转换为序列"""
        tokens = self.clean_text(text).split()
        sequence = [self.word_to_idx.get(token, self.word_to_idx['<UNK>']) 
                   for token in tokens]
        
        # Pad or truncate / 填充或截断
        if len(sequence) > self.max_length:
            return sequence[:self.max_length]
        else:
            return sequence + [0] * (self.max_length - len(sequence))

class SentimentDataset(Dataset):
    """Sentiment dataset / 情感数据集"""
    
    def __init__(self, texts, labels, preprocessor):
        self.texts = texts
        self.labels = labels
        self.preprocessor = preprocessor
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        sequence = self.preprocessor.text_to_sequence(self.texts[idx])
        return torch.tensor(sequence, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

class SentimentRNN(nn.Module):
    """RNN for sentiment analysis / 用于情感分析的RNN"""
    
    def __init__(self, vocab_size, embedding_dim=32, hidden_size=32, num_classes=2):
        super(SentimentRNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        rnn_output, _ = self.rnn(embedded)  # (batch_size, seq_len, hidden_size)
        last_output = rnn_output[:, -1, :]  # (batch_size, hidden_size)
        dropped = self.dropout(last_output)
        return self.classifier(dropped)

def create_sample_data():
    """Create sample data / 创建示例数据"""
    positive_reviews = [
        "This movie is absolutely amazing and wonderful",
        "I love this film so much fantastic",
        "Great story excellent acting highly recommended",
        "Outstanding performance brilliant cinematography",
        "Beautiful film incredible characters",
        "Masterpiece cinema truly inspiring",
        "Loved every minute perfect movie",
        "Excellent direction superb screenplay"
    ]
    
    negative_reviews = [
        "Terrible movie very boring disappointing",
        "Worst film ever seen waste time",
        "Poor acting awful storyline",
        "Boring plot terrible characters",
        "Disappointing movie bad direction",
        "Awful film couldn't wait end",
        "Poor quality uninteresting story",
        "Bad movie terrible acting"
    ]
    
    texts = positive_reviews + negative_reviews
    labels = [1] * len(positive_reviews) + [0] * len(negative_reviews)
    return texts, labels

def train_model(model, train_loader, epochs=50):
    """Train model / 训练模型"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_texts, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_texts)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}')

def evaluate_model(model, test_loader):
    """Evaluate model / 评估模型"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_texts, batch_labels in test_loader:
            outputs = model(batch_texts)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == batch_labels).sum().item()
            total += batch_labels.size(0)
    
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')
    return accuracy

def predict_sentiment(model, text, preprocessor):
    """Predict sentiment / 预测情感"""
    model.eval()
    sequence = preprocessor.text_to_sequence(text)
    input_tensor = torch.tensor([sequence], dtype=torch.long)
    
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)
    
    sentiment = "Positive" if predicted_class.item() == 1 else "Negative"
    confidence = probabilities[0][predicted_class].item()
    return sentiment, confidence

def main():
    """Main function / 主函数"""
    print("=== RNN Sentiment Analysis Demo ===")
    
    # 1. Create data / 创建数据
    texts, labels = create_sample_data()
    print(f"Total samples: {len(texts)}")
    
    # 2. Split data / 分割数据
    split_idx = int(0.75 * len(texts))
    train_texts, train_labels = texts[:split_idx], labels[:split_idx]
    test_texts, test_labels = texts[split_idx:], labels[split_idx:]
    
    # 3. Preprocess / 预处理
    preprocessor = TextPreprocessor()
    preprocessor.build_vocabulary(train_texts)
    
    # 4. Create datasets / 创建数据集
    train_dataset = SentimentDataset(train_texts, train_labels, preprocessor)
    test_dataset = SentimentDataset(test_texts, test_labels, preprocessor)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    # 5. Create and train model / 创建并训练模型
    model = SentimentRNN(vocab_size=preprocessor.vocab_size)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    train_model(model, train_loader)
    
    # 6. Evaluate / 评估
    accuracy = evaluate_model(model, test_loader)
    
    # 7. Test examples / 测试示例
    print("\nTesting custom examples:")
    test_examples = [
        "This movie is fantastic amazing",
        "Terrible film very disappointing", 
        "Great story excellent acting",
        "Boring poorly made"
    ]
    
    for text in test_examples:
        sentiment, confidence = predict_sentiment(model, text, preprocessor)
        print(f"'{text}' -> {sentiment} ({confidence:.3f})")
    
    print("\n=== Demo completed! ===")

if __name__ == "__main__":
    main() 