"""
基于Transformer的文本生成项目
Text Generation Project based on Transformer

这个项目演示如何使用Transformer生成中文文本
This project demonstrates how to use Transformer for Chinese text generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import jieba
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import random
from attention_mechanism import (
    MultiHeadAttention, PositionalEncoding, FeedForward, 
    ScaledDotProductAttention, create_padding_mask
)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class TextDataset(Dataset):
    """
    文本生成数据集
    Text generation dataset
    """
    
    def __init__(self, texts, vocab, seq_len=64):
        self.vocab = vocab
        self.seq_len = seq_len
        self.data = []
        
        # 处理所有文本
        # Process all texts
        for text in texts:
            tokens = list(jieba.cut(text))
            token_ids = [vocab.get(token, vocab['<UNK>']) for token in tokens]
            
            # 创建滑动窗口序列
            # Create sliding window sequences
            for i in range(len(token_ids) - seq_len):
                self.data.append(token_ids[i:i + seq_len + 1])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence = self.data[idx]
        # 输入是前seq_len个token，目标是后seq_len个token
        # Input is first seq_len tokens, target is last seq_len tokens
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        target_ids = torch.tensor(sequence[1:], dtype=torch.long)
        return input_ids, target_ids


class TransformerDecoderLayer(nn.Module):
    """
    Transformer解码器层
    Transformer Decoder Layer
    
    包含掩码自注意力、前馈网络和残差连接
    Contains masked self-attention, feed forward and residual connections
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        解码器层前向传播
        Decoder layer forward pass
        """
        # 1. 掩码自注意力 + 残差连接 + 层归一化
        # Masked self-attention + residual connection + layer normalization
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 2. 前馈网络 + 残差连接 + 层归一化
        # Feed forward + residual connection + layer normalization
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class GPTModel(nn.Module):
    """
    简化的GPT模型（仅解码器的Transformer）
    Simplified GPT model (decoder-only Transformer)
    
    用于自回归文本生成
    For autoregressive text generation
    """
    
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6, 
                 d_ff=2048, max_len=1024, dropout=0.1):
        super(GPTModel, self).__init__()
        
        self.d_model = d_model
        self.max_len = max_len
        
        # 词嵌入和位置编码
        # Word embedding and positional encoding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        
        # Transformer解码器层
        # Transformer decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 输出投影层
        # Output projection layer
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def create_causal_mask(self, seq_len):
        """
        创建因果掩码（下三角矩阵）
        Create causal mask (lower triangular matrix)
        
        确保每个位置只能看到之前的位置
        Ensure each position can only see previous positions
        """
        mask = torch.tril(torch.ones(seq_len, seq_len))
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        return mask
    
    def forward(self, x):
        """
        前向传播
        Forward pass
        
        Args:
            x: 输入token序列 [batch_size, seq_len]
        
        Returns:
            logits: 词汇表上的概率分布 [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = x.shape
        
        # 1. 词嵌入 + 位置编码
        # Word embedding + positional encoding
        x = self.embedding(x) * np.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # 2. 创建因果掩码
        # Create causal mask
        causal_mask = self.create_causal_mask(seq_len).to(x.device)
        
        # 3. 通过所有解码器层
        # Pass through all decoder layers
        for layer in self.layers:
            x = layer(x, causal_mask)
        
        # 4. 输出投影
        # Output projection
        logits = self.output_projection(x)
        
        return logits
    
    def generate(self, start_tokens, max_length=100, temperature=1.0, top_k=50):
        """
        生成文本
        Generate text
        
        Args:
            start_tokens: 开始的token序列 [seq_len]
            max_length: 最大生成长度
            temperature: 温度参数，控制随机性
            top_k: top-k采样
        
        Returns:
            generated_tokens: 生成的token序列
        """
        self.eval()
        
        with torch.no_grad():
            tokens = start_tokens.clone()
            
            for _ in range(max_length):
                # 限制输入长度
                # Limit input length
                if tokens.size(0) > self.max_len:
                    input_tokens = tokens[-self.max_len:]
                else:
                    input_tokens = tokens
                
                # 前向传播
                # Forward pass
                logits = self(input_tokens.unsqueeze(0))  # [1, seq_len, vocab_size]
                next_token_logits = logits[0, -1, :]  # [vocab_size]
                
                # 应用温度
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Top-k采样
                # Top-k sampling
                if top_k > 0:
                    values, indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, -float('inf'))
                    next_token_logits[indices] = values
                
                # 采样下一个token
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # 添加到序列
                # Add to sequence
                tokens = torch.cat([tokens, next_token])
                
                # 检查是否生成结束符（如果有的话）
                # Check for end token (if applicable)
                # 这里可以添加结束条件
                
        return tokens


def create_sample_texts():
    """
    创建示例文本数据
    Create sample text data
    """
    texts = [
        "春天来了，花儿开放了，鸟儿在树上歌唱。",
        "夏天的阳光很温暖，孩子们在海边游泳。",
        "秋天的叶子变黄了，农民在田里收获粮食。",
        "冬天下雪了，大地被白雪覆盖着。",
        "今天天气很好，我们去公园散步吧。",
        "学习深度学习需要耐心和坚持不懈的努力。",
        "人工智能技术正在改变我们的生活方式。",
        "读书是一个很好的习惯，可以增长知识。",
        "运动对身体健康很有好处，每天都应该锻炼。",
        "音乐能够带给人们快乐和放松的感觉。",
        "朋友之间的友谊是珍贵的，需要用心维护。",
        "家人的关爱是最温暖的，让人感到幸福。",
        "工作虽然辛苦，但是能够实现自己的价值。",
        "旅行可以开阔视野，了解不同的文化。",
        "美食能够满足味蕾，带来愉悦的体验。",
        "科技发展日新月异，改变着人类的生活。",
        "教育是社会进步的基石，培养未来的人才。",
        "环境保护是每个人的责任，需要共同努力。",
        "健康的生活方式包括合理饮食和适量运动。",
        "创新思维是解决问题的关键，需要不断培养。"
    ]
    
    return texts


def build_vocabulary(texts, min_freq=1):
    """
    构建词汇表
    Build vocabulary
    """
    counter = Counter()
    
    # 统计词频
    # Count word frequency
    for text in texts:
        tokens = list(jieba.cut(text))
        counter.update(tokens)
    
    # 构建词汇映射
    # Build vocabulary mapping
    vocab = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
    idx = 4
    
    for token, freq in counter.most_common():
        if freq >= min_freq:
            vocab[token] = idx
            idx += 1
    
    # 构建反向映射
    # Build reverse mapping
    id2token = {v: k for k, v in vocab.items()}
    
    return vocab, id2token


def train_model(model, dataloader, num_epochs=50, lr=1e-4, device='cpu'):
    """
    训练模型
    Train model
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.to(device)
    losses = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            # Forward pass
            logits = model(data)  # [batch_size, seq_len, vocab_size]
            
            # 计算损失
            # Compute loss
            loss = criterion(logits.view(-1, logits.size(-1)), target.view(-1))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')
    
    return losses


def generate_text(model, start_text, vocab, id2token, max_length=50, 
                 temperature=1.0, top_k=20, device='cpu'):
    """
    生成文本
    Generate text
    """
    model.eval()
    
    # 分词并转换为ID
    # Tokenize and convert to IDs
    tokens = list(jieba.cut(start_text))
    token_ids = [vocab.get(token, vocab['<UNK>']) for token in tokens]
    start_tokens = torch.tensor(token_ids, dtype=torch.long).to(device)
    
    # 生成
    # Generate
    generated_tokens = model.generate(
        start_tokens, max_length=max_length, 
        temperature=temperature, top_k=top_k
    )
    
    # 转换回文本
    # Convert back to text
    generated_text = ''.join([id2token[token_id.item()] for token_id in generated_tokens])
    
    return generated_text


def plot_loss_curve(losses):
    """
    绘制损失曲线
    Plot loss curve
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('训练损失曲线 (Training Loss Curve)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()


def demo_text_generation():
    """
    文本生成演示
    Text generation demonstration
    """
    print("=== 基于Transformer的文本生成项目 ===")
    print("=== Transformer-based Text Generation Project ===\n")
    
    # 设置设备
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备 (Using device): {device}\n")
    
    # 1. 准备数据
    # Prepare data
    print("1. 准备数据 (Preparing data)...")
    texts = create_sample_texts()
    vocab, id2token = build_vocabulary(texts, min_freq=1)
    vocab_size = len(vocab)
    
    print(f"文本数量 (Number of texts): {len(texts)}")
    print(f"词汇表大小 (Vocabulary size): {vocab_size}")
    print(f"示例词汇 (Sample vocabulary): {list(vocab.keys())[:10]}\n")
    
    # 2. 创建数据集
    # Create dataset
    print("2. 创建数据集 (Creating dataset)...")
    seq_len = 32  # 减小序列长度以适应小数据集
    dataset = TextDataset(texts, vocab, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    print(f"数据集大小 (Dataset size): {len(dataset)}")
    print(f"序列长度 (Sequence length): {seq_len}\n")
    
    # 3. 创建模型
    # Create model
    print("3. 创建模型 (Creating model)...")
    model = GPTModel(
        vocab_size=vocab_size,
        d_model=128,  # 减小模型以适应小数据集
        num_heads=4,
        num_layers=3,
        d_ff=256,
        max_len=128,
        dropout=0.1
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数数量 (Model parameters): {total_params:,}\n")
    
    # 4. 训练模型
    # Train model
    print("4. 训练模型 (Training model)...")
    losses = train_model(model, dataloader, num_epochs=100, lr=1e-3, device=device)
    
    # 5. 绘制训练曲线
    # Plot training curve
    print("\n5. 可视化训练结果 (Visualizing training results)...")
    try:
        plot_loss_curve(losses)
    except Exception as e:
        print(f"绘图错误: {e}")
    
    # 6. 生成文本
    # Generate text
    print("\n6. 生成文本 (Generating text)...")
    
    start_texts = [
        "春天",
        "学习",
        "人工智能",
        "今天天气"
    ]
    
    for start_text in start_texts:
        try:
            generated_text = generate_text(
                model, start_text, vocab, id2token, 
                max_length=30, temperature=0.8, top_k=10, device=device
            )
            print(f"起始文本: {start_text}")
            print(f"生成文本: {generated_text}\n")
        except Exception as e:
            print(f"生成错误: {e}")
    
    print("=== 项目演示完成 (Project demonstration complete) ===")


def interactive_generation(model, vocab, id2token, device='cpu'):
    """
    交互式文本生成
    Interactive text generation
    """
    print("\n=== 交互式文本生成 (Interactive Text Generation) ===")
    print("输入起始文本，模型将继续生成。输入'quit'退出。")
    print("Enter starting text, model will continue generating. Type 'quit' to exit.")
    
    while True:
        start_text = input("\n请输入起始文本 (Enter starting text): ")
        
        if start_text.lower() == 'quit':
            break
        
        try:
            generated_text = generate_text(
                model, start_text, vocab, id2token,
                max_length=50, temperature=0.8, top_k=15, device=device
            )
            print(f"生成的文本 (Generated text): {generated_text}")
        except Exception as e:
            print(f"生成错误 (Generation error): {e}")


if __name__ == "__main__":
    demo_text_generation()
    
    # 如果想要交互式生成，可以取消下面的注释
    # Uncomment below for interactive generation
    # interactive_generation(model, vocab, id2token, device) 