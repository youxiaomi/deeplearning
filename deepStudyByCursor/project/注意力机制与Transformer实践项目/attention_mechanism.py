"""
注意力机制与Transformer的PyTorch实现
Attention Mechanism and Transformer Implementation in PyTorch

这个文件包含了从基础注意力到完整Transformer的实现
This file contains implementations from basic attention to complete Transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class BasicAttention(nn.Module):
    """
    基础注意力机制实现
    Basic Attention Mechanism Implementation
    
    这是最简单的注意力机制，帮助理解核心概念
    This is the simplest attention mechanism to understand core concepts
    """
    
    def __init__(self, input_dim: int):
        super(BasicAttention, self).__init__()
        self.input_dim = input_dim
        
        # 查询、键、值的线性变换层
        # Linear transformation layers for query, key, value
        self.query_layer = nn.Linear(input_dim, input_dim)
        self.key_layer = nn.Linear(input_dim, input_dim)
        self.value_layer = nn.Linear(input_dim, input_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        Forward pass
        
        Args:
            x: 输入张量 [batch_size, seq_len, input_dim]
        
        Returns:
            output: 注意力输出 [batch_size, seq_len, input_dim]
            attention_weights: 注意力权重 [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        
        # 计算查询、键、值
        # Compute query, key, value
        Q = self.query_layer(x)  # [batch_size, seq_len, input_dim]
        K = self.key_layer(x)    # [batch_size, seq_len, input_dim]
        V = self.value_layer(x)  # [batch_size, seq_len, input_dim]
        
        # 计算注意力分数：Q与K的点积
        # Compute attention scores: dot product of Q and K
        attention_scores = torch.bmm(Q, K.transpose(1, 2))  # [batch_size, seq_len, seq_len]
        
        # 缩放（可选，这里为了简单没有缩放）
        # Scaling (optional, not scaled here for simplicity)
        
        # 应用softmax获得注意力权重
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # 应用注意力权重到值向量
        # Apply attention weights to value vectors
        output = torch.bmm(attention_weights, V)  # [batch_size, seq_len, input_dim]
        
        return output, attention_weights


class ScaledDotProductAttention(nn.Module):
    """
    缩放点积注意力（Transformer中使用的版本）
    Scaled Dot-Product Attention (used in Transformer)
    
    这是Transformer论文中提出的标准注意力机制
    This is the standard attention mechanism proposed in the Transformer paper
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        缩放点积注意力计算
        Scaled dot-product attention computation
        
        Args:
            Q: 查询矩阵 [batch_size, seq_len, d_model]
            K: 键矩阵 [batch_size, seq_len, d_model]
            V: 值矩阵 [batch_size, seq_len, d_model]
            mask: 掩码矩阵 [batch_size, seq_len, seq_len] (可选)
        
        Returns:
            output: 注意力输出
            attention_weights: 注意力权重
        """
        d_k = Q.size(-1)
        
        # 1. 计算注意力分数：Q·K^T / √d_k
        # Compute attention scores: Q·K^T / √d_k
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        # 2. 应用掩码（如果提供）
        # Apply mask (if provided)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 3. 应用softmax
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 4. 应用注意力权重到值向量
        # Apply attention weights to value vectors
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    Multi-Head Attention Mechanism
    
    将注意力分解为多个"头"，每个头关注不同的表示子空间
    Decomposes attention into multiple "heads", each focusing on different representation subspaces
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度
        
        # 线性变换层
        # Linear transformation layers
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(d_model, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        多头注意力前向传播
        Multi-head attention forward pass
        
        Args:
            query, key, value: 输入张量 [batch_size, seq_len, d_model]
            mask: 注意力掩码 [batch_size, seq_len, seq_len]
        
        Returns:
            output: 多头注意力输出 [batch_size, seq_len, d_model]
            attention_weights: 注意力权重 [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, d_model = query.shape
        
        # 1. 线性变换并重塑为多头格式
        # Linear transformations and reshape for multi-head format
        Q = self.W_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. 扩展掩码维度以匹配多头
        # Expand mask dimensions to match multi-head
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        
        # 3. 应用缩放点积注意力
        # Apply scaled dot-product attention
        attention_output, attention_weights = self.attention(Q, K, V, mask)
        
        # 4. 拼接所有头的输出
        # Concatenate outputs from all heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # 5. 最终的线性变换
        # Final linear transformation
        output = self.W_o(attention_output)
        
        return output, attention_weights


class PositionalEncoding(nn.Module):
    """
    位置编码
    Positional Encoding
    
    为序列中的每个位置添加位置信息
    Adds positional information to each position in the sequence
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        # 创建位置编码矩阵
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        # 计算角度
        # Compute angles
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(math.log(10000.0) / d_model))
        
        # 应用sin和cos函数
        # Apply sin and cos functions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        添加位置编码到输入
        Add positional encoding to input
        
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
        
        Returns:
            带位置编码的张量 [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class FeedForward(nn.Module):
    """
    前馈网络
    Feed Forward Network
    
    Transformer中的位置级前馈网络
    Position-wise feed forward network in Transformer
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前馈网络前向传播
        Feed forward network forward pass
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    """
    Transformer编码器层
    Transformer Encoder Layer
    
    包含多头注意力和前馈网络，以及残差连接和层归一化
    Contains multi-head attention and feed forward network, with residual connections and layer normalization
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        编码器层前向传播
        Encoder layer forward pass
        """
        # 1. 自注意力 + 残差连接 + 层归一化
        # Self-attention + residual connection + layer normalization
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 2. 前馈网络 + 残差连接 + 层归一化
        # Feed forward + residual connection + layer normalization
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class TransformerEncoder(nn.Module):
    """
    完整的Transformer编码器
    Complete Transformer Encoder
    
    由多个编码器层堆叠而成
    Stacked with multiple encoder layers
    """
    
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, 
                 num_layers: int, d_ff: int, max_len: int = 5000, dropout: float = 0.1):
        super(TransformerEncoder, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        编码器前向传播
        Encoder forward pass
        
        Args:
            x: 输入token序列 [batch_size, seq_len]
            mask: 注意力掩码 [batch_size, seq_len, seq_len]
        
        Returns:
            编码后的表示 [batch_size, seq_len, d_model]
        """
        # 1. 词嵌入 + 位置编码
        # Word embedding + positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # 2. 通过所有编码器层
        # Pass through all encoder layers
        for layer in self.layers:
            x = layer(x, mask)
            
        return x


def create_padding_mask(seq: torch.Tensor, pad_token: int = 0) -> torch.Tensor:
    """
    创建填充掩码
    Create padding mask
    
    Args:
        seq: 输入序列 [batch_size, seq_len]
        pad_token: 填充token的ID
    
    Returns:
        掩码矩阵 [batch_size, seq_len, seq_len]
    """
    batch_size, seq_len = seq.shape
    mask = (seq != pad_token).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
    mask = mask.expand(batch_size, 1, seq_len, seq_len)  # [batch_size, 1, seq_len, seq_len]
    return mask.squeeze(1)  # [batch_size, seq_len, seq_len]


def visualize_attention(attention_weights: torch.Tensor, tokens: list, 
                       head_idx: int = 0, save_path: str = None):
    """
    可视化注意力权重
    Visualize attention weights
    
    Args:
        attention_weights: 注意力权重 [batch_size, num_heads, seq_len, seq_len]
        tokens: 对应的token列表
        head_idx: 要可视化的头索引
        save_path: 保存路径（可选）
    """
    # 获取指定头的注意力权重
    # Get attention weights for specified head
    weights = attention_weights[0, head_idx].detach().cpu().numpy()
    
    # 创建热力图
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(weights, 
                xticklabels=tokens, 
                yticklabels=tokens,
                cmap='Blues',
                annot=True,
                fmt='.2f',
                cbar_kws={'label': '注意力权重 (Attention Weight)'})
    
    plt.title(f'注意力可视化 - 头 {head_idx} (Attention Visualization - Head {head_idx})')
    plt.xlabel('键 (Key)')
    plt.ylabel('查询 (Query)')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# 示例使用函数
# Example usage functions

def demo_basic_attention():
    """
    演示基础注意力机制
    Demonstrate basic attention mechanism
    """
    print("=== 基础注意力机制演示 (Basic Attention Demo) ===")
    
    # 创建模型和数据
    # Create model and data
    d_model = 64
    seq_len = 5
    batch_size = 2
    
    model = BasicAttention(d_model)
    
    # 随机输入数据
    # Random input data
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 前向传播
    # Forward pass
    output, attention_weights = model(x)
    
    print(f"输入形状 (Input shape): {x.shape}")
    print(f"输出形状 (Output shape): {output.shape}")
    print(f"注意力权重形状 (Attention weights shape): {attention_weights.shape}")
    print(f"注意力权重示例 (Attention weights example):\n{attention_weights[0]}")


def demo_transformer_encoder():
    """
    演示完整的Transformer编码器
    Demonstrate complete Transformer encoder
    """
    print("\n=== Transformer编码器演示 (Transformer Encoder Demo) ===")
    
    # 模型参数
    # Model parameters
    vocab_size = 1000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_len = 100
    
    # 创建模型
    # Create model
    model = TransformerEncoder(vocab_size, d_model, num_heads, num_layers, d_ff, max_len)
    
    # 创建示例输入
    # Create example input
    batch_size = 4
    seq_len = 20
    x = torch.randint(1, vocab_size, (batch_size, seq_len))  # 避免使用0（填充token）
    
    # 创建填充掩码
    # Create padding mask
    mask = create_padding_mask(x, pad_token=0)
    
    # 前向传播
    # Forward pass
    output = model(x, mask)
    
    print(f"词汇表大小 (Vocabulary size): {vocab_size}")
    print(f"模型维度 (Model dimension): {d_model}")
    print(f"注意力头数 (Number of heads): {num_heads}")
    print(f"编码器层数 (Number of encoder layers): {num_layers}")
    print(f"输入形状 (Input shape): {x.shape}")
    print(f"输出形状 (Output shape): {output.shape}")
    
    # 计算参数数量
    # Calculate number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数数量 (Total parameters): {total_params:,}")


def demo_attention_visualization():
    """
    演示注意力可视化
    Demonstrate attention visualization
    """
    print("\n=== 注意力可视化演示 (Attention Visualization Demo) ===")
    
    # 创建简单的多头注意力模型
    # Create simple multi-head attention model
    d_model = 64
    num_heads = 4
    
    model = MultiHeadAttention(d_model, num_heads)
    model.eval()
    
    # 创建示例句子
    # Create example sentence
    tokens = ["我", "爱", "深度", "学习", "和", "Transformer"]
    seq_len = len(tokens)
    
    # 创建随机向量表示这些词
    # Create random vectors to represent these words
    x = torch.randn(1, seq_len, d_model)
    
    # 获取注意力权重
    # Get attention weights
    with torch.no_grad():
        output, attention_weights = model(x, x, x)
    
    print(f"句子: {' '.join(tokens)}")
    print(f"注意力权重形状: {attention_weights.shape}")
    
    # 可视化第一个头的注意力
    # Visualize attention for the first head
    try:
        visualize_attention(attention_weights, tokens, head_idx=0)
    except Exception as e:
        print(f"可视化需要matplotlib和seaborn库: {e}")


if __name__ == "__main__":
    # 运行所有演示
    # Run all demonstrations
    demo_basic_attention()
    demo_transformer_encoder()
    demo_attention_visualization()
    
    print("\n=== 演示完成 (Demonstration Complete) ===")
    print("这些例子展示了从基础注意力到完整Transformer的实现")
    print("These examples show implementations from basic attention to complete Transformer") 