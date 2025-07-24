"""
数据处理工具函数
Data Processing Utility Functions

包含序列数据预处理、文本处理、时间序列处理等功能。
Contains sequence data preprocessing, text processing, time series processing functions.
"""

import torch
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Union
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import re
import jieba
from collections import Counter


def set_random_seed(seed: int = 42):
    """
    设置随机种子以确保结果可重现
    Set random seed for reproducible results
    
    Args:
        seed (int): 随机种子值 | Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_sequences(data: np.ndarray, seq_length: int, target_col: int = -1) -> Tuple[np.ndarray, np.ndarray]:
    """
    从时间序列数据创建序列样本
    Create sequence samples from time series data
    
    这就像把一本长书切成许多小段，每一小段都是一个学习样本。
    It's like cutting a long book into many small segments, each segment is a learning sample.
    
    Args:
        data (np.ndarray): 原始时间序列数据 | Original time series data
        seq_length (int): 序列长度 | Sequence length
        target_col (int): 目标列索引 | Target column index
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (输入序列, 目标值) | (Input sequences, Target values)
    """
    sequences = []
    targets = []
    
    for i in range(len(data) - seq_length):
        # 输入序列：从i到i+seq_length-1的数据
        # Input sequence: data from i to i+seq_length-1
        seq = data[i:i + seq_length]
        
        # 目标值：第i+seq_length个时间点的目标列数据
        # Target: target column data at time point i+seq_length
        target = data[i + seq_length, target_col]
        
        sequences.append(seq)
        targets.append(target)
    
    return np.array(sequences), np.array(targets)


def normalize_data(data: Union[np.ndarray, pd.DataFrame], 
                  method: str = 'minmax') -> Tuple[np.ndarray, object]:
    """
    数据标准化
    Data normalization
    
    就像把不同单位的数据（比如身高用cm，体重用kg）转换成统一的比例。
    Like converting data with different units (height in cm, weight in kg) to unified scales.
    
    Args:
        data: 输入数据 | Input data
        method: 标准化方法 ('minmax' 或 'standard') | Normalization method
        
    Returns:
        Tuple: (标准化后的数据, 标准化器对象) | (Normalized data, scaler object)
    """
    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("method必须是'minmax'或'standard' | method must be 'minmax' or 'standard'")
    
    if isinstance(data, pd.DataFrame):
        data = data.values
    
    normalized_data = scaler.fit_transform(data.reshape(-1, 1) if data.ndim == 1 else data)
    
    return normalized_data, scaler


def denormalize_data(normalized_data: np.ndarray, scaler: object) -> np.ndarray:
    """
    反标准化数据
    Denormalize data
    
    Args:
        normalized_data: 标准化后的数据 | Normalized data
        scaler: 标准化器对象 | Scaler object
        
    Returns:
        np.ndarray: 原始尺度的数据 | Data in original scale
    """
    return scaler.inverse_transform(normalized_data.reshape(-1, 1) if normalized_data.ndim == 1 else normalized_data)


class TextProcessor:
    """
    文本处理器
    Text Processor
    
    这就像一个聪明的助手，帮你把杂乱的文字整理成计算机能理解的格式。
    Like a smart assistant that helps organize messy text into formats computers can understand.
    """
    
    def __init__(self, max_vocab_size: int = 10000, min_freq: int = 2):
        """
        初始化文本处理器
        Initialize text processor
        
        Args:
            max_vocab_size: 最大词汇表大小 | Maximum vocabulary size
            min_freq: 最小词频阈值 | Minimum word frequency threshold
        """
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0
        
    def clean_text(self, text: str) -> str:
        """
        清理文本
        Clean text
        
        Args:
            text: 原始文本 | Original text
            
        Returns:
            str: 清理后的文本 | Cleaned text
        """
        # 移除特殊字符，只保留中文、英文、数字和空格
        # Remove special characters, keep only Chinese, English, numbers and spaces
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
        
        # 转换为小写并去除多余空格
        # Convert to lowercase and remove extra spaces
        text = re.sub(r'\s+', ' ', text.lower().strip())
        
        return text
    
    def tokenize(self, text: str, language: str = 'chinese') -> List[str]:
        """
        文本分词
        Text tokenization
        
        Args:
            text: 输入文本 | Input text
            language: 语言类型 | Language type
            
        Returns:
            List[str]: 词语列表 | List of words
        """
        if language == 'chinese':
            return list(jieba.cut(text))
        else:
            return text.split()
    
    def build_vocab(self, texts: List[str], language: str = 'chinese'):
        """
        构建词汇表
        Build vocabulary
        
        就像制作一本字典，把所有出现的词语按重要性排序。
        Like creating a dictionary, sorting all words by importance.
        
        Args:
            texts: 文本列表 | List of texts
            language: 语言类型 | Language type
        """
        word_counts = Counter()
        
        for text in texts:
            cleaned_text = self.clean_text(text)
            words = self.tokenize(cleaned_text, language)
            word_counts.update(words)
        
        # 过滤低频词并限制词汇表大小
        # Filter low-frequency words and limit vocabulary size
        filtered_words = [word for word, count in word_counts.items() if count >= self.min_freq]
        most_common_words = word_counts.most_common(self.max_vocab_size)
        
        # 构建词汇表映射
        # Build vocabulary mapping
        self.word2idx = {
            '<PAD>': 0,  # 填充符 | Padding token
            '<UNK>': 1,  # 未知词 | Unknown token
            '<BOS>': 2,  # 句首 | Beginning of sentence
            '<EOS>': 3,  # 句尾 | End of sentence
            '<MASK>': 4  # 掩码 | Mask
        }
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        
        for word in filtered_words:
            if word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)
        
        self.vocab_size = len(self.word2idx)
        
    def text_to_sequence(self, text: str, language: str = 'chinese') -> List[int]:
        """
        文本转换为序列
        Convert text to sequence
        
        Args:
            text: 输入文本 | Input text
            language: 语言类型 | Language type
            
        Returns:
            List[int]: 序列 | Sequence
        """
        tokens = self.tokenize(text, language)
        sequence = [self.word2idx.get(token, self.word2idx['<UNK>']) for token in tokens]
        
        # 添加句首和句尾标记
        # Add beginning and end of sentence markers
        sequence = [self.word2idx['<BOS>']] + sequence + [self.word2idx['<EOS>']]
        
        return sequence
    
    def sequence_to_text(self, sequence: List[int]) -> str:
        """
        序列转换为文本
        Convert sequence to text
        
        Args:
            sequence: 输入序列 | Input sequence
            
        Returns:
            str: 文本 | Text
        """
        words = [self.idx2word.get(idx, '<UNK>') for idx in sequence]
        return ' '.join(words) 