# Word2Vec详细实现指南
# Word2Vec Detailed Implementation Guide

**词向量革命的起点 - 让词语拥有数学表示**
**The Starting Point of Word Vector Revolution - Giving Words Mathematical Representation**

---

## 🎯 项目概述 | Project Overview

Word2Vec不仅仅是一个算法，它是NLP历史的转折点！2013年，Google的Tomas Mikolov团队提出了这个革命性的方法，第一次让计算机真正"理解"了词语之间的语义关系。

Word2Vec is not just an algorithm, it's a turning point in NLP history! In 2013, Google's Tomas Mikolov team proposed this revolutionary method, making computers truly "understand" semantic relationships between words for the first time.

### 核心洞察 | Core Insights
- **分布式假设**: 词语的含义由其上下文决定
- **Distributional Hypothesis**: A word's meaning is determined by its context
- **低维表示**: 将高维稀疏的one-hot向量转换为低维稠密向量
- **Low-dimensional Representation**: Transform high-dimensional sparse one-hot vectors to low-dimensional dense vectors
- **语义关系**: 向量运算能够捕捉词语间的语义关系
- **Semantic Relations**: Vector operations can capture semantic relationships between words

## 🧠 深度理论解析 | Deep Theoretical Analysis

### 分布式假设的数学表示 | Mathematical Representation of Distributional Hypothesis

**核心思想 | Core Idea:**
如果两个词经常出现在相似的上下文中，那么它们在语义上是相关的。

If two words often appear in similar contexts, they are semantically related.

**数学表达 | Mathematical Expression:**
```
P(w_o | w_c) = exp(u_o^T v_c) / Σ exp(u_w^T v_c)
```

其中：
- `w_c`: 中心词 | center word
- `w_o`: 上下文词 | context word  
- `v_c`: 中心词向量 | center word vector
- `u_o`: 上下文词向量 | context word vector

### Skip-gram vs CBOW架构对比 | Skip-gram vs CBOW Architecture Comparison

#### Skip-gram模型 | Skip-gram Model
```
输入: 中心词 → 输出: 上下文词
Input: Center word → Output: Context words

例子: "I love [deep] learning AI"
Example: "I love [deep] learning AI"

给定"deep"，预测"I", "love", "learning", "AI"
Given "deep", predict "I", "love", "learning", "AI"
```

#### CBOW模型 | CBOW Model  
```
输入: 上下文词 → 输出: 中心词
Input: Context words → Output: Center word

例子: "I love [?] learning AI"
Example: "I love [?] learning AI"

给定"I", "love", "learning", "AI"，预测"deep"
Given "I", "love", "learning", "AI", predict "deep"
```

## 🛠️ 完整实现代码 | Complete Implementation Code

### 第一步: 数据预处理 | Step 1: Data Preprocessing

```python
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import random
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.optim as optim

class Word2VecDataset:
    """
    Word2Vec数据预处理类
    Word2Vec Data Preprocessing Class
    """
    def __init__(self, min_count=5, window_size=5):
        self.min_count = min_count
        self.window_size = window_size
        self.word2idx = {}
        self.idx2word = {}
        self.word_counts = Counter()
        self.vocab_size = 0
        
    def build_vocab(self, sentences):
        """
        构建词汇表
        Build vocabulary
        """
        print("Building vocabulary...")
        
        # 统计词频 | Count word frequencies
        for sentence in tqdm(sentences):
            words = sentence.lower().split()
            self.word_counts.update(words)
        
        # 过滤低频词 | Filter low-frequency words
        filtered_words = {word: count for word, count in self.word_counts.items() 
                         if count >= self.min_count}
        
        # 创建词汇映射 | Create vocabulary mapping
        vocab = list(filtered_words.keys())
        self.vocab_size = len(vocab)
        
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
        print(f"Vocabulary size: {self.vocab_size}")
        return vocab
    
    def create_skipgram_pairs(self, sentences):
        """
        创建Skip-gram训练对
        Create Skip-gram training pairs
        """
        print("Creating Skip-gram pairs...")
        
        pairs = []
        for sentence in tqdm(sentences):
            words = [word for word in sentence.lower().split() 
                    if word in self.word2idx]
            
            for i, center_word in enumerate(words):
                # 定义上下文窗口 | Define context window
                start = max(0, i - self.window_size)
                end = min(len(words), i + self.window_size + 1)
                
                for j in range(start, end):
                    if i != j:  # 跳过中心词本身 | Skip center word itself
                        context_word = words[j]
                        center_idx = self.word2idx[center_word]
                        context_idx = self.word2idx[context_word]
                        pairs.append((center_idx, context_idx))
        
        print(f"Generated {len(pairs)} training pairs")
        return pairs
    
    def create_cbow_pairs(self, sentences):
        """
        创建CBOW训练对
        Create CBOW training pairs
        """
        print("Creating CBOW pairs...")
        
        pairs = []
        for sentence in tqdm(sentences):
            words = [word for word in sentence.lower().split() 
                    if word in self.word2idx]
            
            for i, center_word in enumerate(words):
                # 收集上下文词 | Collect context words
                start = max(0, i - self.window_size)
                end = min(len(words), i + self.window_size + 1)
                
                context_words = []
                for j in range(start, end):
                    if i != j:
                        context_words.append(self.word2idx[words[j]])
                
                if len(context_words) > 0:
                    center_idx = self.word2idx[center_word]
                    pairs.append((context_words, center_idx))
        
        print(f"Generated {len(pairs)} training pairs")
        return pairs
```

### 第二步: Skip-gram模型实现 | Step 2: Skip-gram Model Implementation

```python
class SkipGramModel(nn.Module):
    """
    Skip-gram模型实现
    Skip-gram Model Implementation
    """
    def __init__(self, vocab_size, embedding_dim=100):
        super(SkipGramModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # 中心词嵌入矩阵 | Center word embedding matrix
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # 上下文词嵌入矩阵 | Context word embedding matrix
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # 初始化权重 | Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """
        初始化嵌入权重
        Initialize embedding weights
        """
        initrange = 0.5 / self.embedding_dim
        self.center_embeddings.weight.data.uniform_(-initrange, initrange)
        self.context_embeddings.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, center_words, context_words):
        """
        前向传播
        Forward pass
        """
        # 获取中心词嵌入 | Get center word embeddings
        center_embeds = self.center_embeddings(center_words)  # [batch_size, embed_dim]
        
        # 获取上下文词嵌入 | Get context word embeddings
        context_embeds = self.context_embeddings(context_words)  # [batch_size, embed_dim]
        
        # 计算内积 | Compute dot product
        scores = torch.sum(center_embeds * context_embeds, dim=1)  # [batch_size]
        
        return scores
    
    def get_word_embedding(self, word_idx):
        """
        获取词嵌入向量
        Get word embedding vector
        """
        return self.center_embeddings.weight[word_idx].detach().numpy()
```

### 第三步: 负采样优化 | Step 3: Negative Sampling Optimization

```python
class NegativeSamplingLoss(nn.Module):
    """
    负采样损失函数
    Negative Sampling Loss Function
    """
    def __init__(self, vocab_size, num_negative_samples=5):
        super(NegativeSamplingLoss, self).__init__()
        self.vocab_size = vocab_size
        self.num_negative_samples = num_negative_samples
        
        # 创建负采样分布 | Create negative sampling distribution
        self.noise_distribution = torch.ones(vocab_size)
        
    def set_noise_distribution(self, word_counts):
        """
        设置噪声分布（基于词频的3/4次方）
        Set noise distribution (based on word frequency to the power of 3/4)
        """
        counts = torch.FloatTensor(word_counts)
        self.noise_distribution = torch.pow(counts, 0.75)
        self.noise_distribution = self.noise_distribution / self.noise_distribution.sum()
    
    def forward(self, positive_scores, center_words, context_words):
        """
        计算负采样损失
        Compute negative sampling loss
        """
        batch_size = positive_scores.size(0)
        
        # 正样本损失 | Positive sample loss
        positive_loss = -torch.log(torch.sigmoid(positive_scores)).mean()
        
        # 负样本采样 | Negative sample sampling
        negative_words = torch.multinomial(
            self.noise_distribution, 
            batch_size * self.num_negative_samples, 
            replacement=True
        ).view(batch_size, self.num_negative_samples)
        
        # 计算负样本分数 | Compute negative sample scores
        center_embeds = model.center_embeddings(center_words).unsqueeze(1)  # [batch, 1, embed]
        negative_embeds = model.context_embeddings(negative_words)  # [batch, neg_samples, embed]
        
        negative_scores = torch.sum(center_embeds * negative_embeds, dim=2)  # [batch, neg_samples]
        
        # 负样本损失 | Negative sample loss
        negative_loss = -torch.log(torch.sigmoid(-negative_scores)).mean()
        
        return positive_loss + negative_loss

def create_negative_samples(positive_pairs, vocab_size, num_negative=5):
    """
    为每个正样本创建负样本
    Create negative samples for each positive sample
    """
    negative_pairs = []
    
    for center_word, context_word in positive_pairs:
        # 为每个正样本生成多个负样本
        # Generate multiple negative samples for each positive sample
        for _ in range(num_negative):
            # 随机选择一个词作为负样本
            # Randomly select a word as negative sample
            negative_word = random.randint(0, vocab_size - 1)
            # 确保负样本不是真实的上下文词
            # Ensure negative sample is not the actual context word
            while negative_word == context_word:
                negative_word = random.randint(0, vocab_size - 1)
            
            negative_pairs.append((center_word, negative_word, 0))  # 标签为0表示负样本
    
    return negative_pairs
```

### 第四步: 训练循环 | Step 4: Training Loop

```python
def train_word2vec(sentences, embedding_dim=100, window_size=5, 
                   min_count=5, num_epochs=5, learning_rate=0.001,
                   batch_size=512, num_negative_samples=5):
    """
    Word2Vec训练主函数
    Word2Vec Training Main Function
    """
    
    # 1. 数据预处理 | Data preprocessing
    dataset = Word2VecDataset(min_count=min_count, window_size=window_size)
    vocab = dataset.build_vocab(sentences)
    
    # 2. 创建训练数据 | Create training data
    skipgram_pairs = dataset.create_skipgram_pairs(sentences)
    
    # 3. 初始化模型 | Initialize model
    model = SkipGramModel(dataset.vocab_size, embedding_dim)
    criterion = NegativeSamplingLoss(dataset.vocab_size, num_negative_samples)
    
    # 设置噪声分布 | Set noise distribution
    word_counts = [dataset.word_counts[word] for word in vocab]
    criterion.set_noise_distribution(word_counts)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 4. 训练循环 | Training loop
    model.train()
    total_loss = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # 打乱数据 | Shuffle data
        random.shuffle(skipgram_pairs)
        
        epoch_loss = 0
        num_batches = len(skipgram_pairs) // batch_size
        
        for i in tqdm(range(0, len(skipgram_pairs), batch_size)):
            batch_pairs = skipgram_pairs[i:i + batch_size]
            
            if len(batch_pairs) < batch_size:
                continue
            
            # 准备批次数据 | Prepare batch data
            center_words = torch.LongTensor([pair[0] for pair in batch_pairs])
            context_words = torch.LongTensor([pair[1] for pair in batch_pairs])
            
            # 前向传播 | Forward pass
            positive_scores = model(center_words, context_words)
            
            # 计算损失 | Compute loss
            loss = criterion(positive_scores, center_words, context_words)
            
            # 反向传播 | Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / num_batches
        print(f"Average Loss: {avg_loss:.4f}")
        total_loss += avg_loss
    
    print(f"\nTraining completed! Average total loss: {total_loss/num_epochs:.4f}")
    
    return model, dataset

# 使用示例 | Usage Example
if __name__ == "__main__":
    # 示例语料 | Example corpus
    sentences = [
        "I love deep learning and natural language processing",
        "Machine learning is a subset of artificial intelligence",
        "Neural networks are inspired by biological neurons",
        "Word embeddings capture semantic relationships between words",
        "BERT and GPT are powerful language models",
        # 添加更多句子...
        # Add more sentences...
    ]
    
    # 训练模型 | Train model
    model, dataset = train_word2vec(
        sentences=sentences,
        embedding_dim=100,
        window_size=5,
        min_count=1,  # 由于示例数据较少，降低最小频次
        num_epochs=10,
        learning_rate=0.001,
        batch_size=32
    )
```

### 第五步: 模型评估与可视化 | Step 5: Model Evaluation and Visualization

```python
class Word2VecEvaluator:
    """
    Word2Vec模型评估器
    Word2Vec Model Evaluator
    """
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self.word_vectors = self._extract_word_vectors()
    
    def _extract_word_vectors(self):
        """
        提取所有词的向量表示
        Extract vector representations of all words
        """
        word_vectors = {}
        with torch.no_grad():
            for word, idx in self.dataset.word2idx.items():
                vector = self.model.get_word_embedding(idx)
                word_vectors[word] = vector
        return word_vectors
    
    def cosine_similarity(self, word1, word2):
        """
        计算两个词的余弦相似度
        Compute cosine similarity between two words
        """
        if word1 not in self.word_vectors or word2 not in self.word_vectors:
            return None
        
        vec1 = self.word_vectors[word1]
        vec2 = self.word_vectors[word2]
        
        cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return cos_sim
    
    def find_most_similar(self, word, top_k=5):
        """
        找到与给定词最相似的词
        Find most similar words to a given word
        """
        if word not in self.word_vectors:
            return []
        
        target_vector = self.word_vectors[word]
        similarities = []
        
        for other_word, other_vector in self.word_vectors.items():
            if other_word != word:
                similarity = self.cosine_similarity(word, other_word)
                similarities.append((other_word, similarity))
        
        # 按相似度排序 | Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def word_analogy(self, word_a, word_b, word_c, top_k=1):
        """
        词类比任务: word_a is to word_b as word_c is to ?
        Word analogy task: word_a is to word_b as word_c is to ?
        
        例如: king - man + woman = queen
        Example: king - man + woman = queen
        """
        if not all(word in self.word_vectors for word in [word_a, word_b, word_c]):
            return []
        
        # 计算类比向量 | Compute analogy vector
        vec_a = self.word_vectors[word_a]
        vec_b = self.word_vectors[word_b]
        vec_c = self.word_vectors[word_c]
        
        target_vector = vec_b - vec_a + vec_c
        
        # 找到最接近的词 | Find closest words
        similarities = []
        for word, vector in self.word_vectors.items():
            if word not in [word_a, word_b, word_c]:
                similarity = np.dot(target_vector, vector) / (
                    np.linalg.norm(target_vector) * np.linalg.norm(vector)
                )
                similarities.append((word, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def visualize_embeddings(self, words=None, perplexity=30, n_iter=1000):
        """
        使用t-SNE可视化词嵌入
        Visualize word embeddings using t-SNE
        """
        if words is None:
            words = list(self.word_vectors.keys())[:50]  # 可视化前50个词
        
        # 准备数据 | Prepare data
        vectors = [self.word_vectors[word] for word in words if word in self.word_vectors]
        valid_words = [word for word in words if word in self.word_vectors]
        
        if len(vectors) < 2:
            print("Not enough vectors to visualize")
            return
        
        # t-SNE降维 | t-SNE dimensionality reduction
        tsne = TSNE(n_components=2, perplexity=min(perplexity, len(vectors)-1), 
                   n_iter=n_iter, random_state=42)
        vectors_2d = tsne.fit_transform(np.array(vectors))
        
        # 绘图 | Plot
        plt.figure(figsize=(12, 8))
        plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], alpha=0.7)
        
        # 添加词标签 | Add word labels
        for i, word in enumerate(valid_words):
            plt.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]), 
                        xytext=(5, 2), textcoords='offset points', 
                        ha='left', va='bottom', fontsize=10)
        
        plt.title('Word2Vec Embeddings Visualization (t-SNE)')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def evaluate_word_similarity(self, word_pairs_with_scores):
        """
        评估词相似度任务
        Evaluate word similarity task
        
        word_pairs_with_scores: [(word1, word2, human_score), ...]
        """
        model_scores = []
        human_scores = []
        
        for word1, word2, human_score in word_pairs_with_scores:
            model_score = self.cosine_similarity(word1, word2)
            if model_score is not None:
                model_scores.append(model_score)
                human_scores.append(human_score)
        
        if len(model_scores) == 0:
            return 0.0
        
        # 计算斯皮尔曼相关系数 | Compute Spearman correlation
        from scipy.stats import spearmanr
        correlation, p_value = spearmanr(human_scores, model_scores)
        
        return correlation, p_value

# 评估示例 | Evaluation Example
def evaluate_model():
    """
    模型评估示例
    Model evaluation example
    """
    # 假设已经训练好了模型 | Assume model is already trained
    evaluator = Word2VecEvaluator(model, dataset)
    
    # 1. 词相似度测试 | Word similarity test
    print("=== Word Similarity Test ===")
    test_pairs = [
        ("deep", "learning"),
        ("machine", "artificial"),
        ("neural", "networks"),
        ("language", "processing")
    ]
    
    for word1, word2 in test_pairs:
        similarity = evaluator.cosine_similarity(word1, word2)
        if similarity is not None:
            print(f"Similarity({word1}, {word2}): {similarity:.4f}")
    
    # 2. 最相似词查找 | Most similar words search
    print("\n=== Most Similar Words ===")
    test_words = ["learning", "neural", "language"]
    for word in test_words:
        if word in evaluator.word_vectors:
            similar_words = evaluator.find_most_similar(word, top_k=3)
            print(f"Most similar to '{word}': {similar_words}")
    
    # 3. 词类比测试 | Word analogy test
    print("\n=== Word Analogy Test ===")
    analogy_tests = [
        ("deep", "learning", "machine"),  # deep learning vs machine ?
        ("neural", "networks", "natural")  # neural networks vs natural ?
    ]
    
    for word_a, word_b, word_c in analogy_tests:
        result = evaluator.word_analogy(word_a, word_b, word_c, top_k=3)
        if result:
            print(f"{word_a}:{word_b} :: {word_c}:{result[0][0]} (score: {result[0][1]:.4f})")
    
    # 4. 可视化词嵌入 | Visualize embeddings
    print("\n=== Visualizing Embeddings ===")
    important_words = ["deep", "learning", "machine", "neural", "networks", 
                      "language", "processing", "artificial", "intelligence"]
    evaluator.visualize_embeddings(words=important_words)

if __name__ == "__main__":
    evaluate_model()
```

## 🎨 高级应用与扩展 | Advanced Applications and Extensions

### 1. 子词嵌入 (Subword Embeddings)

```python
class SubwordEmbedding:
    """
    子词嵌入实现，处理未登录词问题
    Subword embedding implementation to handle OOV words
    """
    def __init__(self, min_n=3, max_n=6):
        self.min_n = min_n
        self.max_n = max_n
        self.char_ngrams = {}
    
    def get_char_ngrams(self, word):
        """
        获取单词的字符n-gram
        Get character n-grams of a word
        """
        word = f"<{word}>"  # 添加边界符号
        ngrams = []
        
        for n in range(self.min_n, min(len(word), self.max_n) + 1):
            for i in range(len(word) - n + 1):
                ngrams.append(word[i:i+n])
        
        return ngrams
    
    def get_word_vector(self, word, word2vec_model):
        """
        通过子词向量合成单词向量
        Compose word vector from subword vectors
        """
        if word in word2vec_model.word_vectors:
            return word2vec_model.word_vectors[word]
        
        # 对于未登录词，使用子词向量的平均
        # For OOV words, use average of subword vectors
        ngrams = self.get_char_ngrams(word)
        vectors = []
        
        for ngram in ngrams:
            if ngram in word2vec_model.word_vectors:
                vectors.append(word2vec_model.word_vectors[ngram])
        
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            # 返回随机向量 | Return random vector
            return np.random.normal(0, 0.1, word2vec_model.embedding_dim)
```

### 2. 层次化Softmax优化

```python
class HierarchicalSoftmax:
    """
    层次化Softmax实现
    Hierarchical Softmax Implementation
    """
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # 构建霍夫曼树 | Build Huffman tree
        self.huffman_tree = self.build_huffman_tree()
    
    def build_huffman_tree(self):
        """
        构建霍夫曼树以实现层次化softmax
        Build Huffman tree for hierarchical softmax
        """
        # 简化实现：使用完全二叉树
        # Simplified implementation: use complete binary tree
        import heapq
        
        # 这里应该根据词频构建真正的霍夫曼树
        # Here should build real Huffman tree based on word frequencies
        # 为简化，我们创建一个简单的二叉树结构
        # For simplicity, we create a simple binary tree structure
        
        tree_nodes = {}
        for i in range(self.vocab_size * 2 - 1):
            tree_nodes[i] = {
                'vector': np.random.normal(0, 0.1, self.embedding_dim),
                'left': None,
                'right': None,
                'code': [],
                'path': []
            }
        
        return tree_nodes
    
    def get_path_and_code(self, word_idx):
        """
        获取词到根节点的路径和编码
        Get path and code from word to root node
        """
        # 简化实现：返回预定义的路径
        # Simplified implementation: return predefined path
        path = []
        code = []
        
        # 这里应该实现真正的霍夫曼编码路径
        # Here should implement real Huffman coding path
        current = word_idx + self.vocab_size - 1
        while current > 0:
            parent = (current - 1) // 2
            if current % 2 == 1:  # 左子树
                code.append(0)
            else:  # 右子树
                code.append(1)
            path.append(parent)
            current = parent
        
        return path[::-1], code[::-1]
```

### 3. 词向量质量评估

```python
def intrinsic_evaluation():
    """
    内在评估：词向量质量的定量分析
    Intrinsic evaluation: quantitative analysis of word vector quality
    """
    
    # 1. 词相似度数据集评估
    # Word similarity dataset evaluation
    wordsim353 = [
        ("love", "sex", 6.77),
        ("tiger", "cat", 7.35),
        ("computer", "keyboard", 7.62),
        # 更多数据...
        # More data...
    ]
    
    # 2. 词类比数据集评估
    # Word analogy dataset evaluation
    analogies = [
        ("man", "woman", "king", "queen"),
        ("good", "better", "bad", "worse"),
        ("go", "went", "take", "took"),
        # 更多类比...
        # More analogies...
    ]
    
    return wordsim353, analogies

def extrinsic_evaluation(word_vectors, downstream_task):
    """
    外在评估：在下游任务上的性能
    Extrinsic evaluation: performance on downstream tasks
    """
    
    # 使用词向量作为特征进行文本分类
    # Use word vectors as features for text classification
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    
    # 这里应该实现具体的下游任务评估
    # Here should implement specific downstream task evaluation
    pass
```

---

**🎯 项目完成检查清单 | Project Completion Checklist:**

### 理论理解 | Theoretical Understanding
- [ ] 深入理解分布式假设的数学基础
- [ ] 掌握Skip-gram和CBOW的区别和适用场景
- [ ] 理解负采样和层次化Softmax的优化原理
- [ ] 能够分析词向量的几何性质

### 编程实现 | Programming Implementation  
- [ ] 从零实现完整的Word2Vec训练流程
- [ ] 正确实现负采样优化技术
- [ ] 能够处理大规模文本数据
- [ ] 实现高效的词向量检索和相似度计算

### 实验分析 | Experimental Analysis
- [ ] 在真实数据集上训练高质量词向量
- [ ] 进行全面的内在和外在评估
- [ ] 可视化和分析词向量的语义结构
- [ ] 对比不同超参数设置的影响

### 应用扩展 | Application Extensions
- [ ] 实现子词嵌入处理未登录词
- [ ] 探索跨语言词向量对齐
- [ ] 将词向量应用到实际NLP任务
- [ ] 优化模型以提升训练和推理效率

**记住**: Word2Vec不仅是一个技术，更是理解语言分布式表示的钥匙。掌握了Word2Vec，你就理解了现代NLP的基础！

**Remember**: Word2Vec is not just a technique, but a key to understanding distributed representation of language. Master Word2Vec, and you understand the foundation of modern NLP! 