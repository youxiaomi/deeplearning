# 第12章：自然语言处理预训练 测试题

## 选择题 (Multiple Choice Questions)

### 1. 独热编码词表示的主要问题不包括以下哪项？
**Which of the following is NOT a problem with one-hot word representations?**

A) 高维度和稀疏性 (High dimensionality and sparsity)
B) 无法捕获语义关系 (No semantic relationships captured) 
C) 计算效率低 (Computational inefficiency)
D) 上下文敏感表示 (Context-sensitive representations)

### 2. 在Skip-gram模型中，模型预测什么？
**In the Skip-gram model, what does the model predict?**

A) 从上下文词预测中心词 (Center word from context words)
B) 从中心词预测上下文词 (Context words from center word)
C) 序列中的下一个词 (Next word in sequence)
D) 序列中的前一个词 (Previous word in sequence)

### 3. 负采样在word2vec训练中的主要优势是什么？
**What is the main advantage of negative sampling in word2vec training?**

A) 更好的词表示 (Better word representations)
B) 降低计算复杂度 (Reduced computational complexity)
C) 改善罕见词处理 (Improved rare word handling)
D) 增强语义关系 (Enhanced semantic relationships)

### 4. 在BERT的掩码语言建模中，通常掩盖多少百分比的标记？
**In BERT's Masked Language Modeling, what percentage of tokens are typically masked?**

A) 10%
B) 15%
C) 20%
D) 25%

### 5. GloVe模型的核心思想是什么？
**What is the core idea behind the GloVe model?**

A) 使用循环神经网络 (Using recurrent neural networks)
B) 利用全局词共现统计 (Leveraging global word co-occurrence statistics)
C) 增加模型深度 (Increasing model depth)
D) 使用注意力机制 (Using attention mechanisms)

### 6. FastText相比Word2Vec的主要改进是什么？
**What is the main improvement of FastText over Word2Vec?**

A) 更快的训练速度 (Faster training speed)
B) 更小的模型大小 (Smaller model size)
C) 支持子词信息 (Support for subword information)
D) 更简单的架构 (Simpler architecture)

### 7. BERT的两个主要预训练任务是什么？
**What are the two main pretraining tasks in BERT?**

A) 语言建模和情感分析 (Language modeling and sentiment analysis)
B) 掩码语言建模和下一句预测 (Masked language modeling and next sentence prediction)
C) 词分类和句子分类 (Word classification and sentence classification)
D) 机器翻译和问答 (Machine translation and question answering)

### 8. 在BERT的输入表示中，不包括以下哪种嵌入？
**Which of the following embeddings is NOT included in BERT's input representation?**

A) 词元嵌入 (Token embeddings)
B) 位置嵌入 (Position embeddings)
C) 分段嵌入 (Segment embeddings)
D) 语法嵌入 (Syntactic embeddings)

### 9. Byte Pair Encoding (BPE) 的主要优势是什么？
**What is the main advantage of Byte Pair Encoding (BPE)?**

A) 减少词汇表大小 (Reducing vocabulary size)
B) 提高训练速度 (Improving training speed)
C) 数据驱动的子词分割 (Data-driven subword segmentation)
D) 简化模型架构 (Simplifying model architecture)

### 10. 在负采样中，负样本的采样概率通常使用什么分布？
**In negative sampling, what distribution is typically used for sampling negative examples?**

A) 均匀分布 (Uniform distribution)
B) 词频的3/4次幂分布 (Word frequency to the 3/4 power)
C) 高斯分布 (Gaussian distribution)
D) 指数分布 (Exponential distribution)

## 简答题 (Short Answer Questions)

### 1. 解释Skip-gram和CBOW模型的区别，并说明各自的优缺点。
**Explain the difference between Skip-gram and CBOW models, and describe their respective advantages and disadvantages.**

### 2. 什么是词的多义性问题？传统词嵌入如何处理这个问题？BERT如何改进？
**What is the polysemy problem in words? How do traditional word embeddings handle this issue? How does BERT improve upon this?**

### 3. 描述BERT的掩码语言建模任务的详细步骤和策略。
**Describe the detailed steps and strategies of BERT's Masked Language Modeling task.**

### 4. 解释分层softmax和负采样两种近似训练方法的原理和区别。
**Explain the principles and differences between hierarchical softmax and negative sampling as approximate training methods.**

### 5. 为什么说BERT是"双向"的？这种双向性带来了什么优势？
**Why is BERT called "bidirectional"? What advantages does this bidirectionality bring?**

### 6. 解释GloVe模型中共现概率比率的重要性，并举例说明。
**Explain the importance of co-occurrence probability ratios in the GloVe model and provide examples.**

## 编程题 (Programming Questions)

### 1. 实现余弦相似度计算函数
**Implement a cosine similarity computation function**

编写一个函数来计算两个词向量之间的余弦相似度。
Write a function to compute cosine similarity between two word vectors.

```python
def cosine_similarity(vec1, vec2):
    """
    计算两个向量的余弦相似度
    Compute cosine similarity between two vectors
    
    Args:
        vec1: numpy array, 第一个向量
        vec2: numpy array, 第二个向量
    
    Returns:
        float: 余弦相似度值 (-1 到 1 之间)
    """
    # 你的代码在这里 (Your code here)
    pass
```

### 2. 实现简单的负采样函数
**Implement a simple negative sampling function**

```python
def negative_sampling(target_word, context_word, vocab, word_freq, num_negative=5):
    """
    为word2vec实现负采样
    Implement negative sampling for word2vec
    
    Args:
        target_word: str, 目标词
        context_word: str, 上下文词
        vocab: dict, 词汇表 {word: index}
        word_freq: dict, 词频 {word: frequency}
        num_negative: int, 负样本数量
    
    Returns:
        list: 包含正样本和负样本的列表
    """
    # 你的代码在这里 (Your code here)
    pass
```

### 3. 实现掩码语言建模数据创建
**Implement Masked Language Modeling data creation**

```python
def create_mlm_data(tokens, mask_token_id, vocab_size, mask_prob=0.15):
    """
    为BERT创建掩码语言建模训练数据
    Create Masked Language Modeling training data for BERT
    
    Args:
        tokens: list, 输入标记序列
        mask_token_id: int, [MASK]标记的ID
        vocab_size: int, 词汇表大小
        mask_prob: float, 掩码概率
    
    Returns:
        tuple: (masked_tokens, labels) 掩码后的标记和标签
    """
    # 你的代码在这里 (Your code here)
    pass
```

### 4. 实现词类比求解函数
**Implement word analogy solving function**

```python
def solve_word_analogy(word_vectors, word_a, word_b, word_c, vocab, top_k=1):
    """
    解决词类比问题：word_a之于word_b如同word_c之于？
    Solve word analogy: word_a is to word_b as word_c is to ?
    
    Args:
        word_vectors: dict, 词向量字典
        word_a, word_b, word_c: str, 类比中的三个词
        vocab: list, 词汇表
        top_k: int, 返回前k个最相似的词
    
    Returns:
        list: 最可能的答案词列表
    """
    # 你的代码在这里 (Your code here)
    pass
```

## 答案 (Answers)

### 选择题答案 (Multiple Choice Answers)

1. **D** - 上下文敏感表示 (Context-sensitive representations)
   - 解释：独热编码的问题是固定表示，而不是上下文敏感

2. **B** - 从中心词预测上下文词 (Context words from center word)
   - 解释：Skip-gram使用中心词预测周围的上下文词

3. **B** - 降低计算复杂度 (Reduced computational complexity)
   - 解释：负采样将复杂度从O(|V|)降低到O(K)

4. **B** - 15%
   - 解释：BERT中标准的掩码比例是15%

5. **B** - 利用全局词共现统计 (Leveraging global word co-occurrence statistics)
   - 解释：GloVe的核心是使用全局共现矩阵

6. **C** - 支持子词信息 (Support for subword information)
   - 解释：FastText使用字符n-gram来处理未见过的词

7. **B** - 掩码语言建模和下一句预测 (Masked language modeling and next sentence prediction)
   - 解释：这是BERT的两个核心预训练任务

8. **D** - 语法嵌入 (Syntactic embeddings)
   - 解释：BERT使用词元、位置和分段嵌入，但不包括语法嵌入

9. **C** - 数据驱动的子词分割 (Data-driven subword segmentation)
   - 解释：BPE根据数据学习最优的子词分割

10. **B** - 词频的3/4次幂分布 (Word frequency to the 3/4 power)
    - 解释：这样可以增加罕见词的采样概率

### 简答题答案 (Short Answer Answers)

#### 1. Skip-gram和CBOW模型的区别

**Skip-gram模型：**
- 使用中心词预测上下文词
- 优势：对罕见词表现更好，能学习更丰富的词表示
- 劣势：训练速度较慢，内存使用较多

**CBOW模型：**
- 使用上下文词预测中心词
- 优势：训练速度快，内存使用少
- 劣势：对罕见词表现较差，信息聚合可能丢失细节

#### 2. 词的多义性问题

**多义性问题：**
一个词在不同上下文中可能有不同的含义，如"bank"可以表示银行或河岸。

**传统词嵌入的处理：**
Word2Vec和GloVe为每个词提供固定的向量表示，无法区分不同上下文中的含义。

**BERT的改进：**
BERT使用上下文敏感的表示，同一个词在不同上下文中会产生不同的向量表示，从而解决多义性问题。

#### 3. BERT的掩码语言建模任务

**步骤：**
1. 随机选择15%的标记进行掩盖
2. 对选中的标记：
   - 80%替换为[MASK]
   - 10%替换为随机标记
   - 10%保持不变
3. 训练模型预测被掩盖的原始标记

**策略意义：**
- 使模型学习双向上下文信息
- 避免预训练和微调之间的差异
- 提高模型的鲁棒性

#### 4. 分层softmax vs 负采样

**分层softmax：**
- 使用二叉树结构组织词汇表
- 复杂度：O(log|V|)
- 优势：精确计算，复杂度可控
- 劣势：树结构固定，可能引入偏差

**负采样：**
- 将多分类问题转化为二分类问题
- 复杂度：O(K)，其中K是负样本数
- 优势：灵活，训练效果好
- 劣势：近似方法，需要调整负样本数

#### 5. BERT的双向性

**双向性含义：**
BERT同时使用左侧和右侧的上下文信息来理解每个词的含义。

**优势：**
- 更全面的上下文理解
- 更好的语义表示
- 在各种NLP任务上的优越性能
- 能够捕获长距离依赖关系

#### 6. GloVe中的共现概率比率

**重要性：**
比率能够编码词汇间的语义关系，比绝对概率更有判别性。

**例子：**
- P(solid|ice)/P(solid|steam) >> 1：solid与ice更相关
- P(gas|ice)/P(gas|steam) << 1：gas与steam更相关
- P(water|ice)/P(water|steam) ≈ 1：water与两者关系相当

### 编程题答案 (Programming Answers)

#### 1. 余弦相似度计算

```python
import numpy as np

def cosine_similarity(vec1, vec2):
    """
    计算两个向量的余弦相似度
    """
    # 计算点积
    dot_product = np.dot(vec1, vec2)
    
    # 计算向量的模
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    # 避免除零错误
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    # 计算余弦相似度
    similarity = dot_product / (norm1 * norm2)
    return similarity
```

#### 2. 负采样函数

```python
import random
import numpy as np

def negative_sampling(target_word, context_word, vocab, word_freq, num_negative=5):
    """
    为word2vec实现负采样
    """
    positive_sample = (target_word, context_word, 1)  # 正样本
    samples = [positive_sample]
    
    # 计算采样概率（3/4次幂）
    words = list(word_freq.keys())
    freqs = [word_freq[word] ** 0.75 for word in words]
    total_freq = sum(freqs)
    probs = [freq / total_freq for freq in freqs]
    
    # 采样负样本
    negative_words = np.random.choice(words, size=num_negative, p=probs)
    
    for neg_word in negative_words:
        if neg_word != context_word:  # 避免采样到正样本
            samples.append((target_word, neg_word, 0))  # 负样本
    
    return samples
```

#### 3. 掩码语言建模数据创建

```python
import random

def create_mlm_data(tokens, mask_token_id, vocab_size, mask_prob=0.15):
    """
    为BERT创建掩码语言建模训练数据
    """
    masked_tokens = tokens.copy()
    labels = [-100] * len(tokens)  # -100表示不计算损失
    
    for i, token in enumerate(tokens):
        # 跳过特殊标记
        if token in [0, 1, 2, 3]:  # [PAD], [UNK], [CLS], [SEP]
            continue
            
        # 随机决定是否掩盖
        if random.random() < mask_prob:
            labels[i] = token  # 保存原始标记用于计算损失
            
            rand = random.random()
            if rand < 0.8:
                # 80%的时间替换为[MASK]
                masked_tokens[i] = mask_token_id
            elif rand < 0.9:
                # 10%的时间替换为随机标记
                masked_tokens[i] = random.randint(5, vocab_size - 1)
            # 10%的时间保持不变
    
    return masked_tokens, labels
```

#### 4. 词类比求解函数

```python
import numpy as np
from scipy.spatial.distance import cosine

def solve_word_analogy(word_vectors, word_a, word_b, word_c, vocab, top_k=1):
    """
    解决词类比问题：word_a之于word_b如同word_c之于？
    """
    # 获取词向量
    vec_a = word_vectors.get(word_a)
    vec_b = word_vectors.get(word_b)
    vec_c = word_vectors.get(word_c)
    
    if any(v is None for v in [vec_a, vec_b, vec_c]):
        return []
    
    # 计算目标向量：vec_b - vec_a + vec_c
    target_vector = vec_b - vec_a + vec_c
    
    # 计算与所有词的相似度
    similarities = []
    for word in vocab:
        if word not in [word_a, word_b, word_c] and word in word_vectors:
            word_vec = word_vectors[word]
            similarity = 1 - cosine(target_vector, word_vec)
            similarities.append((word, similarity))
    
    # 按相似度排序并返回前k个
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [word for word, _ in similarities[:top_k]]
```

## 评分标准 (Scoring Criteria)

### 选择题 (Multiple Choice): 40分
每题4分，共10题

### 简答题 (Short Answer): 30分
每题5分，共6题

### 编程题 (Programming): 30分
每题7.5分，共4题

### 总分 (Total): 100分

**评分等级 (Grade Levels):**
- A (优秀): 90-100分
- B (良好): 80-89分  
- C (及格): 70-79分
- D (不及格): <70分 