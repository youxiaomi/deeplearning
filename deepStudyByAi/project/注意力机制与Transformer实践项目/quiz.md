# 注意力机制与Transformer测试题

## 基础概念题 (Basic Concept Questions)

### 1. 选择题：注意力机制的核心作用是什么？
**What is the core function of attention mechanism?**

A) 增加模型参数数量 (Increase model parameters)
B) 让模型专注于最相关的信息 (Let model focus on most relevant information)  
C) 加速模型训练 (Accelerate model training)
D) 减少计算复杂度 (Reduce computational complexity)

**答案：B**
**解释：注意力机制的核心作用是让模型能够选择性地关注输入中最相关的部分，就像人类的注意力一样。**
**Explanation: The core function of attention mechanism is to allow models to selectively focus on the most relevant parts of the input, similar to human attention.**

### 2. 填空题：在注意力机制中，QKV分别代表什么？
**In attention mechanism, what do Q, K, V represent?**

- Q代表：_______ (Query - 查询向量)
- K代表：_______ (Key - 键向量)  
- V代表：_______ (Value - 值向量)

**答案：**
- Q代表：Query/查询向量 (表示"我想要什么信息")
- K代表：Key/键向量 (表示"我有什么信息")
- V代表：Value/值向量 (表示"信息的具体内容")

### 3. 判断题：Transformer完全抛弃了RNN和CNN结构
**True/False: Transformer completely abandons RNN and CNN structures**

**答案：正确 (True)**
**解释：Transformer是第一个完全基于注意力机制的模型，不使用任何循环或卷积结构。**
**Explanation: Transformer is the first model that is completely based on attention mechanisms, without using any recurrent or convolutional structures.**

## 数学原理题 (Mathematical Principle Questions)

### 4. 计算题：给定以下参数，计算注意力权重
**Given the following parameters, calculate attention weights**

假设我们有一个简化的注意力计算：
Suppose we have a simplified attention calculation:

查询向量 Q = [1, 0]
键向量 K₁ = [1, 1], K₂ = [0, 1], K₃ = [1, 0]

计算每个键的注意力权重（使用点积相似度和softmax）
Calculate attention weights for each key (using dot product similarity and softmax)

**解答步骤：**
1. 计算相似度分数：
   - e₁ = Q·K₁ = [1,0]·[1,1] = 1×1 + 0×1 = 1
   - e₂ = Q·K₂ = [1,0]·[0,1] = 1×0 + 0×1 = 0  
   - e₃ = Q·K₃ = [1,0]·[1,0] = 1×1 + 0×0 = 1

2. 应用softmax：
   - α₁ = exp(1)/(exp(1)+exp(0)+exp(1)) = e/(e+1+e) = e/(2e+1) ≈ 0.42
   - α₂ = exp(0)/(exp(1)+exp(0)+exp(1)) = 1/(2e+1) ≈ 0.16
   - α₃ = exp(1)/(exp(1)+exp(0)+exp(1)) = e/(2e+1) ≈ 0.42

### 5. 概念题：为什么Transformer中的注意力要除以√d_k？
**Why is attention in Transformer divided by √d_k?**

**答案：**
当向量维度d_k很大时，点积QK^T的结果会变得很大，这会导致softmax函数进入饱和区域，使得梯度变得非常小，影响训练效果。除以√d_k可以将点积结果缩放到合理范围，保持梯度的稳定性。

When the vector dimension d_k is large, the dot product QK^T results become very large, causing the softmax function to enter saturation regions where gradients become very small, affecting training. Dividing by √d_k scales the dot product results to a reasonable range, maintaining gradient stability.

## 架构理解题 (Architecture Understanding Questions)

### 6. 简答题：解释多头注意力的优势
**Explain the advantages of multi-head attention**

**答案：**
多头注意力的优势包括：
1. **多角度关注**：不同的头可以关注输入的不同方面，如语法关系、语义关系等
2. **增强表达能力**：类似于CNN中的多个滤波器，增加模型的表达能力
3. **并行计算**：多个头可以并行计算，提高效率
4. **降低风险**：即使某些头学习失败，其他头仍可以提供有用信息

Advantages of multi-head attention include:
1. **Multi-perspective focus**: Different heads can focus on different aspects of input, like syntactic and semantic relations
2. **Enhanced expressiveness**: Similar to multiple filters in CNN, increases model expressiveness  
3. **Parallel computation**: Multiple heads can be computed in parallel, improving efficiency
4. **Risk reduction**: Even if some heads fail to learn, other heads can still provide useful information

### 7. 设计题：如果要设计一个用于机器翻译的Transformer模型，编码器和解码器各需要什么类型的注意力？
**If designing a Transformer model for machine translation, what types of attention are needed in encoder and decoder?**

**答案：**
**编码器 (Encoder)：**
- **自注意力 (Self-Attention)**：让源语言句子中的每个词都能关注到句子中的所有其他词，捕捉源语言的内部依赖关系

**解码器 (Decoder)：**
- **掩码自注意力 (Masked Self-Attention)**：让目标语言中的每个词只能关注之前的词，保证解码的自回归特性
- **交叉注意力 (Cross-Attention)**：让目标语言中的每个词都能关注源语言句子中的所有词，实现翻译对齐

## 应用实践题 (Application Practice Questions)

### 8. 分析题：为什么Transformer比RNN更适合处理长序列？
**Why is Transformer better than RNN for processing long sequences?**

**答案：**
1. **并行性**：Transformer可以并行处理序列中的所有位置，而RNN必须逐步处理
2. **长距离依赖**：注意力机制可以直接连接任意两个位置，而RNN需要通过多个时间步传递信息
3. **梯度传播**：Transformer中信息传递路径更短，避免了RNN的梯度消失问题
4. **计算效率**：对于长序列，Transformer的并行计算比RNN的串行计算更高效

1. **Parallelism**: Transformer can process all positions in parallel, while RNN must process sequentially
2. **Long-range dependencies**: Attention can directly connect any two positions, while RNN needs to pass information through multiple time steps  
3. **Gradient propagation**: Shorter information paths in Transformer avoid gradient vanishing problems in RNN
4. **Computational efficiency**: For long sequences, parallel computation in Transformer is more efficient than sequential computation in RNN

### 9. 创新题：如果要将注意力机制应用到图像处理中，你会如何设计？
**If applying attention mechanism to image processing, how would you design it?**

**答案思路：**
1. **图像分块 (Image Patching)**：将图像分割成小的patches，每个patch作为一个token
2. **位置编码 (Positional Encoding)**：为每个patch添加2D位置信息
3. **自注意力 (Self-Attention)**：让每个patch与所有其他patches计算相关性
4. **层次化设计 (Hierarchical Design)**：从低级特征到高级特征逐层抽象

这就是Vision Transformer (ViT)的核心思想。

This is the core idea of Vision Transformer (ViT).

### 10. 深度思考题：注意力机制的计算复杂度是O(n²)，对于处理超长序列有什么问题？如何解决？
**The computational complexity of attention is O(n²). What problems does this cause for very long sequences? How to solve it?**

**答案：**
**问题 (Problems)：**
1. **内存消耗**：注意力矩阵需要O(n²)的内存
2. **计算时间**：随序列长度平方增长
3. **实用性限制**：难以处理文档级别的长文本

**解决方案 (Solutions)：**
1. **稀疏注意力 (Sparse Attention)**：限制每个位置只关注部分位置
2. **局部注意力 (Local Attention)**：只在滑动窗口内计算注意力
3. **长范围注意力 (Longformer)**：结合局部和全局注意力
4. **线性注意力 (Linear Attention)**：使用核方法降低复杂度到O(n)
5. **分层注意力 (Hierarchical Attention)**：在不同层次上应用注意力

**Problems:**
1. **Memory consumption**: Attention matrix requires O(n²) memory
2. **Computation time**: Grows quadratically with sequence length  
3. **Practical limitations**: Difficult to handle document-level long texts

**Solutions:**
1. **Sparse Attention**: Limit each position to attend to only some positions
2. **Local Attention**: Compute attention only within sliding windows
3. **Longformer**: Combine local and global attention
4. **Linear Attention**: Use kernel methods to reduce complexity to O(n)
5. **Hierarchical Attention**: Apply attention at different levels 