# RNN Mathematical Principles
# RNN数学原理

## 1. Introduction to Sequential Data
## 序列数据介绍

Sequential data is everywhere around us! Think about reading a sentence - each word depends on the previous words to make sense.

序列数据在我们周围无处不在！想象一下阅读一句话 - 每个词都依赖于前面的词才能理解。

**Examples / 示例:**
- Text: "I love deep learning" / 文本："我爱深度学习"
- Time series: [100, 102, 98, 105, 110] / 时间序列：股价数据

## 2. RNN Core Mathematical Formula
## RNN核心数学公式

At each time step t, an RNN computes:

在每个时间步t，RNN计算：

```
h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)
y_t = W_hy * h_t + b_y
```

Where / 其中:
- `h_t`: hidden state at time t / 时间t的隐藏状态
- `x_t`: input at time t / 时间t的输入
- `W_hh`: hidden-to-hidden weights / 隐藏到隐藏权重
- `W_xh`: input-to-hidden weights / 输入到隐藏权重

## 3. Step-by-Step Example
## 逐步示例

Processing "I love AI":

处理"我爱AI"：

```
Step 1: h_1 = tanh(W_hh * 0 + W_xh * "I" + b_h)
Step 2: h_2 = tanh(W_hh * h_1 + W_xh * "love" + b_h)  
Step 3: h_3 = tanh(W_hh * h_2 + W_xh * "AI" + b_h)
```

## 4. Backpropagation Through Time (BPTT)
## 时间反向传播

BPTT calculates gradients by unfolding the RNN across time steps. The gradient travels backward through time to update weights.

BPTT通过在时间步上展开RNN来计算梯度。梯度在时间中向后传播以更新权重。

```
∂L/∂W_hh = Σ(t=1 to T) ∂L_t/∂h_t * ∂h_t/∂W_hh
```

## 5. Vanishing Gradient Problem
## 梯度消失问题

The main challenge with RNNs! Gradients become exponentially smaller as they travel back through many time steps.

RNN的主要挑战！梯度在通过许多时间步向后传播时变得指数级地更小。

**Why it happens / 为什么发生:**
- Tanh derivative is ≤ 1
- Product of many small values → very small gradient
- 许多小值的乘积 → 非常小的梯度

**Solution Preview / 解决方案预览:**
- LSTM and GRU (covered in later chapters)
- Gradient clipping / 梯度裁剪

## 6. Sequence Types
## 序列类型

### One-to-Many (1:N) / 一对多
- Input: Single item / 输入：单个项目
- Output: Sequence / 输出：序列
- Example: Image captioning / 示例：图像标题

### Many-to-One (N:1) / 多对一
- Input: Sequence / 输入：序列  
- Output: Single item / 输出：单个项目
- Example: Sentiment analysis / 示例：情感分析

### Many-to-Many (N:M) / 多对多
- Input: Sequence / 输入：序列
- Output: Sequence / 输出：序列
- Example: Translation / 示例：翻译

## Summary
## 总结

RNNs process sequences using hidden states that carry information forward in time. The key mathematical insight is the recurrent connection that allows information to flow from one time step to the next. 