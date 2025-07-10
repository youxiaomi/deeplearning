# RNN Mathematical Principles Quiz
# RNN数学原理测试题

## Question 1: Multiple Choice / 选择题

**English**: What is the core RNN formula?
**中文**: RNN的核心公式是什么？

A) `h_t = sigmoid(W * x_t + b)`
B) `h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)`
C) `h_t = ReLU(W * x_t + b)`

**Answer / 答案**: B

**Explanation / 解释**: 
Option B shows the recurrent connection and input processing with tanh activation.
选项B显示了带有tanh激活的循环连接和输入处理。

---

## Question 2: Fill in the Blank / 填空题

**English**: The term `W_hh * h_{t-1}` represents the _______ from previous time steps.
**中文**: 项 `W_hh * h_{t-1}` 表示来自前一时间步的_______。

**Answer / 答案**: memory / 记忆

---

## Question 3: True or False / 判断题

**English**: RNNs suffer from vanishing gradients because tanh derivatives are ≤ 1.
**中文**: RNN遭受梯度消失是因为tanh导数≤1。

**Answer / 答案**: True / 正确

**Explanation / 解释**:
When many values ≤ 1 are multiplied together, the product becomes very small.
当许多≤1的值相乘时，乘积变得非常小。

---

## Question 4: Sequence Types / 序列类型

**English**: Match the applications with sequence types:
1. Sentiment analysis → ?
2. Translation → ?
3. Image captioning → ?

**中文**: 将应用与序列类型匹配：
1. 情感分析 → ?
2. 翻译 → ?
3. 图像标题 → ?

**Answer / 答案**:
1. Many-to-One (N:1) / 多对一
2. Many-to-Many (N:M) / 多对多  
3. One-to-Many (1:N) / 一对多

---

## Question 5: Math Calculation / 数学计算

**English**: Given W_hh=0.5, W_xh=0.8, b_h=0.1, h_0=0, x_1=1
Calculate h_1 = tanh(W_hh * h_0 + W_xh * x_1 + b_h)

**中文**: 给定参数计算h_1

**Answer / 答案**:
```
h_1 = tanh(0.5 * 0 + 0.8 * 1 + 0.1)
h_1 = tanh(0.9) ≈ 0.72
``` 