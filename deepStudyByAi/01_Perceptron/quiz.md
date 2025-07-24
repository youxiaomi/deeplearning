# 深度学习基础知识测验：线性模型与感知机

## 第一章：线性模型与感知机基础

### 选择题

1.  以下哪种问题类型最适合使用线性回归模型来解决？
    A. 判断一张图片是猫还是狗。
    B. 预测一套房屋的连续价格。
    C. 将电子邮件分为垃圾邮件或非垃圾邮件。
    D. 识别语音指令。

2.  在二分类问题中，线性分类模型通常会使用哪种激活函数将输出转换为概率？
    A. ReLU
    B. Tanh
    C. Sigmoid
    D. None (不需要激活函数)

3.  对于回归问题，衡量模型预测值与真实值之间差距的常用损失函数是？
    A. 二元交叉熵 (Binary Cross-Entropy)
    B. 均方误差 (Mean Squared Error)
    C. 交叉熵 (Cross-Entropy)
    D. Softmax

4.  对于分类问题，尤其是输出为概率值时，为什么二元交叉熵损失函数通常比均方误差更合适？
    A. 二元交叉熵计算速度更快。
    B. 二元交叉熵能更好地惩罚模型对正确类别给出低概率的预测。
    C. 均方误差不能处理概率值。
    D. 二元交叉熵只适用于二分类问题。

5.  一个人工神经元（感知机）的核心组成部分不包括以下哪项？
    A. 输入
    B. 权重
    C. 偏置
    D. 循环层

6.  激活函数的主要目的是什么？
    A. 增加模型的训练速度。
    B. 减少模型参数数量。
    C. 引入非线性，使神经网络能够学习复杂的模式。
    D. 防止模型过拟合。

7.  以下关于ReLU激活函数的描述，哪项是错误的？
    A. 它可以有效缓解梯度消失问题。
    B. 计算效率高。
    C. 当输入为负数时，其梯度为0，可能导致"死亡ReLU"问题。
    D. 它的输出范围在 (0, 1) 之间。


8.  感知机的一个主要局限性是它无法解决什么类型的问题？
    A. 线性可分问题。
    B. 包含大量输入特征的问题。
    C. 非线性可分问题（如异或问题）。
    D. 回归问题。

### 填空题

1.  在线性回归模型中，当只有一个输入特征 $x$ 时，模型可以表示为 $y = wx + b$，其中 $w$ 是________， $b$ 是________。
2.  在二分类问题中，线性模型通过一条或一个________来分隔不同类别的样本。
3.  Sigmoid 激活函数的输出范围是 (________, ________)。
4.  ReLU 激活函数在输入 $z > 0$ 时，其输出为 $z$，在输入 $z \le 0$ 时，其输出为________。
5.  在感知机的训练过程中，调整权重和偏置的步长由________参数控制。

### 计算题

1.  **线性回归预测：**
    假设一个简单的线性回归模型为：`价格 = 2.5 * 面积 + 50`。
    如果一套房屋的面积是 `80` 平米，请计算模型的预测价格。

2.  **均方误差 (MSE) 计算：**
    假设真实价格是 `250` 万，模型预测价格是 `230` 万。请计算单个样本的均方误差。

3.  **Sigmoid 函数计算：**
    如果线性模型的输出 $z = 0$，请计算经过 Sigmoid 函数激活后的输出 $\sigma(z)$。

### 编程题

1.  **实现线性回归预测函数：**
    请编写一个Python函数 `linear_regression_predict(area, weight, bias)`，该函数接收房屋面积、权重和偏置作为输入，并返回预测价格。

    ```python
    def linear_regression_predict(area, weight, bias):
        # 你的代码
        pass

    # 测试你的函数
    # print(linear_regression_predict(80, 2.5, 50)) # 期望输出 250.0
    ```

2.  **实现Sigmoid激活函数：**
    请编写一个Python函数 `sigmoid(z)`，该函数接收一个数值 $z$ 作为输入，并返回其Sigmoid激活后的结果。

    ```python
    import math

    def sigmoid(z):
        # 你的代码
        pass

    # 测试你的函数
    # print(sigmoid(0)) # 期望输出 0.5
    # print(sigmoid(2.5)) # 期望输出 0.924 (约)
    ```

3.  **多分类数据准备函数：**

    ```python
    import numpy as np

    def prepare_multiclass_data(num_samples, num_features, num_classes):
        X = np.random.randn(num_samples, num_features)
        y = np.random.randint(0, num_classes, num_samples)
        return X, y

    # 测试你的函数
    # X_test, y_test = prepare_multiclass_data(100, 4, 3)
    # print(f"Generated X shape: {X_test.shape}, y shape: {y_test.shape}")
    # 期望输出类似: Generated X shape: (100, 4), y shape: (100,)
    ```

---

### 答案与解析

#### 选择题答案

1.  **B. 预测一套房屋的连续价格。**
    *   **解析：** 线性回归用于预测连续的数值输出。A、C、D 都是分类问题。

2.  **C. Sigmoid**
    *   **解析：** Sigmoid 函数能将任意实数映射到 (0, 1) 区间，非常适合表示二分类的概率。ReLU 主要用于隐藏层引入非线性，Tanh 也可用于隐藏层，但不直接输出概率。

3.  **B. 均方误差 (Mean Squared Error)**
    *   **解析：** MSE 是回归问题中最常用的损失函数，它通过计算预测误差的平方和来衡量模型的表现。二元交叉熵和交叉熵用于分类问题。

4.  **B. 二元交叉熵能更好地惩罚模型对正确类别给出低概率的预测。**
    *   **解析：** 交叉熵在模型预测概率与真实标签不一致时会给出更大的惩罚，尤其是在预测概率很低但真实标签是1的情况下。MSE 对误差的惩罚是线性的，不适用于概率分布的衡量。

5.  **D. 循环层**
    *   **解析：** 循环层是循环神经网络 (RNN) 的概念，不属于单个神经元（感知机）的基本组成部分。神经元包含输入、权重和偏置，并通过激活函数产生输出。

6.  **C. 引入非线性，使神经网络能够学习复杂的模式。**
    *   **解析：** 如果没有激活函数，多层神经网络将退化为单层线性模型，无法处理非线性关系。激活函数赋予了神经网络学习复杂模式的能力。

7.  **D. 它的输出范围在 (0, 1) 之间。**
    *   **解析：** ReLU 函数的输出范围是 $[0, +\infty)$，当输入为负数时输出为0，当输入为正数时输出为输入本身。输出范围在 (0, 1) 之间的是Sigmoid函数。

8.  **C. 非线性可分问题（如异或问题）。**
    *   **解析：** 感知机只能找到线性决策边界来分类数据，因此无法解决无法通过一条直线（或超平面）分隔开的非线性可分问题，例如经典的异或问题。

#### 填空题答案

1.  权重，偏置
2.  决策边界 (或：直线、平面)
3.  0, 1
4.  0
5.  学习率

#### 计算题答案

1.  **线性回归预测：**
    `预测价格 = 2.5 * 80 + 50 = 200 + 50 = 250` 万。

2.  **均方误差 (MSE) 计算：**
    `MSE = (250 - 230)^2 = 20^2 = 400`。

3.  **Sigmoid 函数计算：**
    当 $z=0$ 时，$\sigma(0) = \frac{1}{1 + e^{-0}} = \frac{1}{1 + 1} = \frac{1}{2} = 0.5$。

#### 编程题答案

1.  **实现线性回归预测函数：**

    ```python
    def linear_regression_predict(area, weight, bias):
        return weight * area + bias

    # 测试
    # print(linear_regression_predict(80, 2.5, 50))
    ```

2.  **实现Sigmoid激活函数：**

    ```python
    import math

    def sigmoid(z):
        return 1 / (1 + math.exp(-z))

    # 测试
    # print(sigmoid(0))
    # print(sigmoid(2.5))
    ```

3.  **多分类数据准备函数：**

    ```python
    import numpy as np

    def prepare_multiclass_data(num_samples, num_features, num_classes):
        X = np.random.randn(num_samples, num_features)
        y = np.random.randint(0, num_classes, num_samples)
        return X, y

    # 测试你的函数
    # X_test, y_test = prepare_multiclass_data(100, 4, 3)
    # print(f"Generated X shape: {X_test.shape}, y shape: {y_test.shape}")
    # 期望输出类似: Generated X shape: (100, 4), y shape: (100,)
    ```

# Quiz: Matrix Operations and Model Evaluation

小测验：矩阵运算与模型评估

Welcome to the quiz! Test your understanding of matrix operations and the confusion matrix.

欢迎来到小测验！测试你对矩阵运算和混淆矩阵的理解。

---

## Part 1: Matrix Operations (矩阵运算部分)

### Question 1 (问题 1)

Given two matrices $\mathbf{A}$ and $\mathbf{B}$:

给出两个矩阵 $\mathbf{A}$ 和 $\mathbf{B}$：

$$\mathbf{A} = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}, \quad \mathbf{B} = \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix}$$

Calculate their standard matrix product $\mathbf{C} = \mathbf{AB}$.

计算它们的标准矩阵乘积 $\mathbf{C} = \mathbf{AB}$。

**Answer (答案):**

$\mathbf{C} = \begin{pmatrix} 1 \cdot 5 + 2 \cdot 7 & 1 \cdot 6 + 2 \cdot 8 \\ 3 \cdot 5 + 4 \cdot 7 & 3 \cdot 6 + 4 \cdot 8 \end{pmatrix} = \begin{pmatrix} 5 + 14 & 6 + 16 \\ 15 + 28 & 18 + 32 \end{pmatrix} = \begin{pmatrix} 19 & 22 \\ 43 & 50 \end{pmatrix}$

---

### Question 2 (问题 2)

What is the transpose of matrix $\mathbf{D}$?

矩阵 $\mathbf{D}$ 的转置是什么？

$$\mathbf{D} = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{pmatrix}$$

**Answer (答案):**

$$\mathbf{D}^T = \begin{pmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{pmatrix}$$

---

### Question 3 (问题 3)

In deep learning, element-wise multiplication (Hadamard product) is often used for what purpose during backpropagation?

在深度学习中，逐元素乘法（哈达玛积）在反向传播期间通常用于什么目的？

**Answer (答案):**

It's used to apply activation function derivatives to the error signal, like in the formula $\delta = \nabla_a C \odot f'(z)$. This scales the error signal based on the slope of the activation function at that point.

它用于将激活函数导数应用于误差信号，例如公式 $\delta = \nabla_a C \odot f'(z)$ 中。这根据激活函数在该点的斜率来调整误差信号。

---

## Part 2: Model Evaluation (模型评估部分)

### Question 4 (问题 4)

A classification model is evaluated, and its confusion matrix is given below. This is a binary classification problem where the positive class is "Spam" and the negative class is "Not Spam".

对一个分类模型进行了评估，其混淆矩阵如下。这是一个二分类问题，正类是"垃圾邮件"，负类是"非垃圾邮件"。

|                   | Predicted Spam | Predicted Not Spam |
| :---------------- | :------------- | :----------------- |
| **Actual Spam**   | 80             | 20                 |
| **Actual Not Spam** | 10             | 90                 |

Based on this matrix, how many emails were **actually spam** but were **predicted as not spam**? What is this metric called?

根据这个矩阵，有多少邮件**实际是垃圾邮件**但被**预测为非垃圾邮件**？这个指标叫什么？

**Answer (答案):**

20 emails were actually spam but predicted as not spam. This metric is called **False Negative (FN)** or **假阴性**。

---

### Question 5 (问题 5)

Why is a **normalized confusion matrix** particularly useful when dealing with **imbalanced datasets** in classification tasks? Explain with a simple example.

在分类任务中处理**不平衡数据集**时，为什么**归一化混淆矩阵**特别有用？请用一个简单的例子解释。

**Answer (答案):**

A normalized confusion matrix is useful for imbalanced datasets because it shows the *proportions* of correct/incorrect predictions for each class, rather than just raw counts. This prevents a model's good performance on a large majority class from masking poor performance on a small minority class.

**Example (示例):**
Suppose you have 100 images: 90 are of cats and 10 are of dogs. Your model predicts everything as "cat".
-   **Raw Confusion Matrix (原始混淆矩阵):**
    | True \\ Predicted | Cat | Dog |
    | :---------------- | :-- | :-- |
    | **Actual Cat**    | 90  | 0   |
    | **Actual Dog**    | 10  | 0   |
    Overall accuracy would be 90/100 = 90%, which looks good.

-   **Normalized (Row-wise) Confusion Matrix (归一化（按行）混淆矩阵):**
    | True \\ Predicted | Cat    | Dog    |
    | :---------------- | :----- | :----- |
    | **Actual Cat**    | 1.00   | 0.00   |
    | **Actual Dog**    | 1.00   | 0.00   |
    Here, you clearly see that for actual dogs, 100% were misclassified as cats (Recall for Dog is 0%). This reveals the model's failure on the minority class, which was hidden by the high overall accuracy in the raw matrix.

---

### Question 6 (问题 6)

In the context of a normalized confusion matrix (normalized by true class), what do the diagonal elements represent?

在归一化混淆矩阵（按真实类别归一化）的背景下，对角线上的元素代表什么？

**Answer (答案):**

When a confusion matrix is normalized by true class (row-wise), the diagonal elements represent the **Recall (召回率)** or **True Positive Rate (真阳性率)** for each class. It indicates the proportion of actual positive cases that were correctly identified. 