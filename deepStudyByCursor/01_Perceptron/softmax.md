# 深度学习基础知识测验：Softmax算法

## 第一章：Softmax算法基础

### 编程题

1.  **多分类数据准备函数：**
    请编写一个Python函数 `prepare_multiclass_data(num_samples, num_features, num_classes)`，该函数接收样本数量、特征数量和类别数量作为输入，并生成随机的输入特征 `X` 和对应的标签 `y`。

    ```python
    import torch

    def prepare_multiclass_data(num_samples, num_features, num_classes):
        X = torch.randn(num_samples, num_features)
        y = torch.randint(0, num_classes, (num_samples,))
        return X, y

    # 测试你的函数
    # num_samples = 100
    # num_features = 5
    # num_classes = 5
    # X_test, y_test = prepare_multiclass_data(num_samples, num_features, num_classes)
    # print(f"Generated X shape: {X_test.shape}, y shape: {y_test.shape}")
    # 期望输出类似: Generated X shape: (100, 5), y shape: (100,)
    ```

2.  **Softmax激活函数实现：**
    请编写一个Python函数 `softmax(z)`，该函数接收一个数值数组或列表 $z$ 作为输入（通常是线性模型的原始输出，称为 logits），并返回经过Softmax激活后的概率分布。Softmax函数可以表示为：
    $$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$
    其中 $z_i$ 是输入向量的第 $i$ 个元素，$K$ 是类别的总数。

    ```python
    import torch

    def softmax(z):
        # 你的代码
        pass

    # 测试你的函数
    # logits = torch.tensor([1.0, 2.0, 3.0])
    # probabilities = softmax(logits)
    # print(f"Softmax probabilities: {probabilities}")
    # print(f"Sum of probabilities: {torch.sum(probabilities)}") # 期望输出接近 1.0
    # # 期望输出类似: Softmax probabilities: [0.09003057 0.24472847 0.66524096], Sum of probabilities: 1.0
    ```

---

### 答案与解析

#### 编程题答案

1.  **多分类数据准备函数：**

    ```python
    import torch

    def prepare_multiclass_data(num_samples, num_features, num_classes):
        X = torch.randn(num_samples, num_features)
        y = torch.randint(0, num_classes, (num_samples,))
        return X, y
    ```

2.  **Softmax激活函数实现：**

    ```python
    import torch

    def softmax(z):
        exp_z = torch.exp(z)
        return exp_z / torch.sum(exp_z)
    ``` 