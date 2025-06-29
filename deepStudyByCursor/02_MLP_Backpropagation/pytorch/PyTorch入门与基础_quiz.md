# PyTorch Introduction and Basics Quiz

PyTorch 入门与基础小测验

This quiz tests your understanding of the fundamental concepts in PyTorch, especially Tensors.

本小测验将测试你对 PyTorch 基本概念，特别是张量的理解。

## 1. Tensors: The Core Data Structure of PyTorch

### Question 1 (问题 1):

What are the two key advantages of PyTorch Tensors over NumPy arrays?

PyTorch 张量相对于 NumPy 数组的两个主要优势是什么？

**Answer (答案):**

<details>
<summary>Click to reveal answer</summary>

1.  **They can run on GPUs for accelerated computation. (它们可以在 GPU 上运行以加速计算。)**
2.  **They support automatic differentiation (`autograd`), which is crucial for training neural networks. (它们支持自动微分（`autograd`），这对于训练神经网络至关重要。)**

</details>

### Question 2 (问题 2):

Write Python code to create a 3x3 tensor filled with zeros using a PyTorch built-in function.

编写 Python 代码，使用 PyTorch 内置函数创建一个 3x3 的全零张量。

**Answer (答案):**

<details>
<summary>Click to reveal answer</summary>

```python
import torch
zeros_tensor = torch.zeros(3, 3)
print(zeros_tensor)
```

</details>

### Question 3 (问题 3):

Which tensor attribute would you use to check if a tensor is stored on the CPU or GPU?

你会使用哪个张量属性来检查张量是存储在 CPU 还是 GPU 上？

**Answer (答案):**

<details>
<summary>Click to reveal answer</summary>

`tensor.device`

</details>

### Question 4 (问题 4):

Given two tensors `a = torch.tensor([[1, 2], [3, 4]])` and `b = torch.tensor([[5, 6], [7, 8]])`, how would you perform matrix multiplication between `a` and `b` using an operator?

给定两个张量 `a = torch.tensor([[1, 2], [3, 4]])` 和 `b = torch.tensor([[5, 6], [7, 8]])`，你将如何使用运算符对 `a` 和 `b` 执行矩阵乘法？

**Answer (答案):**

<details>
<summary>Click to reveal answer</summary>

`a @ b`

</details>

### Question 5 (问题 5):

Explain in your own words, using the LEGO analogy, why Tensors are considered the fundamental building blocks and the language of all operations in PyTorch.

用你自己的话，结合乐高类比，解释为什么张量被认为是 PyTorch 中所有操作的基本构建块和语言。

**Answer (答案):**

<details>
<summary>Click to reveal answer</summary>

Just as every part of a LEGO castle, from the smallest brick to the largest wall, is made of LEGO bricks, in PyTorch, every piece of data, whether it's the input image, the model's parameters, or the final output, is represented as a Tensor. Tensors are the universal format for all data and computations. Similar to how you perform various operations (connecting, stacking) with LEGO bricks to build a complex structure, in PyTorch, you perform mathematical operations on Tensors to construct and train neural networks. They are the common language that allows different parts of the PyTorch framework to interact and compute.

就像乐高城堡的每一个部分，从最小的积木到最大的墙壁，都是由乐高积木构成一样，在 PyTorch 中，每一份数据，无论是输入的图像、模型的参数还是最终的输出，都表示为张量。张量是所有数据和计算的通用格式。类似于你通过对乐高积木进行各种操作（连接、堆叠）来构建复杂的结构，在 PyTorch 中，你对张量执行数学操作来构建和训练神经网络。它们是允许 PyTorch 框架不同部分进行交互和计算的通用语言。

</details> 