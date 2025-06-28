# PyTorch Introduction and Basics

PyTorch 入门与基础

## Introduction

简介

PyTorch is an open-source machine learning framework that accelerates the path from research prototyping to production deployment. It is widely used in deep learning for building and training neural networks due to its flexibility, Pythonic nature, and strong GPU acceleration support.

PyTorch是一个开源的机器学习框架，它加速了从研究原型到生产部署的进程。由于其灵活性、Pythonic特性以及强大的GPU加速支持，它被广泛用于深度学习中构建和训练神经网络。

This document will cover some fundamental aspects of PyTorch, starting with data preprocessing using `transforms`.

本文将涵盖PyTorch的一些基本方面，从使用`transforms`进行数据预处理开始。

## 1. Data Preprocessing with `torchvision.transforms`

使用 `torchvision.transforms` 进行数据预处理

In deep learning, data preprocessing is a crucial step to ensure that the data fed into the neural network is in the correct format and appropriate numerical range, which helps the model learn faster and better.

在深度学习中，数据预处理是一个非常关键的步骤，它能确保输入到神经网络的数据格式正确、数值范围合适，从而帮助模型更快更好地学习。

Let's look at a common example from data loading:

让我们看一个数据加载中的常见示例：

```python
import torchvision.transforms as transforms

# Example from data_loader.py
# from torchvision import datasets
# import os

# Assuming data_dir is defined, e.g., '../data'
# raw_dir = os.path.join(data_dir, 'raw')

transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to tensor (转换为张量)
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization (MNIST标准化)
])

# Example of applying transform during dataset loading
# train_dataset = datasets.MNIST(
#     root=raw_dir,
#     train=True,
#     download=True,
#     transform=transform
# )
```

Here, `transforms` from `torchvision` provides a suite of common image transformation operations for data preprocessing.

这里，来自`torchvision`的`transforms`提供了一系列常用的图像变换操作，用于数据预处理。

### 1.1 `transforms.Compose()`

**Purpose (用途):**
`transforms.Compose()`'s role is to sequentially combine multiple `transforms` operations. It acts like a "pipeline" or "assembly line" where you put a series of data processing steps, and the data goes through each step in order.

`transforms.Compose()` 的作用是将多个 `transforms` 操作按顺序组合在一起。它就像一个"流水线"或"管道"，你把一系列的数据处理步骤放进去，数据会依次经过这些步骤进行转换。

**Analogy to Real Life (生活中的类比):**
Imagine you're preparing ingredients for cooking. `transforms.Compose()` is like your recipe, which lists every step: first wash the vegetables, then chop them, and then marinate them. `Compose` ensures these steps are followed strictly in the order you set.

想象你在准备食材做饭。`transforms.Compose()` 就像你的一套食谱，里面列出了做菜的每一个步骤：先洗菜、再切菜、然后腌制。`Compose` 就确保这些步骤严格按照你设定的顺序进行。

### 1.2 `transforms.ToTensor()`

**Purpose (用途):**
`transforms.ToTensor()`'s main role is to convert PIL (Python Imaging Library) images or NumPy `ndarray` format data into PyTorch's `Tensor` format.

`transforms.ToTensor()` 的主要作用是将 PIL (Python Imaging Library) 图像或 NumPy `ndarray` 格式的数据转换为 PyTorch 的 `Tensor` 格式。

-   **Data Type Conversion (数据类型转换):** Images typically store pixel values as `uint8` (0-255) integers. `ToTensor()` scales these pixel values from the `[0, 255]` range to the `[0.0, 1.0]` floating-point range. This is because neural networks typically process floating-point inputs, and normalizing pixel values to `[0, 1]` helps with training stability.
    
    **数据类型转换：** 图像通常以 `uint8` (0-255) 的整数形式存储像素值。`ToTensor()` 会将这些像素值从 `[0, 255]` 的范围缩放到 `[0.0, 1.0]` 的浮点数范围。这是因为神经网络通常处理浮点数输入，并且将像素值归一化到 `[0, 1]` 有助于模型的训练稳定性。

-   **Dimension Order Adjustment (维度顺序调整):** In image processing, image dimensions are usually `(Height, Width, Channels)` (e.g., color image is `(H, W, 3)`). However, PyTorch neural networks typically expect image input dimensions to be `(Channels, Height, Width)`. `ToTensor()` automatically performs this dimension conversion.
    
    **维度顺序调整：** 在图像处理中，图像的维度通常是 `(Height, Width, Channels)`（例如，彩色图像是 `(H, W, 3)`）。而 PyTorch 神经网络期望的图像输入维度通常是 `(Channels, Height, Width)`。`ToTensor()` 会自动进行这个维度转换。
    
    *   For example, for a grayscale image like MNIST, the original might be `(28, 28)` or `(28, 28, 1)`, and after `ToTensor()` it will become `(1, 28, 28)`.
    *   例如，对于 MNIST 这样的灰度图像，原始可能是 `(28, 28)` 或 `(28, 28, 1)`，经过 `ToTensor()` 后会变成 `(1, 28, 28)`。

**Analogy to Real Life (生活中的类比):**
You have a physical picture (raw data). `ToTensor()` is like scanning this picture digitally and converting it into a unified format that computers can process more easily (e.g., converting it into binary data consisting of 0s and 1s, and adjusting the data storage method to be more suitable for computer memory reading). At the same time, it also converts the shades of color (0-255) uniformly into decimals between 0 and 1, making it easier for computers to calculate.

你有一张纸质的图片（原始数据）。`ToTensor()` 就像是你将这张图片进行数字化扫描，并将其转换成计算机更容易处理的统一格式（例如，转换为0和1组成的二进制数据，并调整数据的存储方式，使其更符合计算机内存读取的习惯）。同时，它还把颜色的深浅（0-255）统一换算成0到1之间的小数，让计算机更容易计算。

### 1.3 `transforms.Normalize((0.1307,), (0.3081,))`

**Purpose (用途):**
`transforms.Normalize()`'s role is to standardize the tensor image. Standardization is a very important preprocessing technique that adjusts data by subtracting the mean and dividing by the standard deviation.

`transforms.Normalize()` 的作用是对张量图像进行标准化处理。标准化是一种非常重要的预处理技术，它通过减去均值 (mean) 并除以标准差 (standard deviation) 来调整数据。

**Formula (公式):**
For each channel in the image, the new pixel value is `output[channel] = (input[channel] - mean[channel]) / std[channel]`.

对于图像中的每个通道，新的像素值 `output[channel] = (input[channel] - mean[channel]) / std[channel]`。

-   **Parameter Explanation (参数解释):**
    -   `(0.1307,)`: This is the **mean** of all image pixels in the MNIST dataset. Since MNIST is a grayscale image with only one channel, there is only one value.
    -   `(0.1307,)`：这是 MNIST 数据集所有图像像素的**均值**。因为 MNIST 是灰度图像，只有一个通道，所以只有一个值。
    -   `(0.3081,)`: This is the **standard deviation** of all image pixels in the MNIST dataset. Similarly, there is only one value.
    -   `(0.3081,)`：这是 MNIST 数据集所有图像像素的**标准差**。同样，也只有一个值。

**Why Normalize? (为什么要标准化?):

1.  **Faster Convergence (加速收敛):** Neural network optimization algorithms (like gradient descent) typically converge faster when input data is standardized. This is because standardized data often has a zero mean and unit variance distribution, making the loss function's "surface" smoother and gradient directions more stable.
    
    **加速收敛：** 神经网络的优化算法（如梯度下降）在输入数据经过标准化后，通常能更快地收敛。这是因为标准化后的数据通常呈零均值、单位方差分布，使得损失函数的"曲面"更平滑，梯度方向更稳定。

2.  **Prevent Gradient Vanishing/Exploding (避免梯度消失/爆炸):** Standardization helps maintain data within a reasonable numerical range, effectively mitigating the problem of gradients becoming very small (gradient vanishing) or very large (gradient exploding) during backpropagation in deep networks.
    
    **避免梯度消失/爆炸：** 标准化有助于将数据维持在一个合理的数值范围内，从而在深层网络中有效缓解梯度在反向传播过程中变得非常小（梯度消失）或非常大（梯度爆炸）的问题。

3.  **Improve Model Performance (提高模型性能):** Many activation functions (e.g., Sigmoid, Tanh) have the largest gradients when inputs are close to 0. Standardization can ensure more data falls into the sensitive regions of these functions, thereby improving the model's learning capability.
    
    **提高模型性能：** 许多激活函数（如 Sigmoid、Tanh）在输入接近0时梯度最大，标准化可以确保更多的数据落在这些函数的敏感区域，从而提高模型的学习能力。

4.  **Fair Treatment of Features (公平对待特征):** Although MNIST has only one channel, when dealing with multi-channel images (like RGB images), standardization ensures that pixel values across different channels are on the same scale, preventing certain channels from disproportionately influencing model training due to large numerical ranges.
    
    **公平对待特征：** 尽管 MNIST 只有一个通道，但在处理多通道图像（如 RGB 图像）时，标准化可以确保不同通道的像素值在相同的尺度上，避免某些通道因数值范围大而对模型训练产生不成比例的影响。

**Analogy to Real Life (生活中的类比):**
Imagine that in your class, some students' heights are recorded in centimeters, and some in meters, leading to inconsistent numerical ranges. `Normalize` is like uniformly processing all students' height data: first, calculate the average height of the class, then subtract the average height from each person's height, and then divide by the standard deviation of heights. This way, all height data is converted into a standard form centered at 0 with a more concentrated numerical range. This makes it easier and more accurate for the teacher to analyze the data, avoiding misjudging a student as "exceptionally tall" or "exceptionally short" due to numerical differences.

想象你班上有同学的身高数据，有的用厘米表示，有的用米表示，数值范围很不一致。`Normalize` 就像是把所有同学的身高数据统一处理：先计算出班级的平均身高，然后每个人的身高都减去平均身高，再除以身高的标准偏差。这样，所有人的身高数据就都转换成了一个以0为中心、数值范围更集中的标准形式。这样一来，老师在分析这些数据时会更容易、更准确，不会因为数值大小差异而误判某个同学"特别高"或"特别矮"。

## 2. Data Loading with `DataLoader`

使用 `DataLoader` 加载数据

After preprocessing your data using `transforms`, the next step is to efficiently load it into your model for training. PyTorch provides `torch.utils.data.DataLoader` for this purpose. It is a core component that wraps your dataset and provides an iterable over it, making it easy to access data in mini-batches.

使用 `transforms` 对数据进行预处理后，下一步就是高效地将数据加载到模型中进行训练。PyTorch 为此提供了 `torch.utils.data.DataLoader`。它是一个核心组件，用于封装你的数据集并提供一个可迭代接口，使你能够方便地按小批次访问数据。

### 2.1 Core Purpose of `DataLoader` ( `DataLoader` 的核心作用)

`DataLoader` addresses several key data loading challenges in deep learning:

`DataLoader` 解决了深度学习中数据加载的几个关键挑战：

-   **Batching (批量处理):** Neural networks typically process data in small batches rather than one sample at a time. `DataLoader` automatically groups samples into specified `batch_size` chunks.
    -   **批量处理：** 神经网络通常一次处理一小批数据，而不是一次处理一个样本。`DataLoader` 会自动将样本分组为指定 `batch_size` 大小的块。
-   **Shuffling (数据打乱):** To prevent the model from learning the order of samples and to improve generalization, `DataLoader` can randomly shuffle the data at the beginning of each epoch.
    -   **数据打乱：** 为了避免模型学习到数据的顺序和模式并提高泛化能力，`DataLoader` 可以在每个 epoch 开始时随机打乱数据。
-   **Multi-process Data Loading (多进程数据加载):** For large datasets, `DataLoader` can leverage multiple CPU cores to load data in parallel, significantly speeding up data fetching and preventing I/O from becoming a bottleneck during training.
    -   **多进程数据加载：** 对于大型数据集，`DataLoader` 可以利用多个 CPU 核并行加载数据，显著加快数据获取速度，防止 I/O 成为训练过程中的瓶颈。
-   **Memory Management (内存管理):** It loads data on demand (lazy loading) rather than loading the entire dataset into memory at once, which is crucial for handling large datasets that might not fit into RAM.
    -   **内存管理：** 它按需加载数据（惰性加载），而不是一次性将整个数据集加载到内存中，这对于处理可能无法完全载入内存的大型数据集至关重要。

### 2.2 Key Parameters of `DataLoader` ( `DataLoader` 的重要参数)

Let's review the parameters commonly used when initializing a `DataLoader`:

让我们回顾一下初始化 `DataLoader` 时常用的参数：

```python
# Example from data_loader.py for validation loader
# data_loader.py 中验证加载器的示例
val_loader = DataLoader(
    val_subset,           # The dataset (数据集)
    batch_size=64,        # Number of samples per batch (每个批次的样本数量)
    shuffle=False,        # Whether to shuffle data (是否打乱数据)
    num_workers=2         # Number of subprocesses for data loading (用于数据加载的子进程数量)
)
```

-   **`dataset`**: (Required) An instance of `torch.utils.data.Dataset` (or any object implementing `__len__` and `__getitem__`). This object provides individual samples.
    -   **`dataset`**：（必需）`torch.utils.data.Dataset` 的实例（或任何实现了 `__len__` 和 `__getitem__` 方法的对象）。此对象提供单个样本。
-   **`batch_size`**: (Optional, default=1) How many samples per batch to load. Larger batch sizes can lead to faster training but require more memory.
    -   **`batch_size`**：（可选，默认值=1）每个批次要加载的样本数量。更大的批次大小可以加快训练速度，但需要更多内存。
-   **`shuffle`**: (Optional, default=False) Set to `True` to have the data reshuffled at every epoch. This is typically `True` for training data and `False` for validation/test data.
    -   **`shuffle`**：（可选，默认值=False）设置为 `True` 可在每个 epoch 打乱数据。训练数据通常设置为 `True`，验证/测试数据通常设置为 `False`。
-   **`num_workers`**: (Optional, default=0) How many subprocesses to use for data loading. `0` means that the data will be loaded in the main process. Positive values indicate multi-process loading.
    -   **`num_workers`**：（可选，默认值=0）用于数据加载的子进程数量。`0` 表示数据将在主进程中加载。正值表示多进程加载。
-   **`drop_last`**: (Optional, default=False) Set to `True` to drop the last incomplete batch if the dataset size is not divisible by the batch size. This can be useful for ensuring all batches have the same shape for operations like Batch Normalization.
    -   **`drop_last`**：（可选，默认值=False）如果数据集大小不能被批次大小整除，则设置为 `True` 以丢弃最后一个不完整的批次。这对于确保所有批次具有相同的形状（例如用于批归一化）很有用。
-   **`pin_memory`**: (Optional, default=False) Set to `True` to load data into CUDA pinned memory. This can speed up data transfer from CPU to GPU, especially for large datasets and when `num_workers > 0`.
    -   **`pin_memory`**：（可选，默认值=False）设置为 `True` 可将数据加载到 CUDA 锁页内存中。这可以加速从 CPU 到 GPU 的数据传输，特别是对于大型数据集和当 `num_workers > 0` 时。

### 2.3 `DataLoader` Workflow ( `DataLoader` 的工作流程)

The typical workflow when using `DataLoader` is as follows:

使用 `DataLoader` 的典型工作流程如下：

1.  **Instantiate `Dataset` (实例化 `Dataset`):** Create an instance of your `Dataset` class, which defines how to get a single sample and its label.
    -   **实例化 `Dataset`：** 创建你的 `Dataset` 类的实例，该类定义了如何获取单个样本及其标签。
2.  **Instantiate `DataLoader` (实例化 `DataLoader`):** Create a `DataLoader` instance, passing your `Dataset` object and configurations like `batch_size`, `shuffle`, `num_workers`, etc.
    -   **实例化 `DataLoader`：** 创建一个 `DataLoader` 实例，传入你的 `Dataset` 对象以及 `batch_size`、`shuffle`、`num_workers` 等配置。
3.  **Iterate in Training Loop (在训练循环中迭代):** In your training loop, you iterate over the `DataLoader` to get mini-batches of data:
    -   **在训练循环中迭代：** 在你的训练循环中，你将遍历 `DataLoader` 以获取数据小批次：
    ```python
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            # data: Tensor of input features for the current batch
            # target: Tensor of labels for the current batch
            # ... perform forward pass, calculate loss, backpropagation, update parameters ...
            # data：当前批次的输入特征张量
            # target：当前批次的标签张量
            # ... 执行前向传播、计算损失、反向传播、更新参数 ...
    ```

### 2.4 Why is `DataLoader` Crucial? ( `DataLoader` 为何如此重要？)

`DataLoader` is a critical abstraction in PyTorch for efficient and flexible data loading. It significantly streamlines the deep learning model training process. Without `DataLoader`, you would have to manually write extensive code to handle batching, shuffling, parallel loading, and memory management, which would greatly increase development complexity and reduce efficiency.

`DataLoader` 是 PyTorch 中实现高效和灵活数据加载的关键抽象。它极大地简化了深度学习模型的训练过程。没有 `DataLoader`，你将不得不手动编写大量代码来处理批处理、打乱、并行加载和内存管理，这将极大地增加开发复杂度和降低效率。

**Analogy to Real Life (生活中的类比):**
Imagine you are a pizza chef needing to prepare pizzas for many customers.

想象你是一名披萨厨师，需要为大量顾客准备披萨。

-   **Dataset (数据集):** The warehouse is full of various ingredients (each ingredient is a data sample).
    -   **数据集：** 仓库里堆满了各种各样的食材（每个食材都是一个数据样本）。
-   **`DataLoader`**: This is like the pizza shop's **automated ingredient distribution system**.
    -   **`DataLoader`**：这就像是披萨店的**自动化配料分发系统**。
    -   `batch_size`: You set how many pizzas to make per batch.
    -   `batch_size`：你设定每批次做多少个披萨。
    -   `shuffle`: Do you randomly grab ingredients from the warehouse for different flavored pizzas, or do you follow a fixed order?
    -   `shuffle`：每批次是随机从仓库里抓取食材来做不同口味的披萨，还是固定顺序做？
    -   `num_workers`: How many assistants (subprocesses) have you hired to help you simultaneously fetch and prepare ingredients from the warehouse (data preprocessing)? This allows you to make pizzas faster.
    -   **`num_workers`**：你雇佣了多少个助手（子进程）来帮你同时从仓库里拿取食材并切好（数据预处理），这样你就能更快地制作披萨了。

With this system, you don't need to manually go to the warehouse to fetch and prepare ingredients one by one, then make all pizzas at once. You just need to tell the system how many pizzas you need and how to prepare them, and the system will continuously provide you with prepared batches of ingredients, allowing you to focus on baking pizzas (model training).

有了这个系统，你不需要自己跑到仓库里一个一个拿食材、切食材，然后一次性把所有披萨都做好。你只需要告诉系统你需要多少个披萨、按什么方式做，系统就会源源不断地为你提供已经准备好的批次食材，让你专注于烤披萨（模型训练）。

## 3. Reproducibility: Random Seeds and `torch.Generator`

可重现性：随机种子与 `torch.Generator`

In deep learning, many operations involve randomness, such as data splitting, data shuffling, model weight initialization, and Dropout. To ensure that your experiments yield consistent results across multiple runs and can be verified by others, it's crucial to manage this randomness. This is achieved by setting a **random seed**.

在深度学习中，许多操作都涉及随机性，例如数据分割、数据打乱、模型权重初始化和Dropout。为了确保你的实验在多次运行中都能产生一致的结果，并能被他人验证，管理这种随机性至关重要。这通过设置**随机种子**来实现。

### 3.1 What is a Random Seed? (什么是随机种子?)

In computing, "random numbers" are typically **pseudo-random numbers**. This means they are not truly random but are generated by a deterministic algorithm. This algorithm requires an initial starting point, known as a "seed."

在计算机中，我们说的"随机数"实际上是**伪随机数**。这意味着它们并不是真正意义上的随机，而是通过一个确定性的算法生成的。这个算法需要一个起始点，也就是我们说的"种子" (seed)。

-   If the same seed is used repeatedly, the generated "random number" sequence will be identical each time.
    -   如果每次使用的种子都一样，那么生成的"随机数"序列也会完全一样。
-   If different seeds are used, the generated "random number" sequences will differ.
    -   如果种子不同，那么生成的"随机数"序列就会不同。

### 3.2 `torch.Generator().manual_seed(42)` Explanation

`torch.Generator().manual_seed(42)` 解释

Let's look at the example from `data_loader.py` for creating train/validation splits:

让我们看 `data_loader.py` 中用于创建训练/验证分割的示例：

```python
# Inside create_splits method in data_loader.py
# data_loader.py 中 create_splits 方法内部
train_subset, val_subset = random_split(
    train_dataset, 
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)  # For reproducibility (可重现性)
)
```

**Meaning (含义):**

1.  **`torch.Generator()`**: This creates a PyTorch random number generator object. Many PyTorch operations that involve randomness (e.g., data shuffling, weight initialization, data splitting) internally use a random number generator.
    
    **`torch.Generator()`**：这是一个 PyTorch 的随机数生成器对象。PyTorch 中的许多涉及到随机性的操作（例如数据打乱、权重初始化、数据分割等）都会在内部使用一个随机数生成器。

2.  **`.manual_seed(42)`**: This is a method of the `Generator` object that sets the random seed for this specific generator to `42`. The number `42` itself has no special meaning; you can use any integer as a seed, but a fixed number is conventionally chosen.
    
    **`.manual_seed(42)`**：这是 `Generator` 对象的一个方法，用于设置这个特定生成器的随机种子为 `42`。数字 `42` 本身没有特殊含义，你可以用任何整数作为种子，但习惯上会选择一个固定的数字。

### 3.3 Why Set a Random Seed? (为什么要设置随机种子?)

Setting a fixed random seed (like `42`) ensures that every time you run the code, all random operations that use this specific generator will produce the exact same sequence. This provides several benefits:

设置一个固定的随机种子（例如这里的 `42`）能确保每次运行代码时，所有依赖于 `torch.Generator()` 并且使用这个特定生成器的随机操作都会产生完全相同的序列。这样一来：

1.  **Reproducible Experimental Results (可重现的实验结果):** Regardless of how many times you run the code, as long as the code and input data remain unchanged, the model's training process and final performance should be consistent. This is vital for debugging and comparing models.
    
    **可重现的实验结果：** 无论你运行多少次，只要代码和输入数据不变，模型的训练过程和最终性能都应该是一致的。这对于调试和比较模型至关重要。

2.  **Facilitates Debugging and Comparison (便于调试和比较):** When you make changes to your code, if the model's performance changes, you can be confident that it's due to your code modifications, not just randomness.
    
    **便于调试和比较：** 当你对代码进行修改后，如果模型性能发生变化，你可以确信这是你的代码改动造成的，而不是随机性。

3.  **Scientific Verification (科学验证):** In scientific research, the reproducibility of experimental results is crucial. If you publish a model, others can verify your findings by obtaining the same results with the same code and data.
    
    **科学验证：** 在科学研究中，实验结果的可复现性是至关重要的。如果你发布了一个模型，别人可以通过相同的代码和数据得到相同的结果，从而验证你的发现。

**Analogy to Real Life (生活中的类比):**
Imagine you and your friends are playing a card game. Before each deal, you shuffle the cards randomly. If you don't use a seed, each shuffle is completely different. If you had a particularly good hand, you couldn't tell your friends, "Follow my shuffling method, and you'll get the same good hand," because each time it's a completely new random outcome. 

However, if you agree to use a "Shuffling Scheme 42" every time, which is a very complex, seemingly random sequence of steps, then because you all follow the same "Scheme 42," the cards will always be shuffled into the exact same order. This way, if you had a good hand, you could say, "Following 'Shuffling Scheme 42', your hand will be as good as mine!"

In PyTorch, `torch.Generator().manual_seed(42)` defines a specific "shuffling scheme," ensuring that your data splitting (or any other random operation) is "predictably random," thus guaranteeing the reproducibility of your experiments.

想象你和你的朋友在玩一个纸牌游戏。每次发牌前，你们都随机洗牌。如果你没有使用种子，每次洗牌的结果都完全不同。如果你发现这局牌你的手气特别好，你无法告诉朋友"按照我的方法洗牌，你也能有一样的好牌"，因为每次都是全新的随机。

然而，如果你们约定每次都用"洗牌方案42"来洗牌，这个方案是一个非常复杂的、看起来像随机的步骤序列。那么，因为你们都遵循同样的"方案42"，所以每次洗出来的牌序都会一模一样。这样，如果你这局牌手气好，你就可以说："按照'洗牌方案42'来，你的牌也会跟我一样好！"

在 PyTorch 中，`torch.Generator().manual_seed(42)` 就是定义了一个特定的"洗牌方案"，确保你的数据分割（或任何其他随机操作）是"可预测的随机"，从而保证了实验的可重现性。

### 3.4 Operations with Randomness in Deep Learning

深度学习中存在随机性的操作

To further understand the importance of random seeds, let's enumerate the common operations in deep learning that inherently involve randomness. Managing these sources of randomness is key to achieving reproducible results.

为了进一步理解随机种子的重要性，让我们列举深度学习中固有地涉及随机性的常见操作。管理这些随机性来源是实现可重现结果的关键。

1.  **Model Weight Initialization (模型权重初始化):**
    *   **Description (描述):** When you create a neural network, its weights and biases are typically initialized to small random values. This is crucial for breaking symmetry and allowing different neurons to learn different features.
    *   **重要性 (Importance):** 如果每次训练都使用不同的随机初始化，模型的起始点就不同，可能导致训练路径和最终性能的差异。

2.  **Data Shuffling (数据打乱):**
    *   **Description (描述):** During training, it's standard practice to shuffle the training data at the beginning of each epoch (或每个 batch). This prevents the model from learning the order of samples and helps it generalize better.
    *   **重要性 (Importance):** 不同的打乱顺序会影响模型在每个训练步骤中看到的批次数据，从而影响梯度计算和参数更新的轨迹。

3.  **Data Splitting (数据分割):**
    *   **Description (描述):** When you divide your dataset into training, validation, and test sets (e.g., using `torch.utils.data.random_split` or `sklearn.model_selection.train_test_split`), the selection of samples for each split is often random.
    *   **重要性 (Importance):** 如果每次分割方式不同，模型训练和评估的数据集组合就不同，直接影响模型的训练效果和评估指标的稳定性。

4.  **Dropout (随机失活):**
    *   **Description (描述):** Dropout is a regularization technique where, during training, a random subset of neurons are temporarily "dropped out" (set to zero) from the network layer. This prevents overfitting by forcing the network to learn more robust features.
    *   **重要性 (Importance):** 每次前向传播时随机关闭的神经元是不同的，引入了大量的随机性，对训练过程有显著影响。

5.  **Batch Normalization (批归一化) (在训练模式下):**
    *   **Description (描述):** 虽然批归一化本身是确定性的操作，但在训练模式下，它会计算当前批次的均值和方差来对数据进行归一化。如果批次的组成是随机的（由于数据打乱），那么计算出的均值和方差也会有微小的随机性。
    *   **重要性 (Importance):** 这种随机性虽然较小，但在某些情况下也可能对可重现性产生影响，尤其是在小批量训练或分布式训练中。

6.  **Optimization Algorithms (优化算法) (特别是带有随机性的):**
    *   **Description (描述):** 像随机梯度下降 (SGD) 这样的优化器，它的 "随机性" 主要来源于每次迭代时选择的 "mini-batch" 数据是随机的。虽然算法本身是确定性的，但数据选择的随机性会影响梯度计算。
    *   **重要性 (Importance):** 随机性体现在每次参数更新的方向会受到当前随机选取的 mini-batch 的影响。

7.  **Data Augmentation (数据增强):**
    *   **Description (描述):** 为了增加训练数据的多样性并提高模型的泛化能力，我们常常对图像进行随机变换，如随机裁剪、随机翻转、随机旋转、颜色抖动等。
    *   **重要性 (Importance):** 这些随机变换会生成略有不同的训练样本，如果没有固定种子，每次运行都会有不同的增强数据。

By ensuring that the random seed is set correctly for all relevant libraries (e.g., PyTorch, NumPy, Python's `random` module), you can significantly improve the reproducibility of your deep learning experiments. Remember to set the seed at the very beginning of your script, before any operations that might introduce randomness.

通过确保为所有相关库（例如 PyTorch、NumPy、Python 的 `random` 模块）正确设置随机种子，你可以显著提高深度学习实验的可重现性。请记住在脚本的最开始，在任何可能引入随机性的操作之前设置种子。

## 4. Summary

总结

`transforms.Compose` combined with `transforms.ToTensor()` and `transforms.Normalize()` provides a standard and efficient image preprocessing pipeline:

`transforms.Compose` 结合 `transforms.ToTensor()` 和 `transforms.Normalize()` 提供了一个标准且高效的图像预处理流程：

1.  Converts raw images into PyTorch tensors. (将原始图像转换为 PyTorch 张量。)
2.  Scales pixel values from `[0, 255]` to `[0.0, 1.0]`. (将像素值从 `[0, 255]` 缩放到 `[0.0, 1.0]`。)
3.  Adjusts tensor dimension order to fit PyTorch models. (调整张量的维度顺序以适应 PyTorch 模型。)
4.  Standardizes pixel values based on the dataset's statistical properties (mean and standard deviation) to approximately zero mean and unit variance, thereby optimizing the neural network training process. (根据数据集的统计特性（均值和标准差）对像素值进行标准化，使其均值为0、标准差为1（近似），从而优化神经网络的训练过程。)

These transformation steps are fundamental to processing image data in deep learning, ensuring that the data is presented in the most suitable way for model learning. 