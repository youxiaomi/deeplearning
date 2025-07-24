# Quiz: Model Building and Saving

小测验：模型构建与保存

Welcome to the quiz! Test your understanding of model building and saving concepts.

欢迎来到小测验！测试你对模型构建和保存概念的理解。

---

## Part 1: Multiple Choice Questions (选择题)

### Question 1 (问题 1)

Which of the following is **NOT** a primary reason for saving a trained deep learning model?

以下哪项**不是**保存训练好的深度学习模型的主要原因？

A. Reusing the model for new predictions without retraining. (无需重新训练即可将模型用于新预测。)
B. Deploying the model into real-world applications. (将模型部署到实际应用中。)
C. Reducing the total number of parameters in the model. (减少模型中参数的总数。)
D. Sharing the trained model with others. (与他人分享训练好的模型。)

**Answer (答案):**

C. Reducing the total number of parameters in the model.

**Explanation (解析):** Saving a model is about persistence and reusability, not about changing the model's architecture or parameter count.

---

### Question 2 (问题 2)

In Python, which module is commonly used for serializing and de-serializing arbitrary Python objects, making it suitable for saving custom deep learning models implemented from scratch?

在Python中，哪个模块通常用于序列化和反序列化任意Python对象，使其适合保存从零开始实现的自定义深度学习模型？

A. `json`
B. `csv`
C. `pickle`
D. `xml`

**Answer (答案):**

C. `pickle`

**Explanation (解析):** `pickle` is specifically designed for Python object serialization. `json`, `csv`, and `xml` are text-based formats more suited for data exchange, not general Python object serialization.

---

### Question 3 (问题 3)

Which of the following is a **security concern** when loading a `.pkl` file from an untrusted source?

从不可信来源加载 `.pkl` 文件时，以下哪项是**安全问题**？

A. The `.pkl` file might be too large to load.
B. Loading the `.pkl` file could execute arbitrary malicious code.
C. The `.pkl` file might be incompatible with the current Python version.
D. The `.pkl` file might only save model weights, not the full model architecture.

**Answer (答案):**

B. Loading the `.pkl` file could execute arbitrary malicious code.

**Explanation (解析):** `pickle` is known to be insecure against maliciously constructed data. Never load `pickle` files from untrusted sources.

---

### Question 4 (问题 4)

What is the main benefit of saving model **checkpoints** during the training process?

在训练过程中保存模型**检查点**的主要好处是什么？

A. It makes the model train faster.
B. It allows for resuming training from a specific point if interrupted.
C. It automatically performs hyperparameter tuning.
D. It reduces the memory footprint of the model during training.

**Answer (答案):**

B. It allows for resuming training from a specific point if interrupted.

**Explanation (解析):** Checkpoints are crucial for training resilience, allowing you to pick up training from where you left off, saving time and computational resources.

---

## Part 2: Fill-in-the-Blanks / Short Answer Questions (填空/简答题)

### Question 5 (问题 5)

Complete the sentence: "Serializing a Python object means converting it into a ________ stream."

完成句子："序列化Python对象意味着将其转换为一个________流。"

**Answer (答案):**

Byte (字节)

---

### Question 6 (问题 6)

Besides `pickle`, name one alternative model saving format or tool commonly used in deep learning frameworks like TensorFlow/Keras or for cross-framework compatibility.

除了 `pickle`，请说出一种在TensorFlow/Keras等深度学习框架中或为了跨框架兼容性而常用的模型保存格式或工具。

**Answer (答案):**

HDF5 (`.h5`), TensorFlow SavedModel, ONNX (Open Neural Network Exchange).

---

### Question 7 (问题 7)

Explain in your own words the concept of "model deployment" and why saving a model is essential for it.

用你自己的话解释"模型部署"的概念，以及为什么保存模型对它至关重要。

**Answer (答案):**

**Model deployment (模型部署):** Refers to the process of integrating a trained machine learning model into an existing production environment or application so that it can receive new data and generate predictions or decisions in real-time or as needed. 

**Why saving is essential (为什么保存至关重要):** Saving a model is essential for deployment because it allows you to store the learned parameters (weights, biases) and the architecture of the trained model. Without saving, you would have to retrain the model every time you want to use it for predictions, which is impractical, time-consuming, and computationally expensive for production systems. A saved model is a portable artifact that can be loaded into the deployment environment to make predictions immediately.

---

## Part 3: Coding Question (编程题)

### Question 8 (问题 8)

Complete the Python code below to save and then load a simple list using the `pickle` module. Ensure the file is opened in binary write (`'wb'`) and binary read (`'rb'`) mode.

完成以下Python代码，使用 `pickle` 模块保存然后加载一个简单的列表。确保文件以二进制写入 (`'wb'`) 和二进制读取 (`'rb'`) 模式打开。

```python
import pickle
import os

my_list = [1, 2, {'a': 1, 'b': 2}, "hello", 3.14]
file_name = "my_data.pkl"

# --- Save the list --- (保存列表)
# Open the file in binary write mode
with open(file_name, 'wb') as f:
    # Use pickle.dump() to save the list to the file
    # YOUR CODE HERE
    pass

print(f"List saved to {file_name}")

# --- Load the list --- (加载列表)
loaded_list = None
# Open the file in binary read mode
with open(file_name, 'rb') as f:
    # Use pickle.load() to load the list from the file
    # YOUR CODE HERE
    pass

print(f"List loaded: {loaded_list}")

# Clean up the created file
# os.remove(file_name)
# print(f"Cleaned up {file_name}")
```

**Answer (答案):**

```python
import pickle
import os

my_list = [1, 2, {'a': 1, 'b': 2}, "hello", 3.14]
file_name = "my_data.pkl"

# --- Save the list ---
with open(file_name, 'wb') as f:
    pickle.dump(my_list, f)

print(f"List saved to {file_name}")

# --- Load the list ---
loaded_list = None
with open(file_name, 'rb') as f:
    loaded_list = pickle.load(f)

print(f"List loaded: {loaded_list}")

# Clean up the created file
# os.remove(file_name)
# print(f"Cleaned up {file_name}")
``` 