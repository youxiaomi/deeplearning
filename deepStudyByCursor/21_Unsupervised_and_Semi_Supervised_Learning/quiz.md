# Quiz: Unsupervised Learning and Semi-Supervised Learning
# 测试题：无监督学习与半监督学习

## 1. Multiple Choice Questions (选择题)

### Question 1
Which of the following best describes unsupervised learning?
以下哪项最好地描述了无监督学习？

A) Learning with complete labeled datasets
A) 使用完整标记数据集进行学习

B) Learning patterns in data without labeled examples
B) 在没有标记示例的情况下学习数据中的模式

C) Learning with a mix of labeled and unlabeled data
C) 使用标记和未标记数据的混合进行学习

D) Learning only from reinforcement signals
D) 仅从强化信号中学习

**Answer: B**
**答案：B**

**Explanation**: Unsupervised learning involves finding patterns, structures, or relationships in data without any labeled examples or target outputs. The algorithm must discover hidden patterns by itself.
**解释**：无监督学习涉及在没有任何标记示例或目标输出的情况下发现数据中的模式、结构或关系。算法必须自己发现隐藏的模式。

---

### Question 2
In K-means clustering, what does the parameter 'k' represent?
在K-means聚类中，参数'k'代表什么？

A) The number of features in the dataset
A) 数据集中特征的数量

B) The number of iterations to run
B) 运行的迭代次数

C) The number of clusters to form
C) 要形成的簇的数量

D) The learning rate
D) 学习率

**Answer: C**
**答案：C**

**Explanation**: The 'k' in K-means represents the number of clusters that the algorithm will attempt to create. This is a hyperparameter that needs to be specified before running the algorithm.
**解释**：K-means中的'k'代表算法将尝试创建的簇数。这是一个在运行算法之前需要指定的超参数。

---

### Question 3
What is the main goal of Principal Component Analysis (PCA)?
主成分分析(PCA)的主要目标是什么？

A) To increase the number of features
A) 增加特征的数量

B) To find the directions of maximum variance
B) 找到最大方差的方向

C) To classify data into categories
C) 将数据分类为类别

D) To generate new data samples
D) 生成新的数据样本

**Answer: B**
**答案：B**

**Explanation**: PCA aims to find the principal components (directions) that capture the maximum variance in the data, allowing for effective dimensionality reduction while preserving the most important information.
**解释**：PCA的目标是找到捕获数据中最大方差的主成分（方向），允许有效的降维同时保留最重要的信息。

---

### Question 4
In semi-supervised learning, what is the typical ratio of labeled to unlabeled data?
在半监督学习中，标记数据与未标记数据的典型比例是多少？

A) 50:50
B) 90:10 (labeled:unlabeled)
B) 90:10（标记:未标记）
C) 10:90 (labeled:unlabeled)
C) 10:90（标记:未标记）
D) 100:0

**Answer: C**
**答案：C**

**Explanation**: Semi-supervised learning typically involves a small amount of labeled data (often 10% or less) and a large amount of unlabeled data. This reflects real-world scenarios where labeling is expensive or time-consuming.
**解释**：半监督学习通常涉及少量标记数据（通常10%或更少）和大量未标记数据。这反映了标记昂贵或耗时的现实场景。

---

### Question 5
Which autoencoder component is responsible for creating the compressed representation?
自编码器的哪个组件负责创建压缩表示？

A) Decoder
A) 解码器

B) Encoder
B) 编码器

C) Loss function
C) 损失函数

D) Optimizer
D) 优化器

**Answer: B**
**答案：B**

**Explanation**: The encoder is responsible for compressing the input data into a lower-dimensional latent representation. The decoder then attempts to reconstruct the original input from this compressed representation.
**解释**：编码器负责将输入数据压缩成低维潜在表示。然后解码器尝试从这个压缩表示重建原始输入。

---

## 2. Fill in the Blanks (填空题)

### Question 6
The objective function for K-means clustering minimizes the _________ sum of squares, which measures the distance between data points and their assigned _________.
K-means聚类的目标函数最小化_________平方和，它测量数据点与其分配的_________之间的距离。

**Answer**: within-cluster, centroids
**答案**：簇内，质心

---

### Question 7
In a Variational Autoencoder (VAE), the latent space is structured as a _________ distribution, typically a _________ distribution, which allows for generation of new samples.
在变分自编码器(VAE)中，潜在空间被结构化为_________分布，通常是_________分布，这允许生成新样本。

**Answer**: probability, Gaussian
**答案**：概率，高斯

---

### Question 8
Self-training is a semi-supervised method where the model uses its own _________ predictions on unlabeled data to expand the training set, typically by selecting predictions above a certain _________ threshold.
自训练是一种半监督方法，模型使用其在未标记数据上的_________预测来扩展训练集，通常通过选择高于某个_________阈值的预测。

**Answer**: confident, confidence
**答案**：置信，置信度

---

## 3. Short Answer Questions (简答题)

### Question 9
Explain the difference between hierarchical clustering and K-means clustering in terms of their approach and outputs.
解释层次聚类和K-means聚类在方法和输出方面的区别。

**Answer**:
**答案**：

**K-means Clustering:**
**K-means聚类：**
- Requires pre-specification of the number of clusters (k)
- 需要预先指定簇的数量(k)
- Uses an iterative algorithm to minimize within-cluster sum of squares
- 使用迭代算法最小化簇内平方和
- Produces flat clustering (no hierarchy)
- 产生平面聚类（无层次结构）
- Suitable for spherical, well-separated clusters
- 适用于球形、分离良好的簇

**Hierarchical Clustering:**
**层次聚类：**
- Does not require pre-specification of cluster number
- 不需要预先指定簇数量
- Builds a tree-like structure (dendrogram) of clusters
- 构建簇的树状结构（树状图）
- Can be agglomerative (bottom-up) or divisive (top-down)
- 可以是凝聚式（自下而上）或分裂式（自上而下）
- Provides clustering at multiple resolutions
- 在多个分辨率级别提供聚类

---

### Question 10
What are the advantages of using semi-supervised learning over purely supervised or unsupervised approaches?
使用半监督学习相比纯监督或无监督方法有什么优势？

**Answer**:
**答案**：

**Advantages over Supervised Learning:**
**相比监督学习的优势：**
- Requires fewer labeled examples, reducing annotation costs
- 需要更少的标记示例，降低标注成本
- Can leverage large amounts of readily available unlabeled data
- 可以利用大量易于获得的未标记数据
- Often achieves better performance when labeled data is limited
- 当标记数据有限时通常能获得更好的性能

**Advantages over Unsupervised Learning:**
**相比无监督学习的优势：**
- Has access to some ground truth labels for guidance
- 有一些真实标签作为指导
- Can perform specific prediction tasks, not just pattern discovery
- 可以执行特定的预测任务，而不仅仅是模式发现
- Generally more interpretable and evaluable results
- 通常具有更可解释和可评估的结果
- Better suited for practical applications with specific objectives
- 更适合有特定目标的实际应用

---

## 4. Programming Questions (编程题)

### Question 11
Implement a simple K-means algorithm from scratch using PyTorch. Your implementation should include initialization, assignment, and update steps.
使用PyTorch从头实现一个简单的K-means算法。你的实现应该包括初始化、分配和更新步骤。

**Answer**:
**答案**：

```python
import torch
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, k=3, max_iters=100, random_state=42):
        self.k = k
        self.max_iters = max_iters
        torch.manual_seed(random_state)
    
    def fit(self, X):
        """
        拟合K-means算法到数据X
        Fit K-means algorithm to data X
        """
        n_samples, n_features = X.shape
        
        # 初始化质心 - Initialize centroids
        self.centroids = X[torch.randperm(n_samples)[:self.k]]
        
        for iteration in range(self.max_iters):
            # 分配步骤：计算每个点到质心的距离
            # Assignment step: compute distances to centroids
            distances = torch.cdist(X, self.centroids)  # [n_samples, k]
            assignments = torch.argmin(distances, dim=1)  # [n_samples]
            
            # 更新步骤：重新计算质心
            # Update step: recompute centroids
            new_centroids = torch.zeros_like(self.centroids)
            for i in range(self.k):
                mask = assignments == i
                if mask.sum() > 0:
                    new_centroids[i] = X[mask].mean(dim=0)
                else:
                    # 如果某个簇为空，保持原质心
                    # If cluster is empty, keep original centroid
                    new_centroids[i] = self.centroids[i]
            
            # 检查收敛性 - Check convergence
            if torch.allclose(self.centroids, new_centroids, atol=1e-4):
                print(f"Converged after {iteration + 1} iterations")
                break
                
            self.centroids = new_centroids
        
        self.labels_ = assignments
        return self
    
    def predict(self, X):
        """
        预测新数据点的簇分配
        Predict cluster assignments for new data points
        """
        distances = torch.cdist(X, self.centroids)
        return torch.argmin(distances, dim=1)
    
    def fit_predict(self, X):
        """
        拟合模型并返回簇分配
        Fit model and return cluster assignments
        """
        return self.fit(X).labels_

# 使用示例 - Usage example
if __name__ == "__main__":
    # 生成示例数据 - Generate sample data
    from sklearn.datasets import make_blobs
    
    X, true_labels = make_blobs(n_samples=300, centers=3, 
                               cluster_std=1.0, random_state=42)
    X = torch.tensor(X, dtype=torch.float32)
    
    # 应用K-means - Apply K-means
    kmeans = KMeans(k=3, random_state=42)
    predicted_labels = kmeans.fit_predict(X)
    
    print(f"Data shape: {X.shape}")
    print(f"Number of clusters: {kmeans.k}")
    print(f"Final centroids:\n{kmeans.centroids}")
    
    # 可视化结果 - Visualize results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=true_labels, alpha=0.7)
    plt.title("True Labels")
    
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=predicted_labels, alpha=0.7)
    plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], 
               c='red', marker='x', s=200, linewidths=3)
    plt.title("K-means Results")
    
    plt.tight_layout()
    plt.show()
```

---

### Question 12
Create a simple autoencoder for MNIST digit reconstruction. Include training loop and visualization of original vs. reconstructed images.
为MNIST数字重建创建一个简单的自编码器。包括训练循环和原始图像与重建图像的可视化。

**Answer**:
**答案**：

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128):
        super(SimpleAutoencoder, self).__init__()
        
        # 编码器 - Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim),
            nn.ReLU()
        )
        
        # 解码器 - Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()  # 输出范围[0,1]
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)

def train_autoencoder():
    # 数据准备 - Data preparation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # 展平为向量
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, transform=transform, download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, transform=transform, download=True
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=256, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=256, shuffle=False
    )
    
    # 模型初始化 - Model initialization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleAutoencoder(input_dim=784, hidden_dim=32).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练循环 - Training loop
    num_epochs = 20
    train_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            
            optimizer.zero_grad()
            reconstructed = model(data)
            loss = criterion(reconstructed, data)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    # 测试和可视化 - Testing and visualization
    model.eval()
    with torch.no_grad():
        # 获取一些测试样本 - Get some test samples
        test_iter = iter(test_loader)
        test_data, _ = next(test_iter)
        test_data = test_data.to(device)
        
        # 重建图像 - Reconstruct images
        reconstructed = model(test_data)
        
        # 可视化结果 - Visualize results
        fig, axes = plt.subplots(2, 10, figsize=(15, 3))
        
        for i in range(10):
            # 原始图像 - Original images
            axes[0, i].imshow(test_data[i].cpu().view(28, 28), cmap='gray')
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')
            
            # 重建图像 - Reconstructed images
            axes[1, i].imshow(reconstructed[i].cpu().view(28, 28), cmap='gray')
            axes[1, i].set_title('Reconstructed')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # 绘制训练损失 - Plot training loss
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses)
        plt.title('Training Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.grid(True)
        plt.show()
    
    return model

# 运行训练 - Run training
if __name__ == "__main__":
    trained_model = train_autoencoder()
```

---

### Question 13
Implement a self-training algorithm for semi-supervised learning. Use a base classifier and iteratively add high-confidence predictions to the training set.
实现半监督学习的自训练算法。使用基础分类器并迭代地将高置信度预测添加到训练集。

**Answer**:
**答案**：

```python
import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=2):
        super(SimpleClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

class SelfTrainingClassifier:
    def __init__(self, base_model, confidence_threshold=0.9, max_iterations=10):
        self.base_model = base_model
        self.confidence_threshold = confidence_threshold
        self.max_iterations = max_iterations
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_model.to(self.device)
        
    def train_model(self, X, y, epochs=50, lr=0.001):
        """
        训练基础模型
        Train the base model
        """
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.base_model.parameters(), lr=lr)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        
        self.base_model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.base_model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')
    
    def get_confident_predictions(self, X_unlabeled):
        """
        获取高置信度的预测
        Get high-confidence predictions
        """
        self.base_model.eval()
        X_tensor = torch.tensor(X_unlabeled, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            outputs = self.base_model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            max_probs, predicted_labels = torch.max(probabilities, dim=1)
            
            # 选择高置信度的预测
            confident_mask = max_probs >= self.confidence_threshold
            
            if confident_mask.sum() == 0:
                return None, None, None
            
            confident_X = X_unlabeled[confident_mask.cpu().numpy()]
            confident_y = predicted_labels[confident_mask].cpu().numpy()
            confident_probs = max_probs[confident_mask].cpu().numpy()
            
            return confident_X, confident_y, confident_probs
    
    def fit(self, X_labeled, y_labeled, X_unlabeled):
        """
        自训练主循环
        Main self-training loop
        """
        print("Starting self-training...")
        
        # 记录训练过程 - Record training process
        iteration_stats = []
        
        # 初始在标记数据上训练
        print(f"Initial training on {len(X_labeled)} labeled samples")
        self.train_model(X_labeled, y_labeled)
        
        # 复制数据以避免修改原始数据
        current_X_labeled = X_labeled.copy()
        current_y_labeled = y_labeled.copy()
        current_X_unlabeled = X_unlabeled.copy()
        
        for iteration in range(self.max_iterations):
            print(f"\n--- Iteration {iteration + 1} ---")
            
            # 获取高置信度预测
            confident_X, confident_y, confident_probs = self.get_confident_predictions(
                current_X_unlabeled
            )
            
            if confident_X is None:
                print("No confident predictions found. Stopping.")
                break
            
            print(f"Found {len(confident_X)} confident predictions")
            print(f"Average confidence: {confident_probs.mean():.3f}")
            
            # 添加到标记数据集
            current_X_labeled = np.vstack([current_X_labeled, confident_X])
            current_y_labeled = np.hstack([current_y_labeled, confident_y])
            
            # 从未标记数据集中移除
            confident_indices = []
            for i, x_conf in enumerate(confident_X):
                for j, x_unlab in enumerate(current_X_unlabeled):
                    if np.allclose(x_conf, x_unlab):
                        confident_indices.append(j)
                        break
            
            remaining_mask = np.ones(len(current_X_unlabeled), dtype=bool)
            remaining_mask[confident_indices] = False
            current_X_unlabeled = current_X_unlabeled[remaining_mask]
            
            # 重新训练模型
            print(f"Retraining on {len(current_X_labeled)} samples")
            self.train_model(current_X_labeled, current_y_labeled)
            
            # 记录统计信息
            iteration_stats.append({
                'iteration': iteration + 1,
                'labeled_samples': len(current_X_labeled),
                'unlabeled_samples': len(current_X_unlabeled),
                'added_samples': len(confident_X),
                'avg_confidence': confident_probs.mean()
            })
            
            print(f"Labeled samples: {len(current_X_labeled)}")
            print(f"Remaining unlabeled: {len(current_X_unlabeled)}")
            
            if len(current_X_unlabeled) == 0:
                print("No more unlabeled data. Stopping.")
                break
        
        self.iteration_stats = iteration_stats
        return self
    
    def predict(self, X):
        """
        预测新样本
        Predict new samples
        """
        self.base_model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            outputs = self.base_model(X_tensor)
            _, predicted = torch.max(outputs, 1)
            return predicted.cpu().numpy()
    
    def predict_proba(self, X):
        """
        预测概率
        Predict probabilities
        """
        self.base_model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            outputs = self.base_model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            return probabilities.cpu().numpy()

def demo_self_training():
    """
    自训练演示
    Self-training demonstration
    """
    # 生成示例数据 - Generate sample data
    X, y = make_classification(
        n_samples=1000, n_features=20, n_classes=2, 
        n_redundant=5, random_state=42
    )
    
    # 分割数据：小部分标记，大部分未标记
    # Split data: small labeled portion, large unlabeled portion
    X_labeled, X_test, y_labeled, y_test = train_test_split(
        X, y, test_size=0.8, random_state=42
    )
    
    # 进一步分割：只用一小部分作为初始标记数据
    X_labeled, X_unlabeled, y_labeled, _ = train_test_split(
        X_labeled, y_labeled, test_size=0.9, random_state=42
    )
    
    print(f"Initial labeled samples: {len(X_labeled)}")
    print(f"Unlabeled samples: {len(X_unlabeled)}")
    print(f"Test samples: {len(X_test)}")
    
    # 创建基础模型 - Create base model
    base_model = SimpleClassifier(input_dim=X.shape[1], num_classes=2)
    
    # 创建自训练分类器 - Create self-training classifier
    self_trainer = SelfTrainingClassifier(
        base_model=base_model,
        confidence_threshold=0.8,
        max_iterations=5
    )
    
    # 执行自训练 - Perform self-training
    self_trainer.fit(X_labeled, y_labeled, X_unlabeled)
    
    # 评估性能 - Evaluate performance
    test_predictions = self_trainer.predict(X_test)
    test_accuracy = np.mean(test_predictions == y_test)
    print(f"\nFinal test accuracy: {test_accuracy:.3f}")
    
    # 可视化训练过程 - Visualize training process
    if self_trainer.iteration_stats:
        iterations = [stat['iteration'] for stat in self_trainer.iteration_stats]
        labeled_counts = [stat['labeled_samples'] for stat in self_trainer.iteration_stats]
        
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(iterations, labeled_counts, 'bo-')
        plt.xlabel('Iteration')
        plt.ylabel('Number of Labeled Samples')
        plt.title('Growth of Labeled Dataset')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        confidences = [stat['avg_confidence'] for stat in self_trainer.iteration_stats]
        plt.plot(iterations, confidences, 'ro-')
        plt.xlabel('Iteration')
        plt.ylabel('Average Confidence')
        plt.title('Confidence of Added Samples')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    demo_self_training()
```

---

## 5. Conceptual Questions (概念题)

### Question 14
Compare the advantages and disadvantages of different clustering algorithms for the following scenarios:
比较不同聚类算法在以下场景中的优缺点：

a) Customer segmentation for an e-commerce platform
a) 电商平台的客户细分

b) Gene expression analysis in bioinformatics
b) 生物信息学中的基因表达分析

c) Image segmentation for computer vision
c) 计算机视觉中的图像分割

**Answer**:
**答案**：

**a) Customer Segmentation for E-commerce:**
**a) 电商平台的客户细分：**

*K-Means:*
- **Advantages**: Fast, interpretable results, works well with numerical features (purchase amount, frequency)
- **优势**：快速，可解释的结果，适用于数值特征（购买金额、频率）
- **Disadvantages**: Requires knowing number of segments beforehand, assumes spherical clusters
- **劣势**：需要事先知道细分数量，假设球形簇

*Hierarchical Clustering:*
- **Advantages**: Provides multiple segmentation levels, no need to specify cluster count
- **优势**：提供多个细分级别，无需指定簇数量
- **Disadvantages**: Computationally expensive for large customer bases, sensitive to outliers
- **劣势**：对大量客户计算昂贵，对异常值敏感

*DBSCAN:*
- **Advantages**: Can find customers with unusual behavior patterns, handles noise well
- **优势**：可以找到具有异常行为模式的客户，很好地处理噪声
- **Disadvantages**: Difficulty in parameter tuning, less intuitive for business stakeholders
- **劣势**：参数调整困难，对业务相关者不够直观

**b) Gene Expression Analysis:**
**b) 基因表达分析：**

*K-Means:*
- **Advantages**: Suitable for identifying co-expressed gene groups, computationally efficient
- **优势**：适合识别共表达基因组，计算效率高
- **Disadvantages**: May miss complex expression patterns, sensitive to noise in expression data
- **劣势**：可能错过复杂的表达模式，对表达数据中的噪声敏感

*Hierarchical Clustering:*
- **Advantages**: Reveals gene relationship hierarchies, commonly used and well-accepted in biology
- **优势**：揭示基因关系层次，在生物学中常用且被广泛接受
- **Disadvantages**: Can be unstable with noisy biological data, computationally intensive
- **劣势**：对噪声生物数据可能不稳定，计算密集

*DBSCAN:*
- **Advantages**: Can identify tightly co-regulated gene modules, robust to outlier genes
- **优势**：可以识别紧密共调节的基因模块，对异常基因鲁棒
- **Disadvantages**: Parameter selection challenging for biological data, may miss subtle patterns
- **劣势**：生物数据的参数选择具有挑战性，可能错过微妙模式

**c) Image Segmentation:**
**c) 图像分割：**

*K-Means:*
- **Advantages**: Simple color-based segmentation, fast for real-time applications
- **优势**：简单的基于颜色的分割，实时应用中快速
- **Disadvantages**: Poor with complex textures, doesn't consider spatial relationships
- **劣势**：复杂纹理效果差，不考虑空间关系

*Hierarchical Clustering:*
- **Advantages**: Can capture nested object structures, good for hierarchical scene understanding
- **优势**：可以捕获嵌套对象结构，适合层次场景理解
- **Disadvantages**: Very slow for high-resolution images, memory intensive
- **劣势**：高分辨率图像非常慢，内存密集

*DBSCAN:*
- **Advantages**: Good for identifying objects of varying shapes, handles background noise well
- **优势**：适合识别各种形状的对象，很好地处理背景噪声
- **Disadvantages**: Struggles with varying densities within images, parameter tuning complex
- **劣势**：在图像内密度变化时有困难，参数调整复杂

---

### Question 15
Explain the concept of "consistency regularization" in semi-supervised learning and provide a real-world scenario where it would be particularly effective.
解释半监督学习中"一致性正则化"的概念，并提供一个特别有效的现实场景。

**Answer**:
**答案**：

**Consistency Regularization Concept:**
**一致性正则化概念：**

Consistency regularization is based on the **smoothness assumption** that small perturbations to the input should not dramatically change the model's predictions. The key idea is to encourage the model to produce **similar outputs for similar inputs**.
一致性正则化基于**平滑性假设**，即对输入的小扰动不应该显著改变模型的预测。关键思想是鼓励模型对相似输入产生**相似输出**。

**Mathematical Formulation:**
**数学表述：**

The total loss combines supervised loss and consistency loss:
总损失结合了监督损失和一致性损失：

$$L_{total} = L_{supervised} + \lambda L_{consistency}$$

Where the consistency loss is:
其中一致性损失为：

$$L_{consistency} = \frac{1}{|U|} \sum_{x \in U} d(f(x), f(T(x)))$$

- $U$ = unlabeled data / 未标记数据
- $T(x)$ = transformed version of input $x$ / 输入$x$的变换版本
- $d(\cdot, \cdot)$ = distance measure (e.g., MSE, KL divergence) / 距离度量
- $\lambda$ = weight for consistency term / 一致性项的权重

**Real-World Scenario: Medical Image Diagnosis**
**现实场景：医学图像诊断**

**Problem Setting:**
**问题设置：**
- Task: Detect pneumonia in chest X-rays
- 任务：在胸部X光片中检测肺炎
- Labeled data: 1,000 X-rays with expert radiologist annotations
- 标记数据：1,000张有专家放射科医生标注的X光片
- Unlabeled data: 50,000 X-rays without annotations
- 未标记数据：50,000张没有标注的X光片

**Why Consistency Regularization is Effective Here:**
**为什么一致性正则化在这里有效：**

1. **Natural Augmentations**: Small rotations, brightness changes, or slight zooms shouldn't change the diagnosis
1. **自然增强**：小的旋转、亮度变化或轻微缩放不应该改变诊断

2. **Expert Knowledge Scarcity**: Radiologist time is expensive, so we have many unlabeled images
2. **专家知识稀缺**：放射科医生的时间很昂贵，所以我们有很多未标记的图像

3. **High Stakes**: Medical errors are costly, so leveraging all available data is crucial
3. **高风险**：医疗错误代价高昂，所以利用所有可用数据至关重要

**Implementation Approach:**
**实现方法：**

```python
def medical_consistency_training(labeled_batch, unlabeled_batch):
    # 监督损失 - Supervised loss
    predictions_labeled = model(labeled_batch['images'])
    supervised_loss = cross_entropy(predictions_labeled, labeled_batch['labels'])
    
    # 一致性损失 - Consistency loss
    # 原始未标记图像 - Original unlabeled images
    pred_original = model(unlabeled_batch)
    
    # 应用医学相关的增强 - Apply medically-relevant augmentations
    augmented_images = apply_medical_augmentations(unlabeled_batch)
    pred_augmented = model(augmented_images)
    
    # 计算一致性损失 - Compute consistency loss
    consistency_loss = mse_loss(
        torch.softmax(pred_original, dim=1),
        torch.softmax(pred_augmented, dim=1)
    )
    
    total_loss = supervised_loss + lambda_consistency * consistency_loss
    return total_loss

def apply_medical_augmentations(images):
    """
    应用医学图像的合理增强
    Apply reasonable augmentations for medical images
    """
    augmentations = [
        RandomRotation(degrees=5),      # 轻微旋转
        RandomBrightness(factor=0.1),   # 亮度调整
        GaussianNoise(std=0.01),        # 微小噪声
        RandomCrop(padding=4)           # 轻微裁剪
    ]
    return apply_transforms(images, augmentations)
```

**Expected Benefits:**
**预期收益：**
- **Improved Accuracy**: Using 50x more data through consistency regularization
- **提高准确性**：通过一致性正则化使用50倍更多的数据
- **Better Generalization**: Model becomes more robust to imaging variations
- **更好的泛化**：模型对成像变化更加鲁棒
- **Reduced Overfitting**: Large unlabeled dataset acts as implicit regularization
- **减少过拟合**：大型未标记数据集作为隐式正则化
- **Cost Efficiency**: Maximizes value from expensive expert annotations
- **成本效率**：最大化昂贵专家标注的价值

This approach has been successfully used in medical AI systems and has shown significant improvements over supervised learning alone.
这种方法已成功应用于医疗AI系统，并显示出比单纯监督学习显著的改进。 