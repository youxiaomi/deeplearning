# Basic RNN Model Implementation
# 基础RNN模型实现

## 1. Custom RNN Cell
## 自定义RNN单元

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Weight matrices / 权重矩阵
        self.W_xh = nn.Linear(input_size, hidden_size, bias=False)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_size))
    
    def forward(self, x, hidden):
        # h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b)
        new_hidden = torch.tanh(
            self.W_xh(x) + self.W_hh(hidden) + self.bias
        )
        return new_hidden
    
    def init_hidden(self, batch_size, device='cpu'):
        return torch.zeros(batch_size, self.hidden_size, device=device)
```

## 2. Complete RNN Network
## 完整RNN网络

```python
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
    
    def forward(self, x, hidden=None):
        # x: (batch_size, seq_len, input_size)
        output, hidden = self.rnn(x, hidden)
        return output, hidden
```

## 3. RNN for Classification
## 用于分类的RNN

```python
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes):
        super(RNNClassifier, self).__init__()
        
        # Layers / 层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # x: (batch_size, seq_len) - token indices
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        rnn_output, _ = self.rnn(embedded)  # (batch_size, seq_len, hidden_size)
        
        # Use last output for classification / 使用最后输出进行分类
        last_output = rnn_output[:, -1, :]  # (batch_size, hidden_size)
        last_output = self.dropout(last_output)
        
        logits = self.classifier(last_output)  # (batch_size, num_classes)
        return logits

# Example usage / 使用示例
model = RNNClassifier(
    vocab_size=1000, 
    embedding_dim=50, 
    hidden_size=64, 
    num_classes=2
)

# Sample input / 示例输入
batch_size, seq_len = 4, 10
x = torch.randint(0, 1000, (batch_size, seq_len))
output = model(x)
print(f"Output shape: {output.shape}")  # (4, 2)
```

## 4. Training Function
## 训练函数

```python
def train_model(model, train_loader, val_loader, epochs=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        # Training / 训练
        model.train()
        train_loss = 0
        
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping / 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        print(f'Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}')
```

## 5. Prediction Function
## 预测函数

```python
def predict_sentiment(model, text_tokens, device='cpu'):
    """
    Predict sentiment of a text
    预测文本的情感
    """
    model.eval()
    with torch.no_grad():
        # Convert to tensor / 转换为tensor
        x = torch.tensor(text_tokens).unsqueeze(0).to(device)  # Add batch dimension
        
        # Get prediction / 获取预测
        logits = model(x)
        probabilities = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)
        
        return predicted_class.item(), probabilities.squeeze()

# Example usage / 使用示例
# text_tokens = [45, 123, 67, 89, 12]  # Token indices
# predicted_class, probs = predict_sentiment(model, text_tokens)
# print(f"Predicted class: {predicted_class}")
# print(f"Probabilities: {probs}")
```

## 6. Model Saving/Loading
## 模型保存/加载

```python
# Save model / 保存模型
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# Load model / 加载模型
def load_model(model_class, path, **kwargs):
    model = model_class(**kwargs)
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Model loaded from {path}")
    return model

# Example / 示例
# save_model(model, 'rnn_model.pth')
# loaded_model = load_model(RNNClassifier, 'rnn_model.pth', 
#                          vocab_size=1000, embedding_dim=50, 
#                          hidden_size=64, num_classes=2)
```

## 7. Complete Example
## 完整示例

```python
# Create synthetic data for demonstration / 创建演示用的合成数据
def create_sample_data():
    # Positive sentences (class 0) / 积极句子（类别0）
    positive_data = [
        [1, 2, 3, 4, 5],      # "I love this movie"
        [6, 7, 8, 9, 10],     # "Great film amazing"
        [11, 12, 13, 14, 15]  # "Excellent story wonderful"
    ]
    
    # Negative sentences (class 1) / 消极句子（类别1）
    negative_data = [
        [16, 17, 18, 19, 20], # "Bad movie terrible"
        [21, 22, 23, 24, 25], # "Worst film ever"
        [26, 27, 28, 29, 30]  # "Awful story boring"
    ]
    
    # Pad sequences to same length / 填充序列到相同长度
    data = positive_data + negative_data
    labels = [0, 0, 0, 1, 1, 1]
    
    return torch.tensor(data), torch.tensor(labels)

# Run complete example / 运行完整示例
if __name__ == "__main__":
    # Create data / 创建数据
    X, y = create_sample_data()
    
    # Create model / 创建模型
    model = RNNClassifier(
        vocab_size=50,
        embedding_dim=16,
        hidden_size=32,
        num_classes=2
    )
    
    # Simple training loop / 简单训练循环
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
    
    # Test prediction / 测试预测
    model.eval()
    with torch.no_grad():
        test_outputs = model(X)
        predictions = torch.argmax(test_outputs, dim=1)
        accuracy = (predictions == y).float().mean()
        print(f'Accuracy: {accuracy:.4f}')
```

This implementation provides a solid foundation for understanding and building RNN models with PyTorch.

这个实现为理解和构建PyTorch的RNN模型提供了坚实的基础。 