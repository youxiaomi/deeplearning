# Chapter 13: Natural Language Processing Applications - Quiz 测试题

## Section 1: Multiple Choice Questions 选择题

### Question 1: Sentiment Analysis Basics 情感分析基础
Which of the following is NOT a typical preprocessing step for sentiment analysis?
以下哪项不是情感分析的典型预处理步骤？

A) Text cleaning and normalization 文本清洁和规范化
B) Tokenization 分词
C) Vocabulary building 词汇表构建
D) Gradient descent optimization 梯度下降优化

**Answer 答案: D**
**Explanation 解释:** Gradient descent optimization is part of the training process, not preprocessing. Preprocessing steps include text cleaning, tokenization, and vocabulary building to prepare data for the model.
梯度下降优化是训练过程的一部分，不是预处理。预处理步骤包括文本清洁、分词和词汇表构建，为模型准备数据。

### Question 2: RNN for Sentiment Analysis RNN情感分析
What is the main advantage of using RNNs for sentiment analysis?
使用RNN进行情感分析的主要优势是什么？

A) Parallel processing capability 并行处理能力
B) Sequential memory of previous words 对先前单词的顺序记忆
C) Fixed computation time 固定计算时间
D) Simple architecture 简单架构

**Answer 答案: B**
**Explanation 解释:** RNNs excel at sentiment analysis because they can process text sequentially and maintain memory of previous words, allowing them to understand context and build cumulative understanding of the text.
RNN在情感分析方面表现出色，因为它们可以按顺序处理文本并保持对先前单词的记忆，使它们能够理解上下文并建立对文本的累积理解。

### Question 3: CNN Text Processing CNN文本处理
In textCNN, what does max-over-time pooling accomplish?
在textCNN中，时间维度最大池化实现了什么？

A) Increases sequence length 增加序列长度
B) Extracts most important features regardless of position 提取最重要的特征，无论位置如何
C) Reduces model parameters 减少模型参数
D) Improves gradient flow 改善梯度流

**Answer 答案: B**
**Explanation 解释:** Max-over-time pooling extracts the most important features from the convolutional output, keeping only the strongest signals regardless of their position in the text, creating a fixed-size representation.
时间维度最大池化从卷积输出中提取最重要的特征，只保留最强的信号，无论它们在文本中的位置如何，创建固定大小的表示。

### Question 4: Natural Language Inference 自然语言推理
What are the three types of logical relationships in Natural Language Inference?
自然语言推理中的三种逻辑关系类型是什么？

A) Positive, Negative, Neutral 正面、负面、中性
B) Entailment, Contradiction, Neutral 蕴含、矛盾、中性
C) Support, Oppose, Unrelated 支持、反对、无关
D) True, False, Unknown 真、假、未知

**Answer 答案: B**
**Explanation 解释:** NLI tasks classify the relationship between premise and hypothesis as: Entailment (premise implies hypothesis), Contradiction (they cannot both be true), or Neutral (no clear logical relationship).
NLI任务将前提和假设之间的关系分类为：蕴含（前提暗示假设）、矛盾（两者不能同时为真）或中性（没有明确的逻辑关系）。

### Question 5: BERT Fine-tuning BERT微调
What is the typical learning rate range for fine-tuning BERT?
微调BERT的典型学习率范围是什么？

A) 1e-2 to 1e-1
B) 1e-3 to 1e-2  
C) 1e-5 to 5e-5
D) 1e-7 to 1e-6

**Answer 答案: C**
**Explanation 解释:** BERT fine-tuning typically uses much lower learning rates (1e-5 to 5e-5) compared to training from scratch, because the pretrained model already has good representations and only needs gentle adjustments.
BERT微调通常使用比从头训练低得多的学习率（1e-5到5e-5），因为预训练模型已经有良好的表示，只需要轻微调整。

## Section 2: Short Answer Questions 简答题

### Question 6: Data Iterators 数据迭代器
Explain the three main components of data iterators for NLP tasks and why each is important.
解释NLP任务数据迭代器的三个主要组件以及每个组件的重要性。

**Answer 答案:**
The three main components are:
主要的三个组件是：

1. **Batching 批处理:** Groups multiple samples together to enable parallel processing and improve training efficiency. Like processing multiple documents simultaneously instead of one by one.
将多个样本组合在一起以实现并行处理并提高训练效率。就像同时处理多个文档而不是逐个处理。

2. **Padding 填充:** Makes sequences the same length within a batch to handle variable-length inputs. Uses special padding tokens to ensure uniform tensor shapes for computation.
使批次内的序列长度相同以处理可变长度输入。使用特殊的填充标记确保计算的统一张量形状。

3. **Shuffling 洗牌:** Randomizes data order to prevent overfitting to data sequence and improve generalization. Ensures the model doesn't learn spurious patterns based on data ordering.
随机化数据顺序以防止对数据序列过拟合并改善泛化。确保模型不会基于数据排序学习虚假模式。

### Question 7: Attention in NLI NLI中的注意力
Describe how cross-attention works in Natural Language Inference models and why it's beneficial.
描述交叉注意力在自然语言推理模型中如何工作以及为什么有益。

**Answer 答案:**
Cross-attention in NLI allows the model to align and compare relevant parts of the premise and hypothesis:
NLI中的交叉注意力允许模型对齐和比较前提和假设的相关部分：

**How it works 工作原理:**
- The premise attends to relevant parts of the hypothesis
- The hypothesis attends to relevant parts of the premise  
- Creates bidirectional understanding and word alignments
- Identifies which words/phrases are most important for logical reasoning

前提关注假设的相关部分
假设关注前提的相关部分
创建双向理解和词对齐
识别哪些词/短语对逻辑推理最重要

**Benefits 好处:**
- Enables focused comparison of relevant information
- Improves logical reasoning accuracy
- Provides interpretable attention weights
- Handles complex sentence relationships better

支持对相关信息的重点比较
提高逻辑推理准确性
提供可解释的注意力权重
更好地处理复杂的句子关系

### Question 8: BERT Input Format BERT输入格式
Explain the input format for different BERT applications and why special tokens are used.
解释不同BERT应用的输入格式以及为什么使用特殊标记。

**Answer 答案:**
BERT uses specific input formats with special tokens for different tasks:
BERT为不同任务使用带有特殊标记的特定输入格式：

**Single Text Classification 单文本分类:**
```
[CLS] text [SEP]
```

**Text Pair Tasks 文本对任务:**
```
[CLS] text_a [SEP] text_b [SEP]
```

**Question Answering 问答:**
```
[CLS] question [SEP] passage [SEP]
```

**Why special tokens are important 为什么特殊标记很重要:**
- **[CLS]**: Represents the entire sequence for classification tasks, aggregates global information
  代表分类任务的整个序列，聚合全局信息
- **[SEP]**: Separates different text segments, helps model understand boundaries
  分离不同的文本段，帮助模型理解边界
- These tokens have learned representations during pretraining that BERT uses for specific purposes
  这些标记在预训练期间学习了BERT用于特定目的的表示

## Section 3: Practical Application Questions 实践应用题

### Question 9: Sentiment Analysis Pipeline 情感分析管道
Design a complete sentiment analysis pipeline for movie reviews. Include preprocessing, model selection, and evaluation steps.
为电影评论设计一个完整的情感分析管道。包括预处理、模型选择和评估步骤。

**Answer 答案:**

**1. Data Collection and Preprocessing 数据收集和预处理:**
```python
# Preprocessing steps
def preprocess_reviews(reviews):
    # Text cleaning
    reviews = clean_text(reviews)  # Remove URLs, special chars
    reviews = handle_contractions(reviews)  # don't -> do not
    reviews = reviews.lower()  # Normalize case
    
    # Tokenization
    tokens = tokenize(reviews)
    
    # Vocabulary building
    vocab = build_vocabulary(tokens, min_freq=5)
    
    return tokens, vocab
```

**2. Model Selection 模型选择:**
- **RNN approach**: For sequential understanding and context
  RNN方法：用于顺序理解和上下文
- **CNN approach**: For local pattern detection and speed  
  CNN方法：用于局部模式检测和速度
- **BERT approach**: For best performance with pretrained knowledge
  BERT方法：利用预训练知识获得最佳性能

**3. Training Configuration 训练配置:**
- Train/validation/test split: 70/15/15
- Batch size: 32-64 for efficiency
- Learning rate: 1e-3 for CNN/RNN, 2e-5 for BERT
- Early stopping on validation accuracy

**4. Evaluation Metrics 评估指标:**
- Accuracy: Overall performance
- Precision/Recall: Class-specific performance  
- F1-score: Balanced metric
- Confusion matrix: Error analysis

### Question 10: TextCNN Architecture TextCNN架构
Implement a textCNN model architecture for sentiment classification. Explain each component's purpose.
为情感分类实现textCNN模型架构。解释每个组件的目的。

**Answer 答案:**

```python
import torch
import torch.nn as nn

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, 
                 filter_sizes=[3,4,5], num_filters=100):
        super(TextCNN, self).__init__()
        
        # Embedding layer - converts words to dense vectors
        # 嵌入层 - 将单词转换为密集向量
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Multiple convolutional layers with different filter sizes
        # 不同滤波器大小的多个卷积层
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        
        # Dropout for regularization
        # Dropout用于正则化
        self.dropout = nn.Dropout(0.5)
        
        # Classification layer
        # 分类层
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        
        # Embedding: (batch_size, seq_len, embed_dim)
        embedded = self.embedding(x)
        
        # Transpose for Conv1d: (batch_size, embed_dim, seq_len)
        embedded = embedded.transpose(1, 2)
        
        # Apply convolutions and max pooling
        # 应用卷积和最大池化
        conv_outputs = []
        for conv in self.convs:
            # Convolution: (batch_size, num_filters, conv_seq_len)
            conv_out = torch.relu(conv(embedded))
            # Max pooling: (batch_size, num_filters)
            pooled = torch.max(conv_out, dim=2)[0]
            conv_outputs.append(pooled)
        
        # Concatenate all pooled outputs
        # 连接所有池化输出
        combined = torch.cat(conv_outputs, dim=1)
        
        # Apply dropout and classification
        # 应用dropout和分类
        combined = self.dropout(combined)
        output = self.fc(combined)
        
        return output
```

**Component Purposes 组件目的:**

1. **Embedding Layer 嵌入层**: Converts discrete word indices to continuous vector representations
   将离散词索引转换为连续向量表示

2. **Convolutional Layers 卷积层**: Detect local patterns (n-grams) of different lengths (3,4,5 words)
   检测不同长度的局部模式（n-gram）（3、4、5个单词）

3. **Max Pooling 最大池化**: Extracts most important features regardless of position
   提取最重要的特征，无论位置如何

4. **Dropout**: Prevents overfitting during training
   在训练期间防止过拟合

5. **Fully Connected Layer 全连接层**: Maps concatenated features to class predictions
   将连接的特征映射到类别预测

### Question 11: BERT Fine-tuning Strategy BERT微调策略
Describe a comprehensive strategy for fine-tuning BERT on a natural language inference task, including data preparation, training configuration, and evaluation.
描述在自然语言推理任务上微调BERT的综合策略，包括数据准备、训练配置和评估。

**Answer 答案:**

**1. Data Preparation 数据准备:**

```python
def prepare_nli_data(premise, hypothesis, label):
    # Format for BERT input
    # 为BERT输入格式化
    input_text = f"[CLS] {premise} [SEP] {hypothesis} [SEP]"
    
    # Tokenize with BERT tokenizer
    # 使用BERT分词器分词
    tokens = bert_tokenizer.tokenize(input_text)
    
    # Convert to IDs and handle sequence length
    # 转换为ID并处理序列长度
    input_ids = bert_tokenizer.convert_tokens_to_ids(tokens)
    input_ids = input_ids[:max_seq_length]  # Truncate if too long
    
    # Create attention mask
    # 创建注意力掩码
    attention_mask = [1] * len(input_ids)
    
    # Pad to max length
    # 填充到最大长度
    padding_length = max_seq_length - len(input_ids)
    input_ids += [0] * padding_length
    attention_mask += [0] * padding_length
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'label': label
    }
```

**2. Training Configuration 训练配置:**

```python
# Model setup
# 模型设置
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=3  # entailment, contradiction, neutral
)

# Optimizer with different learning rates for different layers
# 不同层使用不同学习率的优化器
optimizer = AdamW([
    {'params': model.bert.parameters(), 'lr': 2e-5},  # Lower for pretrained
    {'params': model.classifier.parameters(), 'lr': 5e-5}  # Higher for new layer
], weight_decay=0.01)

# Learning rate scheduler with warmup
# 带预热的学习率调度器
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=1000,
    num_training_steps=total_steps
)
```

**3. Training Loop 训练循环:**

```python
def train_epoch(model, dataloader, optimizer, scheduler):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        # Forward pass
        # 前向传播
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['label']
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        # Backward pass
        # 反向传播
        loss.backward()
        
        # Gradient clipping for stability
        # 梯度裁剪保持稳定性
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    
    return total_loss / len(dataloader)
```

**4. Evaluation Strategy 评估策略:**

```python
def evaluate_model(model, test_dataloader):
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(batch['label'].cpu().numpy())
    
    # Calculate metrics
    # 计算指标
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='macro')
    
    # Confusion matrix for error analysis
    # 用于错误分析的混淆矩阵
    cm = confusion_matrix(true_labels, predictions)
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'confusion_matrix': cm
    }
```

**5. Best Practices 最佳实践:**

- **Gradient Accumulation 梯度累积**: For larger effective batch sizes with limited memory
  在内存有限的情况下获得更大的有效批次大小

- **Early Stopping 早停**: Monitor validation performance to prevent overfitting
  监控验证性能以防止过拟合

- **Model Checkpointing 模型检查点**: Save best models based on validation metrics
  基于验证指标保存最佳模型

- **Error Analysis 错误分析**: Examine misclassified examples to understand model limitations
  检查误分类示例以了解模型限制

### Question 12: Comparison Analysis 对比分析
Compare RNN, CNN, and BERT approaches for sentiment analysis. Discuss their strengths, weaknesses, and appropriate use cases.
比较RNN、CNN和BERT方法在情感分析中的应用。讨论它们的优势、劣势和适当的用例。

**Answer 答案:**

| Aspect 方面 | RNN | CNN | BERT |
|-------------|-----|-----|------|
| **Sequential Processing 顺序处理** | ✅ Natural sequential flow 自然顺序流 | ❌ Parallel processing 并行处理 | ✅ Bidirectional context 双向上下文 |
| **Training Speed 训练速度** | ⚠️ Slow (sequential) 慢（顺序） | ✅ Fast (parallel) 快（并行） | ❌ Slow (large model) 慢（大模型） |
| **Memory Usage 内存使用** | ✅ Low 低 | ✅ Low 低 | ❌ High 高 |
| **Context Understanding 上下文理解** | ✅ Good for long sequences 长序列表现好 | ⚠️ Limited to filter size 限制于滤波器大小 | ✅ Excellent global context 出色的全局上下文 |
| **Performance 性能** | ⚠️ Good 良好 | ⚠️ Good 良好 | ✅ Excellent 出色 |

**Detailed Comparison 详细比较:**

**1. RNN Approach RNN方法:**

*Strengths 优势:*
- Natural sequential processing mimics human reading
  自然顺序处理模拟人类阅读
- Good memory of previous context
  对先前上下文的良好记忆
- Handles variable-length sequences well
  很好地处理可变长度序列
- Lower computational requirements
  较低的计算要求

*Weaknesses 劣势:*
- Sequential nature prevents parallelization
  顺序性质阻止并行化
- Vanishing gradient problems in long sequences
  长序列中的梯度消失问题
- Slower training compared to CNNs
  与CNN相比训练较慢

*Use Cases 用例:*
- Long document analysis
  长文档分析
- Limited computational resources
  计算资源有限
- When sequence order is crucial
  当序列顺序至关重要时

**2. CNN Approach CNN方法:**

*Strengths 优势:*
- Fast parallel processing
  快速并行处理
- Effective at detecting local patterns
  有效检测局部模式
- Good performance on short texts
  在短文本上表现良好
- Computationally efficient
  计算效率高

*Weaknesses 劣势:*
- Limited long-range dependencies
  有限的长距离依赖关系
- Fixed filter sizes may miss patterns
  固定滤波器大小可能错过模式
- Less natural for sequential data
  对顺序数据不太自然

*Use Cases 用例:*
- Short text classification (tweets, reviews)
  短文本分类（推文、评论）
- Real-time applications requiring speed
  需要速度的实时应用
- When local patterns are most important
  当局部模式最重要时

**3. BERT Approach BERT方法:**

*Strengths 优势:*
- State-of-the-art performance
  最先进的性能
- Rich pretrained representations
  丰富的预训练表示
- Bidirectional context understanding
  双向上下文理解
- Transfer learning benefits
  迁移学习好处

*Weaknesses 劣势:*
- High computational requirements
  高计算要求
- Large memory footprint
  大内存占用
- Slower inference
  推理较慢
- May be overkill for simple tasks
  对简单任务可能过度

*Use Cases 用例:*
- High-accuracy requirements
  高准确性要求
- Complex language understanding
  复杂语言理解
- Sufficient computational resources available
  有足够的计算资源
- When state-of-the-art performance is needed
  当需要最先进的性能时

**Recommendation Guidelines 推荐指南:**

1. **Choose RNN when 选择RNN当:**
   - Sequential context is crucial
   - Limited computational resources
   - Working with long sequences
   - Need interpretable sequential processing

2. **Choose CNN when 选择CNN当:**
   - Speed is priority
   - Working with short texts
   - Local patterns are key
   - Need efficient deployment

3. **Choose BERT when 选择BERT当:**
   - Maximum accuracy required
   - Complex language understanding needed
   - Sufficient computational resources
   - Can afford longer training/inference times

## Section 4: Coding Exercises 编程练习

### Exercise 1: Implement Attention Visualization 实现注意力可视化
Write code to visualize attention weights in a text classification model to understand which words the model focuses on.
编写代码可视化文本分类模型中的注意力权重，以了解模型关注哪些单词。

**Answer 答案:**
```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize_attention(tokens, attention_weights, save_path=None):
    """
    Visualize attention weights for text tokens
    可视化文本标记的注意力权重
    
    Args:
        tokens: List of tokens in the sequence
        attention_weights: Attention weights for each token
        save_path: Optional path to save the visualization
    """
    # Create figure
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # Create heatmap
    # 创建热图
    attention_matrix = attention_weights.reshape(1, -1)
    
    sns.heatmap(
        attention_matrix,
        xticklabels=tokens,
        yticklabels=['Attention'],
        cmap='Blues',
        annot=True,
        fmt='.3f',
        cbar_kws={'label': 'Attention Weight'}
    )
    
    plt.title('Attention Weights Visualization 注意力权重可视化')
    plt.xlabel('Tokens 标记')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

# Example usage
# 使用示例
def extract_attention_from_model(model, input_text, tokenizer):
    """
    Extract attention weights from a trained model
    从训练好的模型中提取注意力权重
    """
    # Tokenize input
    # 分词输入
    tokens = tokenizer.tokenize(input_text)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    # Forward pass with attention
    # 带注意力的前向传播
    model.eval()
    with torch.no_grad():
        outputs = model(
            torch.tensor([input_ids]), 
            output_attentions=True
        )
        
        # Extract attention weights (from last layer, first head)
        # 提取注意力权重（来自最后一层，第一个头）
        attention = outputs.attentions[-1][0, 0, :].numpy()
    
    return tokens, attention

# Demo
# 演示
text = "This movie is absolutely fantastic and entertaining!"
tokens, attention_weights = extract_attention_from_model(model, text, tokenizer)
visualize_attention(tokens, attention_weights)
```

### Exercise 2: Build Custom TextCNN 构建自定义TextCNN
Implement a complete TextCNN training pipeline with data loading, training, and evaluation.
实现完整的TextCNN训练管道，包括数据加载、训练和评估。

**Answer 答案:**
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, classification_report

class SentimentDataset(Dataset):
    """Custom dataset for sentiment analysis"""
    def __init__(self, texts, labels, vocab, max_length=100):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        # Convert text to indices
        # 将文本转换为索引
        text = self.texts[idx]
        tokens = text.split()
        
        # Convert tokens to indices
        # 将标记转换为索引
        indices = [self.vocab.get(token, self.vocab['<UNK>']) 
                  for token in tokens]
        
        # Pad or truncate to max_length
        # 填充或截断到最大长度
        if len(indices) < self.max_length:
            indices += [self.vocab['<PAD>']] * (self.max_length - len(indices))
        else:
            indices = indices[:self.max_length]
        
        return {
            'text': torch.tensor(indices, dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class TextCNN(nn.Module):
    """TextCNN model for sentiment analysis"""
    def __init__(self, vocab_size, embed_dim, num_classes, 
                 filter_sizes=[3,4,5], num_filters=100, dropout=0.5):
        super(TextCNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        
    def forward(self, x):
        # x: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        embedded = embedded.transpose(1, 2)  # (batch_size, embed_dim, seq_len)
        
        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(embedded))
            pooled = torch.max(conv_out, dim=2)[0]
            conv_outputs.append(pooled)
        
        combined = torch.cat(conv_outputs, dim=1)
        combined = self.dropout(combined)
        output = self.fc(combined)
        
        return output

def train_textcnn(train_loader, val_loader, vocab_size, num_classes, 
                  num_epochs=10, learning_rate=0.001):
    """Complete training pipeline for TextCNN"""
    
    # Initialize model
    # 初始化模型
    model = TextCNN(
        vocab_size=vocab_size,
        embed_dim=128,
        num_classes=num_classes,
        filter_sizes=[3, 4, 5],
        num_filters=100
    )
    
    # Loss and optimizer
    # 损失和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    # 训练循环
    for epoch in range(num_epochs):
        # Training phase
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            outputs = model(batch['text'])
            loss = criterion(outputs, batch['label'])
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch['label'].size(0)
            train_correct += (predicted == batch['label']).sum().item()
        
        # Validation phase
        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(batch['text'])
                loss = criterion(outputs, batch['label'])
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch['label'].size(0)
                val_correct += (predicted == batch['label']).sum().item()
        
        # Print progress
        # 打印进度
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, '
              f'Val Acc: {val_acc:.2f}%')
        print('-' * 50)
    
    return model

def evaluate_model(model, test_loader, class_names):
    """Evaluate model performance"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(batch['text'])
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch['label'].cpu().numpy())
    
    # Calculate metrics
    # 计算指标
    accuracy = accuracy_score(all_labels, all_predictions)
    report = classification_report(
        all_labels, all_predictions, 
        target_names=class_names
    )
    
    print(f'Test Accuracy: {accuracy:.4f}')
    print('Classification Report:')
    print(report)
    
    return accuracy, report

# Example usage
# 使用示例
"""
# Prepare data
train_texts = ["This movie is great!", "Terrible film", ...]
train_labels = [1, 0, ...]  # 1: positive, 0: negative
vocab = build_vocabulary(train_texts)

# Create datasets and dataloaders
train_dataset = SentimentDataset(train_texts, train_labels, vocab)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Train model
model = train_textcnn(train_loader, val_loader, len(vocab), num_classes=2)

# Evaluate
accuracy, report = evaluate_model(model, test_loader, ['Negative', 'Positive'])
"""
```

## Summary 总结

This quiz covers the essential concepts and practical applications of Natural Language Processing in deep learning. The questions range from basic understanding of preprocessing and model architectures to advanced topics like BERT fine-tuning and attention mechanisms. The practical exercises provide hands-on experience with implementing and training NLP models.

这个测试涵盖了深度学习中自然语言处理的基本概念和实际应用。问题范围从预处理和模型架构的基本理解到BERT微调和注意力机制等高级主题。实践练习提供了实现和训练NLP模型的实践经验。

Key learning objectives achieved:
达到的关键学习目标：

1. Understanding different approaches to sentiment analysis
   理解情感分析的不同方法
2. Grasping the concepts of natural language inference
   掌握自然语言推理的概念
3. Learning BERT fine-tuning strategies
   学习BERT微调策略
4. Implementing practical NLP solutions
   实现实用的NLP解决方案
5. Evaluating and comparing different models
   评估和比较不同模型 