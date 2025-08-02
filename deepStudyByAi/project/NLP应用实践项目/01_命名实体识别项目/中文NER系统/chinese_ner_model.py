"""
中文命名实体识别模型实现
Chinese Named Entity Recognition Model Implementation

这个模块包含了完整的中文NER系统实现，包括数据处理、模型定义、训练和评估。
This module contains a complete Chinese NER system implementation, including data processing, model definition, training, and evaluation.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from torchcrf import CRF
import json
import numpy as np
from sklearn.metrics import classification_report
from seqeval.metrics import f1_score, classification_report as seq_classification_report
import logging
from tqdm import tqdm
import os

# 设置日志
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChineseNERDataset(Dataset):
    """
    中文NER数据集类
    Chinese NER Dataset Class
    
    负责加载和预处理中文NER数据，支持BIO标注格式
    Responsible for loading and preprocessing Chinese NER data, supporting BIO annotation format
    """
    
    def __init__(self, data_path, tokenizer, max_length=128, label_list=None):
        """
        初始化数据集
        Initialize dataset
        
        Args:
            data_path: 数据文件路径 | Data file path
            tokenizer: BERT tokenizer
            max_length: 最大序列长度 | Maximum sequence length  
            label_list: 标签列表 | Label list
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_data(data_path)
        
        # 默认标签映射 | Default label mapping
        if label_list is None:
            self.label_list = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG']
        else:
            self.label_list = label_list
            
        self.label2id = {label: i for i, label in enumerate(self.label_list)}
        self.id2label = {i: label for label, i in self.label2id.items()}
        
        logger.info(f"数据集加载完成 | Dataset loaded: {len(self.data)} 个样本 | samples")
        logger.info(f"标签体系 | Label system: {self.label_list}")
    
    def load_data(self, data_path):
        """
        加载JSON Lines格式的数据
        Load data in JSON Lines format
        
        数据格式示例 | Data format example:
        {
            "text": "马云创立了阿里巴巴公司",
            "entities": [
                {"start": 0, "end": 2, "type": "PER", "text": "马云"},
                {"start": 5, "end": 10, "type": "ORG", "text": "阿里巴巴公司"}
            ]
        }
        """
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                data.append(item)
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        获取单个数据样本
        Get single data sample
        """
        item = self.data[idx]
        text = item['text']
        entities = item.get('entities', [])
        
        # 字符级别的tokenization
        # Character-level tokenization
        chars = list(text)
        labels = ['O'] * len(chars)
        
        # 根据实体设置BIO标签
        # Set BIO labels based on entities
        for entity in entities:
            start, end, entity_type = entity['start'], entity['end'], entity['type']
            if start < len(chars) and end <= len(chars):
                labels[start] = f'B-{entity_type}'
                for i in range(start + 1, end):
                    labels[i] = f'I-{entity_type}'
        
        # 使用BERT tokenizer处理
        # Process with BERT tokenizer
        encoding = self.tokenizer(
            chars,
            is_split_into_words=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 对齐标签与token
        # Align labels with tokens
        word_ids = encoding.word_ids()
        aligned_labels = []
        
        for word_id in word_ids:
            if word_id is None:
                # 特殊token如[CLS], [SEP], [PAD]
                # Special tokens like [CLS], [SEP], [PAD]
                aligned_labels.append(-100)
            else:
                # 对应原文字符的标签
                # Label corresponding to original character
                if word_id < len(labels):
                    label = labels[word_id]
                    aligned_labels.append(self.label2id.get(label, 0))
                else:
                    aligned_labels.append(-100)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(aligned_labels, dtype=torch.long),
            'original_text': text,
            'original_entities': entities
        }


class ChineseNERModel(nn.Module):
    """
    中文命名实体识别模型
    Chinese Named Entity Recognition Model
    
    结合BERT和CRF的序列标注模型
    Sequence labeling model combining BERT and CRF
    """
    
    def __init__(self, bert_model_name, num_labels, dropout_rate=0.1):
        """
        初始化模型
        Initialize model
        
        Args:
            bert_model_name: BERT模型名称 | BERT model name
            num_labels: 标签数量 | Number of labels
            dropout_rate: Dropout比率 | Dropout rate
        """
        super(ChineseNERModel, self).__init__()
        
        # BERT编码器 | BERT encoder
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout_rate)
        
        # 分类层 | Classification layer
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
        # CRF层用于序列约束 | CRF layer for sequence constraints
        self.crf = CRF(num_labels, batch_first=True)
        
        self.num_labels = num_labels
        
        logger.info(f"模型初始化完成 | Model initialized: {bert_model_name}")
        logger.info(f"标签数量 | Number of labels: {num_labels}")
    
    def forward(self, input_ids, attention_mask, labels=None):
        """
        前向传播
        Forward pass
        
        Args:
            input_ids: 输入token IDs | Input token IDs
            attention_mask: 注意力掩码 | Attention mask
            labels: 标签（训练时提供）| Labels (provided during training)
            
        Returns:
            训练时返回损失和logits | Returns loss and logits during training
            推理时返回预测结果 | Returns predictions during inference
        """
        # BERT编码 | BERT encoding
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 获取序列表示并应用dropout
        # Get sequence representation and apply dropout
        sequence_output = self.dropout(bert_output.last_hidden_state)
        
        # 分类预测 | Classification prediction
        logits = self.classifier(sequence_output)
        
        # 创建CRF掩码 | Create CRF mask
        mask = attention_mask.bool()
        
        if labels is not None:
            # 训练模式：计算CRF损失
            # Training mode: calculate CRF loss
            
            # 处理-100标签（忽略的位置）
            # Handle -100 labels (ignored positions)
            active_mask = (labels != -100) & mask
            
            # 创建有效的标签和logits
            # Create valid labels and logits
            batch_size, seq_len = labels.shape
            active_labels = torch.where(active_mask, labels, torch.zeros_like(labels))
            
            # 计算CRF损失
            # Calculate CRF loss
            loss = -self.crf(logits, active_labels, mask=active_mask, reduction='mean')
            
            return loss, logits
        else:
            # 推理模式：CRF解码
            # Inference mode: CRF decoding
            predictions = self.crf.decode(logits, mask=mask)
            return predictions
    
    def predict(self, input_ids, attention_mask):
        """
        预测方法
        Prediction method
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(input_ids, attention_mask)
        return predictions


class ChineseNERTrainer:
    """
    中文NER模型训练器
    Chinese NER Model Trainer
    
    负责模型的训练、验证和评估
    Responsible for model training, validation, and evaluation
    """
    
    def __init__(self, model, train_dataset, val_dataset, device='cuda'):
        """
        初始化训练器
        Initialize trainer
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        
        # 将模型移动到指定设备
        # Move model to specified device
        self.model.to(self.device)
        
        logger.info(f"训练器初始化完成 | Trainer initialized on device: {self.device}")
    
    def train(self, epochs=3, batch_size=16, learning_rate=2e-5, warmup_steps=100):
        """
        训练模型
        Train model
        
        Args:
            epochs: 训练轮数 | Training epochs
            batch_size: 批次大小 | Batch size
            learning_rate: 学习率 | Learning rate
            warmup_steps: 预热步数 | Warmup steps
        """
        # 创建数据加载器
        # Create data loaders
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self._collate_fn
        )
        
        # 设置优化器和学习率调度器
        # Set up optimizer and learning rate scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        logger.info(f"开始训练 | Start training: {epochs} epochs, {total_steps} total steps")
        
        # 训练循环
        # Training loop
        best_f1 = 0
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            
            # 训练阶段
            # Training phase
            train_loss = self._train_epoch(train_loader, optimizer, scheduler)
            logger.info(f"训练损失 | Training loss: {train_loss:.4f}")
            
            # 验证阶段
            # Validation phase
            val_metrics = self._evaluate(val_loader)
            logger.info(f"验证指标 | Validation metrics: {val_metrics}")
            
            # 保存最佳模型
            # Save best model
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                self._save_model('best_model.pt')
                logger.info(f"保存最佳模型 | Saved best model: F1 = {best_f1:.4f}")
    
    def _train_epoch(self, train_loader, optimizer, scheduler):
        """
        训练一个epoch
        Train one epoch
        """
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            # 数据移动到设备
            # Move data to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 前向传播
            # Forward pass
            optimizer.zero_grad()
            loss, _ = self.model(input_ids, attention_mask, labels)
            
            # 反向传播
            # Backward pass
            loss.backward()
            
            # 梯度裁剪
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(train_loader)
    
    def _evaluate(self, val_loader):
        """
        评估模型
        Evaluate model
        """
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # 获取预测
                # Get predictions
                predictions = self.model(input_ids, attention_mask)
                
                # 处理每个样本
                # Process each sample
                for i, pred_seq in enumerate(predictions):
                    # 获取有效长度（排除padding）
                    # Get valid length (excluding padding)
                    valid_length = attention_mask[i].sum().item()
                    
                    # 转换为标签名称
                    # Convert to label names
                    pred_labels = []
                    true_labels = []
                    
                    for j in range(1, valid_length - 1):  # 跳过[CLS]和[SEP]
                        if labels[i][j].item() != -100:
                            pred_labels.append(self.train_dataset.id2label[pred_seq[j-1]])
                            true_labels.append(self.train_dataset.id2label[labels[i][j].item()])
                    
                    all_predictions.append(pred_labels)
                    all_labels.append(true_labels)
        
        # 计算指标
        # Calculate metrics
        f1 = f1_score(all_labels, all_predictions)
        report = seq_classification_report(all_labels, all_predictions, output_dict=True)
        
        return {
            'f1': f1,
            'precision': report['micro avg']['precision'],
            'recall': report['micro avg']['recall'],
            'report': report
        }
    
    def _collate_fn(self, batch):
        """
        批次数据整理函数
        Batch data collation function
        """
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    def _save_model(self, path):
        """
        保存模型
        Save model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'label2id': self.train_dataset.label2id,
            'id2label': self.train_dataset.id2label
        }, path)


class ChineseNERPredictor:
    """
    中文NER预测器
    Chinese NER Predictor
    
    用于加载训练好的模型并进行预测
    Used to load trained models and make predictions
    """
    
    def __init__(self, model_path, bert_model_name, device='cuda'):
        """
        初始化预测器
        Initialize predictor
        """
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        
        # 加载模型
        # Load model
        checkpoint = torch.load(model_path, map_location=device)
        self.label2id = checkpoint['label2id']
        self.id2label = checkpoint['id2label']
        
        self.model = ChineseNERModel(
            bert_model_name,
            num_labels=len(self.label2id)
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        logger.info(f"预测器加载完成 | Predictor loaded from: {model_path}")
    
    def predict(self, text, max_length=128):
        """
        预测文本中的实体
        Predict entities in text
        
        Args:
            text: 输入文本 | Input text
            max_length: 最大长度 | Maximum length
            
        Returns:
            entities: 实体列表 | Entity list
        """
        # 字符级别tokenization
        # Character-level tokenization
        chars = list(text)
        
        # BERT编码
        # BERT encoding
        encoding = self.tokenizer(
            chars,
            is_split_into_words=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # 模型预测
        # Model prediction
        with torch.no_grad():
            predictions = self.model(input_ids, attention_mask)
        
        # 解析预测结果
        # Parse prediction results
        pred_labels = predictions[0]  # 第一个（也是唯一一个）样本
        word_ids = encoding.word_ids()
        
        # 对齐预测标签与原文字符
        # Align predicted labels with original characters
        char_labels = []
        for i, word_id in enumerate(word_ids):
            if word_id is not None and i > 0 and i < len(pred_labels) + 1:
                # 调整索引以匹配CRF输出
                # Adjust index to match CRF output
                if i - 1 < len(pred_labels):
                    label_id = pred_labels[i - 1]
                    char_labels.append(self.id2label[label_id])
        
        # 确保标签列表长度与字符列表一致
        # Ensure label list length matches character list
        while len(char_labels) < len(chars):
            char_labels.append('O')
        char_labels = char_labels[:len(chars)]
        
        # 将BIO标签转换为实体列表
        # Convert BIO labels to entity list
        entities = self._bio_to_entities(chars, char_labels)
        
        return entities
    
    def _bio_to_entities(self, chars, bio_labels):
        """
        将BIO标签转换为实体列表
        Convert BIO labels to entity list
        """
        entities = []
        current_entity = None
        
        for i, (char, label) in enumerate(zip(chars, bio_labels)):
            if label.startswith('B-'):
                # 保存之前的实体
                # Save previous entity
                if current_entity:
                    entities.append(current_entity)
                
                # 开始新实体
                # Start new entity
                entity_type = label[2:]
                current_entity = {
                    'text': char,
                    'start': i,
                    'end': i + 1,
                    'type': entity_type
                }
            
            elif label.startswith('I-') and current_entity:
                # 继续当前实体
                # Continue current entity
                entity_type = label[2:]
                if entity_type == current_entity['type']:
                    current_entity['text'] += char
                    current_entity['end'] = i + 1
                else:
                    # 类型不匹配，结束当前实体
                    # Type mismatch, end current entity
                    entities.append(current_entity)
                    current_entity = None
            
            elif label == 'O':
                # 结束当前实体
                # End current entity
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        # 处理最后一个实体
        # Handle last entity
        if current_entity:
            entities.append(current_entity)
        
        return entities


def main():
    """
    主函数，演示完整的训练和预测流程
    Main function demonstrating complete training and prediction workflow
    """
    # 设置设备
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备 | Using device: {device}")
    
    # 初始化tokenizer
    # Initialize tokenizer
    bert_model_name = 'bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    
    # 创建示例数据（实际使用时应该从文件加载）
    # Create example data (should load from files in actual use)
    example_data = [
        {
            "text": "马云创立了阿里巴巴公司",
            "entities": [
                {"start": 0, "end": 2, "type": "PER", "text": "马云"},
                {"start": 5, "end": 10, "type": "ORG", "text": "阿里巴巴公司"}
            ]
        },
        {
            "text": "小芳领导人在北京会见了美国总统",
            "entities": [
                {"start": 0, "end": 3, "type": "PER", "text": "小芳"},
                {"start": 6, "end": 8, "type": "LOC", "text": "北京"},
                {"start": 11, "end": 13, "type": "LOC", "text": "美国"}
            ]
        }
    ]
    
    # 保存示例数据
    # Save example data
    with open('train_data.jsonl', 'w', encoding='utf-8') as f:
        for item in example_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 创建数据集
    # Create datasets
    train_dataset = ChineseNERDataset('train_data.jsonl', tokenizer)
    val_dataset = ChineseNERDataset('train_data.jsonl', tokenizer)  # 示例中使用相同数据
    
    # 创建模型
    # Create model
    model = ChineseNERModel(
        bert_model_name=bert_model_name,
        num_labels=len(train_dataset.label_list)
    )
    
    # 创建训练器并训练
    # Create trainer and train
    trainer = ChineseNERTrainer(model, train_dataset, val_dataset, device)
    trainer.train(epochs=2, batch_size=2)  # 小批次用于演示
    
    # 预测示例
    # Prediction example
    predictor = ChineseNERPredictor('best_model.pt', bert_model_name, device)
    
    test_text = "张三在清华大学工作"
    entities = predictor.predict(test_text)
    
    print(f"\n预测结果 | Prediction results:")
    print(f"文本 | Text: {test_text}")
    print(f"实体 | Entities: {entities}")


if __name__ == "__main__":
    main() 