"""
阅读理解问答系统模型实现
Reading Comprehension Question Answering System Model Implementation

这个模块包含了完整的阅读理解QA系统，包括BERT-based模型、双向注意力机制和多跳推理。
This module contains a complete reading comprehension QA system, including BERT-based models, bidirectional attention mechanism, and multi-hop reasoning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
import json
import numpy as np
from tqdm import tqdm
import logging
import re
import string
from collections import defaultdict

# 设置日志
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QADataset(Dataset):
    """
    问答数据集类
    Question Answering Dataset Class
    
    支持SQuAD格式的数据加载和预处理
    Supports SQuAD format data loading and preprocessing
    """
    
    def __init__(self, data_path, tokenizer, max_length=384, doc_stride=128):
        """
        初始化数据集
        Initialize dataset
        
        Args:
            data_path: 数据文件路径 | Data file path
            tokenizer: 分词器 | Tokenizer
            max_length: 最大序列长度 | Maximum sequence length
            doc_stride: 文档滑动窗口步长 | Document sliding window stride
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.doc_stride = doc_stride
        self.examples = self.load_data(data_path)
        self.features = self.convert_examples_to_features()
        
        logger.info(f"数据集加载完成 | Dataset loaded: {len(self.examples)} 个样本 | examples")
        logger.info(f"特征转换完成 | Feature conversion completed: {len(self.features)} 个特征 | features")
    
    def load_data(self, data_path):
        """
        加载SQuAD格式数据
        Load SQuAD format data
        """
        with open(data_path, 'r', encoding='utf-8') as f:
            squad_data = json.load(f)
        
        examples = []
        for article in squad_data['data']:
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    question = qa['question']
                    qa_id = qa['id']
                    is_impossible = qa.get('is_impossible', False)
                    
                    answers = []
                    if not is_impossible and 'answers' in qa:
                        for answer in qa['answers']:
                            answers.append({
                                'text': answer['text'],
                                'answer_start': answer['answer_start']
                            })
                    
                    examples.append({
                        'qa_id': qa_id,
                        'question': question,
                        'context': context,
                        'answers': answers,
                        'is_impossible': is_impossible
                    })
        
        return examples
    
    def convert_examples_to_features(self):
        """
        将样本转换为模型输入特征
        Convert examples to model input features
        """
        features = []
        
        for example in tqdm(self.examples, desc="转换特征 | Converting features"):
            # 使用滑动窗口处理长文档
            # Use sliding window for long documents
            encoded = self.tokenizer(
                example['question'],
                example['context'],
                max_length=self.max_length,
                truncation='only_second',
                stride=self.doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding='max_length',
                return_tensors='pt'
            )
            
            # 处理每个窗口
            # Process each window
            for i in range(len(encoded['input_ids'])):
                input_ids = encoded['input_ids'][i]
                attention_mask = encoded['attention_mask'][i]
                token_type_ids = encoded['token_type_ids'][i] if 'token_type_ids' in encoded else None
                offset_mapping = encoded['offset_mapping'][i]
                
                # 找到context在token序列中的位置
                # Find context position in token sequence
                sequence_ids = encoded.sequence_ids(i)
                context_start_idx = sequence_ids.index(1) if 1 in sequence_ids else 0
                context_end_idx = len(sequence_ids) - 1 - sequence_ids[::-1].index(1) if 1 in sequence_ids else 0
                
                # 处理答案位置
                # Process answer positions
                start_position = 0
                end_position = 0
                
                if not example['is_impossible'] and example['answers']:
                    answer = example['answers'][0]
                    answer_start_char = answer['answer_start']
                    answer_end_char = answer_start_char + len(answer['text'])
                    
                    # 在当前窗口中查找答案位置
                    # Find answer position in current window
                    for token_idx, (start_char, end_char) in enumerate(offset_mapping):
                        if sequence_ids[token_idx] != 1:  # 不在context中
                            continue
                            
                        if start_char <= answer_start_char < end_char:
                            start_position = token_idx
                        if start_char < answer_end_char <= end_char:
                            end_position = token_idx
                            break
                    
                    # 如果答案不在当前窗口中，标记为无答案
                    # If answer is not in current window, mark as no answer
                    if start_position == 0 and end_position == 0:
                        start_position = context_start_idx
                        end_position = context_start_idx
                
                features.append({
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'token_type_ids': token_type_ids,
                    'start_position': torch.tensor(start_position, dtype=torch.long),
                    'end_position': torch.tensor(end_position, dtype=torch.long),
                    'qa_id': example['qa_id'],
                    'is_impossible': example['is_impossible'],
                    'offset_mapping': offset_mapping,
                    'context_start_idx': context_start_idx,
                    'context_end_idx': context_end_idx
                })
        
        return features
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]


class BiDirectionalAttention(nn.Module):
    """
    双向注意力机制
    Bidirectional Attention Mechanism
    
    实现问题到文档和文档到问题的双向注意力
    Implements bidirectional attention from question to context and context to question
    """
    
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.att_weight = nn.Linear(hidden_size * 3, 1, bias=False)
    
    def forward(self, context, question, context_mask, question_mask):
        """
        前向传播
        Forward pass
        
        Args:
            context: [batch_size, context_len, hidden_size]
            question: [batch_size, question_len, hidden_size]
            context_mask: [batch_size, context_len]
            question_mask: [batch_size, question_len]
        """
        batch_size, context_len, hidden_size = context.size()
        question_len = question.size(1)
        
        # 计算相似度矩阵 | Compute similarity matrix
        similarity_matrix = self.compute_similarity_matrix(context, question)
        
        # 应用掩码 | Apply masks
        similarity_matrix = similarity_matrix.masked_fill(
            ~question_mask.unsqueeze(1), -1e9
        )
        
        # 问题到文档注意力 | Question-to-context attention
        q2c_attention = F.softmax(similarity_matrix, dim=-1)
        q2c_attended = torch.bmm(q2c_attention, question)
        
        # 文档到问题注意力 | Context-to-question attention
        c2q_attention = F.softmax(similarity_matrix.max(dim=-1)[0], dim=-1)
        c2q_attended = torch.bmm(c2q_attention.unsqueeze(1), context).squeeze(1)
        c2q_attended = c2q_attended.unsqueeze(1).expand(-1, context_len, -1)
        
        # 特征融合 | Feature fusion
        fused = torch.cat([
            context,
            q2c_attended,
            context * q2c_attended,
            context * c2q_attended
        ], dim=-1)
        
        return fused
    
    def compute_similarity_matrix(self, context, question):
        """
        计算相似度矩阵
        Compute similarity matrix
        """
        batch_size, context_len, hidden_size = context.size()
        question_len = question.size(1)
        
        # 扩展维度
        # Expand dimensions
        context_expanded = context.unsqueeze(2).expand(-1, -1, question_len, -1)
        question_expanded = question.unsqueeze(1).expand(-1, context_len, -1, -1)
        
        # 计算特征
        # Compute features
        elementwise_product = context_expanded * question_expanded
        features = torch.cat([
            context_expanded,
            question_expanded,
            elementwise_product
        ], dim=-1)
        
        # 计算相似度分数
        # Compute similarity scores
        similarity = self.att_weight(features).squeeze(-1)
        
        return similarity


class BERTQuestionAnswering(nn.Module):
    """
    基于BERT的问答模型
    BERT-based Question Answering Model
    
    结合BERT编码器和答案边界预测层
    Combines BERT encoder with answer span prediction layers
    """
    
    def __init__(self, model_name='bert-base-uncased', use_bidirectional_attention=False):
        super().__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        self.use_bidirectional_attention = use_bidirectional_attention
        
        if use_bidirectional_attention:
            # 使用双向注意力机制
            # Use bidirectional attention mechanism
            self.bidirectional_attention = BiDirectionalAttention(self.bert.config.hidden_size)
            self.qa_outputs = nn.Linear(self.bert.config.hidden_size * 4, 2)
        else:
            # 标准BERT QA
            # Standard BERT QA
            self.qa_outputs = nn.Linear(self.bert.config.hidden_size, 2)
        
        # 无答案检测器
        # No answer detector
        self.answerable_classifier = nn.Linear(self.bert.config.hidden_size, 2)
        
        self.dropout = nn.Dropout(0.1)
        
        logger.info(f"模型初始化完成 | Model initialized: {model_name}")
        logger.info(f"使用双向注意力 | Use bidirectional attention: {use_bidirectional_attention}")
    
    def forward(self, input_ids, attention_mask, token_type_ids=None, 
                start_positions=None, end_positions=None, answerable_labels=None):
        """
        前向传播
        Forward pass
        """
        # BERT编码
        # BERT encoding
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output
        
        if self.use_bidirectional_attention:
            # 分离问题和文档表示
            # Separate question and context representations
            question_mask = token_type_ids == 0
            context_mask = token_type_ids == 1
            
            question_output = sequence_output * question_mask.unsqueeze(-1).float()
            context_output = sequence_output * context_mask.unsqueeze(-1).float()
            
            # 应用双向注意力
            # Apply bidirectional attention
            attended_output = self.bidirectional_attention(
                context_output, question_output, context_mask, question_mask
            )
            
            # 只保留文档部分用于预测
            # Keep only context part for prediction
            sequence_output = attended_output * context_mask.unsqueeze(-1).float()
        
        sequence_output = self.dropout(sequence_output)
        
        # 答案边界预测
        # Answer span prediction
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        # 可回答性预测
        # Answerability prediction
        answerable_logits = self.answerable_classifier(pooled_output)
        
        outputs = {
            'start_logits': start_logits,
            'end_logits': end_logits,
            'answerable_logits': answerable_logits
        }
        
        # 计算损失
        # Compute loss
        if start_positions is not None and end_positions is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            span_loss = (start_loss + end_loss) / 2
            
            total_loss = span_loss
            
            if answerable_labels is not None:
                answerable_loss = loss_fct(answerable_logits, answerable_labels)
                total_loss = span_loss + 0.5 * answerable_loss
            
            outputs['loss'] = total_loss
        
        return outputs


class MultiHopQAModel(nn.Module):
    """
    多跳推理问答模型
    Multi-hop Reasoning Question Answering Model
    
    实现多步推理机制来处理复杂问题
    Implements multi-step reasoning mechanism for complex questions
    """
    
    def __init__(self, model_name='bert-base-uncased', num_hops=3):
        super().__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        self.num_hops = num_hops
        
        # 推理步骤的注意力机制
        # Attention mechanism for reasoning steps
        self.reasoning_attention = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.bert.config.hidden_size,
                num_heads=8,
                batch_first=True
            ) for _ in range(num_hops)
        ])
        
        # 门控网络用于信息融合
        # Gating networks for information fusion
        self.gates = nn.ModuleList([
            nn.Linear(self.bert.config.hidden_size * 2, self.bert.config.hidden_size)
            for _ in range(num_hops)
        ])
        
        # 最终预测层
        # Final prediction layers
        self.qa_outputs = nn.Linear(self.bert.config.hidden_size, 2)
        self.answerable_classifier = nn.Linear(self.bert.config.hidden_size, 2)
        
        self.dropout = nn.Dropout(0.1)
        
        logger.info(f"多跳推理模型初始化完成 | Multi-hop model initialized: {num_hops} hops")
    
    def forward(self, input_ids, attention_mask, token_type_ids=None,
                start_positions=None, end_positions=None, answerable_labels=None):
        """
        前向传播
        Forward pass
        """
        # BERT编码
        # BERT encoding
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output
        
        # 多跳推理过程
        # Multi-hop reasoning process
        reasoning_state = sequence_output
        
        for hop in range(self.num_hops):
            # 自注意力推理
            # Self-attention reasoning
            attended_output, attention_weights = self.reasoning_attention[hop](
                reasoning_state, reasoning_state, reasoning_state,
                key_padding_mask=~attention_mask.bool()
            )
            
            # 门控融合
            # Gated fusion
            gate_input = torch.cat([reasoning_state, attended_output], dim=-1)
            gate = torch.sigmoid(self.gates[hop](gate_input))
            reasoning_state = gate * attended_output + (1 - gate) * reasoning_state
        
        reasoning_state = self.dropout(reasoning_state)
        
        # 最终预测
        # Final prediction
        logits = self.qa_outputs(reasoning_state)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        # 可回答性预测
        # Answerability prediction
        answerable_logits = self.answerable_classifier(pooled_output)
        
        outputs = {
            'start_logits': start_logits,
            'end_logits': end_logits,
            'answerable_logits': answerable_logits,
            'reasoning_states': reasoning_state
        }
        
        # 计算损失
        # Compute loss
        if start_positions is not None and end_positions is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            span_loss = (start_loss + end_loss) / 2
            
            total_loss = span_loss
            
            if answerable_labels is not None:
                answerable_loss = loss_fct(answerable_logits, answerable_labels)
                total_loss = span_loss + 0.5 * answerable_loss
            
            outputs['loss'] = total_loss
        
        return outputs


class QATrainer:
    """
    问答模型训练器
    Question Answering Model Trainer
    
    负责模型训练、验证和评估
    Responsible for model training, validation, and evaluation
    """
    
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        
        logger.info(f"训练器初始化完成 | Trainer initialized on device: {device}")
    
    def train(self, train_dataset, val_dataset, epochs=3, batch_size=16, 
              learning_rate=2e-5, warmup_steps=100):
        """
        训练模型
        Train model
        """
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self._collate_fn
        )
        
        # 设置优化器和调度器
        # Set up optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, eps=1e-8)
        
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        logger.info(f"开始训练 | Start training: {epochs} epochs, {total_steps} total steps")
        
        best_f1 = 0
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            
            # 训练阶段
            # Training phase
            train_loss = self._train_epoch(train_loader, optimizer, scheduler)
            logger.info(f"训练损失 | Training loss: {train_loss:.4f}")
            
            # 验证阶段
            # Validation phase
            val_metrics = self._evaluate(val_loader, val_dataset)
            logger.info(f"验证指标 | Validation metrics: {val_metrics}")
            
            # 保存最佳模型
            # Save best model
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                self._save_model('best_qa_model.pt')
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
            batch = {k: v.to(self.device) for k, v in batch.items() 
                    if k not in ['qa_id', 'offset_mapping', 'context_start_idx', 'context_end_idx']}
            
            # 前向传播
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(**batch)
            loss = outputs['loss']
            
            # 反向传播
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(train_loader)
    
    def _evaluate(self, val_loader, val_dataset):
        """
        评估模型
        Evaluate model
        """
        self.model.eval()
        
        all_predictions = {}
        all_references = {}
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                qa_ids = batch['qa_id']
                
                # 移除非tensor字段
                # Remove non-tensor fields
                model_inputs = {k: v.to(self.device) for k, v in batch.items() 
                              if k not in ['qa_id', 'offset_mapping', 'context_start_idx', 'context_end_idx']}
                
                outputs = self.model(**model_inputs)
                start_logits = outputs['start_logits']
                end_logits = outputs['end_logits']
                
                # 处理预测结果
                # Process predictions
                for i, qa_id in enumerate(qa_ids):
                    start_idx = torch.argmax(start_logits[i]).item()
                    end_idx = torch.argmax(end_logits[i]).item()
                    
                    # 确保end >= start
                    # Ensure end >= start
                    if end_idx < start_idx:
                        end_idx = start_idx
                    
                    # 解码答案
                    # Decode answer
                    input_ids = batch['input_ids'][i]
                    answer_tokens = input_ids[start_idx:end_idx+1]
                    predicted_answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
                    
                    all_predictions[qa_id] = predicted_answer
        
        # 获取真实答案
        # Get ground truth answers
        for feature in val_dataset.features:
            qa_id = feature['qa_id']
            if qa_id not in all_references:
                # 从原始样本中找到真实答案
                # Find ground truth answer from original examples
                for example in val_dataset.examples:
                    if example['qa_id'] == qa_id:
                        if example['is_impossible']:
                            all_references[qa_id] = ""
                        elif example['answers']:
                            all_references[qa_id] = example['answers'][0]['text']
                        else:
                            all_references[qa_id] = ""
                        break
        
        # 计算指标
        # Compute metrics
        metrics = self._compute_metrics(all_predictions, all_references)
        
        return metrics
    
    def _compute_metrics(self, predictions, references):
        """
        计算评估指标
        Compute evaluation metrics
        """
        exact_matches = 0
        f1_scores = []
        
        for qa_id in predictions:
            if qa_id in references:
                pred = self._normalize_answer(predictions[qa_id])
                ref = self._normalize_answer(references[qa_id])
                
                # 精确匹配
                # Exact match
                if pred == ref:
                    exact_matches += 1
                
                # F1分数
                # F1 score
                f1 = self._compute_f1(pred, ref)
                f1_scores.append(f1)
        
        total_samples = len(predictions)
        exact_match = exact_matches / total_samples if total_samples > 0 else 0
        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
        
        return {
            'exact_match': exact_match,
            'f1': avg_f1,
            'total_samples': total_samples
        }
    
    def _normalize_answer(self, answer):
        """
        标准化答案
        Normalize answer
        """
        # 去除冠词
        # Remove articles
        answer = re.sub(r'\b(a|an|the)\b', ' ', answer.lower())
        
        # 去除标点
        # Remove punctuation
        answer = ''.join(char for char in answer if char not in string.punctuation)
        
        # 去除多余空格
        # Remove extra spaces
        answer = ' '.join(answer.split())
        
        return answer
    
    def _compute_f1(self, pred_answer, true_answer):
        """
        计算F1分数
        Compute F1 score
        """
        pred_tokens = pred_answer.split()
        true_tokens = true_answer.split()
        
        if len(pred_tokens) == 0 and len(true_tokens) == 0:
            return 1.0
        if len(pred_tokens) == 0 or len(true_tokens) == 0:
            return 0.0
        
        common_tokens = set(pred_tokens) & set(true_tokens)
        
        if len(common_tokens) == 0:
            return 0.0
        
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(true_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        
        return f1
    
    def _collate_fn(self, batch):
        """
        批次数据整理函数
        Batch data collation function
        """
        collated = {}
        
        for key in batch[0].keys():
            if key in ['qa_id']:
                collated[key] = [item[key] for item in batch]
            elif key in ['offset_mapping', 'context_start_idx', 'context_end_idx']:
                collated[key] = [item[key] for item in batch]
            else:
                collated[key] = torch.stack([item[key] for item in batch])
        
        return collated
    
    def _save_model(self, path):
        """
        保存模型
        Save model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'tokenizer': self.tokenizer
        }, path)
    
    def predict_answer(self, question, context, max_length=384):
        """
        预测单个问题的答案
        Predict answer for a single question
        """
        self.model.eval()
        
        # 编码输入
        # Encode input
        encoding = self.tokenizer(
            question,
            context,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 移动到设备
        # Move to device
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        
        with torch.no_grad():
            outputs = self.model(**encoding)
            
            start_logits = outputs['start_logits']
            end_logits = outputs['end_logits']
            answerable_logits = outputs['answerable_logits']
            
            # 预测答案边界
            # Predict answer boundaries
            start_idx = torch.argmax(start_logits, dim=-1).item()
            end_idx = torch.argmax(end_logits, dim=-1).item()
            
            # 预测可回答性
            # Predict answerability
            answerable_prob = torch.softmax(answerable_logits, dim=-1)[0][1].item()
            
            # 确保答案合理性
            # Ensure answer validity
            if end_idx < start_idx:
                end_idx = start_idx
            
            # 解码答案
            # Decode answer
            answer_tokens = encoding['input_ids'][0][start_idx:end_idx+1]
            answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
            
            # 计算置信度
            # Compute confidence
            start_prob = torch.softmax(start_logits, dim=-1)[0][start_idx].item()
            end_prob = torch.softmax(end_logits, dim=-1)[0][end_idx].item()
            span_confidence = start_prob * end_prob
            
            return {
                'answer': answer,
                'confidence': span_confidence,
                'answerable_prob': answerable_prob,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'is_answerable': answerable_prob > 0.5
            }


def main():
    """
    主函数，演示QA系统的使用
    Main function demonstrating QA system usage
    """
    # 设置设备
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备 | Using device: {device}")
    
    # 初始化tokenizer
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # 创建模型
    # Create model
    model = BERTQuestionAnswering('bert-base-uncased', use_bidirectional_attention=True)
    
    # 创建训练器
    # Create trainer
    trainer = QATrainer(model, tokenizer, device)
    
    # 演示预测功能
    # Demonstrate prediction functionality
    context = """
    The Amazon rainforest, also known as Amazonia or the Amazon Jungle, is a moist broadleaf forest that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 square kilometers, of which 5,500,000 square kilometers are covered by the rainforest. The majority of the forest is contained within Brazil, with 60% of the rainforest, followed by Peru with 13%, Colombia with 10%, and with minor amounts in Venezuela, Ecuador, Bolivia, Guyana, Suriname and French Guiana.
    """
    
    questions = [
        "What is the Amazon rainforest also known as?",
        "How much area does the Amazon basin cover?",
        "Which country contains most of the Amazon rainforest?",
        "What percentage of the Amazon rainforest is in Peru?"
    ]
    
    print("\n=== 问答预测演示 | QA Prediction Demo ===")
    for question in questions:
        result = trainer.predict_answer(question, context)
        print(f"\n问题 | Question: {question}")
        print(f"答案 | Answer: {result['answer']}")
        print(f"置信度 | Confidence: {result['confidence']:.4f}")
        print(f"可回答性 | Answerability: {result['answerable_prob']:.4f}")


if __name__ == "__main__":
    main() 