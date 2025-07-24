# 中文NER系统测试题
# Chinese NER System Quiz

---

## 📝 理论知识测试 | Theoretical Knowledge Test

### 1. 基础概念题 | Basic Concept Questions

**Question 1:** 
在中文NER任务中，为什么通常采用字符级别而不是词级别的tokenization？请列举至少3个原因。

In Chinese NER tasks, why is character-level tokenization usually used instead of word-level tokenization? Please list at least 3 reasons.

**Answer:**
1. **避免分词错误传播** | **Avoid word segmentation error propagation**: 中文分词本身就有错误，这些错误会直接影响NER的准确性
2. **处理未登录词** | **Handle out-of-vocabulary words**: 字符级别可以更好地处理训练时未见过的新词
3. **实体边界更精确** | **More precise entity boundaries**: 字符级别可以更准确地定位实体的起始和结束位置
4. **统一处理方式** | **Unified processing approach**: 不需要针对不同领域重新训练分词器

---

**Question 2:**
请解释BIO标注体系，并给出标注示例："北京大学的张教授"

Please explain the BIO tagging system and provide a tagging example for: "北京大学的张教授"

**Answer:**
**BIO标注体系说明 | BIO Tagging System Explanation:**
- **B-TYPE**: 实体的开始字符 | Beginning of entity
- **I-TYPE**: 实体的内部字符 | Inside of entity  
- **O**: 不属于任何实体 | Outside any entity

**标注示例 | Tagging Example:**
```
北  B-ORG
京  I-ORG  
大  I-ORG
学  I-ORG
的  O
张  B-PER
教  I-PER
授  I-PER
```

---

### 2. 技术原理题 | Technical Principle Questions

**Question 3:**
为什么在BERT+CRF架构中需要使用CRF层？直接使用BERT的输出做分类有什么问题？

Why is a CRF layer needed in the BERT+CRF architecture? What problems exist with directly using BERT output for classification?

**Answer:**
**CRF层的必要性 | Necessity of CRF Layer:**

1. **序列一致性约束** | **Sequence consistency constraints**: 
   - 防止无效的标签转移，如"I-PER"后直接跟"B-ORG"
   - Prevent invalid label transitions like "I-PER" directly followed by "B-ORG"

2. **全局优化** | **Global optimization**:
   - BERT单独分类是局部决策，CRF考虑整个序列的最优标注
   - BERT alone makes local decisions, CRF considers optimal labeling for entire sequence

3. **标注规则强制** | **Tagging rule enforcement**:
   - 确保实体必须以B开头，不能有孤立的I标签
   - Ensure entities must start with B, no isolated I labels

**直接使用BERT分类的问题 | Problems with Direct BERT Classification:**
- 可能产生不合法的标签序列
- May produce illegal label sequences
- 无法利用标签间的依赖关系
- Cannot utilize dependencies between labels

---

**Question 4:**
在处理类别不平衡的NER数据时，有哪些策略可以改善模型性能？

What strategies can improve model performance when dealing with class-imbalanced NER data?

**Answer:**
**类别不平衡处理策略 | Class Imbalance Handling Strategies:**

1. **数据层面 | Data Level:**
   - **过采样** | **Oversampling**: 增加少数类实体的样本
   - **数据增强** | **Data Augmentation**: 实体替换、上下文扰动
   - **合成数据生成** | **Synthetic Data Generation**: 使用模板生成新样本

2. **损失函数层面 | Loss Function Level:**
   - **Focal Loss**: 降低易分类样本的权重
   - **类别权重** | **Class Weights**: 给少数类更高的权重
   - **标签平滑** | **Label Smoothing**: 减少过拟合

3. **模型层面 | Model Level:**
   - **集成学习** | **Ensemble Learning**: 多模型投票
   - **主动学习** | **Active Learning**: 主动选择有价值的样本标注

---

## 💻 编程实践题 | Programming Practice Questions

### 3. 代码实现题 | Code Implementation Questions

**Question 5:**
实现一个函数，将BIO格式的标注转换为实体列表。

Implement a function to convert BIO format annotations to entity list.

```python
def bio_to_entities(chars, bio_tags):
    """
    将BIO标注转换为实体列表
    Convert BIO annotations to entity list
    
    Args:
        chars: 字符列表 | Character list
        bio_tags: BIO标签列表 | BIO tag list
    
    Returns:
        entities: 实体列表 | Entity list
        格式: [{'text': '实体文本', 'start': 开始位置, 'end': 结束位置, 'type': '实体类型'}]
    """
    # 请实现此函数 | Please implement this function
    pass
```

**Answer:**
```python
def bio_to_entities(chars, bio_tags):
    """
    将BIO标注转换为实体列表
    Convert BIO annotations to entity list
    """
    entities = []
    current_entity = None
    
    for i, (char, tag) in enumerate(zip(chars, bio_tags)):
        if tag.startswith('B-'):
            # 如果之前有实体，先保存
            # If there was a previous entity, save it first
            if current_entity:
                entities.append(current_entity)
            
            # 开始新实体
            # Start new entity
            entity_type = tag[2:]  # 去掉'B-'前缀
            current_entity = {
                'text': char,
                'start': i,
                'end': i + 1,
                'type': entity_type
            }
        
        elif tag.startswith('I-') and current_entity:
            # 继续当前实体
            # Continue current entity
            entity_type = tag[2:]
            if entity_type == current_entity['type']:
                current_entity['text'] += char
                current_entity['end'] = i + 1
            else:
                # 类型不匹配，保存当前实体并开始新实体
                # Type mismatch, save current entity and start new one
                entities.append(current_entity)
                current_entity = {
                    'text': char,
                    'start': i,
                    'end': i + 1,
                    'type': entity_type
                }
        
        elif tag == 'O':
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

# 测试示例 | Test Example
chars = ['北', '京', '大', '学', '的', '张', '教', '授']
tags = ['B-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'O', 'B-PER', 'I-PER', 'I-PER']

result = bio_to_entities(chars, tags)
print(result)
# 期望输出 | Expected Output:
# [
#     {'text': '北京大学', 'start': 0, 'end': 4, 'type': 'ORG'},
#     {'text': '张教授', 'start': 5, 'end': 8, 'type': 'PER'}
# ]
```

---

**Question 6:**
实现一个数据增强函数，通过实体替换来扩充训练数据。

Implement a data augmentation function to expand training data through entity substitution.

```python
def entity_substitution_augment(text, entities, entity_dict, num_augments=3):
    """
    通过实体替换进行数据增强
    Data augmentation through entity substitution
    
    Args:
        text: 原始文本 | Original text
        entities: 实体列表 | Entity list  
        entity_dict: 实体替换词典 | Entity substitution dictionary
        num_augments: 增强样本数量 | Number of augmented samples
        
    Returns:
        augmented_samples: 增强后的样本列表 | List of augmented samples
    """
    # 请实现此函数 | Please implement this function
    pass
```

**Answer:**
```python
import random

def entity_substitution_augment(text, entities, entity_dict, num_augments=3):
    """
    通过实体替换进行数据增强
    Data augmentation through entity substitution
    """
    augmented_samples = []
    
    for _ in range(num_augments):
        augmented_text = text
        augmented_entities = []
        
        # 从后往前替换，避免位置偏移
        # Replace from back to front to avoid position offset
        for entity in sorted(entities, key=lambda x: x['start'], reverse=True):
            entity_type = entity['type']
            original_text = entity['text']
            
            # 如果该类型有替换词典
            # If substitution dictionary exists for this type
            if entity_type in entity_dict and len(entity_dict[entity_type]) > 0:
                # 随机选择替换词
                # Randomly select replacement word
                replacement = random.choice(entity_dict[entity_type])
                
                # 执行替换
                # Perform replacement
                start, end = entity['start'], entity['end']
                augmented_text = augmented_text[:start] + replacement + augmented_text[end:]
                
                # 更新实体信息
                # Update entity information
                new_entity = {
                    'text': replacement,
                    'start': start,
                    'end': start + len(replacement),
                    'type': entity_type
                }
                augmented_entities.insert(0, new_entity)  # 插入到前面保持顺序
            else:
                # 保持原实体
                # Keep original entity
                augmented_entities.insert(0, entity)
        
        augmented_samples.append({
            'text': augmented_text,
            'entities': augmented_entities
        })
    
    return augmented_samples

# 测试示例 | Test Example
text = "张三在北京大学工作"
entities = [
    {'text': '张三', 'start': 0, 'end': 2, 'type': 'PER'},
    {'text': '北京大学', 'start': 3, 'end': 7, 'type': 'ORG'}
]

entity_dict = {
    'PER': ['李四', '王五', '赵六'],
    'ORG': ['清华大学', '复旦大学', '浙江大学']
}

augmented = entity_substitution_augment(text, entities, entity_dict, num_augments=2)
for i, sample in enumerate(augmented):
    print(f"增强样本{i+1} | Augmented Sample {i+1}:")
    print(f"文本: {sample['text']}")
    print(f"实体: {sample['entities']}")
    print()
```

---

### 4. 案例分析题 | Case Analysis Questions

**Question 7:**
分析以下中文NER的错误案例，说明可能的原因和改进方法：

Analyze the following Chinese NER error cases, explain possible causes and improvement methods:

**案例1 | Case 1:**
- 输入文本 | Input Text: "苹果公司的新iPhone手机"
- 正确标注 | Correct Annotation: 苹果公司(ORG), iPhone(PRODUCT)  
- 模型预测 | Model Prediction: 苹果(O), 公司(O), iPhone(O)

**案例2 | Case 2:**
- 输入文本 | Input Text: "王建国家很富有"
- 正确标注 | Correct Annotation: 王建国(PER), 家(O)
- 模型预测 | Model Prediction: 王建国家(LOC)

**Answer:**

**案例1分析 | Case 1 Analysis:**

**可能原因 | Possible Causes:**
1. **训练数据不足** | **Insufficient Training Data**: 缺少相关的公司名和产品名样本
2. **实体边界识别困难** | **Entity Boundary Recognition Difficulty**: "苹果公司"作为一个整体实体的训练不够
3. **多义性问题** | **Polysemy Issue**: "苹果"既可以是水果也可以是公司名

**改进方法 | Improvement Methods:**
1. **数据增强** | **Data Augmentation**: 增加更多公司名和产品名的训练样本
2. **预训练模型优化** | **Pretrained Model Optimization**: 使用在商业文本上预训练的模型
3. **上下文特征增强** | **Context Feature Enhancement**: 利用"公司"等关键词作为强特征

**案例2分析 | Case 2 Analysis:**

**可能原因 | Possible Causes:**
1. **分词歧义** | **Segmentation Ambiguity**: "王建国家"可能被理解为地名
2. **语义理解不足** | **Insufficient Semantic Understanding**: 模型没有理解"家"在此处的含义
3. **训练数据偏差** | **Training Data Bias**: 训练数据中"XX国家"形式的地名较多

**改进方法 | Improvement Methods:**
1. **语法特征融入** | **Grammar Feature Integration**: 考虑词性和语法结构
2. **上下文窗口扩大** | **Context Window Expansion**: 增大模型看到的上下文范围
3. **多任务学习** | **Multi-task Learning**: 结合分词和NER联合训练

---

**Question 8:**
设计一个评估中文NER系统的完整方案，包括数据集选择、评估指标和测试场景。

Design a complete evaluation scheme for Chinese NER systems, including dataset selection, evaluation metrics, and test scenarios.

**Answer:**

**完整评估方案 | Complete Evaluation Scheme:**

### 1. 数据集选择 | Dataset Selection

**标准数据集 | Standard Datasets:**
- **MSRA NER**: 微软亚洲研究院中文NER数据集
- **People's Daily**: 人民日报标注数据集  
- **Weibo NER**: 社交媒体文本数据集
- **CLUENER**: 中文细粒度NER数据集

**领域特定数据集 | Domain-specific Datasets:**
- 医疗领域：病历文本、医学论文
- Medical Domain: Medical records, medical papers
- 金融领域：财经新闻、公司公告
- Financial Domain: Financial news, company announcements
- 法律领域：法律文书、合同文本
- Legal Domain: Legal documents, contract texts

### 2. 评估指标 | Evaluation Metrics

**核心指标 | Core Metrics:**
```python
# 实体级别精确匹配
# Entity-level exact match
def entity_level_metrics(true_entities, pred_entities):
    """
    计算实体级别的P/R/F1
    Calculate entity-level P/R/F1
    """
    true_set = set(true_entities)  # (start, end, type)
    pred_set = set(pred_entities)
    
    precision = len(true_set & pred_set) / len(pred_set) if pred_set else 0
    recall = len(true_set & pred_set) / len(true_set) if true_set else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    
    return precision, recall, f1

# 松弛匹配评估
# Relaxed matching evaluation  
def relaxed_matching_metrics(true_entities, pred_entities, overlap_threshold=0.5):
    """
    允许部分重叠的松弛匹配
    Relaxed matching allowing partial overlap
    """
    matches = 0
    for true_entity in true_entities:
        for pred_entity in pred_entities:
            if entities_overlap(true_entity, pred_entity, overlap_threshold):
                matches += 1
                break
    
    precision = matches / len(pred_entities) if pred_entities else 0
    recall = matches / len(true_entities) if true_entities else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    
    return precision, recall, f1
```

**细分指标 | Detailed Metrics:**
- **按实体类型评估** | **Evaluation by Entity Type**: 分别计算PER、LOC、ORG等的性能
- **按实体长度评估** | **Evaluation by Entity Length**: 短实体vs长实体的识别效果
- **按频率评估** | **Evaluation by Frequency**: 高频实体vs低频实体的性能

### 3. 测试场景 | Test Scenarios

**基础功能测试 | Basic Functionality Tests:**
```python
def basic_functionality_tests():
    """
    基础功能测试用例
    Basic functionality test cases
    """
    test_cases = [
        # 标准实体识别
        # Standard entity recognition
        {
            'text': '习近平主席在北京会见了美国总统拜登',
            'expected': [('习近平', 'PER'), ('北京', 'LOC'), ('美国', 'LOC'), ('拜登', 'PER')]
        },
        
        # 嵌套实体
        # Nested entities
        {
            'text': '北京大学计算机学院',
            'expected': [('北京大学', 'ORG'), ('计算机学院', 'ORG')]
        },
        
        # 边界情况
        # Boundary cases
        {
            'text': '王建国家很富有',
            'expected': [('王建国', 'PER')]
        }
    ]
    
    return test_cases
```

**鲁棒性测试 | Robustness Tests:**
```python
def robustness_tests():
    """
    鲁棒性测试用例
    Robustness test cases
    """
    return [
        # 噪声文本
        # Noisy text
        '马@云#创#立$了%阿#里&巴*巴',
        
        # 非标准格式  
        # Non-standard format
        'JACK MA创立了alibaba公司',
        
        # 长文本
        # Long text
        '在这个充满挑战的时代，企业家马云先生凭借其卓越的商业眼光...' * 100
    ]
```

**性能压力测试 | Performance Stress Tests:**
```python
def performance_tests():
    """
    性能压力测试
    Performance stress tests
    """
    return {
        'batch_processing': '测试批量处理能力',
        'memory_usage': '监控内存使用情况', 
        'inference_speed': '推理速度基准测试',
        'concurrent_requests': '并发请求处理能力'
    }
```

**领域适应性测试 | Domain Adaptation Tests:**
- **跨领域泛化** | **Cross-domain Generalization**: 在金融数据上训练，在医疗数据上测试
- **时间稳定性** | **Temporal Stability**: 在不同时间段的数据上测试
- **文体适应性** | **Genre Adaptability**: 新闻、小说、社交媒体等不同文体

这个完整的测试题集涵盖了中文NER系统的理论基础、技术实现和实际应用，帮助学习者全面掌握相关知识和技能！

This complete quiz set covers the theoretical foundations, technical implementation, and practical applications of Chinese NER systems, helping learners comprehensively master relevant knowledge and skills! 