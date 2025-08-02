"""
中文NER数据处理工具
Chinese NER Data Processing Tools

包含数据预处理、格式转换、增强等功能
Contains data preprocessing, format conversion, augmentation functions
"""

import json
import random
import re
from typing import List, Dict, Tuple
import jieba
from collections import defaultdict, Counter

class ChineseNERDataProcessor:
    """
    中文NER数据处理器
    Chinese NER Data Processor
    """
    
    def __init__(self):
        # 实体类型映射 | Entity type mapping
        self.entity_types = {
            'PER': '人名',    # Person
            'LOC': '地名',    # Location  
            'ORG': '机构名',  # Organization
            'MISC': '其他'    # Miscellaneous
        }
        
        # 数据增强字典 | Data augmentation dictionary
        self.augment_dict = {
            'PER': ['张三', '李四', '王五', '赵六', '孙七', '周八', '吴九', '郑十'],
            'LOC': ['北京', '上海', '广州', '深圳', '南京', '杭州', '成都', '西安'],
            'ORG': ['清华大学', '北京大学', '复旦大学', '浙江大学', '中山大学', '华中科技大学']
        }
    
    def load_raw_data(self, file_path: str) -> List[Dict]:
        """
        加载原始数据
        Load raw data
        """
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))
        return data
    
    def text_to_bio_labels(self, text: str, entities: List[Dict]) -> Tuple[List[str], List[str]]:
        """
        将文本和实体转换为BIO标注格式
        Convert text and entities to BIO labeling format
        """
        chars = list(text)
        labels = ['O'] * len(chars)
        
        # 按开始位置排序实体 | Sort entities by start position
        sorted_entities = sorted(entities, key=lambda x: x['start'])
        
        for entity in sorted_entities:
            start = entity['start']
            end = entity['end']
            entity_type = entity['type']
            
            if start < len(chars) and end <= len(chars):
                # 设置B标签 | Set B label
                labels[start] = f'B-{entity_type}'
                # 设置I标签 | Set I labels
                for i in range(start + 1, end):
                    labels[i] = f'I-{entity_type}'
        
        return chars, labels
    
    def bio_labels_to_entities(self, chars: List[str], bio_labels: List[str]) -> List[Dict]:
        """
        将BIO标注转换为实体列表
        Convert BIO labels to entity list
        """
        entities = []
        current_entity = None
        
        for i, (char, label) in enumerate(zip(chars, bio_labels)):
            if label.startswith('B-'):
                # 保存之前的实体 | Save previous entity
                if current_entity:
                    entities.append(current_entity)
                
                # 开始新实体 | Start new entity
                entity_type = label[2:]
                current_entity = {
                    'text': char,
                    'start': i,
                    'end': i + 1,
                    'type': entity_type
                }
            
            elif label.startswith('I-') and current_entity:
                # 继续当前实体 | Continue current entity
                entity_type = label[2:]
                if entity_type == current_entity['type']:
                    current_entity['text'] += char
                    current_entity['end'] = i + 1
                else:
                    # 类型不匹配，结束当前实体 | Type mismatch, end current entity
                    entities.append(current_entity)
                    current_entity = None
            
            elif label == 'O':
                # 结束当前实体 | End current entity
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        # 处理最后一个实体 | Handle last entity
        if current_entity:
            entities.append(current_entity)
        
        return entities
    
    def validate_bio_sequence(self, bio_labels: List[str]) -> bool:
        """
        验证BIO序列的有效性
        Validate BIO sequence validity
        """
        for i, label in enumerate(bio_labels):
            if label.startswith('I-'):
                # I标签前面必须是相同类型的B或I标签
                # I label must be preceded by B or I label of same type
                if i == 0:
                    return False
                
                entity_type = label[2:]
                prev_label = bio_labels[i-1]
                
                if not (prev_label == f'B-{entity_type}' or prev_label == f'I-{entity_type}'):
                    return False
        
        return True
    
    def clean_text(self, text: str) -> str:
        """
        清理文本数据
        Clean text data
        """
        # 去除多余空格 | Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        
        # 去除特殊字符 | Remove special characters
        text = re.sub(r'[^\u4e00-\u9fff\w\s.,!?;:()""''【】]', '', text)
        
        # 统一标点符号 | Normalize punctuation
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text.strip()
    
    def split_long_text(self, text: str, entities: List[Dict], max_length: int = 100) -> List[Tuple[str, List[Dict]]]:
        """
        分割长文本为短片段
        Split long text into short segments
        """
        if len(text) <= max_length:
            return [(text, entities)]
        
        segments = []
        start = 0
        
        while start < len(text):
            end = min(start + max_length, len(text))
            
            # 尝试在句号、问号、感叹号处分割
            # Try to split at period, question mark, exclamation mark
            if end < len(text):
                for punct in ['。', '！', '？']:
                    punct_pos = text.rfind(punct, start, end)
                    if punct_pos > start:
                        end = punct_pos + 1
                        break
            
            segment_text = text[start:end]
            
            # 找到该片段中的实体 | Find entities in this segment
            segment_entities = []
            for entity in entities:
                if entity['start'] >= start and entity['end'] <= end:
                    # 调整实体位置 | Adjust entity positions
                    new_entity = entity.copy()
                    new_entity['start'] -= start
                    new_entity['end'] -= start
                    segment_entities.append(new_entity)
            
            segments.append((segment_text, segment_entities))
            start = end
        
        return segments
    
    def entity_substitution_augment(self, text: str, entities: List[Dict], num_augments: int = 3) -> List[Tuple[str, List[Dict]]]:
        """
        实体替换数据增强
        Entity substitution data augmentation
        """
        augmented_samples = []
        
        for _ in range(num_augments):
            new_text = text
            new_entities = []
            offset = 0  # 位置偏移量 | Position offset
            
            # 从前往后处理实体 | Process entities from front to back
            for entity in sorted(entities, key=lambda x: x['start']):
                entity_type = entity['type']
                original_text = entity['text']
                
                if entity_type in self.augment_dict:
                    # 随机选择替换词 | Randomly select replacement
                    replacement = random.choice(self.augment_dict[entity_type])
                    
                    # 计算调整后的位置 | Calculate adjusted positions
                    adjusted_start = entity['start'] + offset
                    adjusted_end = entity['end'] + offset
                    
                    # 执行替换 | Perform replacement
                    new_text = new_text[:adjusted_start] + replacement + new_text[adjusted_end:]
                    
                    # 更新偏移量 | Update offset
                    offset += len(replacement) - len(original_text)
                    
                    # 创建新实体 | Create new entity
                    new_entity = {
                        'text': replacement,
                        'start': adjusted_start,
                        'end': adjusted_start + len(replacement),
                        'type': entity_type
                    }
                    new_entities.append(new_entity)
                else:
                    # 保持原实体，但调整位置 | Keep original entity but adjust position
                    new_entity = entity.copy()
                    new_entity['start'] += offset
                    new_entity['end'] += offset
                    new_entities.append(new_entity)
            
            augmented_samples.append((new_text, new_entities))
        
        return augmented_samples
    
    def context_perturbation_augment(self, text: str, entities: List[Dict]) -> str:
        """
        上下文扰动增强
        Context perturbation augmentation
        """
        # 同义词替换字典 | Synonym replacement dictionary
        synonyms = {
            '访问': ['拜访', '探访', '造访'],
            '会见': ['接见', '会面', '见面'],
            '签署': ['签订', '签约', '签定'],
            '成立': ['创立', '建立', '设立'],
            '发布': ['公布', '宣布', '发表'],
            '举行': ['召开', '进行', '开展']
        }
        
        # 使用jieba分词 | Use jieba word segmentation
        words = list(jieba.cut(text))
        new_words = []
        
        # 获取实体位置以避免替换实体内容
        # Get entity positions to avoid replacing entity content
        entity_positions = set()
        for entity in entities:
            for pos in range(entity['start'], entity['end']):
                entity_positions.add(pos)
        
        current_pos = 0
        for word in words:
            # 检查当前词是否在实体中 | Check if current word is in entity
            word_in_entity = any(pos in entity_positions for pos in range(current_pos, current_pos + len(word)))
            
            if not word_in_entity and word in synonyms and random.random() < 0.3:
                # 30%概率进行同义词替换 | 30% chance for synonym replacement
                new_words.append(random.choice(synonyms[word]))
            else:
                new_words.append(word)
            
            current_pos += len(word)
        
        return ''.join(new_words)
    
    def analyze_data_statistics(self, data: List[Dict]) -> Dict:
        """
        分析数据统计信息
        Analyze data statistics
        """
        stats = {
            'total_samples': len(data),
            'total_entities': 0,
            'entity_type_counts': defaultdict(int),
            'entity_length_dist': defaultdict(int),
            'text_length_dist': defaultdict(int),
            'avg_text_length': 0,
            'avg_entities_per_text': 0
        }
        
        total_text_length = 0
        total_entities = 0
        
        for item in data:
            text = item['text']
            entities = item.get('entities', [])
            
            # 文本长度统计 | Text length statistics
            text_length = len(text)
            total_text_length += text_length
            stats['text_length_dist'][text_length // 10 * 10] += 1  # 按10字符分组
            
            # 实体统计 | Entity statistics
            entity_count = len(entities)
            total_entities += entity_count
            
            for entity in entities:
                entity_type = entity['type']
                entity_length = len(entity['text'])
                
                stats['entity_type_counts'][entity_type] += 1
                stats['entity_length_dist'][entity_length] += 1
        
        stats['total_entities'] = total_entities
        stats['avg_text_length'] = total_text_length / len(data) if data else 0
        stats['avg_entities_per_text'] = total_entities / len(data) if data else 0
        
        return dict(stats)
    
    def convert_to_conll_format(self, data: List[Dict], output_path: str):
        """
        转换为CoNLL格式
        Convert to CoNLL format
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                text = item['text']
                entities = item.get('entities', [])
                
                chars, bio_labels = self.text_to_bio_labels(text, entities)
                
                for char, label in zip(chars, bio_labels):
                    f.write(f"{char}\t{label}\n")
                f.write("\n")  # 空行分隔句子
    
    def load_from_conll_format(self, file_path: str) -> List[Dict]:
        """
        从CoNLL格式加载数据
        Load data from CoNLL format
        """
        data = []
        current_chars = []
        current_labels = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    if len(parts) == 2:
                        char, label = parts
                        current_chars.append(char)
                        current_labels.append(label)
                else:
                    # 空行，处理当前句子 | Empty line, process current sentence
                    if current_chars and current_labels:
                        text = ''.join(current_chars)
                        entities = self.bio_labels_to_entities(current_chars, current_labels)
                        
                        data.append({
                            'text': text,
                            'entities': entities
                        })
                        
                        current_chars = []
                        current_labels = []
        
        # 处理最后一个句子 | Process last sentence
        if current_chars and current_labels:
            text = ''.join(current_chars)
            entities = self.bio_labels_to_entities(current_chars, current_labels)
            
            data.append({
                'text': text,
                'entities': entities
            })
        
        return data
    
    def balance_dataset(self, data: List[Dict], min_samples_per_type: int = 100) -> List[Dict]:
        """
        平衡数据集中各实体类型的分布
        Balance distribution of entity types in dataset
        """
        # 统计各类型的样本数 | Count samples for each type
        type_samples = defaultdict(list)
        
        for item in data:
            entities = item.get('entities', [])
            entity_types_in_item = set(entity['type'] for entity in entities)
            
            for entity_type in entity_types_in_item:
                type_samples[entity_type].append(item)
        
        # 找到需要增强的类型 | Find types that need augmentation
        balanced_data = list(data)  # 开始时包含所有原始数据
        
        for entity_type, samples in type_samples.items():
            if len(samples) < min_samples_per_type:
                # 需要增强 | Need augmentation
                needed = min_samples_per_type - len(samples)
                
                # 通过实体替换增强 | Augment through entity substitution
                for _ in range(needed):
                    source_item = random.choice(samples)
                    augmented_samples = self.entity_substitution_augment(
                        source_item['text'], 
                        source_item['entities'], 
                        num_augments=1
                    )
                    
                    if augmented_samples:
                        aug_text, aug_entities = augmented_samples[0]
                        balanced_data.append({
                            'text': aug_text,
                            'entities': aug_entities
                        })
        
        return balanced_data


def main():
    """
    演示数据处理功能
    Demonstrate data processing functions
    """
    processor = ChineseNERDataProcessor()
    
    # 创建示例数据 | Create example data
    sample_data = [
        {
            "text": "马云创立了阿里巴巴集团，总部位于杭州",
            "entities": [
                {"start": 0, "end": 2, "type": "PER", "text": "马云"},
                {"start": 5, "end": 10, "type": "ORG", "text": "阿里巴巴集团"},
                {"start": 15, "end": 17, "type": "LOC", "text": "杭州"}
            ]
        },
        {
            "text": "小芳领导人在北京人民大会堂会见了美国总统拜登",
            "entities": [
                {"start": 0, "end": 3, "type": "PER", "text": "小芳"},
                {"start": 6, "end": 12, "type": "LOC", "text": "北京人民大会堂"},
                {"start": 16, "end": 18, "type": "LOC", "text": "美国"},
                {"start": 20, "end": 22, "type": "PER", "text": "拜登"}
            ]
        }
    ]
    
    print("=== 数据统计分析 | Data Statistics Analysis ===")
    stats = processor.analyze_data_statistics(sample_data)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n=== BIO标注转换 | BIO Labeling Conversion ===")
    for item in sample_data:
        text = item['text']
        entities = item['entities']
        chars, bio_labels = processor.text_to_bio_labels(text, entities)
        
        print(f"文本: {text}")
        print("BIO标注:")
        for char, label in zip(chars, bio_labels):
            print(f"  {char}: {label}")
        print()
    
    print("=== 数据增强示例 | Data Augmentation Example ===")
    item = sample_data[0]
    augmented = processor.entity_substitution_augment(
        item['text'], item['entities'], num_augments=2
    )
    
    print(f"原始: {item['text']}")
    for i, (aug_text, aug_entities) in enumerate(augmented):
        print(f"增强{i+1}: {aug_text}")
        print(f"实体: {aug_entities}")
        print()


if __name__ == "__main__":
    main() 