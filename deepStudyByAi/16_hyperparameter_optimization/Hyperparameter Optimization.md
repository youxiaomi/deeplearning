# Chapter 16: Hyperparameter Optimization 第16章：超参数优化

## 16.1. What Is Hyperparameter Optimization? 什么是超参数优化？

### Introduction 介绍

Hyperparameter optimization (HPO) is one of the most critical yet challenging aspects of machine learning model development. 超参数优化（HPO）是机器学习模型开发中最关键但也最具挑战性的方面之一。

Unlike model parameters that are learned during training (such as weights and biases in neural networks), hyperparameters are configuration settings that must be set before training begins. 与在训练过程中学习的模型参数（如神经网络中的权重和偏置）不同，超参数是必须在训练开始前设置的配置参数。

Think of hyperparameters as the "recipe settings" for baking a cake. 将超参数想象成烘焙蛋糕的"配方设置"。Just as you need to decide the oven temperature, baking time, and ingredient proportions before you start baking, you need to set hyperparameters like learning rate, batch size, and network architecture before training your model. 就像您需要在开始烘焙前决定烤箱温度、烘焙时间和配料比例一样，您需要在训练模型前设置学习率、批次大小和网络架构等超参数。

### Common Hyperparameters 常见超参数

**Learning Rate (学习率)**: Controls how quickly the model learns. 控制模型学习的速度。Too high and the model might overshoot the optimal solution; too low and training becomes extremely slow. 过高可能导致模型错过最优解；过低则训练变得极其缓慢。

**Batch Size (批次大小)**: Number of training examples processed together. 一起处理的训练样本数量。Larger batches provide more stable gradients but require more memory. 较大的批次提供更稳定的梯度但需要更多内存。

**Network Architecture (网络架构)**: Number of layers, neurons per layer, activation functions. 层数、每层神经元数量、激活函数。This is like deciding the complexity of your model's "brain structure". 这就像决定模型"大脑结构"的复杂性。

**Regularization Parameters (正则化参数)**: L1/L2 regularization coefficients, dropout rates. L1/L2正则化系数、dropout率。These prevent overfitting, similar to adding constraints to prevent memorization instead of learning. 这些防止过拟合，类似于添加约束以防止记忆而非学习。

### 16.1.1. The Optimization Problem 优化问题

Hyperparameter optimization can be formalized as finding the configuration that minimizes a validation error: 超参数优化可以形式化为找到最小化验证误差的配置：

```
θ* = argmin_θ L_val(A_θ(D_train), D_val)
```

Where: 其中：
- `θ` represents the hyperparameter configuration 表示超参数配置
- `A_θ` is the learning algorithm with hyperparameters θ 是使用超参数θ的学习算法
- `D_train` and `D_val` are training and validation datasets 是训练和验证数据集
- `L_val` is the validation loss function 是验证损失函数

**Real-world Example 实际例子**: Imagine you're tuning a neural network to recognize handwritten digits. 想象您正在调整神经网络以识别手写数字。Your hyperparameters might include:
- Learning rate: 0.001, 0.01, 0.1 学习率：0.001, 0.01, 0.1
- Hidden layers: 1, 2, 3 隐藏层：1, 2, 3
- Neurons per layer: 64, 128, 256 每层神经元：64, 128, 256

Each combination creates a different "recipe" for your model, and you need to find the best one. 每种组合为您的模型创建不同的"配方"，您需要找到最佳组合。

### Challenges in Hyperparameter Optimization 超参数优化的挑战

**1. High Dimensional Search Space (高维搜索空间)**: With multiple hyperparameters, the search space grows exponentially. 随着多个超参数的存在，搜索空间呈指数增长。If you have 5 hyperparameters each with 10 possible values, that's 10^5 = 100,000 combinations! 如果您有5个超参数，每个有10个可能值，那就是10^5 = 100,000种组合！

**2. Expensive Function Evaluation (昂贵的函数评估)**: Each hyperparameter configuration requires training a complete model, which can take hours or days. 每个超参数配置都需要训练一个完整的模型，这可能需要数小时或数天。

**3. Noisy Objective Function (噪声目标函数)**: Model performance can vary due to random initialization and data shuffling. 由于随机初始化和数据打乱，模型性能可能会发生变化。

**4. Mixed Variable Types (混合变量类型)**: Some hyperparameters are continuous (learning rate), others discrete (number of layers), and some categorical (optimizer type). 一些超参数是连续的（学习率），其他是离散的（层数），还有一些是分类的（优化器类型）。

### 16.1.2. Random Search 随机搜索

Random search is one of the simplest yet surprisingly effective methods for hyperparameter optimization. 随机搜索是超参数优化中最简单但出人意料地有效的方法之一。

**How Random Search Works 随机搜索的工作原理**:

1. **Define Search Space (定义搜索空间)**: Specify the range or set of possible values for each hyperparameter. 为每个超参数指定可能值的范围或集合。

2. **Random Sampling (随机采样)**: Randomly sample hyperparameter configurations from the defined space. 从定义的空间中随机采样超参数配置。

3. **Evaluate Performance (评估性能)**: Train and evaluate the model for each sampled configuration. 为每个采样配置训练和评估模型。

4. **Select Best Configuration (选择最佳配置)**: Choose the configuration that achieves the best validation performance. 选择实现最佳验证性能的配置。

**Example Implementation 示例实现**:

```python
import random
import numpy as np

def random_search(search_space, num_trials=50):
    """
    Perform random search for hyperparameter optimization
    执行超参数优化的随机搜索
    
    Args:
        search_space: Dictionary defining parameter ranges
                     定义参数范围的字典
        num_trials: Number of random configurations to try
                   要尝试的随机配置数量
    """
    best_config = None
    best_score = float('inf')
    
    for trial in range(num_trials):
        # Sample random configuration 采样随机配置
        config = {}
        for param_name, param_range in search_space.items():
            if param_range['type'] == 'uniform':
                config[param_name] = random.uniform(
                    param_range['low'], param_range['high']
                )
            elif param_range['type'] == 'choice':
                config[param_name] = random.choice(param_range['values'])
        
        # Evaluate configuration (placeholder) 评估配置（占位符）
        score = evaluate_model(config)
        
        # Update best configuration 更新最佳配置
        if score < best_score:
            best_score = score
            best_config = config
    
    return best_config, best_score

# Example search space 示例搜索空间
search_space = {
    'learning_rate': {'type': 'uniform', 'low': 0.0001, 'high': 0.1},
    'batch_size': {'type': 'choice', 'values': [16, 32, 64, 128]},
    'hidden_size': {'type': 'choice', 'values': [64, 128, 256, 512]},
    'dropout_rate': {'type': 'uniform', 'low': 0.0, 'high': 0.5}
}
```

**Why Random Search Works Well 为什么随机搜索效果很好**:

Random search is surprisingly effective because many hyperparameters don't significantly affect model performance. 随机搜索出人意料地有效，因为许多超参数不会显著影响模型性能。The key insight is that if only a few hyperparameters really matter, random search is likely to find good values for these important parameters. 关键洞察是，如果只有少数超参数真正重要，随机搜索很可能为这些重要参数找到好的值。

**Comparison with Grid Search 与网格搜索的比较**:

Grid search evaluates all combinations in a regular grid, while random search samples randomly. 网格搜索评估规则网格中的所有组合，而随机搜索随机采样。

Consider this analogy: 考虑这个类比：If you're looking for a lost item in a dark room, grid search is like systematically checking every corner in a predetermined pattern, while random search is like randomly shining a flashlight around. 如果您在黑暗房间中寻找丢失的物品，网格搜索就像按预定模式系统地检查每个角落，而随机搜索就像随机照射手电筒。

For many problems, random search finds good solutions faster because it doesn't waste time on unimportant parameter combinations. 对于许多问题，随机搜索更快找到好的解决方案，因为它不会在不重要的参数组合上浪费时间。

### 16.1.3. Summary 总结

Hyperparameter optimization is essential for achieving good model performance, but it comes with significant challenges: 超参数优化对于实现良好的模型性能至关重要，但它带来了重大挑战：

**Key Takeaways 关键要点**:

1. **Hyperparameters are crucial (超参数至关重要)**: The difference between good and bad hyperparameters can make or break your model's performance. 好的和坏的超参数之间的差异可以决定模型性能的成败。

2. **Random search is a strong baseline (随机搜索是强基线)**: Before trying complex optimization methods, random search often provides surprisingly good results. 在尝试复杂优化方法之前，随机搜索通常提供出人意料的好结果。

3. **Evaluation is expensive (评估代价昂贵)**: Each configuration requires full model training, making efficient search strategies crucial. 每个配置都需要完整的模型训练，使得高效的搜索策略至关重要。

4. **Multiple objectives matter (多个目标很重要)**: Besides accuracy, consider training time, model size, and inference speed. 除了准确性，还要考虑训练时间、模型大小和推理速度。

### 16.1.4. Exercises 练习

**Exercise 1 练习1**: Implement a simple random search for a linear regression problem. 为线性回归问题实现简单的随机搜索。Compare the performance of different learning rates and regularization strengths. 比较不同学习率和正则化强度的性能。

**Exercise 2 练习2**: Analyze the effect of the number of random trials on the quality of found hyperparameters. 分析随机试验次数对找到的超参数质量的影响。Plot the best validation score vs. number of trials. 绘制最佳验证分数与试验次数的关系图。

**Exercise 3 练习3**: Design a search space for a convolutional neural network for image classification. 为图像分类的卷积神经网络设计搜索空间。Include architecture choices, optimization parameters, and regularization options. 包括架构选择、优化参数和正则化选项。

## 16.2. Hyperparameter Optimization API 超参数优化API

### Introduction to HPO API HPO API介绍

A well-designed hyperparameter optimization API provides a structured framework for conducting systematic hyperparameter search. 设计良好的超参数优化API为进行系统性超参数搜索提供了结构化框架。Think of it as a standardized toolkit that separates the concerns of search strategy, evaluation scheduling, and result tracking. 将其视为标准化工具包，分离了搜索策略、评估调度和结果跟踪的关注点。

The API typically consists of three main components: 该API通常由三个主要组件组成：
- **Searcher**: Decides which hyperparameters to try next 决定接下来尝试哪些超参数
- **Scheduler**: Manages when and how evaluations are performed 管理何时以及如何执行评估
- **Tuner**: Orchestrates the overall optimization process 协调整体优化过程

### 16.2.1. Searcher 搜索器

The Searcher component is responsible for proposing new hyperparameter configurations to evaluate. 搜索器组件负责提出要评估的新超参数配置。It's like a strategic advisor that suggests the next experiment based on previous results. 它就像一个战略顾问，根据以前的结果建议下一个实验。

**Abstract Searcher Interface 抽象搜索器接口**:

```python
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional

class BaseSearcher(ABC):
    """
    Abstract base class for hyperparameter searchers
    超参数搜索器的抽象基类
    """
    
    def __init__(self, search_space: Dict[str, Any]):
        """
        Initialize searcher with search space definition
        使用搜索空间定义初始化搜索器
        
        Args:
            search_space: Dictionary defining parameter ranges and types
                         定义参数范围和类型的字典
        """
        self.search_space = search_space
        self.evaluated_configs = []  # Store evaluated configurations 存储已评估的配置
        self.scores = []  # Store corresponding scores 存储相应的分数
    
    @abstractmethod
    def suggest(self) -> Dict[str, Any]:
        """
        Suggest next hyperparameter configuration to evaluate
        建议下一个要评估的超参数配置
        
        Returns:
            Dictionary containing hyperparameter values
            包含超参数值的字典
        """
        pass
    
    def update(self, config: Dict[str, Any], score: float):
        """
        Update searcher with evaluation result
        使用评估结果更新搜索器
        
        Args:
            config: Evaluated hyperparameter configuration
                   已评估的超参数配置
            score: Performance score (lower is better)
                  性能分数（越低越好）
        """
        self.evaluated_configs.append(config)
        self.scores.append(score)
```

**Random Searcher Implementation 随机搜索器实现**:

```python
import random

class RandomSearcher(BaseSearcher):
    """
    Random search implementation
    随机搜索实现
    """
    
    def suggest(self) -> Dict[str, Any]:
        """
        Generate random hyperparameter configuration
        生成随机超参数配置
        """
        config = {}
        
        for param_name, param_spec in self.search_space.items():
            if param_spec['type'] == 'uniform':
                # Continuous parameter 连续参数
                value = random.uniform(param_spec['low'], param_spec['high'])
                config[param_name] = value
                
            elif param_spec['type'] == 'log_uniform':
                # Log-scale continuous parameter 对数尺度连续参数
                log_low = np.log(param_spec['low'])
                log_high = np.log(param_spec['high'])
                log_value = random.uniform(log_low, log_high)
                config[param_name] = np.exp(log_value)
                
            elif param_spec['type'] == 'choice':
                # Categorical parameter 分类参数
                config[param_name] = random.choice(param_spec['values'])
                
            elif param_spec['type'] == 'int_uniform':
                # Integer parameter 整数参数
                config[param_name] = random.randint(
                    param_spec['low'], param_spec['high']
                )
        
        return config
```

**Grid Searcher Implementation 网格搜索器实现**:

```python
from itertools import product

class GridSearcher(BaseSearcher):
    """
    Grid search implementation
    网格搜索实现
    """
    
    def __init__(self, search_space: Dict[str, Any]):
        super().__init__(search_space)
        self.grid_configs = self._generate_grid()
        self.current_index = 0
    
    def _generate_grid(self):
        """
        Generate all possible combinations in grid
        生成网格中所有可能的组合
        """
        param_names = []
        param_values = []
        
        for name, spec in self.search_space.items():
            param_names.append(name)
            if spec['type'] == 'choice':
                param_values.append(spec['values'])
            elif spec['type'] == 'uniform':
                # Discretize continuous space 离散化连续空间
                num_points = spec.get('num_points', 10)
                values = np.linspace(spec['low'], spec['high'], num_points)
                param_values.append(values.tolist())
        
        # Generate all combinations 生成所有组合
        configs = []
        for combination in product(*param_values):
            config = dict(zip(param_names, combination))
            configs.append(config)
        
        return configs
    
    def suggest(self) -> Dict[str, Any]:
        """
        Return next configuration in grid
        返回网格中的下一个配置
        """
        if self.current_index >= len(self.grid_configs):
            raise StopIteration("All grid configurations have been evaluated")
        
        config = self.grid_configs[self.current_index]
        self.current_index += 1
        return config
```

### 16.2.2. Scheduler 调度器

The Scheduler component manages the execution and resource allocation of hyperparameter evaluations. 调度器组件管理超参数评估的执行和资源分配。Think of it as a project manager that decides when to start, pause, or stop different experiments. 将其视为项目经理，决定何时开始、暂停或停止不同的实验。

**Basic Scheduler Implementation 基本调度器实现**:

```python
import time
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Callable, List, Tuple

class BaseScheduler:
    """
    Base scheduler for managing hyperparameter evaluations
    管理超参数评估的基础调度器
    """
    
    def __init__(self, 
                 objective_function: Callable[[Dict[str, Any]], float],
                 max_trials: int = 100,
                 max_workers: int = 1):
        """
        Initialize scheduler
        初始化调度器
        
        Args:
            objective_function: Function to evaluate hyperparameters
                              评估超参数的函数
            max_trials: Maximum number of trials to run
                       运行的最大试验次数
            max_workers: Number of parallel workers
                        并行工作者数量
        """
        self.objective_function = objective_function
        self.max_trials = max_trials
        self.max_workers = max_workers
        self.completed_trials = []
        self.running_trials = []
        
    def run_trial(self, config: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        """
        Execute single hyperparameter evaluation
        执行单个超参数评估
        
        Args:
            config: Hyperparameter configuration to evaluate
                   要评估的超参数配置
        
        Returns:
            Tuple of (config, score)
            (配置, 分数)的元组
        """
        start_time = time.time()
        
        try:
            score = self.objective_function(config)
            end_time = time.time()
            
            result = {
                'config': config,
                'score': score,
                'training_time': end_time - start_time,
                'status': 'completed'
            }
            
        except Exception as e:
            result = {
                'config': config,
                'score': float('inf'),
                'training_time': time.time() - start_time,
                'status': 'failed',
                'error': str(e)
            }
        
        return config, result
```

**Synchronous Scheduler 同步调度器**:

```python
class SynchronousScheduler(BaseScheduler):
    """
    Synchronous scheduler that evaluates configurations sequentially
    按顺序评估配置的同步调度器
    """
    
    def run_optimization(self, searcher: BaseSearcher) -> List[Dict[str, Any]]:
        """
        Run hyperparameter optimization
        运行超参数优化
        
        Args:
            searcher: Hyperparameter searcher instance
                     超参数搜索器实例
        
        Returns:
            List of evaluation results
            评估结果列表
        """
        results = []
        
        for trial_id in range(self.max_trials):
            # Get next configuration from searcher
            # 从搜索器获取下一个配置
            config = searcher.suggest()
            
            # Evaluate configuration 评估配置
            _, result = self.run_trial(config)
            results.append(result)
            
            # Update searcher with result 使用结果更新搜索器
            searcher.update(config, result['score'])
            
            print(f"Trial {trial_id + 1}/{self.max_trials}: "
                  f"Score = {result['score']:.4f}, "
                  f"Time = {result['training_time']:.2f}s")
        
        return results
```

**Asynchronous Scheduler 异步调度器**:

```python
class AsynchronousScheduler(BaseScheduler):
    """
    Asynchronous scheduler that can run multiple evaluations in parallel
    可以并行运行多个评估的异步调度器
    """
    
    def run_optimization(self, searcher: BaseSearcher) -> List[Dict[str, Any]]:
        """
        Run asynchronous hyperparameter optimization
        运行异步超参数优化
        """
        results = []
        completed_trials = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit initial batch of trials 提交初始试验批次
            futures = {}
            
            # Start initial trials 开始初始试验
            for _ in range(min(self.max_workers, self.max_trials)):
                config = searcher.suggest()
                future = executor.submit(self.run_trial, config)
                futures[future] = config
            
            while completed_trials < self.max_trials and futures:
                # Wait for at least one trial to complete
                # 等待至少一个试验完成
                from concurrent.futures import as_completed
                
                for completed_future in as_completed(futures):
                    config, result = completed_future.result()
                    results.append(result)
                    completed_trials += 1
                    
                    # Update searcher 更新搜索器
                    searcher.update(config, result['score'])
                    
                    # Remove completed future 移除已完成的future
                    del futures[completed_future]
                    
                    # Submit new trial if budget allows
                    # 如果预算允许则提交新试验
                    if completed_trials < self.max_trials:
                        new_config = searcher.suggest()
                        new_future = executor.submit(self.run_trial, new_config)
                        futures[new_future] = new_config
                    
                    print(f"Completed {completed_trials}/{self.max_trials} trials, "
                          f"Best score: {min(r['score'] for r in results):.4f}")
                    
                    break  # Process one completion at a time
        
        return results
```

### 16.2.3. Tuner 调优器

The Tuner is the orchestrator that brings together the Searcher and Scheduler components. 调优器是将搜索器和调度器组件结合在一起的协调者。It's like a conductor leading an orchestra, ensuring all components work harmoniously together. 它就像指挥家指挥管弦乐队，确保所有组件和谐地协同工作。

```python
class HPOTuner:
    """
    Main tuner class that orchestrates hyperparameter optimization
    协调超参数优化的主要调优器类
    """
    
    def __init__(self, 
                 searcher: BaseSearcher,
                 scheduler: BaseScheduler,
                 save_results: bool = True,
                 results_file: str = "hpo_results.json"):
        """
        Initialize HPO tuner
        初始化HPO调优器
        
        Args:
            searcher: Hyperparameter search strategy
                     超参数搜索策略
            scheduler: Evaluation scheduler
                      评估调度器
            save_results: Whether to save results to file
                         是否将结果保存到文件
            results_file: File to save results
                         保存结果的文件
        """
        self.searcher = searcher
        self.scheduler = scheduler
        self.save_results = save_results
        self.results_file = results_file
        self.optimization_results = None
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run hyperparameter optimization
        运行超参数优化
        
        Returns:
            Optimization results including best configuration
            包括最佳配置的优化结果
        """
        print("Starting hyperparameter optimization...")
        print(f"Search space: {self.searcher.search_space}")
        print(f"Max trials: {self.scheduler.max_trials}")
        
        # Run optimization 运行优化
        start_time = time.time()
        results = self.scheduler.run_optimization(self.searcher)
        total_time = time.time() - start_time
        
        # Find best configuration 找到最佳配置
        best_result = min(results, key=lambda x: x['score'])
        
        # Compile optimization summary 编译优化摘要
        optimization_summary = {
            'best_config': best_result['config'],
            'best_score': best_result['score'],
            'total_trials': len(results),
            'total_time': total_time,
            'successful_trials': len([r for r in results if r['status'] == 'completed']),
            'failed_trials': len([r for r in results if r['status'] == 'failed']),
            'all_results': results
        }
        
        self.optimization_results = optimization_summary
        
        # Save results if requested 如果请求则保存结果
        if self.save_results:
            self._save_results(optimization_summary)
        
        # Print summary 打印摘要
        self._print_summary(optimization_summary)
        
        return optimization_summary
    
    def _save_results(self, results: Dict[str, Any]):
        """Save optimization results to JSON file 将优化结果保存到JSON文件"""
        import json
        
        # Convert numpy types to native Python types for JSON serialization
        # 将numpy类型转换为原生Python类型以进行JSON序列化
        def convert_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            return obj
        
        converted_results = convert_types(results)
        
        with open(self.results_file, 'w') as f:
            json.dump(converted_results, f, indent=2)
        
        print(f"Results saved to {self.results_file}")
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print optimization summary 打印优化摘要"""
        print("\n" + "="*50)
        print("HYPERPARAMETER OPTIMIZATION SUMMARY")
        print("超参数优化摘要")
        print("="*50)
        print(f"Best configuration 最佳配置:")
        for param, value in results['best_config'].items():
            print(f"  {param}: {value}")
        print(f"\nBest score 最佳分数: {results['best_score']:.6f}")
        print(f"Total trials 总试验次数: {results['total_trials']}")
        print(f"Successful trials 成功试验次数: {results['successful_trials']}")
        print(f"Failed trials 失败试验次数: {results['failed_trials']}")
        print(f"Total time 总时间: {results['total_time']:.2f} seconds")
        print("="*50)
```

### 16.2.4. Bookkeeping the Performance of HPO Algorithms HPO算法性能记录

Tracking and analyzing the performance of different HPO algorithms is crucial for understanding their effectiveness and making informed decisions about which approach to use. 跟踪和分析不同HPO算法的性能对于理解其有效性和做出明智的方法选择决策至关重要。

**Performance Metrics 性能指标**:

```python
class HPOAnalyzer:
    """
    Analyzer for comparing different HPO algorithms
    用于比较不同HPO算法的分析器
    """
    
    def __init__(self):
        self.algorithm_results = {}
    
    def add_algorithm_results(self, 
                            algorithm_name: str, 
                            results: Dict[str, Any]):
        """
        Add results from an HPO algorithm
        添加HPO算法的结果
        
        Args:
            algorithm_name: Name of the algorithm
                           算法名称
            results: Optimization results
                    优化结果
        """
        self.algorithm_results[algorithm_name] = results
    
    def compute_performance_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute performance metrics for all algorithms
        计算所有算法的性能指标
        
        Returns:
            Dictionary with metrics for each algorithm
            每个算法指标的字典
        """
        metrics = {}
        
        for alg_name, results in self.algorithm_results.items():
            all_results = results['all_results']
            scores = [r['score'] for r in all_results if r['status'] == 'completed']
            times = [r['training_time'] for r in all_results if r['status'] == 'completed']
            
            # Calculate cumulative best scores 计算累积最佳分数
            cumulative_best = []
            current_best = float('inf')
            for score in scores:
                current_best = min(current_best, score)
                cumulative_best.append(current_best)
            
            metrics[alg_name] = {
                'best_score': results['best_score'],
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'success_rate': results['successful_trials'] / results['total_trials'],
                'avg_training_time': np.mean(times),
                'total_time': results['total_time'],
                'convergence_trial': self._find_convergence_trial(cumulative_best),
                'improvement_rate': self._calculate_improvement_rate(cumulative_best)
            }
        
        return metrics
    
    def _find_convergence_trial(self, cumulative_best: List[float], 
                               tolerance: float = 1e-4) -> int:
        """
        Find trial number where algorithm converged
        找到算法收敛的试验次数
        """
        if len(cumulative_best) < 2:
            return len(cumulative_best)
        
        final_best = cumulative_best[-1]
        for i, score in enumerate(cumulative_best):
            if abs(score - final_best) <= tolerance:
                return i + 1
        return len(cumulative_best)
    
    def _calculate_improvement_rate(self, cumulative_best: List[float]) -> float:
        """
        Calculate rate of improvement over trials
        计算试验过程中的改进率
        """
        if len(cumulative_best) < 2:
            return 0.0
        
        initial_score = cumulative_best[0]
        final_score = cumulative_best[-1]
        
        if initial_score == final_score:
            return 0.0
        
        improvement = (initial_score - final_score) / initial_score
        return improvement
    
    def plot_convergence(self, save_path: str = None):
        """
        Plot convergence curves for all algorithms
        绘制所有算法的收敛曲线
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        
        for alg_name, results in self.algorithm_results.items():
            all_results = results['all_results']
            scores = [r['score'] for r in all_results if r['status'] == 'completed']
            
            # Calculate cumulative best 计算累积最佳
            cumulative_best = []
            current_best = float('inf')
            for score in scores:
                current_best = min(current_best, score)
                cumulative_best.append(current_best)
            
            plt.plot(range(1, len(cumulative_best) + 1), 
                    cumulative_best, 
                    label=f'{alg_name}',
                    linewidth=2,
                    marker='o',
                    markersize=4)
        
        plt.xlabel('Trial Number 试验次数')
        plt.ylabel('Best Score So Far 目前最佳分数')
        plt.title('HPO Algorithm Convergence Comparison\nHPO算法收敛比较')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Use log scale for better visualization
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_comparison_report(self) -> str:
        """
        Generate detailed comparison report
        生成详细比较报告
        """
        metrics = self.compute_performance_metrics()
        
        report = "HYPERPARAMETER OPTIMIZATION ALGORITHMS COMPARISON\n"
        report += "超参数优化算法比较\n"
        report += "=" * 60 + "\n\n"
        
        # Best score comparison 最佳分数比较
        report += "Best Scores 最佳分数:\n"
        sorted_algos = sorted(metrics.items(), key=lambda x: x[1]['best_score'])
        for i, (alg_name, metrics_dict) in enumerate(sorted_algos):
            report += f"{i+1}. {alg_name}: {metrics_dict['best_score']:.6f}\n"
        
        report += "\nDetailed Metrics 详细指标:\n"
        report += "-" * 40 + "\n"
        
        for alg_name, metrics_dict in metrics.items():
            report += f"\n{alg_name}:\n"
            report += f"  Best Score 最佳分数: {metrics_dict['best_score']:.6f}\n"
            report += f"  Mean Score 平均分数: {metrics_dict['mean_score']:.6f}\n"
            report += f"  Score Std 分数标准差: {metrics_dict['std_score']:.6f}\n"
            report += f"  Success Rate 成功率: {metrics_dict['success_rate']:.2%}\n"
            report += f"  Avg Training Time 平均训练时间: {metrics_dict['avg_training_time']:.2f}s\n"
            report += f"  Total Time 总时间: {metrics_dict['total_time']:.2f}s\n"
            report += f"  Convergence Trial 收敛试验: {metrics_dict['convergence_trial']}\n"
            report += f"  Improvement Rate 改进率: {metrics_dict['improvement_rate']:.2%}\n"
        
        return report
```

这个详细的HPO API设计提供了一个完整的框架，将搜索策略、评估调度和结果分析分离开来，使得整个超参数优化过程更加模块化和可扩展。通过这种设计，研究人员和实践者可以轻松地试验不同的搜索算法，比较它们的性能，并根据具体需求选择最合适的方法。

继续到下一部分... 

## 16.3. Asynchronous Random Search 异步随机搜索

### Introduction to Asynchronous Optimization 异步优化介绍

In traditional hyperparameter optimization, we typically wait for one configuration to finish training before starting the next one. 在传统的超参数优化中，我们通常等待一个配置完成训练后再开始下一个。This is like having a restaurant kitchen where only one chef can work at a time – highly inefficient when you have multiple stoves available! 这就像餐厅厨房中一次只能有一个厨师工作一样——当您有多个炉灶可用时，这是非常低效的！

Asynchronous hyperparameter optimization allows multiple configurations to be evaluated simultaneously, significantly reducing the wall-clock time required for optimization. 异步超参数优化允许同时评估多个配置，显著减少优化所需的实际时间。Think of it as coordinating multiple chefs in a kitchen, where each chef can work on different dishes simultaneously. 将其想象为协调厨房中的多个厨师，每个厨师可以同时制作不同的菜肴。

### Key Concepts 关键概念

**Parallelization vs Asynchrony 并行化与异步性**: 
- **Parallelization**: Running multiple evaluations at the same time 并行化：同时运行多个评估
- **Asynchrony**: Starting new evaluations without waiting for all previous ones to complete 异步性：无需等待所有先前评估完成就开始新评估

**Worker Management 工作器管理**: Managing multiple computational resources (GPUs, CPUs, machines) that can train models independently. 管理多个可以独立训练模型的计算资源（GPU、CPU、机器）。

**Load Balancing 负载均衡**: Distributing work efficiently across available resources, considering that different configurations may have different training times. 考虑到不同配置可能有不同的训练时间，在可用资源间高效分配工作。

## 16.3.1. Objective Function 目标函数

### Designing Asynchronous-Friendly Objectives 设计异步友好的目标

When implementing asynchronous optimization, the objective function must be designed to handle concurrent evaluations effectively. 在实施异步优化时，目标函数必须设计为能够有效处理并发评估。Here's how to structure such functions: 以下是如何构建此类函数的方法：

```python
import asyncio
import concurrent.futures
import torch
import torch.nn as nn
import time
import threading
from queue import Queue, Empty
import uuid
import logging
from typing import Dict, List, Optional, Callable
import json
from dataclasses import dataclass
from datetime import datetime

@dataclass
class EvaluationResult:
    """评估结果数据类"""
    config_id: str
    config: Dict
    performance: float
    training_time: float
    worker_id: str
    start_time: datetime
    end_time: datetime
    metadata: Dict = None

class AsyncObjectiveFunction:
    """异步目标函数实现"""
    
    def __init__(self, dataset_loader, model_factory, max_workers=4):
        self.dataset_loader = dataset_loader
        self.model_factory = model_factory
        self.max_workers = max_workers
        self.evaluation_history = []
        self.active_evaluations = {}
        self.worker_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.result_queue = Queue()
        self.lock = threading.Lock()
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def submit_evaluation(self, config: Dict) -> str:
        """提交配置进行异步评估"""
        config_id = str(uuid.uuid4())
        
        # 提交到线程池
        future = self.worker_pool.submit(self._evaluate_config, config_id, config)
        
        with self.lock:
            self.active_evaluations[config_id] = {
                'config': config,
                'future': future,
                'start_time': datetime.now()
            }
        
        self.logger.info(f"Submitted evaluation {config_id} with config: {config}")
        return config_id
    
    def _evaluate_config(self, config_id: str, config: Dict) -> EvaluationResult:
        """在工作线程中评估配置"""
        worker_id = threading.current_thread().name
        start_time = datetime.now()
        
        try:
            # 模拟模型训练
            performance = self._train_model(config, worker_id)
            
            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()
            
            result = EvaluationResult(
                config_id=config_id,
                config=config.copy(),
                performance=performance,
                training_time=training_time,
                worker_id=worker_id,
                start_time=start_time,
                end_time=end_time,
                metadata={'status': 'success'}
            )
            
            # 将结果放入队列
            self.result_queue.put(result)
            return result
            
        except Exception as e:
            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()
            
            self.logger.error(f"Error evaluating config {config_id}: {e}")
            
            result = EvaluationResult(
                config_id=config_id,
                config=config.copy(),
                performance=0.0,
                training_time=training_time,
                worker_id=worker_id,
                start_time=start_time,
                end_time=end_time,
                metadata={'status': 'error', 'error': str(e)}
            )
            
            self.result_queue.put(result)
            return result
    
    def _train_model(self, config: Dict, worker_id: str) -> float:
        """实际的模型训练函数"""
        self.logger.info(f"Worker {worker_id} training with config: {config}")
        
        # 创建模型
        model = self.model_factory(config)
        
        # 获取数据加载器
        train_loader, val_loader = self.dataset_loader(config['batch_size'])
        
        # 设置优化器
        if config['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9)
        
        criterion = nn.CrossEntropyLoss()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        # 训练循环
        best_val_acc = 0.0
        patience_counter = 0
        patience_limit = 5
        
        for epoch in range(config.get('max_epochs', 20)):
            # 训练阶段
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                if batch_idx > 50:  # 限制训练步数以加快演示
                    break
                    
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            
            # 验证阶段
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(val_loader):
                    if batch_idx > 20:  # 限制验证步数
                        break
                        
                    data, target = data.to(device), target.to(device)
                    outputs = model(data)
                    _, predicted = torch.max(outputs.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
            
            val_acc = correct / total if total > 0 else 0.0
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience_limit:
                    break
            
            # 模拟不同配置的不同训练时间
            time.sleep(0.1 + config.get('complexity_factor', 0.1))
        
        return best_val_acc
    
    def get_completed_results(self) -> List[EvaluationResult]:
        """获取已完成的评估结果"""
        results = []
        
        # 从队列中获取所有结果
        while True:
            try:
                result = self.result_queue.get_nowait()
                results.append(result)
                
                # 从活跃评估中移除
                with self.lock:
                    if result.config_id in self.active_evaluations:
                        del self.active_evaluations[result.config_id]
                        
            except Empty:
                break
        
        return results
    
    def get_active_evaluations(self) -> Dict:
        """获取当前活跃的评估"""
        with self.lock:
            return {k: {'config': v['config'], 'start_time': v['start_time']} 
                   for k, v in self.active_evaluations.items()}
    
    def wait_for_completion(self, timeout: Optional[float] = None) -> List[EvaluationResult]:
        """等待所有评估完成"""
        with self.lock:
            futures = [eval_info['future'] for eval_info in self.active_evaluations.values()]
        
        if not futures:
            return self.get_completed_results()
        
        # 等待所有Future完成
        concurrent.futures.wait(futures, timeout=timeout)
        
        return self.get_completed_results()
    
    def shutdown(self):
        """关闭工作器池"""
        self.worker_pool.shutdown(wait=True)

# 示例模型工厂
def simple_cnn_factory(config):
    """创建简单CNN模型的工厂函数"""
    
    class SimpleCNN(nn.Module):
        def __init__(self, config):
            super(SimpleCNN, self).__init__()
            
            # 根据配置构建网络
            self.features = nn.Sequential(
                nn.Conv2d(3, config['filters'], 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(config['filters'], config['filters']*2, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d(1)
            )
            
            self.classifier = nn.Sequential(
                nn.Dropout(config.get('dropout', 0.5)),
                nn.Linear(config['filters']*2, 10)  # CIFAR-10有10个类
            )
        
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
    
    return SimpleCNN(config)

# 示例数据加载器工厂
def cifar10_loader_factory(batch_size):
    """创建CIFAR-10数据加载器"""
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # 创建小型数据集用于快速演示
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)
    valset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    
    # 使用子集以加快训练
    train_subset = torch.utils.data.Subset(trainset, range(1000))
    val_subset = torch.utils.data.Subset(valset, range(200))
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

# 使用示例
async_objective = AsyncObjectiveFunction(
    dataset_loader=cifar10_loader_factory,
    model_factory=simple_cnn_factory,
    max_workers=4
)
```

### Handling Different Evaluation Times 处理不同的评估时间

One of the key challenges in asynchronous optimization is that different configurations can have vastly different training times. 异步优化的关键挑战之一是不同配置可能有很大不同的训练时间。For example: 例如：

```python
class TimeAwareAsyncObjective(AsyncObjectiveFunction):
    """时间感知的异步目标函数"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_estimates = {}  # 配置到时间的映射
        self.config_complexity_analyzer = ConfigComplexityAnalyzer()
    
    def estimate_training_time(self, config: Dict) -> float:
        """估计配置的训练时间"""
        complexity_score = self.config_complexity_analyzer.analyze(config)
        
        # 基于配置复杂性估计时间
        base_time = 60  # 基础60秒
        complexity_multiplier = 1 + complexity_score
        
        # 考虑历史数据
        if self.time_estimates:
            similar_configs = self._find_similar_configs(config)
            if similar_configs:
                historical_avg = np.mean([self.time_estimates[c] for c in similar_configs])
                return (base_time * complexity_multiplier + historical_avg) / 2
        
        return base_time * complexity_multiplier
    
    def _find_similar_configs(self, config: Dict) -> List[str]:
        """找到相似的配置以估计时间"""
        similar = []
        
        for config_id, hist_config in self.evaluation_history:
            similarity = self._calculate_config_similarity(config, hist_config)
            if similarity > 0.8:  # 80%相似度阈值
                similar.append(config_id)
        
        return similar
    
    def _calculate_config_similarity(self, config1: Dict, config2: Dict) -> float:
        """计算两个配置的相似度"""
        common_keys = set(config1.keys()) & set(config2.keys())
        if not common_keys:
            return 0.0
        
        similarities = []
        for key in common_keys:
            val1, val2 = config1[key], config2[key]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # 数值相似度
                max_val = max(abs(val1), abs(val2), 1e-8)
                sim = 1 - abs(val1 - val2) / max_val
            elif val1 == val2:
                sim = 1.0
            else:
                sim = 0.0
            
            similarities.append(sim)
        
        return np.mean(similarities)

class ConfigComplexityAnalyzer:
    """配置复杂性分析器"""
    
    def analyze(self, config: Dict) -> float:
        """分析配置的计算复杂性"""
        complexity = 0.0
        
        # 网络深度复杂性
        if 'num_layers' in config:
            complexity += config['num_layers'] * 0.1
        
        # 网络宽度复杂性
        if 'filters' in config:
            complexity += config['filters'] / 100
        
        # 批次大小复杂性（反比）
        if 'batch_size' in config:
            complexity += 100 / config['batch_size'] * 0.01
        
        # 学习率复杂性（可能影响收敛速度）
        if 'learning_rate' in config:
            lr = config['learning_rate']
            if lr < 0.001 or lr > 0.1:
                complexity += 0.2  # 极端学习率需要更多epoch
        
        return complexity
```

## 16.3.2. Asynchronous Scheduler 异步调度器

### Scheduler Architecture 调度器架构

The heart of asynchronous HPO is the scheduler that coordinates multiple workers and decides which configurations to evaluate next. 异步HPO的核心是协调多个工作器并决定下一步评估哪些配置的调度器。Think of it as an air traffic controller managing multiple planes (configurations) landing and taking off (starting and completing evaluations). 将其想象为管理多架飞机（配置）起降（开始和完成评估）的空中交通管制员。

```python
import heapq
import time
from abc import ABC, abstractmethod
from enum import Enum
from collections import defaultdict, deque
import numpy as np
from typing import Optional, Tuple

class SchedulerState(Enum):
    """调度器状态枚举"""
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"

class AsyncHPOScheduler:
    """异步超参数优化调度器"""
    
    def __init__(self, 
                 objective_function: AsyncObjectiveFunction,
                 search_algorithm: 'SearchAlgorithm',
                 max_concurrent_evaluations: int = 4,
                 total_budget: int = 100,
                 time_budget_hours: Optional[float] = None):
        
        self.objective_function = objective_function
        self.search_algorithm = search_algorithm
        self.max_concurrent = max_concurrent_evaluations
        self.total_budget = total_budget
        self.time_budget = time_budget_hours * 3600 if time_budget_hours else None
        
        # 状态跟踪
        self.state = SchedulerState.STOPPED
        self.start_time = None
        self.evaluations_submitted = 0
        self.evaluations_completed = 0
        self.active_evaluations = {}
        self.completed_results = []
        
        # 性能监控
        self.performance_history = []
        self.best_result = None
        self.worker_utilization = defaultdict(list)
        
        # 调度策略
        self.scheduling_strategy = AdaptiveSchedulingStrategy()
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
    
    def start_optimization(self):
        """开始优化过程"""
        self.state = SchedulerState.RUNNING
        self.start_time = time.time()
        self.logger.info("Starting asynchronous hyperparameter optimization")
        
        # 初始配置提交
        self._submit_initial_configurations()
        
        # 主优化循环
        self._optimization_loop()
    
    def _submit_initial_configurations(self):
        """提交初始配置"""
        initial_configs = self.search_algorithm.get_initial_configurations(
            min(self.max_concurrent, self.total_budget)
        )
        
        for config in initial_configs:
            if self._can_submit_new_evaluation():
                self._submit_configuration(config)
    
    def _optimization_loop(self):
        """主优化循环"""
        while self.state == SchedulerState.RUNNING and not self._should_stop():
            try:
                # 检查完成的评估
                self._process_completed_evaluations()
                
                # 提交新的配置
                self._submit_new_configurations()
                
                # 监控工作器状态
                self._monitor_workers()
                
                # 更新调度策略
                self._update_scheduling_strategy()
                
                # 短暂休眠以避免忙等待
                time.sleep(0.1)
                
            except KeyboardInterrupt:
                self.logger.info("Optimization interrupted by user")
                self.pause_optimization()
                break
            except Exception as e:
                self.logger.error(f"Error in optimization loop: {e}")
                break
        
        self._finalize_optimization()
    
    def _process_completed_evaluations(self):
        """处理完成的评估"""
        completed = self.objective_function.get_completed_results()
        
        for result in completed:
            self.evaluations_completed += 1
            self.completed_results.append(result)
            
            # 更新最佳结果
            if self.best_result is None or result.performance > self.best_result.performance:
                self.best_result = result
                self.logger.info(f"New best result: {result.performance:.4f}")
            
            # 更新搜索算法
            self.search_algorithm.update_with_result(result)
            
            # 记录性能历史
            self.performance_history.append({
                'timestamp': time.time() - self.start_time,
                'performance': result.performance,
                'config_id': result.config_id,
                'worker_id': result.worker_id
            })
            
            # 更新工作器利用率
            self.worker_utilization[result.worker_id].append(result.training_time)
            
            self.logger.info(f"Completed evaluation {result.config_id}: {result.performance:.4f}")
    
    def _submit_new_configurations(self):
        """提交新配置"""
        while (self._can_submit_new_evaluation() and 
               self.evaluations_submitted < self.total_budget):
            
            # 从搜索算法获取下一个配置
            next_config = self.search_algorithm.suggest_next_configuration()
            
            if next_config is None:
                break
            
            self._submit_configuration(next_config)
    
    def _submit_configuration(self, config: Dict):
        """提交单个配置"""
        config_id = self.objective_function.submit_evaluation(config)
        self.evaluations_submitted += 1
        
        self.active_evaluations[config_id] = {
            'config': config,
            'submit_time': time.time()
        }
        
        self.logger.info(f"Submitted configuration {config_id} ({self.evaluations_submitted}/{self.total_budget})")
    
    def _can_submit_new_evaluation(self) -> bool:
        """检查是否可以提交新评估"""
        current_active = len(self.objective_function.get_active_evaluations())
        return current_active < self.max_concurrent
    
    def _should_stop(self) -> bool:
        """检查是否应该停止优化"""
        # 预算用完
        if self.evaluations_submitted >= self.total_budget:
            return True
        
        # 时间预算用完
        if self.time_budget and (time.time() - self.start_time) >= self.time_budget:
            return True
        
        # 没有更多配置可以尝试
        if not self.search_algorithm.has_more_configurations():
            return True
        
        return False
    
    def _monitor_workers(self):
        """监控工作器状态"""
        active = self.objective_function.get_active_evaluations()
        current_time = time.time()
        
        for config_id, eval_info in active.items():
            elapsed = current_time - eval_info['start_time'].timestamp()
            
            # 检查超长运行的评估
            if elapsed > 3600:  # 1小时超时
                self.logger.warning(f"Evaluation {config_id} running for {elapsed/3600:.1f} hours")
    
    def _update_scheduling_strategy(self):
        """更新调度策略"""
        if len(self.completed_results) > 0:
            self.scheduling_strategy.update(
                completed_results=self.completed_results,
                active_evaluations=len(self.objective_function.get_active_evaluations()),
                remaining_budget=self.total_budget - self.evaluations_submitted
            )
    
    def _finalize_optimization(self):
        """完成优化"""
        self.state = SchedulerState.STOPPED
        
        # 等待所有活跃评估完成
        self.logger.info("Waiting for remaining evaluations to complete...")
        remaining_results = self.objective_function.wait_for_completion(timeout=300)
        
        for result in remaining_results:
            self.evaluations_completed += 1
            self.completed_results.append(result)
            
            if self.best_result is None or result.performance > self.best_result.performance:
                self.best_result = result
        
        # 生成最终报告
        self._generate_final_report()
    
    def _generate_final_report(self):
        """生成最终优化报告"""
        total_time = time.time() - self.start_time
        
        report = {
            'optimization_summary': {
                'total_evaluations': self.evaluations_completed,
                'total_time_seconds': total_time,
                'best_performance': self.best_result.performance if self.best_result else None,
                'best_config': self.best_result.config if self.best_result else None
            },
            'worker_utilization': {
                worker_id: {
                    'total_jobs': len(times),
                    'avg_time': np.mean(times),
                    'total_time': sum(times)
                }
                for worker_id, times in self.worker_utilization.items()
            },
            'performance_progression': self.performance_history
        }
        
        self.logger.info("=== OPTIMIZATION COMPLETE ===")
        self.logger.info(f"Best performance: {report['optimization_summary']['best_performance']:.4f}")
        self.logger.info(f"Total evaluations: {report['optimization_summary']['total_evaluations']}")
        self.logger.info(f"Total time: {total_time/3600:.2f} hours")
        
        return report
    
    def pause_optimization(self):
        """暂停优化"""
        self.state = SchedulerState.PAUSED
        self.logger.info("Optimization paused")
    
    def resume_optimization(self):
        """恢复优化"""
        if self.state == SchedulerState.PAUSED:
            self.state = SchedulerState.RUNNING
            self.logger.info("Optimization resumed")
            self._optimization_loop()
    
    def get_current_status(self) -> Dict:
        """获取当前状态"""
        if self.start_time is None:
            return {'status': 'not_started'}
        
        elapsed_time = time.time() - self.start_time
        active_count = len(self.objective_function.get_active_evaluations())
        
        return {
            'status': self.state.value,
            'elapsed_time_seconds': elapsed_time,
            'evaluations_submitted': self.evaluations_submitted,
            'evaluations_completed': self.evaluations_completed,
            'active_evaluations': active_count,
            'best_performance': self.best_result.performance if self.best_result else None,
            'remaining_budget': self.total_budget - self.evaluations_submitted
        }

class AdaptiveSchedulingStrategy:
    """自适应调度策略"""
    
    def __init__(self):
        self.recent_completion_times = deque(maxlen=10)
        self.performance_trend = deque(maxlen=20)
        self.worker_efficiency = defaultdict(float)
    
    def update(self, completed_results: List, active_evaluations: int, remaining_budget: int):
        """更新调度策略"""
        if completed_results:
            # 更新完成时间统计
            recent_result = completed_results[-1]
            self.recent_completion_times.append(recent_result.training_time)
            self.performance_trend.append(recent_result.performance)
            
            # 更新工作器效率
            efficiency = recent_result.performance / max(recent_result.training_time, 1)
            self.worker_efficiency[recent_result.worker_id] = efficiency
    
    def should_increase_parallelism(self) -> bool:
        """判断是否应该增加并行度"""
        if len(self.recent_completion_times) < 3:
            return True
        
        avg_completion_time = np.mean(self.recent_completion_times)
        return avg_completion_time < 300  # 如果平均完成时间少于5分钟，增加并行度
    
    def get_optimal_batch_size(self, remaining_budget: int) -> int:
        """获取最优批次大小"""
        if remaining_budget < 10:
            return min(2, remaining_budget)
        elif remaining_budget < 50:
            return min(4, remaining_budget)
        else:
            return min(8, remaining_budget)
```

### Search Algorithm Integration 搜索算法集成

The scheduler needs to integrate with different search algorithms. Here's how to make this modular: 调度器需要与不同的搜索算法集成。以下是如何使其模块化：

```python
class SearchAlgorithm(ABC):
    """搜索算法抽象基类"""
    
    @abstractmethod
    def get_initial_configurations(self, count: int) -> List[Dict]:
        """获取初始配置"""
        pass
    
    @abstractmethod
    def suggest_next_configuration(self) -> Optional[Dict]:
        """建议下一个配置"""
        pass
    
    @abstractmethod
    def update_with_result(self, result: EvaluationResult):
        """使用结果更新算法"""
        pass
    
    @abstractmethod
    def has_more_configurations(self) -> bool:
        """检查是否还有更多配置"""
        pass

class AsyncRandomSearch(SearchAlgorithm):
    """异步随机搜索算法"""
    
    def __init__(self, param_space: Dict, seed: int = 42):
        self.param_space = param_space
        self.rng = np.random.RandomState(seed)
        self.generated_count = 0
        self.max_suggestions = 1000  # 最大建议数
        
    def get_initial_configurations(self, count: int) -> List[Dict]:
        """获取初始随机配置"""
        configs = []
        for _ in range(count):
            config = self._generate_random_config()
            configs.append(config)
        return configs
    
    def suggest_next_configuration(self) -> Optional[Dict]:
        """建议下一个随机配置"""
        if self.generated_count >= self.max_suggestions:
            return None
        
        config = self._generate_random_config()
        return config
    
    def update_with_result(self, result: EvaluationResult):
        """随机搜索不需要学习，但可以记录统计信息"""
        pass
    
    def has_more_configurations(self) -> bool:
        """检查是否还有更多配置"""
        return self.generated_count < self.max_suggestions
    
    def _generate_random_config(self) -> Dict:
        """生成随机配置"""
        config = {}
        
        for param_name, param_values in self.param_space.items():
            if isinstance(param_values, list):
                # 离散参数
                config[param_name] = self.rng.choice(param_values)
            elif isinstance(param_values, tuple) and len(param_values) == 2:
                # 连续参数 (min, max)
                min_val, max_val = param_values
                if isinstance(min_val, int) and isinstance(max_val, int):
                    config[param_name] = self.rng.randint(min_val, max_val + 1)
                else:
                    config[param_name] = self.rng.uniform(min_val, max_val)
            else:
                raise ValueError(f"Unsupported parameter format for {param_name}: {param_values}")
        
        self.generated_count += 1
        return config

class AsyncBayesianOptimization(SearchAlgorithm):
    """异步贝叶斯优化算法"""
    
    def __init__(self, param_space: Dict, acquisition_function: str = 'ucb'):
        self.param_space = param_space
        self.acquisition_function = acquisition_function
        self.observations = []
        self.configs_suggested = 0
        self.max_suggestions = 500
        
        # 简化的高斯过程代理模型
        self.surrogate_model = None
        self._initialize_parameter_encoding()
    
    def _initialize_parameter_encoding(self):
        """初始化参数编码"""
        self.param_bounds = []
        self.param_names = []
        
        for param_name, param_values in self.param_space.items():
            self.param_names.append(param_name)
            
            if isinstance(param_values, list):
                # 离散参数编码为[0, len-1]
                self.param_bounds.append((0, len(param_values) - 1))
            elif isinstance(param_values, tuple):
                # 连续参数
                self.param_bounds.append(param_values)
    
    def get_initial_configurations(self, count: int) -> List[Dict]:
        """获取初始配置（随机采样）"""
        configs = []
        for _ in range(count):
            config = self._generate_random_config()
            configs.append(config)
        return configs
    
    def suggest_next_configuration(self) -> Optional[Dict]:
        """使用获取函数建议下一个配置"""
        if self.configs_suggested >= self.max_suggestions:
            return None
        
        if len(self.observations) < 3:
            # 初期使用随机采样
            config = self._generate_random_config()
        else:
            # 使用获取函数
            config = self._optimize_acquisition_function()
        
        self.configs_suggested += 1
        return config
    
    def update_with_result(self, result: EvaluationResult):
        """使用新结果更新代理模型"""
        # 编码配置
        encoded_config = self._encode_config(result.config)
        
        self.observations.append({
            'config': encoded_config,
            'performance': result.performance,
            'config_id': result.config_id
        })
        
        # 更新代理模型
        self._update_surrogate_model()
    
    def has_more_configurations(self) -> bool:
        """检查是否还有更多配置"""
        return self.configs_suggested < self.max_suggestions
    
    def _generate_random_config(self) -> Dict:
        """生成随机配置"""
        config = {}
        
        for i, param_name in enumerate(self.param_names):
            min_val, max_val = self.param_bounds[i]
            param_values = self.param_space[param_name]
            
            if isinstance(param_values, list):
                # 离散参数
                idx = np.random.randint(min_val, max_val + 1)
                config[param_name] = param_values[idx]
            else:
                # 连续参数
                if isinstance(min_val, int) and isinstance(max_val, int):
                    config[param_name] = np.random.randint(min_val, max_val + 1)
                else:
                    config[param_name] = np.random.uniform(min_val, max_val)
        
        return config
    
    def _encode_config(self, config: Dict) -> np.ndarray:
        """将配置编码为数值向量"""
        encoded = []
        
        for param_name in self.param_names:
            value = config[param_name]
            param_values = self.param_space[param_name]
            
            if isinstance(param_values, list):
                # 离散参数：编码为索引
                encoded.append(param_values.index(value))
            else:
                # 连续参数：直接使用值
                encoded.append(value)
        
        return np.array(encoded)
    
    def _optimize_acquisition_function(self) -> Dict:
        """优化获取函数寻找下一个配置"""
        # 简化实现：在最佳配置附近搜索
        if not self.observations:
            return self._generate_random_config()
        
        # 找到最佳观察
        best_obs = max(self.observations, key=lambda x: x['performance'])
        best_config = best_obs['config']
        
        # 在最佳配置附近添加噪声
        noise_scale = 0.1
        perturbed_config = best_config + np.random.normal(0, noise_scale, size=best_config.shape)
        
        # 将扰动后的配置投影到有效范围
        for i, (min_val, max_val) in enumerate(self.param_bounds):
            perturbed_config[i] = np.clip(perturbed_config[i], min_val, max_val)
        
        # 解码回原始格式
        return self._decode_config(perturbed_config)
    
    def _decode_config(self, encoded_config: np.ndarray) -> Dict:
        """将编码向量解码为配置字典"""
        config = {}
        
        for i, param_name in enumerate(self.param_names):
            value = encoded_config[i]
            param_values = self.param_space[param_name]
            
            if isinstance(param_values, list):
                # 离散参数：四舍五入到最近的索引
                idx = int(round(np.clip(value, 0, len(param_values) - 1)))
                config[param_name] = param_values[idx]
            else:
                # 连续参数
                min_val, max_val = self.param_bounds[i]
                if isinstance(param_values[0], int) and isinstance(param_values[1], int):
                    config[param_name] = int(round(np.clip(value, min_val, max_val)))
                else:
                    config[param_name] = float(np.clip(value, min_val, max_val))
        
        return config
    
    def _update_surrogate_model(self):
        """更新代理模型（简化实现）"""
        # 在实际实现中，这里会训练高斯过程或其他代理模型
        # 这里使用简化的实现
        pass

# 使用示例
def run_async_hpo_example():
    """运行异步HPO示例"""
    
    # 定义参数空间
    param_space = {
        'learning_rate': [0.0001, 0.001, 0.01, 0.1],
        'batch_size': [32, 64, 128, 256],
        'filters': [16, 32, 64, 128],
        'dropout': (0.0, 0.5),
        'optimizer': ['adam', 'sgd'],
        'max_epochs': (10, 50)
    }
    
    # 创建异步目标函数
    async_objective = AsyncObjectiveFunction(
        dataset_loader=cifar10_loader_factory,
        model_factory=simple_cnn_factory,
        max_workers=4
    )
    
    # 创建搜索算法
    search_algorithm = AsyncRandomSearch(param_space, seed=42)
    # 或者使用贝叶斯优化
    # search_algorithm = AsyncBayesianOptimization(param_space)
    
    # 创建调度器
    scheduler = AsyncHPOScheduler(
        objective_function=async_objective,
        search_algorithm=search_algorithm,
        max_concurrent_evaluations=4,
        total_budget=20,  # 小预算用于演示
        time_budget_hours=0.5  # 30分钟时间限制
    )
    
    try:
        # 开始优化
        scheduler.start_optimization()
        
        # 获取最终结果
        final_report = scheduler._generate_final_report()
        
        print("\n=== FINAL RESULTS ===")
        print(f"Best configuration: {scheduler.best_result.config}")
        print(f"Best performance: {scheduler.best_result.performance:.4f}")
        
    finally:
        # 清理资源
        async_objective.shutdown()

if __name__ == "__main__":
    run_async_hpo_example()
```

This comprehensive implementation of asynchronous optimization provides: 这个异步优化的综合实现提供了：

1. **Parallel Evaluation**: Multiple configurations can be trained simultaneously 并行评估：多个配置可以同时训练
2. **Resource Management**: Intelligent allocation of computational resources 资源管理：计算资源的智能分配
3. **Adaptive Scheduling**: Dynamic adjustment based on current performance and resource utilization 自适应调度：基于当前性能和资源利用率的动态调整
4. **Algorithm Flexibility**: Support for different search algorithms (Random Search, Bayesian Optimization) 算法灵活性：支持不同的搜索算法（随机搜索、贝叶斯优化）

The key advantages of asynchronous optimization include: 异步优化的主要优势包括：
- **Reduced Wall-Clock Time**: Better utilization of available hardware 减少实际时间：更好地利用可用硬件
- **Better Resource Utilization**: Keeps all computational resources busy 更好的资源利用：保持所有计算资源忙碌
- **Fault Tolerance**: Can continue optimization even if some evaluations fail 容错性：即使某些评估失败也可以继续优化 

## 16.3.3. Visualize the Asynchronous Optimization Process 可视化异步优化过程

### Real-Time Monitoring and Visualization 实时监控和可视化

Understanding what's happening during asynchronous optimization is crucial for debugging, performance analysis, and gaining insights. 理解异步优化过程中发生的事情对于调试、性能分析和获得见解至关重要。Think of it as having a dashboard in your car that shows speed, fuel consumption, and engine temperature – you need to see what's happening to make informed decisions. 将其想象为汽车中显示速度、油耗和发动机温度的仪表板——您需要了解正在发生的事情以做出明智的决定。

```python
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import time
from collections import deque
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

class AsyncOptimizationVisualizer:
    """异步优化过程可视化器"""
    
    def __init__(self, scheduler: AsyncHPOScheduler, update_interval: float = 1.0):
        self.scheduler = scheduler
        self.update_interval = update_interval
        self.visualization_thread = None
        self.is_monitoring = False
        
        # 数据缓存
        self.performance_history = deque(maxlen=1000)
        self.worker_status = {}
        self.resource_utilization = deque(maxlen=100)
        self.convergence_data = deque(maxlen=1000)
        
        # 可视化组件
        self.fig = None
        self.axes = None
        self.animation_obj = None
        
        # 设置样式
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def start_monitoring(self):
        """开始实时监控"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.visualization_thread = threading.Thread(
            target=self._monitoring_loop, 
            daemon=True
        )
        self.visualization_thread.start()
        
        # 初始化可视化
        self._setup_visualization()
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        if self.visualization_thread:
            self.visualization_thread.join(timeout=2)
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                self._collect_data()
                time.sleep(self.update_interval)
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                break
    
    def _collect_data(self):
        """收集监控数据"""
        current_time = time.time()
        
        # 收集性能数据
        status = self.scheduler.get_current_status()
        
        self.performance_history.append({
            'timestamp': current_time,
            'best_performance': status.get('best_performance', 0),
            'evaluations_completed': status.get('evaluations_completed', 0),
            'active_evaluations': status.get('active_evaluations', 0),
            'remaining_budget': status.get('remaining_budget', 0)
        })
        
        # 收集工作器状态
        active_evals = self.scheduler.objective_function.get_active_evaluations()
        self.worker_status = {}
        
        for config_id, eval_info in active_evals.items():
            elapsed = current_time - eval_info['start_time'].timestamp()
            self.worker_status[config_id] = {
                'elapsed_time': elapsed,
                'config': eval_info['config'],
                'status': 'running'
            }
        
        # 计算资源利用率
        total_workers = self.scheduler.objective_function.max_workers
        active_workers = len(active_evals)
        utilization = active_workers / total_workers if total_workers > 0 else 0
        
        self.resource_utilization.append({
            'timestamp': current_time,
            'utilization': utilization,
            'active_workers': active_workers,
            'total_workers': total_workers
        })
        
        # 收集收敛数据
        if self.scheduler.completed_results:
            best_so_far = max(
                result.performance for result in self.scheduler.completed_results
            )
            self.convergence_data.append({
                'evaluation': len(self.scheduler.completed_results),
                'best_performance': best_so_far,
                'timestamp': current_time
            })
    
    def _setup_visualization(self):
        """设置可视化界面"""
        self.fig, self.axes = plt.subplots(2, 3, figsize=(18, 12))
        self.fig.suptitle('Asynchronous Hyperparameter Optimization Monitor', 
                         fontsize=16, fontweight='bold')
        
        # 调整子图间距
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # 启动动画
        self.animation_obj = animation.FuncAnimation(
            self.fig, self._update_plots, interval=1000, blit=False
        )
        
        plt.show()
    
    def _update_plots(self, frame):
        """更新所有图表"""
        if not self.performance_history:
            return
        
        # 清除所有子图
        for ax in self.axes.flat:
            ax.clear()
        
        self._plot_convergence_curve()
        self._plot_worker_timeline()
        self._plot_resource_utilization()
        self._plot_performance_distribution()
        self._plot_hyperparameter_progress()
        self._plot_optimization_statistics()
        
        self.fig.canvas.draw()
    
    def _plot_convergence_curve(self):
        """绘制收敛曲线"""
        ax = self.axes[0, 0]
        
        if len(self.convergence_data) > 1:
            data = list(self.convergence_data)
            evaluations = [d['evaluation'] for d in data]
            performances = [d['best_performance'] for d in data]
            
            ax.plot(evaluations, performances, 'b-', linewidth=2, marker='o', markersize=4)
            ax.fill_between(evaluations, performances, alpha=0.3)
            
        ax.set_title('Convergence Curve 收敛曲线', fontweight='bold')
        ax.set_xlabel('Evaluations 评估次数')
        ax.set_ylabel('Best Performance 最佳性能')
        ax.grid(True, alpha=0.3)
        ax.legend(['Best Performance'])
    
    def _plot_worker_timeline(self):
        """绘制工作器时间线"""
        ax = self.axes[0, 1]
        
        if self.worker_status:
            worker_ids = list(self.worker_status.keys())
            y_positions = range(len(worker_ids))
            
            for i, (worker_id, status) in enumerate(self.worker_status.items()):
                elapsed = status['elapsed_time']
                color = 'green' if elapsed < 300 else 'orange' if elapsed < 600 else 'red'
                
                # 绘制运行时间条
                ax.barh(i, elapsed, color=color, alpha=0.7)
                
                # 添加配置信息
                config_text = f"LR:{status['config'].get('learning_rate', 'N/A')}"
                ax.text(elapsed/2, i, config_text, ha='center', va='center', 
                       fontsize=8, fontweight='bold')
            
            ax.set_yticks(y_positions)
            ax.set_yticklabels([f"Worker {i+1}" for i in y_positions])
        
        ax.set_title('Worker Timeline 工作器时间线', fontweight='bold')
        ax.set_xlabel('Elapsed Time (seconds) 运行时间（秒）')
        ax.grid(True, alpha=0.3)
    
    def _plot_resource_utilization(self):
        """绘制资源利用率"""
        ax = self.axes[0, 2]
        
        if len(self.resource_utilization) > 1:
            data = list(self.resource_utilization)
            timestamps = [d['timestamp'] for d in data]
            utilizations = [d['utilization'] * 100 for d in data]
            
            # 转换为相对时间
            start_time = timestamps[0]
            relative_times = [(t - start_time) / 60 for t in timestamps]  # 转换为分钟
            
            ax.plot(relative_times, utilizations, 'g-', linewidth=2)
            ax.fill_between(relative_times, utilizations, alpha=0.3, color='green')
            
            # 添加平均线
            avg_util = np.mean(utilizations)
            ax.axhline(y=avg_util, color='red', linestyle='--', 
                      label=f'Average: {avg_util:.1f}%')
        
        ax.set_title('Resource Utilization 资源利用率', fontweight='bold')
        ax.set_xlabel('Time (minutes) 时间（分钟）')
        ax.set_ylabel('Utilization (%) 利用率（%）')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _plot_performance_distribution(self):
        """绘制性能分布"""
        ax = self.axes[1, 0]
        
        if self.scheduler.completed_results:
            performances = [result.performance for result in self.scheduler.completed_results]
            
            # 直方图
            ax.hist(performances, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            
            # 添加统计信息
            mean_perf = np.mean(performances)
            std_perf = np.std(performances)
            ax.axvline(mean_perf, color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {mean_perf:.3f}')
            ax.axvline(mean_perf + std_perf, color='orange', linestyle=':', 
                      label=f'+1σ: {mean_perf + std_perf:.3f}')
            ax.axvline(mean_perf - std_perf, color='orange', linestyle=':', 
                      label=f'-1σ: {mean_perf - std_perf:.3f}')
        
        ax.set_title('Performance Distribution 性能分布', fontweight='bold')
        ax.set_xlabel('Performance 性能')
        ax.set_ylabel('Frequency 频率')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _plot_hyperparameter_progress(self):
        """绘制超参数进展"""
        ax = self.axes[1, 1]
        
        if self.scheduler.completed_results and len(self.scheduler.completed_results) > 5:
            # 获取最近的结果
            recent_results = self.scheduler.completed_results[-20:]
            
            # 提取学习率和性能
            learning_rates = []
            performances = []
            
            for result in recent_results:
                if 'learning_rate' in result.config:
                    learning_rates.append(result.config['learning_rate'])
                    performances.append(result.performance)
            
            if learning_rates:
                # 散点图显示学习率vs性能
                scatter = ax.scatter(learning_rates, performances, 
                                   c=range(len(performances)), 
                                   cmap='viridis', alpha=0.7, s=50)
                
                # 添加颜色条
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Evaluation Order 评估顺序')
        
        ax.set_title('Hyperparameter Progress 超参数进展', fontweight='bold')
        ax.set_xlabel('Learning Rate 学习率')
        ax.set_ylabel('Performance 性能')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
    
    def _plot_optimization_statistics(self):
        """绘制优化统计信息"""
        ax = self.axes[1, 2]
        
        # 统计信息
        stats_data = []
        if self.performance_history:
            latest = list(self.performance_history)[-1]
            stats_data = [
                ('Completed', latest['evaluations_completed']),
                ('Active', latest['active_evaluations']),
                ('Remaining', latest['remaining_budget'])
            ]
        
        if stats_data:
            labels, values = zip(*stats_data)
            colors = ['green', 'orange', 'gray']
            
            # 饼图
            wedges, texts, autotexts = ax.pie(values, labels=labels, colors=colors, 
                                             autopct='%1.1f%%', startangle=90)
            
            # 美化文字
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        
        ax.set_title('Optimization Statistics 优化统计', fontweight='bold')
    
    def generate_interactive_dashboard(self):
        """生成交互式仪表板"""
        if not self.scheduler.completed_results:
            print("No completed results to visualize")
            return
        
        # 创建子图
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Convergence Progress 收敛进展',
                'Performance vs Time 性能vs时间',
                'Hyperparameter Sensitivity 超参数敏感性',
                'Worker Efficiency 工作器效率',
                'Resource Timeline 资源时间线',
                'Performance Statistics 性能统计'
            ),
            specs=[[{"secondary_y": True}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "box"}]]
        )
        
        self._add_interactive_plots(fig)
        
        # 更新布局
        fig.update_layout(
            title="Asynchronous HPO Interactive Dashboard 异步HPO交互式仪表板",
            height=1000,
            showlegend=True,
            template="plotly_white"
        )
        
        # 显示
        fig.show()
    
    def _add_interactive_plots(self, fig):
        """添加交互式图表"""
        results = self.scheduler.completed_results
        
        # 1. 收敛进展
        evaluations = list(range(1, len(results) + 1))
        best_performances = []
        current_best = 0
        
        for result in results:
            current_best = max(current_best, result.performance)
            best_performances.append(current_best)
        
        fig.add_trace(
            go.Scatter(x=evaluations, y=best_performances, 
                      mode='lines+markers', name='Best Performance',
                      line=dict(color='blue', width=3)),
            row=1, col=1
        )
        
        # 2. 性能vs时间
        training_times = [result.training_time for result in results]
        performances = [result.performance for result in results]
        
        fig.add_trace(
            go.Scatter(x=training_times, y=performances, 
                      mode='markers', name='Evaluations',
                      marker=dict(size=8, color=performances, 
                                colorscale='viridis', showscale=True)),
            row=1, col=2
        )
        
        # 3. 超参数敏感性（学习率）
        learning_rates = []
        lr_performances = []
        
        for result in results:
            if 'learning_rate' in result.config:
                learning_rates.append(result.config['learning_rate'])
                lr_performances.append(result.performance)
        
        if learning_rates:
            # 按学习率分组计算平均性能
            lr_groups = {}
            for lr, perf in zip(learning_rates, lr_performances):
                if lr not in lr_groups:
                    lr_groups[lr] = []
                lr_groups[lr].append(perf)
            
            lr_values = list(lr_groups.keys())
            avg_performances = [np.mean(lr_groups[lr]) for lr in lr_values]
            
            fig.add_trace(
                go.Bar(x=[str(lr) for lr in lr_values], y=avg_performances,
                      name='Avg Performance by LR'),
                row=2, col=1
            )
        
        # 4. 工作器效率
        worker_performances = {}
        for result in results:
            worker_id = result.worker_id
            if worker_id not in worker_performances:
                worker_performances[worker_id] = []
            worker_performances[worker_id].append(result.performance)
        
        for worker_id, perfs in worker_performances.items():
            fig.add_trace(
                go.Scatter(x=list(range(len(perfs))), y=perfs,
                          mode='lines+markers', name=f'Worker {worker_id}'),
                row=2, col=2
            )
        
        # 5. 资源时间线
        start_times = [(result.start_time.timestamp() - results[0].start_time.timestamp()) / 60 
                      for result in results]  # 转换为相对分钟
        end_times = [(result.end_time.timestamp() - results[0].start_time.timestamp()) / 60 
                    for result in results]
        
        for i, result in enumerate(results):
            fig.add_trace(
                go.Scatter(x=[start_times[i], end_times[i]], 
                          y=[i, i], mode='lines',
                          line=dict(width=8, color=f'rgba({int(result.performance*255)}, 100, 150, 0.8)'),
                          name=f'Eval {i+1}',
                          hovertemplate=f'Performance: {result.performance:.3f}<extra></extra>'),
                row=3, col=1
            )
        
        # 6. 性能统计
        fig.add_trace(
            go.Box(y=performances, name='Performance Distribution'),
            row=3, col=2
        )

# 使用示例
def run_visualization_example():
    """运行可视化示例"""
    
    # ... (使用之前定义的调度器设置)
    param_space = {
        'learning_rate': [0.0001, 0.001, 0.01, 0.1],
        'batch_size': [32, 64, 128],
        'filters': [16, 32, 64],
        'dropout': (0.0, 0.5),
        'optimizer': ['adam', 'sgd']
    }
    
    async_objective = AsyncObjectiveFunction(
        dataset_loader=cifar10_loader_factory,
        model_factory=simple_cnn_factory,
        max_workers=3
    )
    
    search_algorithm = AsyncRandomSearch(param_space, seed=42)
    
    scheduler = AsyncHPOScheduler(
        objective_function=async_objective,
        search_algorithm=search_algorithm,
        max_concurrent_evaluations=3,
        total_budget=15,
        time_budget_hours=0.25
    )
    
    # 创建可视化器
    visualizer = AsyncOptimizationVisualizer(scheduler, update_interval=2.0)
    
    try:
        # 启动监控
        visualizer.start_monitoring()
        
        # 启动优化（在另一个线程中）
        optimization_thread = threading.Thread(
            target=scheduler.start_optimization,
            daemon=True
        )
        optimization_thread.start()
        
        # 等待优化完成
        optimization_thread.join()
        
        # 生成最终的交互式仪表板
        print("Generating interactive dashboard...")
        visualizer.generate_interactive_dashboard()
        
    finally:
        visualizer.stop_monitoring()
        async_objective.shutdown()

# 高级可视化功能
class HPOAnalyzer:
    """HPO结果分析器"""
    
    def __init__(self, results: List[EvaluationResult]):
        self.results = results
        self.df = self._create_dataframe()
    
    def _create_dataframe(self) -> pd.DataFrame:
        """创建结果DataFrame"""
        data = []
        for result in self.results:
            row = {
                'config_id': result.config_id,
                'performance': result.performance,
                'training_time': result.training_time,
                'worker_id': result.worker_id
            }
            # 添加超参数
            row.update(result.config)
            data.append(row)
        
        return pd.DataFrame(data)
    
    def analyze_hyperparameter_importance(self):
        """分析超参数重要性"""
        if self.df.empty:
            return
        
        # 找出数值型超参数
        numeric_params = []
        for col in self.df.columns:
            if col not in ['config_id', 'performance', 'training_time', 'worker_id']:
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    numeric_params.append(col)
        
        if not numeric_params:
            return
        
        # 计算相关性
        correlations = {}
        for param in numeric_params:
            corr = self.df[param].corr(self.df['performance'])
            if not pd.isna(corr):
                correlations[param] = abs(corr)
        
        # 可视化
        if correlations:
            plt.figure(figsize=(10, 6))
            params = list(correlations.keys())
            values = list(correlations.values())
            
            bars = plt.bar(params, values, color='skyblue', edgecolor='navy', alpha=0.7)
            plt.title('Hyperparameter Importance (Correlation with Performance)\n超参数重要性（与性能的相关性）', 
                     fontweight='bold', fontsize=14)
            plt.xlabel('Hyperparameters 超参数')
            plt.ylabel('Absolute Correlation 绝对相关性')
            plt.xticks(rotation=45)
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
    
    def plot_parameter_sweep(self, param_name: str):
        """绘制参数扫描图"""
        if param_name not in self.df.columns:
            print(f"Parameter {param_name} not found in results")
            return
        
        plt.figure(figsize=(12, 8))
        
        # 根据参数类型选择不同的可视化方式
        if pd.api.types.is_numeric_dtype(self.df[param_name]):
            # 数值参数：散点图
            plt.scatter(self.df[param_name], self.df['performance'], 
                       alpha=0.6, s=50, c=self.df['training_time'], cmap='viridis')
            plt.colorbar(label='Training Time (seconds) 训练时间（秒）')
            
            # 添加趋势线
            z = np.polyfit(self.df[param_name], self.df['performance'], 1)
            p = np.poly1d(z)
            plt.plot(self.df[param_name], p(self.df[param_name]), "r--", alpha=0.8, linewidth=2)
            
        else:
            # 分类参数：箱线图
            unique_values = self.df[param_name].unique()
            data_by_value = [self.df[self.df[param_name] == val]['performance'].values 
                           for val in unique_values]
            
            plt.boxplot(data_by_value, labels=unique_values)
            
            # 添加散点
            for i, val in enumerate(unique_values):
                val_data = self.df[self.df[param_name] == val]['performance']
                x = np.random.normal(i+1, 0.04, size=len(val_data))
                plt.scatter(x, val_data, alpha=0.6, s=30)
        
        plt.title(f'Performance vs {param_name}\n性能与{param_name}的关系', 
                 fontweight='bold', fontsize=14)
        plt.xlabel(f'{param_name}')
        plt.ylabel('Performance 性能')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def generate_summary_report(self):
        """生成总结报告"""
        print("="*60)
        print("ASYNCHRONOUS HPO SUMMARY REPORT")
        print("异步超参数优化总结报告")
        print("="*60)
        
        print(f"\n📊 Overall Statistics 总体统计:")
        print(f"   Total evaluations: {len(self.results)} 总评估次数")
        print(f"   Best performance: {self.df['performance'].max():.4f} 最佳性能")
        print(f"   Average performance: {self.df['performance'].mean():.4f} 平均性能")
        print(f"   Performance std: {self.df['performance'].std():.4f} 性能标准差")
        
        print(f"\n⏱️  Timing Analysis 时间分析:")
        print(f"   Total training time: {self.df['training_time'].sum():.1f} seconds 总训练时间")
        print(f"   Average training time: {self.df['training_time'].mean():.1f} seconds 平均训练时间")
        print(f"   Fastest evaluation: {self.df['training_time'].min():.1f} seconds 最快评估")
        print(f"   Slowest evaluation: {self.df['training_time'].max():.1f} seconds 最慢评估")
        
        print(f"\n🏆 Best Configuration 最佳配置:")
        best_idx = self.df['performance'].idxmax()
        best_result = self.results[best_idx]
        for key, value in best_result.config.items():
            print(f"   {key}: {value}")
        
        print(f"\n👥 Worker Statistics 工作器统计:")
        worker_stats = self.df.groupby('worker_id').agg({
            'performance': ['count', 'mean', 'std'],
            'training_time': ['mean', 'sum']
        }).round(4)
        print(worker_stats)
        
        print("\n" + "="*60)

if __name__ == "__main__":
    # 运行可视化示例
    run_visualization_example()
```

## 16.3.4. Summary 总结

### Asynchronous Random Search Overview 异步随机搜索概述

Asynchronous Random Search represents a significant advancement over traditional sequential hyperparameter optimization. 异步随机搜索代表了相对于传统顺序超参数优化的重大进步。By leveraging parallel computation effectively, it addresses one of the most practical limitations of hyperparameter optimization: wall-clock time. 通过有效利用并行计算，它解决了超参数优化最实际的限制之一：实际时间。

### Key Advantages 主要优势

**1. Computational Efficiency 计算效率**
- **Parallel Evaluation**: Multiple configurations evaluated simultaneously 并行评估：同时评估多个配置
- **Resource Utilization**: Better use of available computational resources 资源利用：更好地利用可用的计算资源
- **Time Reduction**: Significant reduction in total optimization time 时间缩减：显著减少总优化时间

**2. Scalability 可扩展性**
- **Hardware Scaling**: Easily scales with available computational resources 硬件扩展：轻松扩展到可用的计算资源
- **Dynamic Resource Management**: Adaptive allocation based on current workload 动态资源管理：基于当前工作负载的自适应分配
- **Fault Tolerance**: Continues optimization even if some evaluations fail 容错性：即使某些评估失败也继续优化

**3. Practical Benefits 实际好处**
- **Real-world Applicability**: Suitable for modern multi-GPU/multi-node environments 现实世界适用性：适合现代多GPU/多节点环境
- **Cost Effectiveness**: Better return on computational investment 成本效益：计算投资的更好回报
- **Flexibility**: Compatible with various search algorithms 灵活性：与各种搜索算法兼容

### Implementation Considerations 实现考虑

**1. Load Balancing 负载均衡**
Different configurations may have vastly different training times, requiring intelligent work distribution. 不同配置可能有截然不同的训练时间，需要智能的工作分配。

**2. Communication Overhead 通信开销**
Managing multiple workers requires careful consideration of communication costs and synchronization points. 管理多个工作器需要仔细考虑通信成本和同步点。

**3. Resource Contention 资源竞争**
Multiple simultaneous evaluations may compete for shared resources (memory, I/O, network). 多个同时评估可能竞争共享资源（内存、I/O、网络）。

### Best Practices 最佳实践

```python
class AsyncHPOBestPractices:
    """异步HPO最佳实践指南"""
    
    @staticmethod
    def estimate_optimal_parallelism(avg_eval_time: float, 
                                   total_budget: int, 
                                   available_workers: int) -> int:
        """估计最优并行度"""
        # 基于评估时间和预算的启发式方法
        if avg_eval_time < 300:  # 5分钟以下
            return min(available_workers, max(2, total_budget // 10))
        elif avg_eval_time < 1800:  # 30分钟以下
            return min(available_workers, max(1, total_budget // 20))
        else:  # 长时间评估
            return min(available_workers, 2)
    
    @staticmethod
    def configure_worker_allocation(evaluation_complexity: Dict) -> Dict:
        """配置工作器分配策略"""
        strategies = {
            'low_complexity': {
                'max_concurrent': 8,
                'timeout_hours': 1,
                'retry_failed': True
            },
            'medium_complexity': {
                'max_concurrent': 4,
                'timeout_hours': 4,
                'retry_failed': False
            },
            'high_complexity': {
                'max_concurrent': 2,
                'timeout_hours': 12,
                'retry_failed': False
            }
        }
        
        complexity_level = evaluation_complexity.get('level', 'medium')
        return strategies.get(complexity_level, strategies['medium_complexity'])
    
    @staticmethod
    def design_early_stopping_strategy(budget_utilization: float) -> Dict:
        """设计早期停止策略"""
        if budget_utilization < 0.3:  # 预算充足
            return {
                'patience': 10,
                'min_improvement': 0.001,
                'warmup_epochs': 5
            }
        elif budget_utilization < 0.7:  # 预算适中
            return {
                'patience': 5,
                'min_improvement': 0.005,
                'warmup_epochs': 3
            }
        else:  # 预算紧张
            return {
                'patience': 3,
                'min_improvement': 0.01,
                'warmup_epochs': 2
            }
```

### Performance Metrics 性能指标

To evaluate the effectiveness of asynchronous optimization, consider these metrics: 要评估异步优化的有效性，请考虑这些指标：

```python
def calculate_async_efficiency_metrics(sync_time: float, 
                                     async_time: float, 
                                     num_workers: int) -> Dict:
    """计算异步效率指标"""
    
    speedup = sync_time / async_time
    efficiency = speedup / num_workers
    
    # 理论最大加速比（假设完美并行化）
    theoretical_speedup = min(num_workers, sync_time / min_eval_time)
    
    return {
        'speedup': speedup,
        'efficiency': efficiency,
        'theoretical_speedup': theoretical_speedup,
        'parallel_efficiency': speedup / theoretical_speedup,
        'time_savings_percent': (1 - async_time / sync_time) * 100
    }
```

### When to Use Asynchronous Random Search 何时使用异步随机搜索

**Ideal Scenarios 理想场景:**
- Multiple computational resources available 有多个计算资源可用
- Evaluation time varies significantly across configurations 不同配置的评估时间差异很大
- Large hyperparameter search spaces 大型超参数搜索空间
- Time constraints more critical than computational cost 时间约束比计算成本更关键

**Not Recommended 不推荐的情况:**
- Single computational resource 单一计算资源
- Very small hyperparameter spaces 非常小的超参数空间
- Highly coupled evaluations requiring sequential information 需要顺序信息的高度耦合评估
- Memory or storage constraints that prevent parallel execution 阻止并行执行的内存或存储约束

## 16.3.5. Exercises 练习题

### Exercise 1: Implementing Basic Async Random Search 练习1：实现基本异步随机搜索

**Objective 目标**: Implement a simplified asynchronous random search system. 实现一个简化的异步随机搜索系统。

**Task 任务**:
Create an async HPO system that can handle 2-4 concurrent evaluations with the following requirements: 创建一个可以处理2-4个并发评估的异步HPO系统，要求如下：

```python
# 实现这个类
class SimpleAsyncHPO:
    def __init__(self, objective_function, param_space, max_workers=2):
        pass
    
    def optimize(self, budget=20):
        """运行异步优化"""
        pass
    
    def get_status(self):
        """获取当前状态"""
        pass
    
    def get_best_result(self):
        """获取最佳结果"""
        pass

# 测试你的实现
param_space = {
    'learning_rate': [0.001, 0.01, 0.1],
    'batch_size': [32, 64, 128],
    'hidden_size': [64, 128, 256]
}

def simple_objective(config):
    # 模拟训练时间
    time.sleep(random.uniform(1, 5))
    # 模拟性能
    return random.random() * 0.9 + 0.1

hpo = SimpleAsyncHPO(simple_objective, param_space, max_workers=3)
results = hpo.optimize(budget=15)
```

**Requirements 要求**:
1. Use Python's `concurrent.futures` for parallelization 使用Python的`concurrent.futures`进行并行化
2. Implement proper result collection and status monitoring 实现适当的结果收集和状态监控
3. Handle failed evaluations gracefully 优雅地处理失败的评估
4. Track timing and resource utilization 跟踪时间和资源利用率

### Exercise 2: Comparing Sync vs Async Performance 练习2：比较同步与异步性能

**Objective 目标**: Quantitatively compare synchronous and asynchronous optimization performance. 定量比较同步和异步优化性能。

**Task 任务**:
```python
def compare_sync_vs_async(param_space, objective_function, budget=30):
    """比较同步和异步优化性能"""
    
    # 实现同步版本
    def run_sync_optimization():
        # 你的实现
        pass
    
    # 实现异步版本
    def run_async_optimization(max_workers=4):
        # 你的实现
        pass
    
    # 运行比较
    sync_results = run_sync_optimization()
    async_results = run_async_optimization()
    
    # 分析和可视化结果
    analyze_comparison(sync_results, async_results)

def analyze_comparison(sync_results, async_results):
    """分析比较结果"""
    # 计算指标：总时间、最佳性能、收敛速度等
    pass
```

**Expected Analysis 预期分析**:
- Total wall-clock time comparison 总实际时间比较
- Best performance achieved 达到的最佳性能
- Convergence speed analysis 收敛速度分析
- Resource utilization efficiency 资源利用效率

### Exercise 3: Advanced Async Scheduler with Load Balancing 练习3：带负载均衡的高级异步调度器

**Objective 目标**: Implement an advanced scheduler that handles variable evaluation times intelligently. 实现一个智能处理可变评估时间的高级调度器。

**Task 任务**:
Design a scheduler that:
- Estimates evaluation time for new configurations 估计新配置的评估时间
- Balances load across workers 在工作器间平衡负载
- Adapts parallelism based on current performance 基于当前性能调整并行度

```python
class AdvancedAsyncScheduler:
    def __init__(self, objective_function, param_space):
        pass
    
    def estimate_evaluation_time(self, config):
        """估计配置的评估时间"""
        pass
    
    def balance_workload(self):
        """平衡工作负载"""
        pass
    
    def adapt_parallelism(self):
        """自适应调整并行度"""
        pass
    
    def optimize_with_adaptation(self, budget=50):
        """带自适应的优化"""
        pass
```

**Advanced Features 高级功能**:
- Dynamic worker allocation 动态工作器分配
- Intelligent configuration queuing 智能配置排队
- Performance-based scheduling 基于性能的调度

### Exercise 4: Visualization and Monitoring System 练习4：可视化和监控系统

**Objective 目标**: Create a comprehensive monitoring system for async HPO. 为异步HPO创建综合监控系统。

**Task 任务**:
```python
class AsyncHPOMonitor:
    def __init__(self, scheduler):
        pass
    
    def start_real_time_monitoring(self):
        """启动实时监控"""
        pass
    
    def create_live_dashboard(self):
        """创建实时仪表板"""
        pass
    
    def generate_performance_report(self):
        """生成性能报告"""
        pass
    
    def plot_worker_utilization(self):
        """绘制工作器利用率"""
        pass
    
    def analyze_bottlenecks(self):
        """分析瓶颈"""
        pass
```

**Visualization Requirements 可视化要求**:
- Real-time convergence curves 实时收敛曲线
- Worker timeline and utilization 工作器时间线和利用率
- Resource consumption monitoring 资源消耗监控
- Interactive parameter exploration 交互式参数探索

### Exercise 5: Fault-Tolerant Async HPO 练习5：容错异步HPO

**Objective 目标**: Implement fault tolerance in asynchronous optimization. 在异步优化中实现容错性。

**Task 任务**:
Handle various failure scenarios:
- Worker crashes during evaluation 评估期间工作器崩溃
- Network interruptions 网络中断
- Resource exhaustion 资源耗尽
- Configuration-specific failures 特定配置的失败

```python
class FaultTolerantAsyncHPO:
    def __init__(self, objective_function, param_space):
        self.retry_policy = RetryPolicy()
        self.checkpoint_manager = CheckpointManager()
        pass
    
    def handle_worker_failure(self, worker_id, config):
        """处理工作器失败"""
        pass
    
    def implement_checkpointing(self):
        """实现检查点"""
        pass
    
    def recover_from_failure(self):
        """从失败中恢复"""
        pass
    
    def robust_optimize(self, budget=100):
        """稳健的优化"""
        pass

class RetryPolicy:
    def should_retry(self, failure_type, attempt_count):
        """判断是否应该重试"""
        pass

class CheckpointManager:
    def save_state(self, state):
        """保存状态"""
        pass
    
    def load_state(self):
        """加载状态"""
        pass
```

**Fault Tolerance Features 容错功能**:
- Automatic retry mechanisms 自动重试机制
- State checkpointing and recovery 状态检查点和恢复
- Graceful degradation under resource constraints 资源约束下的优雅降级
- Comprehensive error logging and analysis 全面的错误记录和分析

### Answer Guidelines 答案指导

**For Exercise 1 练习1**:
- Focus on clean thread management and result aggregation 专注于清洁的线程管理和结果聚合
- Implement proper synchronization mechanisms 实现适当的同步机制
- Include comprehensive error handling 包含全面的错误处理

**For Exercise 2 练习2**:
- Use statistical analysis to compare performance 使用统计分析比较性能
- Consider both efficiency and effectiveness metrics 考虑效率和有效性指标
- Visualize results with clear before/after comparisons 用清晰的前后比较可视化结果

**For Exercise 3 练习3**:
- Implement machine learning-based time estimation 实现基于机器学习的时间估计
- Design adaptive algorithms that respond to changing conditions 设计响应变化条件的自适应算法
- Consider real-world constraints like memory and network limitations 考虑内存和网络限制等现实约束

**For Exercise 4 练习4**:
- Use modern visualization libraries (Plotly, Bokeh) for interactivity 使用现代可视化库（Plotly，Bokeh）实现交互性
- Implement real-time updates without performance degradation 实现不影响性能的实时更新
- Design user-friendly interfaces for non-technical users 为非技术用户设计用户友好的界面

**For Exercise 5 练习5**:
- Implement robust error classification and handling 实现强大的错误分类和处理
- Design comprehensive testing scenarios for failure modes 为失败模式设计全面的测试场景
- Consider distributed systems concepts like consensus and coordination 考虑分布式系统概念，如共识和协调

These exercises provide hands-on experience with the practical challenges and solutions in asynchronous hyperparameter optimization. 这些练习提供了异步超参数优化中实际挑战和解决方案的实践经验。

## 16.4. Multi-Fidelity Hyperparameter Optimization 多保真度超参数优化

### Introduction to Multi-Fidelity Optimization 多保真度优化介绍

Multi-fidelity hyperparameter optimization is like having a preview system when shopping for clothes. 多保真度超参数优化就像购买衣服时的预览系统。Instead of trying on every outfit completely (which takes time), you first look at how clothes appear on a mannequin or in photos (low fidelity), then try on the most promising ones briefly (medium fidelity), and finally spend time properly fitting only the best candidates (high fidelity). 与其完全试穿每件衣服（这很耗时），您首先查看衣服在人体模型或照片中的外观（低保真度），然后简单试穿最有前景的（中保真度），最后只花时间适当试穿最佳候选者（高保真度）。

The core idea is to use cheaper, faster approximations of the true objective function to guide the search, reserving expensive full evaluations only for the most promising configurations. 核心思想是使用更便宜、更快的真实目标函数近似来指导搜索，只为最有前景的配置保留昂贵的完整评估。

### Types of Fidelity 保真度类型

**1. Training Time/Epochs 训练时间/轮次**
- Low fidelity: 5-10 epochs 低保真度：5-10轮次
- Medium fidelity: 25-50 epochs 中保真度：25-50轮次  
- High fidelity: 100+ epochs 高保真度：100+轮次

**2. Dataset Size 数据集大小**
- Low fidelity: 10-20% of data 低保真度：10-20%的数据
- Medium fidelity: 50% of data 中保真度：50%的数据
- High fidelity: Full dataset 高保真度：完整数据集

**3. Model Complexity 模型复杂度**
- Low fidelity: Simplified architecture 低保真度：简化架构
- Medium fidelity: Intermediate complexity 中保真度：中等复杂度
- High fidelity: Full target architecture 高保真度：完整目标架构

**4. Resolution/Precision 分辨率/精度**
- Low fidelity: Lower input resolution, reduced precision 低保真度：较低输入分辨率，降低精度
- Medium fidelity: Standard settings 中保真度：标准设置
- High fidelity: Full resolution and precision 高保真度：完整分辨率和精度

## 16.4.1. Successive Halving 连续减半

### The Successive Halving Algorithm 连续减半算法

Successive Halving is like organizing a tournament where we start with many participants and progressively eliminate the weaker ones. 连续减半就像组织一场锦标赛，我们从许多参与者开始，逐步淘汰较弱的参与者。At each round, we give the remaining participants more resources (higher fidelity) but eliminate half of them based on their performance. 在每一轮中，我们给剩余的参与者更多资源（更高保真度），但根据他们的表现淘汰一半。

```python
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Callable
import time
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import random

@dataclass
class FidelityConfig:
    """保真度配置"""
    name: str
    epochs: int
    dataset_fraction: float
    batch_size_multiplier: float = 1.0
    early_stopping_patience: int = 5
    
    @property
    def relative_cost(self) -> float:
        """相对计算成本"""
        return self.epochs * self.dataset_fraction * (1 / self.batch_size_multiplier)

class SuccessiveHalvingScheduler:
    """连续减半调度器"""
    
    def __init__(self, 
                 param_space: Dict,
                 fidelity_configs: List[FidelityConfig],
                 initial_budget: int = 100,
                 elimination_rate: float = 0.5,
                 random_seed: int = 42):
        
        self.param_space = param_space
        self.fidelity_configs = fidelity_configs
        self.initial_budget = initial_budget
        self.elimination_rate = elimination_rate
        self.random_seed = random_seed
        
        # 设置随机种子
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # 初始化状态
        self.all_configs = []
        self.results_history = []
        self.current_round = 0
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def generate_initial_configurations(self, num_configs: int) -> List[Dict]:
        """生成初始配置"""
        configs = []
        
        for _ in range(num_configs):
            config = {}
            for param_name, param_values in self.param_space.items():
                if isinstance(param_values, list):
                    config[param_name] = random.choice(param_values)
                elif isinstance(param_values, tuple) and len(param_values) == 2:
                    min_val, max_val = param_values
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        config[param_name] = random.randint(min_val, max_val)
                    else:
                        config[param_name] = random.uniform(min_val, max_val)
            configs.append(config)
        
        return configs
    
    def run_successive_halving(self, 
                             objective_function: Callable,
                             min_fidelity_index: int = 0) -> Tuple[Dict, float]:
        """运行连续减半算法"""
        
        self.logger.info(f"Starting Successive Halving with {self.initial_budget} initial configurations")
        
        # 生成初始配置
        current_configs = self.generate_initial_configurations(self.initial_budget)
        self.all_configs = current_configs.copy()
        
        # 逐步提高保真度
        for fidelity_idx in range(min_fidelity_index, len(self.fidelity_configs)):
            fidelity = self.fidelity_configs[fidelity_idx]
            
            self.logger.info(f"\n--- Round {self.current_round + 1}: {fidelity.name} ---")
            self.logger.info(f"Evaluating {len(current_configs)} configurations")
            self.logger.info(f"Fidelity: {fidelity.epochs} epochs, {fidelity.dataset_fraction:.1%} data")
            
            # 评估当前配置
            round_results = []
            for i, config in enumerate(current_configs):
                self.logger.info(f"Evaluating configuration {i+1}/{len(current_configs)}")
                
                performance = objective_function(config, fidelity)
                result = {
                    'config': config,
                    'performance': performance,
                    'fidelity': fidelity,
                    'round': self.current_round,
                    'config_id': f"round_{self.current_round}_config_{i}"
                }
                round_results.append(result)
                self.results_history.append(result)
                
                self.logger.info(f"Performance: {performance:.4f}")
            
            # 按性能排序
            round_results.sort(key=lambda x: x['performance'], reverse=True)
            
            # 显示当前轮次结果
            self._display_round_results(round_results, fidelity)
            
            # 选择继续的配置（除最后一轮外）
            if fidelity_idx < len(self.fidelity_configs) - 1:
                num_survivors = max(1, int(len(current_configs) * self.elimination_rate))
                current_configs = [r['config'] for r in round_results[:num_survivors]]
                
                self.logger.info(f"Advancing {num_survivors} configurations to next round")
            
            self.current_round += 1
        
        # 返回最佳配置
        best_result = max(self.results_history, key=lambda x: x['performance'])
        
        self.logger.info(f"\n=== SUCCESSIVE HALVING COMPLETE ===")
        self.logger.info(f"Best performance: {best_result['performance']:.4f}")
        self.logger.info(f"Best configuration: {best_result['config']}")
        
        return best_result['config'], best_result['performance']
    
    def _display_round_results(self, round_results: List[Dict], fidelity: FidelityConfig):
        """显示当前轮次结果"""
        self.logger.info(f"\nTop 5 configurations in {fidelity.name}:")
        for i, result in enumerate(round_results[:5]):
            config_str = ', '.join([f"{k}={v}" for k, v in result['config'].items()])
            self.logger.info(f"  {i+1}. Performance: {result['performance']:.4f} | {config_str}")
    
    def plot_successive_halving_progress(self):
        """绘制连续减半进展"""
        if not self.results_history:
            return
        
        # 按轮次组织数据
        rounds_data = {}
        for result in self.results_history:
            round_num = result['round']
            if round_num not in rounds_data:
                rounds_data[round_num] = []
            rounds_data[round_num].append(result['performance'])
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 每轮最佳性能
        ax1 = axes[0, 0]
        rounds = sorted(rounds_data.keys())
        best_performances = [max(rounds_data[r]) for r in rounds]
        fidelity_names = [self.fidelity_configs[r].name for r in rounds]
        
        bars = ax1.bar(range(len(rounds)), best_performances, 
                      color=['lightblue', 'lightgreen', 'lightcoral', 'lightyellow'][:len(rounds)])
        ax1.set_xlabel('Round (Fidelity Level) 轮次（保真度级别）')
        ax1.set_ylabel('Best Performance 最佳性能')
        ax1.set_title('Best Performance by Fidelity Level 各保真度级别的最佳性能')
        ax1.set_xticks(range(len(rounds)))
        ax1.set_xticklabels(fidelity_names, rotation=45, ha='right')
        
        # 添加数值标签
        for bar, perf in zip(bars, best_performances):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{perf:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. 配置数量随轮次变化
        ax2 = axes[0, 1]
        config_counts = [len(rounds_data[r]) for r in rounds]
        ax2.plot(range(len(rounds)), config_counts, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Round 轮次')
        ax2.set_ylabel('Number of Configurations 配置数量')
        ax2.set_title('Configuration Elimination Progress 配置淘汰进展')
        ax2.set_xticks(range(len(rounds)))
        ax2.set_xticklabels(fidelity_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # 3. 性能分布比较
        ax3 = axes[1, 0]
        positions = []
        data_to_plot = []
        labels = []
        
        for i, round_num in enumerate(rounds):
            positions.append(i + 1)
            data_to_plot.append(rounds_data[round_num])
            labels.append(f'Round {round_num+1}')
        
        bp = ax3.boxplot(data_to_plot, positions=positions, patch_artist=True)
        
        # 美化箱线图
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax3.set_xlabel('Round 轮次')
        ax3.set_ylabel('Performance Distribution 性能分布')
        ax3.set_title('Performance Distribution by Round 各轮次性能分布')
        ax3.set_xticklabels(labels, rotation=45, ha='right')
        
        # 4. 计算效率分析
        ax4 = axes[1, 1]
        
        # 计算累积计算成本和累积最佳性能
        cumulative_cost = 0
        cumulative_costs = []
        cumulative_best = []
        
        for round_num in rounds:
            fidelity = self.fidelity_configs[round_num]
            round_cost = len(rounds_data[round_num]) * fidelity.relative_cost
            cumulative_cost += round_cost
            cumulative_costs.append(cumulative_cost)
            
            # 到目前为止的最佳性能
            all_perfs_so_far = []
            for r in range(round_num + 1):
                all_perfs_so_far.extend(rounds_data[r])
            cumulative_best.append(max(all_perfs_so_far))
        
        ax4.plot(cumulative_costs, cumulative_best, 'go-', linewidth=3, markersize=8)
        ax4.set_xlabel('Cumulative Computational Cost 累积计算成本')
        ax4.set_ylabel('Best Performance Found 找到的最佳性能')
        ax4.set_title('Efficiency: Performance vs Cost 效率：性能vs成本')
        ax4.grid(True, alpha=0.3)
        
        # 添加轮次标注
        for i, (cost, perf) in enumerate(zip(cumulative_costs, cumulative_best)):
            ax4.annotate(f'Round {i+1}', (cost, perf), 
                        textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_configuration_survival(self):
        """分析配置存活情况"""
        if not self.results_history:
            return
        
        # 跟踪每个配置在各轮次的表现
        config_tracking = {}
        
        for result in self.results_history:
            config_str = str(sorted(result['config'].items()))
            if config_str not in config_tracking:
                config_tracking[config_str] = {
                    'config': result['config'],
                    'rounds': [],
                    'performances': []
                }
            
            config_tracking[config_str]['rounds'].append(result['round'])
            config_tracking[config_str]['performances'].append(result['performance'])
        
        # 分析存活配置的特征
        survivors = [cfg for cfg in config_tracking.values() 
                    if len(cfg['rounds']) == len(self.fidelity_configs)]
        
        print(f"\n=== CONFIGURATION SURVIVAL ANALYSIS ===")
        print(f"Total initial configurations: {self.initial_budget}")
        print(f"Configurations that survived all rounds: {len(survivors)}")
        
        if survivors:
            print(f"\nSurviving configurations:")
            for i, survivor in enumerate(survivors):
                config_str = ', '.join([f"{k}={v}" for k, v in survivor['config'].items()])
                final_performance = survivor['performances'][-1]
                print(f"  {i+1}. {config_str} | Final performance: {final_performance:.4f}")
                
                # 显示性能改进
                if len(survivor['performances']) > 1:
                    improvement = survivor['performances'][-1] - survivor['performances'][0]
                    print(f"      Performance improvement: {improvement:+.4f}")

# 多保真度目标函数示例
class MultiFidelityObjective:
    """多保真度目标函数"""
    
    def __init__(self, dataset_factory, model_factory):
        self.dataset_factory = dataset_factory
        self.model_factory = model_factory
        self.cache = {}  # 缓存结果以避免重复计算
    
    def __call__(self, config: Dict, fidelity: FidelityConfig) -> float:
        """评估配置在给定保真度下的性能"""
        
        # 创建缓存键
        cache_key = (str(sorted(config.items())), fidelity.name)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # 创建模型
        model = self.model_factory(config)
        
        # 获取数据
        train_loader, val_loader = self.dataset_factory(
            batch_size=int(config.get('batch_size', 64) * fidelity.batch_size_multiplier),
            dataset_fraction=fidelity.dataset_fraction
        )
        
        # 训练模型
        performance = self._train_model(
            model, train_loader, val_loader, config, fidelity
        )
        
        # 缓存结果
        self.cache[cache_key] = performance
        
        return performance
    
    def _train_model(self, model, train_loader, val_loader, config, fidelity):
        """训练模型并返回验证性能"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        # 设置优化器
        if config.get('optimizer', 'adam') == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9)
        
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(fidelity.epochs):
            # 训练阶段
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                # 限制训练步数以加快演示
                if batch_idx > 20:
                    break
                    
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            
            # 验证阶段
            if epoch % max(1, fidelity.epochs // 5) == 0:  # 定期验证
                val_acc = self._evaluate_model(model, val_loader, device)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= fidelity.early_stopping_patience:
                        break
            
            # 模拟训练时间
            time.sleep(0.05)  # 缩短以加快演示
        
        return best_val_acc
    
    def _evaluate_model(self, model, val_loader, device):
        """评估模型性能"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                if batch_idx > 10:  # 限制验证步数
                    break
                    
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        return correct / total if total > 0 else 0.0

# 多保真度数据加载器工厂
def multi_fidelity_cifar10_factory(batch_size=64, dataset_fraction=1.0):
    """创建多保真度CIFAR-10数据加载器"""
    import torchvision
    import torchvision.transforms as transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # 加载完整数据集
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)
    valset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    
    # 根据保真度选择数据子集
    if dataset_fraction < 1.0:
        train_size = int(len(trainset) * dataset_fraction)
        val_size = int(len(valset) * dataset_fraction)
        
        # 随机选择子集
        train_indices = random.sample(range(len(trainset)), train_size)
        val_indices = random.sample(range(len(valset)), val_size)
        
        train_subset = Subset(trainset, train_indices)
        val_subset = Subset(valset, val_indices)
    else:
        train_subset = trainset
        val_subset = valset
    
    # 为演示使用更小的数据集
    if len(train_subset) > 2000:
        train_subset = Subset(train_subset, range(2000))
    if len(val_subset) > 400:
        val_subset = Subset(val_subset, range(400))
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

# 使用示例
def run_successive_halving_example():
    """运行连续减半示例"""
    
    # 定义保真度级别
    fidelity_configs = [
        FidelityConfig("Low Fidelity", epochs=3, dataset_fraction=0.2, 
                      batch_size_multiplier=2.0, early_stopping_patience=2),
        FidelityConfig("Medium Fidelity", epochs=8, dataset_fraction=0.5, 
                      batch_size_multiplier=1.5, early_stopping_patience=3),
        FidelityConfig("High Fidelity", epochs=15, dataset_fraction=1.0, 
                      batch_size_multiplier=1.0, early_stopping_patience=5)
    ]
    
    # 定义参数空间
    param_space = {
        'learning_rate': [0.0001, 0.001, 0.01, 0.1],
        'batch_size': [32, 64, 128],
        'filters': [16, 32, 64],
        'dropout': (0.0, 0.5),
        'optimizer': ['adam', 'sgd']
    }
    
    # 创建多保真度目标函数
    objective = MultiFidelityObjective(
        dataset_factory=multi_fidelity_cifar10_factory,
        model_factory=simple_cnn_factory  # 使用之前定义的模型工厂
    )
    
    # 创建连续减半调度器
    scheduler = SuccessiveHalvingScheduler(
        param_space=param_space,
        fidelity_configs=fidelity_configs,
        initial_budget=16,  # 较小的预算用于演示
        elimination_rate=0.5,
        random_seed=42
    )
    
    # 运行连续减半
    print("Starting Successive Halving optimization...")
    best_config, best_performance = scheduler.run_successive_halving(objective)
    
    # 分析结果
    scheduler.analyze_configuration_survival()
    
    # 可视化结果
    scheduler.plot_successive_halving_progress()
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Best configuration: {best_config}")
    print(f"Best performance: {best_performance:.4f}")

if __name__ == "__main__":
    run_successive_halving_example()
```

### Advanced Successive Halving Variants 高级连续减半变体

```python
class AdaptiveSuccessiveHalving(SuccessiveHalvingScheduler):
    """自适应连续减半"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.performance_variance_threshold = 0.05
        self.minimum_survivors = 2
    
    def adaptive_elimination(self, round_results: List[Dict]) -> List[Dict]:
        """自适应淘汰策略"""
        performances = [r['performance'] for r in round_results]
        
        # 计算性能方差
        perf_variance = np.var(performances)
        perf_mean = np.mean(performances)
        coefficient_of_variation = np.sqrt(perf_variance) / perf_mean if perf_mean > 0 else 0
        
        # 如果性能差异很小，保留更多配置
        if coefficient_of_variation < self.performance_variance_threshold:
            elimination_rate = 0.3  # 更保守的淘汰率
        else:
            elimination_rate = self.elimination_rate
        
        num_survivors = max(self.minimum_survivors, 
                           int(len(round_results) * elimination_rate))
        
        self.logger.info(f"Performance variance: {perf_variance:.4f}")
        self.logger.info(f"Coefficient of variation: {coefficient_of_variation:.4f}")
        self.logger.info(f"Adaptive elimination rate: {1-elimination_rate:.2f}")
        
        return round_results[:num_survivors]

class BanditBasedSuccessiveHalving(SuccessiveHalvingScheduler):
    """基于老虎机的连续减半"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.confidence_level = 0.95
        self.exploration_bonus = 0.1
    
    def calculate_upper_confidence_bound(self, performance_history: List[float], 
                                       total_evaluations: int) -> float:
        """计算上置信界"""
        if not performance_history:
            return float('inf')
        
        mean_performance = np.mean(performance_history)
        n_evaluations = len(performance_history)
        
        # UCB1公式
        exploration_term = np.sqrt(2 * np.log(total_evaluations) / n_evaluations)
        ucb = mean_performance + self.exploration_bonus * exploration_term
        
        return ucb
    
    def bandit_based_selection(self, round_results: List[Dict], 
                             num_survivors: int) -> List[Dict]:
        """基于老虎机算法的配置选择"""
        total_evaluations = len(self.results_history)
        
        # 为每个配置计算UCB
        for result in round_results:
            config_id = result['config_id']
            
            # 获取该配置的历史性能
            config_history = [r['performance'] for r in self.results_history 
                            if r.get('config_id') == config_id]
            
            ucb = self.calculate_upper_confidence_bound(config_history, total_evaluations)
            result['ucb'] = ucb
        
        # 按UCB排序
        round_results.sort(key=lambda x: x['ucb'], reverse=True)
        
        return round_results[:num_survivors]
```

## 16.4.2. Summary 总结

### Multi-Fidelity HPO Advantages 多保真度HPO优势

**1. Computational Efficiency 计算效率**
- **Early Elimination**: Poor configurations identified and eliminated quickly 早期淘汰：快速识别和淘汰不良配置
- **Resource Allocation**: More resources devoted to promising configurations 资源分配：更多资源投入到有前景的配置
- **Cost Reduction**: Significant reduction in total computational cost 成本降低：显著减少总计算成本

**2. Practical Benefits 实际好处**
- **Faster Convergence**: Reaches good solutions more quickly 更快收敛：更快达到好的解决方案
- **Better Resource Utilization**: Efficient use of computational budget 更好的资源利用：高效使用计算预算
- **Scalability**: Works well with large hyperparameter spaces 可扩展性：在大型超参数空间中表现良好

**3. Risk Management 风险管理**
- **Robust Performance**: Less sensitive to initial random configurations 稳健性能：对初始随机配置不太敏感
- **Progressive Validation**: Multiple validation stages reduce overfitting risk 渐进验证：多个验证阶段降低过拟合风险

### When to Use Multi-Fidelity Methods 何时使用多保真度方法

**Ideal Scenarios 理想场景:**
- Training time scales significantly with epochs/data size 训练时间随轮次/数据大小显著扩展
- Large hyperparameter search spaces 大型超参数搜索空间
- Limited computational budget 有限的计算预算
- Clear correlation between low and high fidelity performance 低保真度和高保真度性能之间有明确相关性

**Not Recommended 不推荐的情况:**
- Very fast evaluations (where fidelity savings are minimal) 非常快的评估（保真度节省很少）
- Poor correlation between fidelities 保真度之间相关性差
- Small hyperparameter spaces 小型超参数空间

The successive halving algorithm provides an elegant balance between exploration and exploitation while maintaining computational efficiency. 连续减半算法在保持计算效率的同时，在探索和利用之间提供了优雅的平衡。Its tournament-style elimination ensures that computational resources are increasingly focused on the most promising configurations. 其锦标赛式淘汰确保计算资源越来越集中在最有前景的配置上。

## 16.5. Asynchronous Successive Halving 异步连续减半

### Introduction to Asynchronous Multi-Fidelity Optimization 异步多保真度优化介绍

Asynchronous Successive Halving combines the best of both worlds: the computational efficiency of multi-fidelity optimization with the time efficiency of asynchronous execution. 异步连续减半结合了两个世界的最佳特性：多保真度优化的计算效率和异步执行的时间效率。Think of it as having multiple cooking shows running simultaneously, where each show has different rounds (fidelities), and chefs (configurations) can be eliminated from any show at any time based on their performance. 将其想象为同时运行多个烹饪节目，每个节目有不同的轮次（保真度），厨师（配置）可以根据他们的表现随时从任何节目中被淘汰。

This approach is particularly powerful when: 这种方法在以下情况下特别强大：
- Different configurations have varying training times 不同配置有不同的训练时间
- You have multiple computational resources available 您有多个可用的计算资源
- You want to maximize both computational and time efficiency 您想要最大化计算和时间效率

## 16.5.1. Objective Function 目标函数

### Designing Async Multi-Fidelity Objectives 设计异步多保真度目标

The objective function for asynchronous successive halving must handle both multiple fidelities and concurrent evaluations. 异步连续减半的目标函数必须处理多重保真度和并发评估。Here's a comprehensive implementation: 以下是一个综合实现：

```python
import asyncio
import concurrent.futures
import threading
import queue
import time
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import random

class EvaluationStatus(Enum):
    """评估状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class AsyncEvaluationResult:
    """异步评估结果"""
    config_id: str
    config: Dict
    fidelity: 'FidelityConfig'
    performance: float
    training_time: float
    worker_id: str
    status: EvaluationStatus
    start_time: float
    end_time: float
    round_number: int
    metadata: Dict = field(default_factory=dict)
    
    @property
    def efficiency_score(self) -> float:
        """计算效率分数（性能/时间）"""
        return self.performance / max(self.training_time, 0.1)

class AsyncMultiFidelityObjective:
    """异步多保真度目标函数"""
    
    def __init__(self, 
                 dataset_factory: Callable,
                 model_factory: Callable,
                 max_workers: int = 4,
                 timeout_minutes: float = 30):
        
        self.dataset_factory = dataset_factory
        self.model_factory = model_factory
        self.max_workers = max_workers
        self.timeout_seconds = timeout_minutes * 60
        
        # 工作器管理
        self.worker_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.active_evaluations = {}
        self.completed_results = queue.Queue()
        self.evaluation_cache = {}
        
        # 线程安全锁
        self.lock = threading.Lock()
        
        # 性能监控
        self.worker_stats = {i: {'jobs_completed': 0, 'total_time': 0.0} 
                           for i in range(max_workers)}
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def submit_evaluation(self, config: Dict, fidelity: 'FidelityConfig', 
                         round_number: int) -> str:
        """提交异步评估"""
        config_id = str(uuid.uuid4())
        
        # 检查缓存
        cache_key = self._create_cache_key(config, fidelity)
        if cache_key in self.evaluation_cache:
            cached_result = self.evaluation_cache[cache_key]
            result = AsyncEvaluationResult(
                config_id=config_id,
                config=config.copy(),
                fidelity=fidelity,
                performance=cached_result['performance'],
                training_time=cached_result['training_time'],
                worker_id='cache',
                status=EvaluationStatus.COMPLETED,
                start_time=time.time(),
                end_time=time.time(),
                round_number=round_number,
                metadata={'cached': True}
            )
            self.completed_results.put(result)
            return config_id
        
        # 提交到工作器池
        future = self.worker_pool.submit(
            self._evaluate_async, config_id, config, fidelity, round_number
        )
        
        with self.lock:
            self.active_evaluations[config_id] = {
                'future': future,
                'config': config,
                'fidelity': fidelity,
                'round_number': round_number,
                'submit_time': time.time()
            }
        
        self.logger.info(f"Submitted evaluation {config_id[:8]} for round {round_number}")
        return config_id
    
    def _evaluate_async(self, config_id: str, config: Dict, 
                       fidelity: 'FidelityConfig', round_number: int) -> AsyncEvaluationResult:
        """在工作线程中执行异步评估"""
        worker_id = threading.current_thread().name
        start_time = time.time()
        
        try:
            self.logger.info(f"Worker {worker_id} starting evaluation {config_id[:8]}")
            
            # 实际训练
            performance = self._train_model_with_fidelity(config, fidelity, config_id)
            
            end_time = time.time()
            training_time = end_time - start_time
            
            # 缓存结果
            cache_key = self._create_cache_key(config, fidelity)
            self.evaluation_cache[cache_key] = {
                'performance': performance,
                'training_time': training_time
            }
            
            # 更新工作器统计
            with self.lock:
                worker_idx = int(worker_id.split('-')[-1]) % self.max_workers
                self.worker_stats[worker_idx]['jobs_completed'] += 1
                self.worker_stats[worker_idx]['total_time'] += training_time
            
            result = AsyncEvaluationResult(
                config_id=config_id,
                config=config.copy(),
                fidelity=fidelity,
                performance=performance,
                training_time=training_time,
                worker_id=worker_id,
                status=EvaluationStatus.COMPLETED,
                start_time=start_time,
                end_time=end_time,
                round_number=round_number
            )
            
            self.completed_results.put(result)
            self.logger.info(f"Worker {worker_id} completed evaluation {config_id[:8]}: {performance:.4f}")
            
            return result
            
        except Exception as e:
            end_time = time.time()
            training_time = end_time - start_time
            
            self.logger.error(f"Worker {worker_id} failed evaluation {config_id[:8]}: {e}")
            
            result = AsyncEvaluationResult(
                config_id=config_id,
                config=config.copy(),
                fidelity=fidelity,
                performance=0.0,
                training_time=training_time,
                worker_id=worker_id,
                status=EvaluationStatus.FAILED,
                start_time=start_time,
                end_time=end_time,
                round_number=round_number,
                metadata={'error': str(e)}
            )
            
            self.completed_results.put(result)
            return result
    
    def _train_model_with_fidelity(self, config: Dict, fidelity: 'FidelityConfig', 
                                  config_id: str) -> float:
        """使用指定保真度训练模型"""
        # 创建模型
        model = self.model_factory(config)
        
        # 获取数据加载器
        train_loader, val_loader = self.dataset_factory(
            batch_size=int(config.get('batch_size', 64) * fidelity.batch_size_multiplier),
            dataset_fraction=fidelity.dataset_fraction
        )
        
        # 设置设备和优化器
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        if config.get('optimizer', 'adam') == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9)
        
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)
        
        # 训练循环
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(fidelity.epochs):
            # 检查是否应该停止（超时或取消）
            if self._should_stop_evaluation(config_id):
                break
            
            # 训练阶段
            model.train()
            train_loss = 0.0
            train_samples = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                # 限制每个epoch的批次数以控制时间
                if batch_idx >= fidelity.max_batches_per_epoch:
                    break
                
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * data.size(0)
                train_samples += data.size(0)
            
            # 验证阶段（每几个epoch进行一次）
            if epoch % max(1, fidelity.epochs // 5) == 0 or epoch == fidelity.epochs - 1:
                val_acc = self._evaluate_model(model, val_loader, device, fidelity)
                scheduler.step(val_acc)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= fidelity.early_stopping_patience:
                        self.logger.info(f"Early stopping for {config_id[:8]} at epoch {epoch}")
                        break
                
                avg_train_loss = train_loss / train_samples if train_samples > 0 else 0
                self.logger.debug(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # 模拟训练时间（可调整）
            time.sleep(0.02 * fidelity.relative_cost)
        
        return best_val_acc
    
    def _evaluate_model(self, model, val_loader, device, fidelity):
        """评估模型性能"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                if batch_idx >= fidelity.max_batches_per_epoch // 2:  # 验证使用较少批次
                    break
                
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        return correct / total if total > 0 else 0.0
    
    def _should_stop_evaluation(self, config_id: str) -> bool:
        """检查是否应该停止评估"""
        with self.lock:
            if config_id in self.active_evaluations:
                submit_time = self.active_evaluations[config_id]['submit_time']
                elapsed = time.time() - submit_time
                return elapsed > self.timeout_seconds
        return False
    
    def _create_cache_key(self, config: Dict, fidelity: 'FidelityConfig') -> str:
        """创建缓存键"""
        config_str = str(sorted(config.items()))
        fidelity_str = f"{fidelity.name}_{fidelity.epochs}_{fidelity.dataset_fraction}"
        return f"{config_str}_{fidelity_str}"
    
    def get_completed_results(self) -> List[AsyncEvaluationResult]:
        """获取已完成的结果"""
        results = []
        while True:
            try:
                result = self.completed_results.get_nowait()
                results.append(result)
                
                # 从活跃评估中移除
                with self.lock:
                    if result.config_id in self.active_evaluations:
                        del self.active_evaluations[result.config_id]
            except queue.Empty:
                break
        
        return results
    
    def get_active_evaluations(self) -> Dict:
        """获取当前活跃的评估"""
        with self.lock:
            return {k: {
                'config': v['config'],
                'fidelity': v['fidelity'].name,
                'round_number': v['round_number'],
                'elapsed_time': time.time() - v['submit_time']
            } for k, v in self.active_evaluations.items()}
    
    def cancel_evaluation(self, config_id: str) -> bool:
        """取消评估"""
        with self.lock:
            if config_id in self.active_evaluations:
                future = self.active_evaluations[config_id]['future']
                if future.cancel():
                    del self.active_evaluations[config_id]
                    return True
        return False
    
    def get_worker_utilization(self) -> Dict:
        """获取工作器利用率统计"""
        with self.lock:
            stats = {}
            for worker_id, data in self.worker_stats.items():
                if data['jobs_completed'] > 0:
                    avg_time = data['total_time'] / data['jobs_completed']
                    stats[f'Worker-{worker_id}'] = {
                        'jobs_completed': data['jobs_completed'],
                        'total_time': data['total_time'],
                        'average_time': avg_time,
                        'efficiency': data['jobs_completed'] / max(data['total_time'], 0.1)
                    }
            return stats
    
    def shutdown(self):
        """关闭工作器池"""
        self.worker_pool.shutdown(wait=True)

# 扩展的保真度配置
@dataclass 
class EnhancedFidelityConfig:
    """增强的保真度配置"""
    name: str
    epochs: int
    dataset_fraction: float
    batch_size_multiplier: float = 1.0
    early_stopping_patience: int = 5
    max_batches_per_epoch: int = 50
    
    @property
    def relative_cost(self) -> float:
        """相对计算成本"""
        return self.epochs * self.dataset_fraction * (1 / self.batch_size_multiplier)
    
    @property
    def estimated_time_minutes(self) -> float:
        """估计运行时间（分钟）"""
        base_time = 0.5  # 基础时间
        return base_time * self.relative_cost

# 增强的数据加载器工厂
def enhanced_cifar10_factory(batch_size=64, dataset_fraction=1.0):
    """创建增强的CIFAR-10数据加载器"""
    import torchvision
    import torchvision.transforms as transforms
    
    # 更丰富的数据增强
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # 加载数据集
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=train_transform)
    valset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=val_transform)
    
    # 根据保真度选择子集
    if dataset_fraction < 1.0:
        train_size = int(len(trainset) * dataset_fraction)
        val_size = int(len(valset) * dataset_fraction)
        
        train_indices = random.sample(range(len(trainset)), train_size)
        val_indices = random.sample(range(len(valset)), val_size)
        
        train_subset = Subset(trainset, train_indices)
        val_subset = Subset(valset, val_indices)
    else:
        train_subset = trainset
        val_subset = valset
    
    # 为演示限制数据集大小
    if len(train_subset) > 3000:
        train_subset = Subset(train_subset, range(3000))
    if len(val_subset) > 600:
        val_subset = Subset(val_subset, range(600))
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader

# 使用示例
def test_async_multi_fidelity_objective():
    """测试异步多保真度目标函数"""
    
    # 定义保真度配置
    fidelity_configs = [
        EnhancedFidelityConfig("Quick Test", epochs=2, dataset_fraction=0.1, 
                             batch_size_multiplier=2.0, early_stopping_patience=1, max_batches_per_epoch=10),
        EnhancedFidelityConfig("Medium Test", epochs=5, dataset_fraction=0.3, 
                             batch_size_multiplier=1.5, early_stopping_patience=2, max_batches_per_epoch=25),
        EnhancedFidelityConfig("Full Test", epochs=10, dataset_fraction=1.0, 
                             batch_size_multiplier=1.0, early_stopping_patience=3, max_batches_per_epoch=50)
    ]
    
    # 创建异步目标函数
    objective = AsyncMultiFidelityObjective(
        dataset_factory=enhanced_cifar10_factory,
        model_factory=simple_cnn_factory,
        max_workers=3,
        timeout_minutes=10
    )
    
    # 测试配置
    test_configs = [
        {'learning_rate': 0.001, 'batch_size': 64, 'filters': 32, 'dropout': 0.2, 'optimizer': 'adam'},
        {'learning_rate': 0.01, 'batch_size': 32, 'filters': 64, 'dropout': 0.3, 'optimizer': 'sgd'},
        {'learning_rate': 0.005, 'batch_size': 128, 'filters': 16, 'dropout': 0.1, 'optimizer': 'adam'}
    ]
    
    try:
        # 提交评估
        config_ids = []
        for i, config in enumerate(test_configs):
            for j, fidelity in enumerate(fidelity_configs):
                config_id = objective.submit_evaluation(config, fidelity, round_number=j)
                config_ids.append(config_id)
                print(f"Submitted config {i+1} with fidelity {fidelity.name}: {config_id[:8]}")
        
        # 等待结果
        print(f"\nWaiting for {len(config_ids)} evaluations to complete...")
        
        completed_count = 0
        while completed_count < len(config_ids):
            time.sleep(2)
            
            # 检查新完成的结果
            new_results = objective.get_completed_results()
            completed_count += len(new_results)
            
            for result in new_results:
                print(f"Completed {result.config_id[:8]}: "
                      f"Performance={result.performance:.4f}, "
                      f"Time={result.training_time:.1f}s, "
                      f"Fidelity={result.fidelity.name}")
            
            # 显示活跃评估状态
            active = objective.get_active_evaluations()
            if active:
                print(f"Active evaluations: {len(active)}")
                for config_id, info in list(active.items())[:3]:  # 显示前3个
                    print(f"  {config_id[:8]}: {info['fidelity']} (running {info['elapsed_time']:.1f}s)")
        
        # 显示工作器统计
        print(f"\nWorker Utilization:")
        worker_stats = objective.get_worker_utilization()
        for worker_id, stats in worker_stats.items():
            print(f"  {worker_id}: {stats['jobs_completed']} jobs, "
                  f"avg time: {stats['average_time']:.1f}s, "
                  f"efficiency: {stats['efficiency']:.2f}")
        
    finally:
        objective.shutdown()

if __name__ == "__main__":
    test_async_multi_fidelity_objective()
```

## 16.5.2. Asynchronous Scheduler 异步调度器

### Advanced Async Successive Halving Scheduler 高级异步连续减半调度器

The asynchronous successive halving scheduler coordinates multiple fidelity levels and worker resources simultaneously. 异步连续减半调度器同时协调多个保真度级别和工作器资源。Here's a comprehensive implementation: 以下是一个综合实现：

```python
from collections import defaultdict, deque
from typing import Set
import heapq
import json

class AsyncSuccessiveHalvingScheduler:
    """异步连续减半调度器"""
    
    def __init__(self,
                 param_space: Dict,
                 fidelity_configs: List[EnhancedFidelityConfig],
                 objective_function: AsyncMultiFidelityObjective,
                 initial_budget: int = 50,
                 elimination_rate: float = 0.5,
                 max_concurrent_per_fidelity: int = 10,
                 min_configs_for_promotion: int = 2,
                 random_seed: int = 42):
        
        self.param_space = param_space
        self.fidelity_configs = fidelity_configs
        self.objective_function = objective_function
        self.initial_budget = initial_budget
        self.elimination_rate = elimination_rate
        self.max_concurrent_per_fidelity = max_concurrent_per_fidelity
        self.min_configs_for_promotion = min_configs_for_promotion
        
        # 设置随机种子
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # 状态管理
        self.round_managers = {i: RoundManager(i, fidelity) 
                              for i, fidelity in enumerate(fidelity_configs)}
        self.global_results = []
        self.config_genealogy = {}  # 跟踪配置在不同保真度下的表现
        
        # 调度状态
        self.is_running = False
        self.start_time = None
        self.promotion_queue = deque()  # 等待提升到下一保真度的配置
        
        # 性能监控
        self.performance_tracker = PerformanceTracker()
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def run_async_successive_halving(self, max_runtime_hours: float = 2.0) -> Dict:
        """运行异步连续减半优化"""
        self.is_running = True
        self.start_time = time.time()
        max_runtime_seconds = max_runtime_hours * 3600
        
        self.logger.info(f"Starting Async Successive Halving with {self.initial_budget} initial configs")
        
        # 生成初始配置并提交到第一个保真度级别
        initial_configs = self._generate_initial_configurations()
        self._submit_initial_round(initial_configs)
        
        try:
            # 主调度循环
            while self.is_running and (time.time() - self.start_time) < max_runtime_seconds:
                # 处理完成的评估
                self._process_completed_evaluations()
                
                # 处理提升队列
                self._process_promotion_queue()
                
                # 检查是否可以进行轮次内淘汰
                self._process_round_eliminations()
                
                # 更新性能跟踪
                self._update_performance_tracking()
                
                # 检查终止条件
                if self._should_terminate():
                    break
                
                # 短暂休眠
                time.sleep(0.5)
        
        except KeyboardInterrupt:
            self.logger.info("Optimization interrupted by user")
        finally:
            self.is_running = False
            self._finalize_optimization()
        
        return self._generate_final_report()
    
    def _generate_initial_configurations(self) -> List[Dict]:
        """生成初始配置"""
        configs = []
        for _ in range(self.initial_budget):
            config = {}
            for param_name, param_values in self.param_space.items():
                if isinstance(param_values, list):
                    config[param_name] = random.choice(param_values)
                elif isinstance(param_values, tuple) and len(param_values) == 2:
                    min_val, max_val = param_values
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        config[param_name] = random.randint(min_val, max_val)
                    else:
                        config[param_name] = random.uniform(min_val, max_val)
            configs.append(config)
        return configs
    
    def _submit_initial_round(self, configs: List[Dict]):
        """提交初始轮次的配置"""
        round_manager = self.round_managers[0]
        fidelity = self.fidelity_configs[0]
        
        for config in configs:
            config_id = str(uuid.uuid4())
            
            # 创建配置谱系
            self.config_genealogy[config_id] = {
                'config': config,
                'fidelity_results': {},
                'promotion_history': [0],
                'elimination_round': None
            }
            
            # 提交评估
            eval_id = self.objective_function.submit_evaluation(config, fidelity, round_number=0)
            round_manager.add_pending_evaluation(config_id, eval_id, config)
        
        self.logger.info(f"Submitted {len(configs)} configurations to round 0")
    
    def _process_completed_evaluations(self):
        """处理完成的评估"""
        completed_results = self.objective_function.get_completed_results()
        
        for result in completed_results:
            self.global_results.append(result)
            
            # 找到对应的轮次管理器
            round_manager = self.round_managers[result.round_number]
            config_id = round_manager.find_config_by_eval_id(result.config_id)
            
            if config_id:
                # 更新配置谱系
                if config_id in self.config_genealogy:
                    self.config_genealogy[config_id]['fidelity_results'][result.round_number] = {
                        'performance': result.performance,
                        'training_time': result.training_time,
                        'efficiency_score': result.efficiency_score
                    }
                
                # 添加到轮次管理器
                round_manager.add_completed_result(config_id, result)
                
                self.logger.info(f"Round {result.round_number}: Config {config_id[:8]} "
                               f"achieved {result.performance:.4f}")
    
    def _process_promotion_queue(self):
        """处理提升队列"""
        while self.promotion_queue and self._can_submit_more_evaluations():
            config_id, target_round = self.promotion_queue.popleft()
            
            if target_round < len(self.fidelity_configs):
                self._promote_configuration(config_id, target_round)
    
    def _promote_configuration(self, config_id: str, target_round: int):
        """将配置提升到更高保真度"""
        if config_id not in self.config_genealogy:
            return
        
        config = self.config_genealogy[config_id]['config']
        fidelity = self.fidelity_configs[target_round]
        
        # 提交到新的保真度级别
        eval_id = self.objective_function.submit_evaluation(config, fidelity, target_round)
        
        # 更新轮次管理器
        round_manager = self.round_managers[target_round]
        round_manager.add_pending_evaluation(config_id, eval_id, config)
        
        # 更新谱系
        self.config_genealogy[config_id]['promotion_history'].append(target_round)
        
        self.logger.info(f"Promoted config {config_id[:8]} to round {target_round}")
    
    def _process_round_eliminations(self):
        """处理轮次内的淘汰"""
        for round_num, round_manager in self.round_managers.items():
            if round_manager.can_perform_elimination(self.min_configs_for_promotion):
                eliminated, promoted = round_manager.perform_elimination(self.elimination_rate)
                
                # 记录被淘汰的配置
                for config_id in eliminated:
                    if config_id in self.config_genealogy:
                        self.config_genealogy[config_id]['elimination_round'] = round_num
                
                # 将晋级的配置加入提升队列
                if round_num < len(self.fidelity_configs) - 1:
                    for config_id in promoted:
                        self.promotion_queue.append((config_id, round_num + 1))
                
                self.logger.info(f"Round {round_num}: Eliminated {len(eliminated)}, "
                               f"promoted {len(promoted)} configurations")
    
    def _can_submit_more_evaluations(self) -> bool:
        """检查是否可以提交更多评估"""
        active_evaluations = self.objective_function.get_active_evaluations()
        total_active = len(active_evaluations)
        
        # 检查每个保真度级别的并发限制
        active_by_round = defaultdict(int)
        for eval_info in active_evaluations.values():
            round_num = eval_info['round_number']
            active_by_round[round_num] += 1
        
        # 查看是否有任何保真度级别未达到限制
        for round_num in range(len(self.fidelity_configs)):
            if active_by_round[round_num] < self.max_concurrent_per_fidelity:
                return True
        
        return False
    
    def _should_terminate(self) -> bool:
        """检查是否应该终止优化"""
        # 检查是否所有轮次都已完成
        all_rounds_complete = True
        for round_manager in self.round_managers.values():
            if not round_manager.is_complete():
                all_rounds_complete = False
                break
        
        # 检查是否没有活跃的评估和等待的提升
        no_active_work = (len(self.objective_function.get_active_evaluations()) == 0 and 
                         len(self.promotion_queue) == 0)
        
        return all_rounds_complete and no_active_work
    
    def _update_performance_tracking(self):
        """更新性能跟踪"""
        self.performance_tracker.update(self.global_results, self.config_genealogy)
    
    def _finalize_optimization(self):
        """完成优化"""
        # 等待剩余的评估完成
        self.logger.info("Finalizing optimization...")
        
        # 取消所有待处理的评估
        active_evaluations = self.objective_function.get_active_evaluations()
        for eval_id in active_evaluations.keys():
            self.objective_function.cancel_evaluation(eval_id)
        
        # 收集最终结果
        final_results = self.objective_function.get_completed_results()
        self.global_results.extend(final_results)
    
    def _generate_final_report(self) -> Dict:
        """生成最终报告"""
        if not self.global_results:
            return {'error': 'No results available'}
        
        # 找到最佳配置
        best_result = max(self.global_results, key=lambda x: x.performance)
        
        # 计算统计信息
        total_evaluations = len(self.global_results)
        total_time = time.time() - self.start_time if self.start_time else 0
        
        # 按保真度级别统计
        results_by_fidelity = defaultdict(list)
        for result in self.global_results:
            results_by_fidelity[result.round_number].append(result)
        
        fidelity_stats = {}
        for round_num, results in results_by_fidelity.items():
            fidelity_name = self.fidelity_configs[round_num].name
            performances = [r.performance for r in results]
            fidelity_stats[fidelity_name] = {
                'evaluations': len(results),
                'best_performance': max(performances),
                'mean_performance': np.mean(performances),
                'std_performance': np.std(performances)
            }
        
        # 生成存活分析
        survival_analysis = self._analyze_configuration_survival()
        
        report = {
            'optimization_summary': {
                'total_evaluations': total_evaluations,
                'total_time_seconds': total_time,
                'best_performance': best_result.performance,
                'best_config': best_result.config,
                'best_fidelity': best_result.fidelity.name
            },
            'fidelity_statistics': fidelity_stats,
            'survival_analysis': survival_analysis,
            'worker_utilization': self.objective_function.get_worker_utilization(),
            'performance_progression': self.performance_tracker.get_progression_data()
        }
        
        self.logger.info("=== ASYNC SUCCESSIVE HALVING COMPLETE ===")
        self.logger.info(f"Best performance: {best_result.performance:.4f}")
        self.logger.info(f"Total evaluations: {total_evaluations}")
        self.logger.info(f"Total time: {total_time/60:.1f} minutes")
        
        return report
    
    def _analyze_configuration_survival(self) -> Dict:
        """分析配置存活情况"""
        survival_stats = {
            'total_configs': len(self.config_genealogy),
            'survivors_by_round': defaultdict(int),
            'elimination_by_round': defaultdict(int),
            'avg_fidelities_tested': 0,
            'best_survivors': []
        }
        
        total_fidelities = 0
        for config_id, genealogy in self.config_genealogy.items():
            fidelities_tested = len(genealogy['fidelity_results'])
            total_fidelities += fidelities_tested
            
            # 统计存活到各轮次的配置数
            max_round = max(genealogy['promotion_history'])
            survival_stats['survivors_by_round'][max_round] += 1
            
            # 统计在各轮次被淘汰的配置数
            if genealogy['elimination_round'] is not None:
                survival_stats['elimination_by_round'][genealogy['elimination_round']] += 1
        
        if len(self.config_genealogy) > 0:
            survival_stats['avg_fidelities_tested'] = total_fidelities / len(self.config_genealogy)
        
        # 找到最佳存活配置
        final_round = len(self.fidelity_configs) - 1
        final_survivors = []
        
        for config_id, genealogy in self.config_genealogy.items():
            if final_round in genealogy['fidelity_results']:
                final_perf = genealogy['fidelity_results'][final_round]['performance']
                final_survivors.append((config_id, genealogy['config'], final_perf))
        
        final_survivors.sort(key=lambda x: x[2], reverse=True)
        survival_stats['best_survivors'] = final_survivors[:5]  # Top 5
        
        return survival_stats

class RoundManager:
    """轮次管理器"""
    
    def __init__(self, round_number: int, fidelity: EnhancedFidelityConfig):
        self.round_number = round_number
        self.fidelity = fidelity
        self.pending_evaluations = {}  # eval_id -> config_id
        self.config_to_eval = {}       # config_id -> eval_id
        self.completed_results = {}    # config_id -> result
        self.eliminated_configs = set()
        
    def add_pending_evaluation(self, config_id: str, eval_id: str, config: Dict):
        """添加待处理的评估"""
        self.pending_evaluations[eval_id] = config_id
        self.config_to_eval[config_id] = eval_id
    
    def add_completed_result(self, config_id: str, result: AsyncEvaluationResult):
        """添加完成的结果"""
        if config_id in self.config_to_eval:
            eval_id = self.config_to_eval[config_id]
            if eval_id in self.pending_evaluations:
                del self.pending_evaluations[eval_id]
        
        self.completed_results[config_id] = result
    
    def find_config_by_eval_id(self, eval_id: str) -> Optional[str]:
        """通过评估ID查找配置ID"""
        return self.pending_evaluations.get(eval_id)
    
    def can_perform_elimination(self, min_configs: int) -> bool:
        """检查是否可以执行淘汰"""
        active_configs = len(self.completed_results)
        return active_configs >= min_configs and len(self.pending_evaluations) == 0
    
    def perform_elimination(self, elimination_rate: float) -> Tuple[List[str], List[str]]:
        """执行淘汰并返回被淘汰和晋级的配置"""
        if not self.completed_results:
            return [], []
        
        # 按性能排序
        sorted_results = sorted(
            self.completed_results.items(),
            key=lambda x: x[1].performance,
            reverse=True
        )
        
        # 计算存活数量
        total_configs = len(sorted_results)
        num_survivors = max(1, int(total_configs * elimination_rate))
        
        # 分离存活和淘汰的配置
        survivors = [config_id for config_id, _ in sorted_results[:num_survivors]]
        eliminated = [config_id for config_id, _ in sorted_results[num_survivors:]]
        
        # 更新淘汰列表
        self.eliminated_configs.update(eliminated)
        
        return eliminated, survivors
    
    def is_complete(self) -> bool:
        """检查轮次是否完成"""
        return len(self.pending_evaluations) == 0
    
    def get_status(self) -> Dict:
        """获取轮次状态"""
        return {
            'round_number': self.round_number,
            'fidelity_name': self.fidelity.name,
            'pending_evaluations': len(self.pending_evaluations),
            'completed_results': len(self.completed_results),
            'eliminated_configs': len(self.eliminated_configs)
        }

class PerformanceTracker:
    """性能跟踪器"""
    
    def __init__(self):
        self.progression_data = []
        self.best_performance_history = []
        self.efficiency_history = []
        
    def update(self, global_results: List[AsyncEvaluationResult], 
              config_genealogy: Dict):
        """更新性能跟踪数据"""
        if not global_results:
            return
        
        # 更新最佳性能历史
        current_best = max(result.performance for result in global_results)
        timestamp = time.time()
        
        if not self.best_performance_history or current_best > self.best_performance_history[-1]['performance']:
            self.best_performance_history.append({
                'timestamp': timestamp,
                'performance': current_best,
                'evaluation_count': len(global_results)
            })
        
        # 计算效率指标
        total_time = sum(result.training_time for result in global_results)
        efficiency = current_best / max(total_time, 0.1)
        
        self.efficiency_history.append({
            'timestamp': timestamp,
            'efficiency': efficiency,
            'total_evaluations': len(global_results)
        })
    
    def get_progression_data(self) -> Dict:
        """获取进展数据"""
        return {
            'best_performance_history': self.best_performance_history,
            'efficiency_history': self.efficiency_history
        }

# 使用示例
def run_async_successive_halving_example():
    """运行异步连续减半示例"""
    
    # 定义保真度配置
    fidelity_configs = [
        EnhancedFidelityConfig("Quick", epochs=2, dataset_fraction=0.1, 
                             batch_size_multiplier=2.0, max_batches_per_epoch=5),
        EnhancedFidelityConfig("Medium", epochs=5, dataset_fraction=0.3, 
                             batch_size_multiplier=1.5, max_batches_per_epoch=15),
        EnhancedFidelityConfig("Full", epochs=8, dataset_fraction=1.0, 
                             batch_size_multiplier=1.0, max_batches_per_epoch=30)
    ]
    
    # 参数空间
    param_space = {
        'learning_rate': [0.0001, 0.001, 0.01, 0.1],
        'batch_size': [32, 64, 128],
        'filters': [16, 32, 64],
        'dropout': (0.0, 0.5),
        'optimizer': ['adam', 'sgd']
    }
    
    # 创建异步目标函数
    objective = AsyncMultiFidelityObjective(
        dataset_factory=enhanced_cifar10_factory,
        model_factory=simple_cnn_factory,
        max_workers=4,
        timeout_minutes=15
    )
    
    # 创建调度器
    scheduler = AsyncSuccessiveHalvingScheduler(
        param_space=param_space,
        fidelity_configs=fidelity_configs,
        objective_function=objective,
        initial_budget=12,  # 较小的预算用于演示
        elimination_rate=0.5,
        max_concurrent_per_fidelity=6,
        min_configs_for_promotion=3
    )
    
    try:
        # 运行优化
        print("Starting Asynchronous Successive Halving...")
        final_report = scheduler.run_async_successive_halving(max_runtime_hours=0.25)
        
        # 显示结果
        print("\n" + "="*80)
        print("FINAL OPTIMIZATION REPORT")
        print("="*80)
        
        opt_summary = final_report['optimization_summary']
        print(f"Best Performance: {opt_summary['best_performance']:.4f}")
        print(f"Best Configuration: {opt_summary['best_config']}")
        print(f"Total Evaluations: {opt_summary['total_evaluations']}")
        print(f"Total Time: {opt_summary['total_time_seconds']/60:.1f} minutes")
        
        print(f"\nFidelity Statistics:")
        for fidelity_name, stats in final_report['fidelity_statistics'].items():
            print(f"  {fidelity_name}: {stats['evaluations']} evals, "
                  f"best: {stats['best_performance']:.4f}, "
                  f"mean: {stats['mean_performance']:.4f}")
        
        survival = final_report['survival_analysis']
        print(f"\nSurvival Analysis:")
        print(f"  Total configurations: {survival['total_configs']}")
        print(f"  Average fidelities tested: {survival['avg_fidelities_tested']:.1f}")
        
        print(f"\nTop survivors:")
        for i, (config_id, config, perf) in enumerate(survival['best_survivors'][:3]):
            print(f"  {i+1}. {config_id[:8]}: {perf:.4f} - {config}")
        
    finally:
        objective.shutdown()

if __name__ == "__main__":
    run_async_successive_halving_example()
```

## 16.5.3. Visualize the Optimization Process 可视化优化过程

### Real-Time Async Multi-Fidelity Visualization 实时异步多保真度可视化

Visualizing asynchronous successive halving requires showing multiple dimensions: time, fidelity levels, configuration survival, and resource utilization. 可视化异步连续减半需要显示多个维度：时间、保真度级别、配置存活和资源利用率。Here's a comprehensive visualization system: 以下是一个综合可视化系统：

```python
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
import pandas as pd
from datetime import datetime
import threading
import time

class AsyncSHVisualizer:
    """异步连续减半可视化器"""
    
    def __init__(self, scheduler: AsyncSuccessiveHalvingScheduler, update_interval: float = 2.0):
        self.scheduler = scheduler
        self.update_interval = update_interval
        self.monitoring_thread = None
        self.is_monitoring = False
        
        # 数据收集
        self.timeline_data = []
        self.fidelity_progression = {i: [] for i in range(len(scheduler.fidelity_configs))}
        self.resource_utilization = []
        self.elimination_events = []
        
        # 可视化状态
        self.fig = None
        self.axes = None
        
        # 设置样式
        plt.style.use('default')
        sns.set_palette("husl")
    
    def start_monitoring(self):
        """开始实时监控"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=3)
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                self._collect_monitoring_data()
                time.sleep(self.update_interval)
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                break
    
    def _collect_monitoring_data(self):
        """收集监控数据"""
        current_time = time.time()
        
        # 收集时间线数据
        active_evaluations = self.scheduler.objective_function.get_active_evaluations()
        completed_results = self.scheduler.global_results
        
        self.timeline_data.append({
            'timestamp': current_time,
            'active_evaluations': len(active_evaluations),
            'completed_evaluations': len(completed_results),
            'best_performance': max([r.performance for r in completed_results]) if completed_results else 0
        })
        
        # 收集保真度级别进展
        for round_num, round_manager in self.scheduler.round_managers.items():
            status = round_manager.get_status()
            self.fidelity_progression[round_num].append({
                'timestamp': current_time,
                'pending': status['pending_evaluations'],
                'completed': status['completed_results'],
                'eliminated': status['eliminated_configs']
            })
        
        # 收集资源利用率
        worker_stats = self.scheduler.objective_function.get_worker_utilization()
        total_workers = self.scheduler.objective_function.max_workers
        active_workers = len([w for w in worker_stats.values() if w['jobs_completed'] > 0])
        
        self.resource_utilization.append({
            'timestamp': current_time,
            'utilization_rate': active_workers / total_workers if total_workers > 0 else 0,
            'active_workers': active_workers,
            'total_workers': total_workers
        })
    
    def create_live_dashboard(self):
        """创建实时仪表板"""
        self.fig, self.axes = plt.subplots(3, 3, figsize=(20, 15))
        self.fig.suptitle('Asynchronous Successive Halving Live Dashboard', 
                         fontsize=16, fontweight='bold')
        
        # 启动动画
        self.animation = animation.FuncAnimation(
            self.fig, self._update_dashboard, interval=2000, blit=False
        )
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.show()
    
    def _update_dashboard(self, frame):
        """更新仪表板"""
        # 清除所有子图
        for ax_row in self.axes:
            for ax in ax_row:
                ax.clear()
        
        if not self.timeline_data:
            return
        
        self._plot_convergence_timeline()
        self._plot_fidelity_progression()
        self._plot_resource_utilization()
        self._plot_configuration_survival()
        self._plot_worker_performance()
        self._plot_efficiency_metrics()
        self._plot_elimination_timeline()
        self._plot_current_status()
        self._plot_performance_distribution()
    
    def _plot_convergence_timeline(self):
        """绘制收敛时间线"""
        ax = self.axes[0, 0]
        
        if len(self.timeline_data) > 1:
            data = self.timeline_data
            timestamps = [d['timestamp'] for d in data]
            best_performances = [d['best_performance'] for d in data]
            
            # 转换为相对时间（分钟）
            start_time = timestamps[0]
            relative_times = [(t - start_time) / 60 for t in timestamps]
            
            ax.plot(relative_times, best_performances, 'b-', linewidth=2, marker='o', markersize=4)
            ax.fill_between(relative_times, best_performances, alpha=0.3)
        
        ax.set_title('Convergence Timeline 收敛时间线', fontweight='bold')
        ax.set_xlabel('Time (minutes) 时间（分钟）')
        ax.set_ylabel('Best Performance 最佳性能')
        ax.grid(True, alpha=0.3)
    
    def _plot_fidelity_progression(self):
        """绘制保真度级别进展"""
        ax = self.axes[0, 1]
        
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
        fidelity_names = [f.name for f in self.scheduler.fidelity_configs]
        
        if self.fidelity_progression[0]:  # 如果有数据
            current_data = []
            for round_num, progression in self.fidelity_progression.items():
                if progression:
                    latest = progression[-1]
                    current_data.append({
                        'fidelity': fidelity_names[round_num],
                        'completed': latest['completed'],
                        'pending': latest['pending'],
                        'eliminated': latest['eliminated']
                    })
            
            if current_data:
                df = pd.DataFrame(current_data)
                x_pos = range(len(df))
                
                # 堆叠条形图
                ax.bar(x_pos, df['completed'], label='Completed', color=colors[0], alpha=0.8)
                ax.bar(x_pos, df['pending'], bottom=df['completed'], 
                      label='Pending', color=colors[1], alpha=0.8)
                ax.bar(x_pos, df['eliminated'], 
                      bottom=df['completed'] + df['pending'],
                      label='Eliminated', color=colors[2], alpha=0.8)
                
                ax.set_xticks(x_pos)
                ax.set_xticklabels(df['fidelity'], rotation=45, ha='right')
        
        ax.set_title('Fidelity Level Progression 保真度级别进展', fontweight='bold')
        ax.set_ylabel('Number of Configurations 配置数量')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_resource_utilization(self):
        """绘制资源利用率"""
        ax = self.axes[0, 2]
        
        if len(self.resource_utilization) > 1:
            data = self.resource_utilization
            timestamps = [d['timestamp'] for d in data]
            utilizations = [d['utilization_rate'] * 100 for d in data]
            
            start_time = timestamps[0]
            relative_times = [(t - start_time) / 60 for t in timestamps]
            
            ax.plot(relative_times, utilizations, 'g-', linewidth=2)
            ax.fill_between(relative_times, utilizations, alpha=0.3, color='green')
            
            # 添加平均线
            if utilizations:
                avg_util = np.mean(utilizations)
                ax.axhline(y=avg_util, color='red', linestyle='--', 
                          label=f'Average: {avg_util:.1f}%')
        
        ax.set_title('Resource Utilization 资源利用率', fontweight='bold')
        ax.set_xlabel('Time (minutes) 时间（分钟）')
        ax.set_ylabel('Utilization (%) 利用率（%）')
        ax.set_ylim(0, 100)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_configuration_survival(self):
        """绘制配置存活情况"""
        ax = self.axes[1, 0]
        
        if self.scheduler.config_genealogy:
            survival_by_round = defaultdict(int)
            
            for config_id, genealogy in self.scheduler.config_genealogy.items():
                max_round = max(genealogy['promotion_history'])
                survival_by_round[max_round] += 1
            
            rounds = sorted(survival_by_round.keys())
            counts = [survival_by_round[r] for r in rounds]
            fidelity_names = [self.scheduler.fidelity_configs[r].name for r in rounds]
            
            bars = ax.bar(range(len(rounds)), counts, color='skyblue', alpha=0.7)
            ax.set_xticks(range(len(rounds)))
            ax.set_xticklabels(fidelity_names, rotation=45, ha='right')
            
            # 添加数值标签
            for bar, count in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       str(count), ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('Configuration Survival by Round 各轮次配置存活情况', fontweight='bold')
        ax.set_ylabel('Number of Configurations 配置数量')
        ax.grid(True, alpha=0.3)
    
    def _plot_worker_performance(self):
        """绘制工作器性能"""
        ax = self.axes[1, 1]
        
        worker_stats = self.scheduler.objective_function.get_worker_utilization()
        
        if worker_stats:
            worker_ids = list(worker_stats.keys())
            efficiencies = [stats['efficiency'] for stats in worker_stats.values()]
            job_counts = [stats['jobs_completed'] for stats in worker_stats.values()]
            
            # 创建散点图，点的大小表示完成的任务数
            scatter = ax.scatter(range(len(worker_ids)), efficiencies, 
                               s=[count * 50 for count in job_counts], 
                               alpha=0.7, c=efficiencies, cmap='viridis')
            
            ax.set_xticks(range(len(worker_ids)))
            ax.set_xticklabels([w.replace('Worker-', 'W') for w in worker_ids], rotation=45)
            
            # 添加颜色条
            plt.colorbar(scatter, ax=ax, label='Efficiency')
        
        ax.set_title('Worker Performance 工作器性能', fontweight='bold')
        ax.set_ylabel('Efficiency (jobs/time) 效率')
        ax.grid(True, alpha=0.3)
    
    def _plot_efficiency_metrics(self):
        """绘制效率指标"""
        ax = self.axes[1, 2]
        
        if self.scheduler.global_results:
            # 计算累积效率
            results = self.scheduler.global_results
            cumulative_performance = []
            cumulative_time = []
            
            best_so_far = 0
            total_time = 0
            
            for result in results:
                best_so_far = max(best_so_far, result.performance)
                total_time += result.training_time
                
                cumulative_performance.append(best_so_far)
                cumulative_time.append(total_time / 60)  # 转换为分钟
            
            if cumulative_time:
                ax.plot(cumulative_time, cumulative_performance, 'purple', linewidth=2, marker='o', markersize=3)
        
        ax.set_title('Efficiency: Performance vs Time 效率：性能vs时间', fontweight='bold')
        ax.set_xlabel('Cumulative Time (minutes) 累积时间（分钟）')
        ax.set_ylabel('Best Performance 最佳性能')
        ax.grid(True, alpha=0.3)
    
    def _plot_elimination_timeline(self):
        """绘制淘汰时间线"""
        ax = self.axes[2, 0]
        
        if self.scheduler.config_genealogy:
            elimination_data = []
            for config_id, genealogy in self.scheduler.config_genealogy.items():
                if genealogy['elimination_round'] is not None:
                    elimination_data.append(genealogy['elimination_round'])
            
            if elimination_data:
                rounds = range(len(self.scheduler.fidelity_configs))
                elimination_counts = [elimination_data.count(r) for r in rounds]
                fidelity_names = [f.name for f in self.scheduler.fidelity_configs]
                
                bars = ax.bar(rounds, elimination_counts, color='red', alpha=0.6)
                ax.set_xticks(rounds)
                ax.set_xticklabels(fidelity_names, rotation=45, ha='right')
                
                # 添加数值标签
                for bar, count in zip(bars, elimination_counts):
                    if count > 0:
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                               str(count), ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('Elimination Timeline 淘汰时间线', fontweight='bold')
        ax.set_ylabel('Configurations Eliminated 被淘汰的配置')
        ax.grid(True, alpha=0.3)
    
    def _plot_current_status(self):
        """绘制当前状态"""
        ax = self.axes[2, 1]
        
        # 统计当前状态
        active_evals = self.scheduler.objective_function.get_active_evaluations()
        completed_count = len(self.scheduler.global_results)
        total_configs = len(self.scheduler.config_genealogy)
        
        if self.timeline_data:
            latest = self.timeline_data[-1]
            current_active = latest['active_evaluations']
        else:
            current_active = 0
        
        # 饼图显示状态分布
        labels = ['Completed', 'Active', 'Pending']
        sizes = [completed_count, current_active, 
                max(0, total_configs - completed_count - current_active)]
        colors = ['lightgreen', 'orange', 'lightgray']
        
        # 只显示非零的部分
        non_zero_data = [(label, size, color) for label, size, color in zip(labels, sizes, colors) if size > 0]
        if non_zero_data:
            labels, sizes, colors = zip(*non_zero_data)
            
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, 
                                             autopct='%1.1f%%', startangle=90)
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        
        ax.set_title('Current Status 当前状态', fontweight='bold')
    
    def _plot_performance_distribution(self):
        """绘制性能分布"""
        ax = self.axes[2, 2]
        
        if self.scheduler.global_results:
            performances = [result.performance for result in self.scheduler.global_results]
            
            # 按保真度级别分组
            perf_by_fidelity = defaultdict(list)
            for result in self.scheduler.global_results:
                perf_by_fidelity[result.round_number].append(result.performance)
            
            # 箱线图
            data_to_plot = []
            labels = []
            for round_num in sorted(perf_by_fidelity.keys()):
                data_to_plot.append(perf_by_fidelity[round_num])
                labels.append(self.scheduler.fidelity_configs[round_num].name)
            
            if data_to_plot:
                bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
                
                # 美化箱线图
                colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
                for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
        
        ax.set_title('Performance Distribution by Fidelity 各保真度性能分布', fontweight='bold')
        ax.set_ylabel('Performance 性能')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    def generate_final_interactive_report(self):
        """生成最终交互式报告"""
        if not self.scheduler.global_results:
            print("No results available for visualization")
            return
        
        # 创建综合的交互式报告
        fig = make_subplots(
            rows=4, cols=3,
            subplot_titles=(
                'Performance Evolution', 'Fidelity Progression', 'Resource Timeline',
                'Configuration Survival', 'Worker Efficiency', 'Elimination Pattern',
                'Performance vs Training Time', 'Hyperparameter Sensitivity', 'Cost-Benefit Analysis',
                'Final Performance Distribution', 'Survival Genealogy', 'Optimization Summary'
            ),
            specs=[[{"type": "scatter"}, {"type": "bar"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "scatter"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "bar"}, {"type": "scatter"}],
                   [{"type": "histogram"}, {"type": "scatter"}, {"type": "table"}]]
        )
        
        self._add_interactive_plots(fig)
        
        fig.update_layout(
            title="Asynchronous Successive Halving - Comprehensive Analysis Report",
            height=1600,
            showlegend=True,
            template="plotly_white"
        )
        
        fig.show()
    
    def _add_interactive_plots(self, fig):
        """添加交互式图表到报告"""
        results = self.scheduler.global_results
        
        # 1. 性能演化
        performances = [result.performance for result in results]
        times = [result.end_time for result in results]
        
        fig.add_trace(
            go.Scatter(x=list(range(len(performances))), y=performances,
                      mode='lines+markers', name='Performance Evolution',
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        # 2. 保真度进展
        fidelity_counts = defaultdict(int)
        for result in results:
            fidelity_counts[result.round_number] += 1
        
        fidelity_names = [self.scheduler.fidelity_configs[i].name for i in sorted(fidelity_counts.keys())]
        counts = [fidelity_counts[i] for i in sorted(fidelity_counts.keys())]
        
        fig.add_trace(
            go.Bar(x=fidelity_names, y=counts, name='Evaluations per Fidelity'),
            row=1, col=2
        )
        
        # 3. 资源时间线
        if self.resource_utilization:
            timestamps = [d['timestamp'] for d in self.resource_utilization]
            utilizations = [d['utilization_rate'] * 100 for d in self.resource_utilization]
            
            fig.add_trace(
                go.Scatter(x=timestamps, y=utilizations,
                          mode='lines', name='Resource Utilization',
                          line=dict(color='green', width=2)),
                row=1, col=3
            )
        
        # 添加更多图表...
        # (为了简洁，这里只显示前几个图表的示例)
        
        # 最终性能分布
        fig.add_trace(
            go.Histogram(x=performances, name='Performance Distribution'),
            row=4, col=1
        )

# 使用示例
def run_visualization_demo():
    """运行可视化演示"""
    
    # 创建调度器（使用之前的设置）
    fidelity_configs = [
        EnhancedFidelityConfig("Quick", epochs=1, dataset_fraction=0.1),
        EnhancedFidelityConfig("Medium", epochs=3, dataset_fraction=0.3),
        EnhancedFidelityConfig("Full", epochs=5, dataset_fraction=1.0)
    ]
    
    param_space = {
        'learning_rate': [0.001, 0.01, 0.1],
        'batch_size': [32, 64, 128],
        'filters': [16, 32, 64]
    }
    
    objective = AsyncMultiFidelityObjective(
        dataset_factory=enhanced_cifar10_factory,
        model_factory=simple_cnn_factory,
        max_workers=3
    )
    
    scheduler = AsyncSuccessiveHalvingScheduler(
        param_space=param_space,
        fidelity_configs=fidelity_configs,
        objective_function=objective,
        initial_budget=8
    )
    
    # 创建可视化器
    visualizer = AsyncSHVisualizer(scheduler, update_interval=1.0)
    
    try:
        # 启动监控
        visualizer.start_monitoring()
        
        # 启动优化（在后台线程）
        optimization_thread = threading.Thread(
            target=lambda: scheduler.run_async_successive_halving(max_runtime_hours=0.1),
            daemon=True
        )
        optimization_thread.start()
        
        # 创建实时仪表板
        visualizer.create_live_dashboard()
        
        # 等待优化完成
        optimization_thread.join()
        
        # 生成最终报告
        visualizer.generate_final_interactive_report()
        
    finally:
        visualizer.stop_monitoring()
        objective.shutdown()

if __name__ == "__main__":
    run_visualization_demo()
```

## 16.5.4. Summary 总结

### Asynchronous Successive Halving Overview 异步连续减半概述

Asynchronous Successive Halving represents the pinnacle of efficient hyperparameter optimization, combining the computational savings of multi-fidelity methods with the time efficiency of parallel execution. 异步连续减半代表了高效超参数优化的顶峰，结合了多保真度方法的计算节省和并行执行的时间效率。

### Key Innovations 关键创新

**1. Dual Efficiency 双重效率**
- **Computational Efficiency**: Progressive elimination reduces wasted computation on poor configurations 计算效率：渐进式淘汰减少在差配置上的计算浪费
- **Time Efficiency**: Asynchronous execution maximizes hardware utilization 时间效率：异步执行最大化硬件利用率

**2. Dynamic Resource Management 动态资源管理**
- **Adaptive Allocation**: Resources automatically shift to promising configurations 自适应分配：资源自动转向有前景的配置
- **Load Balancing**: Intelligent distribution across available workers 负载均衡：在可用工作器间智能分配

**3. Progressive Validation 渐进验证**
- **Risk Mitigation**: Multiple fidelity stages reduce overfitting to validation data 风险缓解：多保真度阶段减少对验证数据的过拟合
- **Early Detection**: Poor configurations identified and eliminated quickly 早期检测：快速识别和淘汰差配置

### Performance Characteristics 性能特征

```python
class ASHPerformanceMetrics:
    """异步连续减半性能指标"""
    
    @staticmethod
    def calculate_efficiency_gains(ash_results: Dict, baseline_results: Dict) -> Dict:
        """计算相对于基线的效率增益"""
        
        # 时间效率
        time_speedup = baseline_results['total_time'] / ash_results['total_time']
        
        # 计算效率
        baseline_evaluations = baseline_results['total_evaluations']
        ash_evaluations = ash_results['total_evaluations']
        computational_savings = 1 - (ash_evaluations / baseline_evaluations)
        
        # 性能比较
        performance_ratio = ash_results['best_performance'] / baseline_results['best_performance']
        
        # 资源利用率
        resource_efficiency = ash_results['average_worker_utilization']
        
        return {
            'time_speedup': time_speedup,
            'computational_savings': computational_savings,
            'performance_ratio': performance_ratio,
            'resource_efficiency': resource_efficiency,
            'overall_efficiency': time_speedup * (1 + computational_savings) * performance_ratio
        }
    
    @staticmethod
    def analyze_scaling_behavior(results_by_workers: Dict) -> Dict:
        """分析扩展行为"""
        scaling_analysis = {}
        
        for num_workers, results in results_by_workers.items():
            if num_workers == 1:
                baseline_time = results['total_time']
                continue
            
            speedup = baseline_time / results['total_time']
            efficiency = speedup / num_workers
            
            scaling_analysis[num_workers] = {
                'speedup': speedup,
                'efficiency': efficiency,
                'parallel_efficiency': min(1.0, efficiency)
            }
        
        return scaling_analysis
```

### Best Practices and Guidelines 最佳实践和指导原则

**1. Fidelity Design 保真度设计**
- **Correlation Validation**: Ensure strong correlation between fidelities 相关性验证：确保保真度之间有强相关性
- **Progressive Difficulty**: Each fidelity should be meaningfully more expensive 渐进难度：每个保真度都应该明显更昂贵
- **Early Stopping**: Implement aggressive early stopping at lower fidelities 早期停止：在较低保真度实施积极的早期停止

**2. Resource Allocation 资源分配**
- **Worker Scaling**: Scale workers based on evaluation time variance 工作器扩展：根据评估时间方差扩展工作器
- **Memory Management**: Consider memory requirements for concurrent evaluations 内存管理：考虑并发评估的内存需求
- **Load Balancing**: Monitor and redistribute work dynamically 负载均衡：动态监控和重新分配工作

**3. Elimination Strategy 淘汰策略**
- **Conservative Early Rounds**: Use higher survival rates in early fidelities 保守的早期轮次：在早期保真度使用更高的存活率
- **Aggressive Final Rounds**: Eliminate more aggressively at higher fidelities 积极的最终轮次：在较高保真度更积极地淘汰
- **Statistical Significance**: Consider confidence intervals for elimination decisions 统计显著性：考虑淘汰决定的置信区间

### When to Use Asynchronous Successive Halving 何时使用异步连续减半

**Ideal Scenarios 理想场景**:
- Large hyperparameter spaces (>20 configurations) 大型超参数空间（>20个配置）
- Variable training times across configurations 不同配置的可变训练时间
- Multiple computational resources available 有多个计算资源可用
- Clear fidelity hierarchy (epochs, data size, etc.) 明确的保真度层次结构（轮次、数据大小等）
- Time constraints are critical 时间约束很关键

**Limitations 局限性**:
- Requires strong correlation between fidelities 需要保真度间的强相关性
- Implementation complexity is high 实现复杂度高
- May not be optimal for very fast evaluations 对于非常快的评估可能不是最优的
- Requires careful resource management 需要仔细的资源管理

### Future Directions 未来方向

The field continues to evolve with exciting developments: 该领域继续随着令人兴奋的发展而演进：

**1. Adaptive Fidelities 自适应保真度**: Dynamic fidelity adjustment based on search progress 基于搜索进展的动态保真度调整

**2. Multi-Objective ASH 多目标ASH**: Extending to multiple competing objectives (accuracy, speed, memory) 扩展到多个竞争目标（准确性、速度、内存）

**3. Transfer Learning Integration 迁移学习集成**: Using knowledge from previous optimization runs 使用先前优化运行的知识

**4. Automated Resource Provisioning 自动资源配置**: Cloud-native implementations with elastic scaling 具有弹性扩展的云原生实现

Asynchronous Successive Halving represents a mature and powerful approach to hyperparameter optimization that effectively addresses the practical challenges of modern machine learning development. 异步连续减半代表了一种成熟而强大的超参数优化方法，有效解决了现代机器学习开发的实际挑战。Its combination of computational efficiency, time optimization, and practical scalability makes it an essential tool for practitioners working with complex models and limited computational budgets. 它结合了计算效率、时间优化和实际可扩展性，使其成为处理复杂模型和有限计算预算的从业者的必备工具。