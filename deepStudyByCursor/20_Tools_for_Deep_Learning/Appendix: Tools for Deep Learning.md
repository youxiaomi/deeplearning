# Chapter 20: Appendix: Tools for Deep Learning 
# 第20章：附录：深度学习工具

## Overview 概述

Deep learning has revolutionized the field of artificial intelligence, but mastering it requires not only understanding the theoretical concepts but also becoming proficient with the practical tools and platforms that make implementation possible. This appendix provides a comprehensive guide to the essential tools, platforms, and environments that every deep learning practitioner should know.

深度学习已经彻底改革了人工智能领域，但要掌握它不仅需要理解理论概念，还需要熟练掌握使实现成为可能的实用工具和平台。这个附录为每个深度学习从业者应该了解的基本工具、平台和环境提供了全面的指南。

In this chapter, we will explore various development environments, from local Jupyter notebooks to cloud-based solutions like Amazon SageMaker, AWS EC2, and Google Colab. We'll also discuss hardware considerations, collaboration tools, and utility functions that can significantly enhance your deep learning workflow.

在本章中，我们将探索各种开发环境，从本地Jupyter笔记本到基于云的解决方案，如Amazon SageMaker、AWS EC2和Google Colab。我们还将讨论硬件考虑因素、协作工具和可以显著增强深度学习工作流程的实用函数。

## 20.1 Using Jupyter Notebooks 使用Jupyter笔记本

Jupyter Notebooks have become the de facto standard for interactive data science and machine learning development. They provide an excellent environment for experimenting with code, visualizing data, and documenting your thought process all in one place.

Jupyter笔记本已成为交互式数据科学和机器学习开发的事实标准。它们为在一个地方试验代码、可视化数据和记录思维过程提供了优秀的环境。

### 20.1.1 Editing and Running the Code Locally 本地编辑和运行代码

#### Installation and Setup 安装和设置

To get started with Jupyter notebooks locally, you need to install the necessary software. The easiest way is through Anaconda, which comes with Jupyter pre-installed, or you can install it via pip.

要在本地开始使用Jupyter笔记本，你需要安装必要的软件。最简单的方法是通过Anaconda，它预装了Jupyter，或者你可以通过pip安装它。

**Method 1: Using Anaconda (Recommended) 方法1：使用Anaconda（推荐）**

```bash
# Download and install Anaconda from https://www.anaconda.com/
# 从 https://www.anaconda.com/ 下载并安装Anaconda

# Launch Jupyter Notebook
# 启动Jupyter笔记本
jupyter notebook
```

**Method 2: Using pip 方法2：使用pip**

```bash
# Install Jupyter
# 安装Jupyter
pip install jupyter

# Install additional packages for deep learning
# 安装深度学习的额外包
pip install torch torchvision matplotlib numpy pandas scikit-learn

# Launch Jupyter Notebook
# 启动Jupyter笔记本
jupyter notebook
```

#### Creating Your First Notebook 创建你的第一个笔记本

When you launch Jupyter, it opens in your web browser. You can create a new notebook by clicking "New" → "Python 3". Let's create a simple deep learning example:

当你启动Jupyter时，它在网页浏览器中打开。你可以通过点击"新建"→"Python 3"来创建新笔记本。让我们创建一个简单的深度学习示例：

```python
# Cell 1: Import necessary libraries
# 单元格1：导入必要的库
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
```

```python
# Cell 2: Create a simple neural network
# 单元格2：创建一个简单的神经网络
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Create model instance
# 创建模型实例
model = SimpleNet(input_size=10, hidden_size=20, output_size=1)
print(model)
```

#### Best Practices for Local Development 本地开发的最佳实践

1. **Organize your notebooks** 组织你的笔记本
   - Create separate folders for different projects
   - 为不同项目创建单独的文件夹
   - Use descriptive filenames with dates or version numbers
   - 使用带有日期或版本号的描述性文件名

2. **Use virtual environments** 使用虚拟环境
   - Keep dependencies isolated between projects
   - 保持项目间依赖关系的隔离
   ```bash
   # Create virtual environment
   # 创建虚拟环境
   conda create -n deeplearning python=3.8
   conda activate deeplearning
   ```

3. **Document your code** 记录你的代码
   - Use markdown cells to explain your thought process
   - 使用markdown单元格解释你的思维过程
   - Include comments in code cells
   - 在代码单元格中包含注释

### 20.1.2 Advanced Options 高级选项

#### Jupyter Extensions �jupyter扩展

Jupyter extensions can significantly enhance your productivity. Here are some essential ones:

Jupyter扩展可以显著提高你的生产力。以下是一些必要的扩展：

```bash
# Install Jupyter extensions
# 安装Jupyter扩展
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user

# Enable useful extensions
# 启用有用的扩展
jupyter nbextension enable --py widgetsnbextension
```

**Popular Extensions 热门扩展：**

1. **Variable Inspector** 变量检查器
   - Shows all variables in your namespace
   - 显示命名空间中的所有变量
   - Useful for debugging and understanding data structures
   - 对调试和理解数据结构很有用

2. **Table of Contents** 目录
   - Automatically generates a table of contents from markdown headers
   - 从markdown标题自动生成目录
   - Makes navigation easier in long notebooks
   - 使长笔记本中的导航更容易

3. **Code Folding** 代码折叠
   - Allows you to collapse code cells
   - 允许你折叠代码单元格
   - Helps with organization and focus
   - 有助于组织和专注

#### Magic Commands 魔法命令

Jupyter notebooks support magic commands that provide powerful functionality:

Jupyter笔记本支持提供强大功能的魔法命令：

```python
# Time execution of a cell
# 计时单元格的执行时间
%%time
import time
time.sleep(1)
print("This took about 1 second")

# Profile memory usage
# 分析内存使用情况
%load_ext memory_profiler
%memit torch.randn(1000, 1000)

# Load external Python file
# 加载外部Python文件
# %load external_script.py

# Show matplotlib plots inline
# 内联显示matplotlib图表
%matplotlib inline
```

#### GPU Configuration GPU配置

When working with deep learning models locally, GPU acceleration is crucial:

在本地处理深度学习模型时，GPU加速是至关重要的：

```python
# Check GPU availability and configure device
# 检查GPU可用性并配置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Move model to GPU
# 将模型移动到GPU
model = model.to(device)
```

### 20.1.3 Summary 总结

Jupyter notebooks provide an ideal environment for deep learning experimentation and development. They combine code execution, visualization, and documentation in a single interface, making them perfect for iterative development and research.

Jupyter笔记本为深度学习实验和开发提供了理想的环境。它们在单个界面中结合了代码执行、可视化和文档，使其非常适合迭代开发和研究。

Key advantages include:
主要优势包括：
- Interactive development and immediate feedback
- 交互式开发和即时反馈
- Rich output formatting (plots, tables, images)
- 丰富的输出格式（图表、表格、图像）
- Easy sharing and collaboration
- 易于分享和协作
- Extensive ecosystem of extensions
- 扩展的广泛生态系统

### 20.1.4 Exercises 练习

1. **Basic Setup Exercise** 基本设置练习
   - Install Jupyter notebook on your local machine
   - 在本地机器上安装Jupyter笔记本
   - Create a new notebook and verify PyTorch installation
   - 创建新笔记本并验证PyTorch安装
   - Implement a simple linear regression model
   - 实现一个简单的线性回归模型

2. **Extension Practice** 扩展练习
   - Install and configure at least 3 Jupyter extensions
   - 安装并配置至少3个Jupyter扩展
   - Use magic commands to profile a deep learning training loop
   - 使用魔法命令分析深度学习训练循环

3. **Documentation Challenge** 文档挑战
   - Create a well-documented notebook that explains a deep learning concept
   - 创建一个解释深度学习概念的详细文档笔记本
   - Include mathematical formulas, code examples, and visualizations
   - 包括数学公式、代码示例和可视化

## 20.2 Using Amazon SageMaker 使用Amazon SageMaker

Amazon SageMaker is a fully managed service that provides every developer and data scientist with the ability to build, train, and deploy machine learning models quickly. It eliminates the heavy lifting from each step of the machine learning process to make it easier to develop high-quality models.

Amazon SageMaker是一项完全托管的服务，为每个开发者和数据科学家提供快速构建、训练和部署机器学习模型的能力。它消除了机器学习过程每个步骤的繁重工作，使开发高质量模型变得更容易。

### 20.2.1 Signing Up 注册

To get started with Amazon SageMaker, you need an AWS account. Here's the step-by-step process:

要开始使用Amazon SageMaker，你需要一个AWS账户。以下是逐步过程：

#### Step 1: Create AWS Account 步骤1：创建AWS账户

1. Go to https://aws.amazon.com/
   访问 https://aws.amazon.com/
2. Click "Create an AWS Account"
   点击"创建AWS账户"
3. Provide your email, password, and AWS account name
   提供你的邮箱、密码和AWS账户名
4. Add payment information (required for verification)
   添加付款信息（验证所需）
5. Verify your identity via phone
   通过电话验证身份

#### Step 2: Access SageMaker 步骤2：访问SageMaker

```bash
# Once logged into AWS Console
# 登录AWS控制台后
# 1. Search for "SageMaker" in the AWS services search bar
# 1. 在AWS服务搜索栏中搜索"SageMaker"
# 2. Click on "Amazon SageMaker"
# 2. 点击"Amazon SageMaker"
# 3. You'll be taken to the SageMaker dashboard
# 3. 你将被带到SageMaker仪表板
```

#### Step 3: Understanding Pricing 步骤3：了解定价

SageMaker pricing is based on usage:
SageMaker定价基于使用情况：

- **Notebook instances**: Charged per hour of usage
- **笔记本实例**：按使用小时收费
- **Training**: Charged per second of training time
- **训练**：按训练时间秒计费
- **Inference**: Charged per hour for hosted endpoints
- **推理**：托管端点按小时收费

*Important*: Always stop instances when not in use to avoid unnecessary charges.
*重要提示*：不使用时务必停止实例以避免不必要的费用。 

### 20.2.2 Creating a SageMaker Instance 创建SageMaker实例

Once you have access to SageMaker, creating a notebook instance is straightforward. A notebook instance is a machine learning compute instance running the Jupyter Notebook App, where you can prepare and process data, write code to train models, deploy models, and test or validate your models.

一旦你可以访问SageMaker，创建笔记本实例就很简单了。笔记本实例是运行Jupyter笔记本应用程序的机器学习计算实例，你可以在其中准备和处理数据、编写代码来训练模型、部署模型以及测试或验证模型。

#### Step-by-Step Instance Creation 逐步创建实例

```bash
# Navigate to SageMaker Console
# 导航到SageMaker控制台
# 1. In AWS Console, go to SageMaker service
# 1. 在AWS控制台中，转到SageMaker服务
# 2. Click "Notebook instances" in the left sidebar
# 2. 在左侧边栏点击"笔记本实例"
# 3. Click "Create notebook instance"
# 3. 点击"创建笔记本实例"
```

#### Instance Configuration 实例配置

When creating a SageMaker notebook instance, you need to configure several important settings:

创建SageMaker笔记本实例时，你需要配置几个重要设置：

**1. Basic Settings 基本设置**

```python
# Instance configuration parameters
# 实例配置参数
instance_config = {
    "notebook_instance_name": "my-deep-learning-notebook",  # 实例名称
    "instance_type": "ml.t3.medium",  # 实例类型（CPU实例，适合开始）
    "role_arn": "arn:aws:iam::account:role/service-role/AmazonSageMaker-ExecutionRole"
}

# For GPU instances (more expensive but faster for training)
# GPU实例（更昂贵但训练更快）
gpu_instance_config = {
    "instance_type": "ml.p3.2xlarge",  # GPU实例类型
    "volume_size": 20  # 存储大小（GB）
}
```

**2. IAM Role Configuration IAM角色配置**

SageMaker needs permissions to access other AWS services. You can either create a new role or use an existing one:

SageMaker需要权限来访问其他AWS服务。你可以创建新角色或使用现有角色：

```python
# Creating a SageMaker execution role
# 创建SageMaker执行角色
import boto3

def create_sagemaker_role():
    """
    Create IAM role for SageMaker with necessary permissions
    为SageMaker创建具有必要权限的IAM角色
    """
    iam = boto3.client('iam')
    
    # Define trust policy
    # 定义信任策略
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "sagemaker.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }
        ]
    }
    
    # Create role
    # 创建角色
    role_name = "SageMakerExecutionRole"
    
    try:
        response = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=str(trust_policy),
            Description="Role for SageMaker notebook instances"
        )
        print(f"创建角色成功: {response['Role']['Arn']}")
        return response['Role']['Arn']
    except Exception as e:
        print(f"创建角色失败: {e}")
        return None
```

#### Instance Types and Pricing 实例类型和定价

Understanding different instance types helps you choose the right one for your needs:

了解不同的实例类型有助于你选择适合需求的实例：

```python
# SageMaker instance types comparison
# SageMaker实例类型比较
instance_types = {
    "ml.t3.medium": {
        "vcpu": 2,
        "memory_gb": 4,
        "gpu": 0,
        "price_per_hour": 0.0464,
        "use_case": "开发和测试"
    },
    "ml.m5.large": {
        "vcpu": 2,
        "memory_gb": 8,
        "gpu": 0,
        "price_per_hour": 0.096,
        "use_case": "中等计算需求"
    },
    "ml.p3.2xlarge": {
        "vcpu": 8,
        "memory_gb": 61,
        "gpu": 1,  # Tesla V100
        "price_per_hour": 3.06,
        "use_case": "GPU加速训练"
    },
    "ml.p4d.24xlarge": {
        "vcpu": 96,
        "memory_gb": 1152,
        "gpu": 8,  # A100
        "price_per_hour": 32.77,
        "use_case": "大规模深度学习"
    }
}

# Display comparison
# 显示比较
print("SageMaker实例类型比较:")
print("="*80)
print(f"{'类型':<15} {'vCPU':<5} {'内存(GB)':<8} {'GPU':<3} {'每小时价格($)':<12} {'用途'}")
print("-"*80)

for instance_type, specs in instance_types.items():
    print(f"{instance_type:<15} {specs['vcpu']:<5} {specs['memory_gb']:<8} "
          f"{specs['gpu']:<3} {specs['price_per_hour']:<12} {specs['use_case']}")
```

### 20.2.3 Running and Stopping an Instance 运行和停止实例

Managing your SageMaker instances properly is crucial for cost control and efficient workflow. Unlike EC2 instances, SageMaker notebook instances are designed to be started and stopped as needed.

正确管理SageMaker实例对于成本控制和高效工作流程至关重要。与EC2实例不同，SageMaker笔记本实例设计为根据需要启动和停止。

#### Starting an Instance 启动实例

```python
import boto3
import time

def start_notebook_instance(instance_name):
    """
    Start a SageMaker notebook instance
    启动SageMaker笔记本实例
    """
    sagemaker = boto3.client('sagemaker')
    
    try:
        # Check current status
        # 检查当前状态
        response = sagemaker.describe_notebook_instance(
            NotebookInstanceName=instance_name
        )
        current_status = response['NotebookInstanceStatus']
        
        if current_status == 'InService':
            print(f"实例 {instance_name} 已经在运行")
            return response['Url']
        elif current_status == 'Stopped':
            # Start the instance
            # 启动实例
            sagemaker.start_notebook_instance(
                NotebookInstanceName=instance_name
            )
            print(f"正在启动实例 {instance_name}...")
            
            # Wait for instance to be ready
            # 等待实例准备就绪
            while True:
                response = sagemaker.describe_notebook_instance(
                    NotebookInstanceName=instance_name
                )
                status = response['NotebookInstanceStatus']
                print(f"当前状态: {status}")
                
                if status == 'InService':
                    print("实例启动成功!")
                    return response['Url']
                elif status == 'Failed':
                    print("实例启动失败!")
                    return None
                
                time.sleep(30)  # Wait 30 seconds before checking again
                
    except Exception as e:
        print(f"启动实例时出错: {e}")
        return None

# Example usage
# 使用示例
# notebook_url = start_notebook_instance("my-deep-learning-notebook")
# if notebook_url:
#     print(f"笔记本URL: {notebook_url}")
```

#### Stopping an Instance 停止实例

**Important**: Always stop your instances when you're not using them to avoid unnecessary charges.

**重要提示**：不使用时务必停止实例，以避免不必要的费用。

```python
def stop_notebook_instance(instance_name):
    """
    Stop a SageMaker notebook instance
    停止SageMaker笔记本实例
    """
    sagemaker = boto3.client('sagemaker')
    
    try:
        # Check current status
        # 检查当前状态
        response = sagemaker.describe_notebook_instance(
            NotebookInstanceName=instance_name
        )
        current_status = response['NotebookInstanceStatus']
        
        if current_status == 'Stopped':
            print(f"实例 {instance_name} 已经停止")
        elif current_status == 'InService':
            # Stop the instance
            # 停止实例
            sagemaker.stop_notebook_instance(
                NotebookInstanceName=instance_name
            )
            print(f"正在停止实例 {instance_name}...")
            
            # Wait for instance to stop
            # 等待实例停止
            while True:
                response = sagemaker.describe_notebook_instance(
                    NotebookInstanceName=instance_name
                )
                status = response['NotebookInstanceStatus']
                print(f"当前状态: {status}")
                
                if status == 'Stopped':
                    print("实例停止成功!")
                    break
                elif status == 'Failed':
                    print("停止实例失败!")
                    break
                
                time.sleep(30)
                
    except Exception as e:
        print(f"停止实例时出错: {e}")

# Automated stop scheduler
# 自动停止调度器
def schedule_auto_stop(instance_name, hours=2):
    """
    Schedule automatic stop of instance after specified hours
    在指定小时后安排自动停止实例
    """
    import threading
    
    def auto_stop():
        time.sleep(hours * 3600)  # Convert hours to seconds
        print(f"自动停止实例 {instance_name} (运行了 {hours} 小时)")
        stop_notebook_instance(instance_name)
    
    # Start auto-stop timer in background
    # 在后台启动自动停止计时器
    timer_thread = threading.Thread(target=auto_stop)
    timer_thread.daemon = True
    timer_thread.start()
    
    print(f"已设置 {hours} 小时后自动停止实例")
```

#### Instance Lifecycle Management 实例生命周期管理

```python
def get_instance_metrics(instance_name, days=7):
    """
    Get usage metrics for cost analysis
    获取用于成本分析的使用指标
    """
    import boto3
    from datetime import datetime, timedelta
    
    cloudwatch = boto3.client('cloudwatch')
    
    # Calculate time range
    # 计算时间范围
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=days)
    
    # Get CPU utilization metrics
    # 获取CPU利用率指标
    response = cloudwatch.get_metric_statistics(
        Namespace='AWS/SageMaker',
        MetricName='CPUUtilization',
        Dimensions=[
            {
                'Name': 'NotebookInstanceName',
                'Value': instance_name
            }
        ],
        StartTime=start_time,
        EndTime=end_time,
        Period=3600,  # 1 hour intervals
        Statistics=['Average', 'Maximum']
    )
    
    # Calculate estimated costs
    # 计算估算成本
    instance_hours = len(response['Datapoints'])
    estimated_cost = instance_hours * 0.0464  # ml.t3.medium price
    
    print(f"实例 {instance_name} 最近 {days} 天的使用情况:")
    print(f"运行小时数: {instance_hours}")
    print(f"估算成本: ${estimated_cost:.2f}")
    print(f"平均CPU利用率: {sum(dp['Average'] for dp in response['Datapoints'])/len(response['Datapoints']):.1f}%" if response['Datapoints'] else "无数据")

# Cost optimization recommendations
# 成本优化建议
def analyze_instance_usage(instance_name):
    """
    Analyze instance usage and provide cost optimization recommendations
    分析实例使用情况并提供成本优化建议
    """
    sagemaker = boto3.client('sagemaker')
    
    # Get instance details
    # 获取实例详情
    response = sagemaker.describe_notebook_instance(
        NotebookInstanceName=instance_name
    )
    
    instance_type = response['InstanceType']
    creation_time = response['CreationTime']
    last_modified = response['LastModifiedTime']
    
    # Calculate age
    # 计算使用时长
    age = datetime.now(creation_time.tzinfo) - creation_time
    
    print(f"实例分析报告:")
    print(f"实例类型: {instance_type}")
    print(f"创建时间: {creation_time}")
    print(f"实例年龄: {age.days} 天")
    
    # Provide recommendations
    # 提供建议
    if age.days > 30 and instance_type.startswith('ml.p'):
        print("💡 建议: GPU实例已使用超过30天，考虑是否需要降级到CPU实例以节省成本")
    elif age.days > 7 and 'large' in instance_type:
        print("💡 建议: 检查是否可以使用更小的实例类型")
    
    print("💡 成本优化提示:")
    print("- 不使用时立即停止实例")
    print("- 定期备份重要笔记本到S3")
    print("- 考虑使用SageMaker Studio作为现代替代方案")
```

### 20.2.4 Updating Notebooks 更新笔记本

Keeping your SageMaker environment up to date is important for security, performance, and access to the latest features. There are several ways to update and manage your notebooks effectively.

保持SageMaker环境更新对于安全性、性能和访问最新功能很重要。有几种方法可以有效地更新和管理笔记本。

#### Git Integration Git集成

SageMaker supports Git integration, allowing you to clone repositories directly into your notebook instance:

SageMaker支持Git集成，允许你直接将存储库克隆到笔记本实例中：

```python
# Setting up Git repository in SageMaker
# 在SageMaker中设置Git存储库
import subprocess
import os

def setup_git_repository(repo_url, local_path="/home/ec2-user/SageMaker"):
    """
    Clone a Git repository to SageMaker notebook instance
    将Git存储库克隆到SageMaker笔记本实例
    """
    try:
        # Change to SageMaker directory
        # 切换到SageMaker目录
        os.chdir(local_path)
        
        # Clone repository
        # 克隆存储库
        result = subprocess.run(
            ["git", "clone", repo_url],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"成功克隆存储库: {repo_url}")
            print(f"本地路径: {local_path}")
        else:
            print(f"克隆失败: {result.stderr}")
            
    except Exception as e:
        print(f"设置Git存储库时出错: {e}")

# Example: Clone a deep learning course repository
# 示例：克隆深度学习课程存储库
# setup_git_repository("https://github.com/d2l-ai/d2l-en.git")
```

#### Package Management 包管理

Managing Python packages in SageMaker requires understanding the environment structure:

在SageMaker中管理Python包需要了解环境结构：

```python
# Package installation and management
# 包安装和管理
import sys
import subprocess

def install_packages(packages):
    """
    Install Python packages in SageMaker environment
    在SageMaker环境中安装Python包
    """
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ 成功安装: {package}")
        except subprocess.CalledProcessError as e:
            print(f"❌ 安装失败 {package}: {e}")

# Essential packages for deep learning
# 深度学习必需包
essential_packages = [
    "torch",
    "torchvision",
    "transformers",
    "datasets",
    "matplotlib",
    "seaborn",
    "scikit-learn",
    "pandas",
    "numpy",
    "tensorboard"
]

# Install packages
# 安装包
# install_packages(essential_packages)

# Check installed packages
# 检查已安装的包
def list_installed_packages():
    """
    List all installed packages with versions
    列出所有已安装包及其版本
    """
    result = subprocess.run([sys.executable, "-m", "pip", "list"], 
                          capture_output=True, text=True)
    print("已安装的包:")
    print(result.stdout)

# Create requirements.txt for reproducibility
# 创建requirements.txt以确保可重复性
def create_requirements_file():
    """
    Create requirements.txt file for environment reproducibility
    创建requirements.txt文件以确保环境可重复性
    """
    result = subprocess.run([sys.executable, "-m", "pip", "freeze"], 
                          capture_output=True, text=True)
    
    with open("requirements.txt", "w") as f:
        f.write(result.stdout)
    
    print("已创建 requirements.txt 文件")
    print("使用 'pip install -r requirements.txt' 来重新创建环境")
```

#### Notebook Lifecycle Configuration 笔记本生命周期配置

Lifecycle configurations allow you to automate setup tasks when instances start:

生命周期配置允许你在实例启动时自动化设置任务：

```bash
#!/bin/bash
# Lifecycle configuration script
# 生命周期配置脚本

# This script runs when the notebook instance starts
# 此脚本在笔记本实例启动时运行

set -e

# Update system packages
# 更新系统包
sudo yum update -y

# Install additional system dependencies
# 安装额外的系统依赖
sudo yum install -y htop tree

# Install conda packages
# 安装conda包
conda install -y -c conda-forge jupyterlab-git

# Install pip packages
# 安装pip包
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install transformers datasets
pip install wandb tensorboard

# Set up Jupyter extensions
# 设置Jupyter扩展
jupyter labextension install @jupyter-widgets/jupyterlab-manager

# Create common directories
# 创建常用目录
mkdir -p /home/ec2-user/SageMaker/data
mkdir -p /home/ec2-user/SageMaker/models
mkdir -p /home/ec2-user/SageMaker/notebooks

# Set up Git configuration (replace with your details)
# 设置Git配置（替换为你的详细信息）
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

echo "Lifecycle configuration completed successfully!"
echo "生命周期配置成功完成!"
```

#### Environment Management 环境管理

```python
# Environment setup and management
# 环境设置和管理
import json
import boto3

def create_lifecycle_config(config_name, script_content):
    """
    Create a lifecycle configuration for SageMaker
    为SageMaker创建生命周期配置
    """
    sagemaker = boto3.client('sagemaker')
    
    try:
        response = sagemaker.create_notebook_instance_lifecycle_config(
            NotebookInstanceLifecycleConfigName=config_name,
            OnStart=[
                {
                    'Content': script_content
                }
            ]
        )
        print(f"成功创建生命周期配置: {config_name}")
        return response['NotebookInstanceLifecycleConfigArn']
    except Exception as e:
        print(f"创建生命周期配置失败: {e}")
        return None

# Environment monitoring
# 环境监控
def monitor_environment():
    """
    Monitor the current environment status
    监控当前环境状态
    """
    import psutil
    import GPUtil
    
    # CPU and Memory info
    # CPU和内存信息
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    print("环境状态监控:")
    print("="*50)
    print(f"CPU 使用率: {cpu_percent:.1f}%")
    print(f"内存使用率: {memory.percent:.1f}% ({memory.used/1024**3:.1f}GB / {memory.total/1024**3:.1f}GB)")
    print(f"磁盘使用率: {disk.percent:.1f}% ({disk.used/1024**3:.1f}GB / {disk.total/1024**3:.1f}GB)")
    
    # GPU info (if available)
    # GPU信息（如果可用）
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            print(f"GPU 数量: {len(gpus)}")
            for i, gpu in enumerate(gpus):
                print(f"GPU {i}: {gpu.name}")
                print(f"  显存使用率: {gpu.memoryUtil*100:.1f}%")
                print(f"  GPU使用率: {gpu.load*100:.1f}%")
                print(f"  温度: {gpu.temperature}°C")
    except:
        print("未检测到GPU或GPUtil不可用")

# Run monitoring
# 运行监控
# monitor_environment()
```

### 20.2.5 Summary 总结

Amazon SageMaker provides a powerful, managed environment for deep learning development. Here are the key takeaways:

Amazon SageMaker为深度学习开发提供了强大的托管环境。以下是关键要点：

**Advantages 优势:**
- Fully managed Jupyter notebook environment 完全托管的Jupyter笔记本环境
- Easy scalability from CPU to GPU instances 从CPU到GPU实例的轻松扩展
- Integrated with AWS ecosystem AWS生态系统集成
- Built-in security and compliance features 内置安全和合规功能
- No infrastructure management required 无需基础设施管理

**Best Practices 最佳实践:**
- Always stop instances when not in use 不使用时务必停止实例
- Use lifecycle configurations for consistent environments 使用生命周期配置确保环境一致性
- Implement proper IAM roles and policies 实施适当的IAM角色和策略
- Regular backup of important notebooks 定期备份重要笔记本
- Monitor costs and usage patterns 监控成本和使用模式

**Cost Optimization 成本优化:**
- Start with smaller instances and scale up as needed 从较小实例开始，根据需要扩展
- Use Spot instances for training jobs 使用Spot实例进行训练作业
- Leverage S3 for data storage instead of instance storage 利用S3进行数据存储而不是实例存储
- Set up billing alerts and usage monitoring 设置账单警报和使用监控

### 20.2.6 Exercises 练习

1. **Basic Setup Exercise 基本设置练习**
   - Create your first SageMaker notebook instance 创建你的第一个SageMaker笔记本实例
   - Install PyTorch and verify GPU availability 安装PyTorch并验证GPU可用性
   - Run a simple neural network training example 运行简单的神经网络训练示例

2. **Cost Management Exercise 成本管理练习**
   - Set up CloudWatch billing alerts 设置CloudWatch账单警报
   - Create a script to automatically stop instances after inactivity 创建脚本在不活动后自动停止实例
   - Compare costs between different instance types 比较不同实例类型的成本

3. **Environment Configuration Exercise 环境配置练习**
   - Create a lifecycle configuration script 创建生命周期配置脚本
   - Set up Git integration with your favorite repository 设置与你喜欢的存储库的Git集成
   - Create a custom conda environment for your project 为你的项目创建自定义conda环境

## 20.3 Using AWS EC2 Instances 使用AWS EC2实例

Amazon Elastic Compute Cloud (EC2) provides scalable computing capacity in the cloud. Unlike SageMaker, EC2 gives you complete control over the computing environment, making it ideal when you need custom configurations or want to optimize costs for long-running workloads.

Amazon弹性计算云（EC2）在云中提供可扩展的计算能力。与SageMaker不同，EC2让你完全控制计算环境，这使其在需要自定义配置或想要为长时间运行的工作负载优化成本时非常理想。

Think of EC2 as renting a virtual computer in the cloud - you get to choose its specifications, install whatever software you need, and configure it exactly how you want.

把EC2想象成在云中租用虚拟计算机——你可以选择其规格、安装所需的任何软件，并完全按照你的需要配置它。

### 20.3.1 Creating and Running an EC2 Instance 创建和运行EC2实例

Creating an EC2 instance for deep learning requires careful consideration of instance types, storage, networking, and security configurations.

为深度学习创建EC2实例需要仔细考虑实例类型、存储、网络和安全配置。

#### Step 1: Choosing the Right Instance Type 步骤1：选择正确的实例类型

```python
# EC2 Instance types for deep learning
# 用于深度学习的EC2实例类型
ec2_instances = {
    # CPU-optimized instances for development
    # 用于开发的CPU优化实例
    "t3.large": {
        "vcpu": 2,
        "memory_gb": 8,
        "network": "Up to 5 Gbps",
        "price_per_hour": 0.0832,
        "use_case": "开发和小规模实验"
    },
    "m5.xlarge": {
        "vcpu": 4,
        "memory_gb": 16,
        "network": "Up to 10 Gbps",
        "price_per_hour": 0.192,
        "use_case": "中等计算需求"
    },
    
    # GPU instances for training
    # 用于训练的GPU实例
    "p3.2xlarge": {
        "vcpu": 8,
        "memory_gb": 61,
        "gpu": "1x Tesla V100 (16GB)",
        "gpu_memory": "16 GB",
        "price_per_hour": 3.06,
        "use_case": "单GPU训练"
    },
    "p3.8xlarge": {
        "vcpu": 32,
        "memory_gb": 244,
        "gpu": "4x Tesla V100 (16GB each)",
        "gpu_memory": "64 GB total",
        "price_per_hour": 12.24,
        "use_case": "多GPU训练"
    },
    "p4d.24xlarge": {
        "vcpu": 96,
        "memory_gb": 1152,
        "gpu": "8x A100 (40GB each)",
        "gpu_memory": "320 GB total",
        "price_per_hour": 32.77,
        "use_case": "大规模分布式训练"
    },
    
    # Cost-effective GPU instances
    # 性价比高的GPU实例
    "g4dn.xlarge": {
        "vcpu": 4,
        "memory_gb": 16,
        "gpu": "1x T4 (16GB)",
        "gpu_memory": "16 GB",
        "price_per_hour": 0.526,
        "use_case": "推理和轻量级训练"
    }
}

# Display comparison
# 显示比较
print("EC2深度学习实例类型比较:")
print("="*100)
print(f"{'实例类型':<12} {'vCPU':<5} {'内存(GB)':<8} {'GPU':<25} {'每小时($)':<10} {'推荐用途'}")
print("-"*100)

for instance, specs in ec2_instances.items():
    gpu_info = specs.get('gpu', 'None')
    print(f"{instance:<12} {specs['vcpu']:<5} {specs['memory_gb']:<8} "
          f"{gpu_info:<25} {specs['price_per_hour']:<10} {specs['use_case']}")
```

#### Step 2: Launching an EC2 Instance 步骤2：启动EC2实例

```python
import boto3
import time

def launch_ec2_instance(instance_type="g4dn.xlarge", key_name="my-key-pair"):
    """
    Launch an EC2 instance optimized for deep learning
    启动针对深度学习优化的EC2实例
    """
    ec2 = boto3.client('ec2')
    
    # Deep Learning AMI (Amazon Machine Image)
    # 深度学习AMI（Amazon机器映像）
    # This AMI comes pre-installed with popular ML frameworks
    # 此AMI预装了流行的机器学习框架
    ami_id = "ami-0c94855ba95b798c7"  # Deep Learning AMI (Ubuntu 20.04)
    
    # Security group configuration
    # 安全组配置
    security_group_config = {
        'GroupName': 'deep-learning-sg',
        'Description': 'Security group for deep learning instances',
        'IpPermissions': [
            {
                'IpProtocol': 'tcp',
                'FromPort': 22,
                'ToPort': 22,
                'IpRanges': [{'CidrIp': '0.0.0.0/0', 'Description': 'SSH access'}]
            },
            {
                'IpProtocol': 'tcp',
                'FromPort': 8888,
                'ToPort': 8888,
                'IpRanges': [{'CidrIp': '0.0.0.0/0', 'Description': 'Jupyter Notebook'}]
            },
            {
                'IpProtocol': 'tcp',
                'FromPort': 6006,
                'ToPort': 6006,
                'IpRanges': [{'CidrIp': '0.0.0.0/0', 'Description': 'TensorBoard'}]
            }
        ]
    }
    
    try:
        # Create security group
        # 创建安全组
        try:
            sg_response = ec2.create_security_group(**security_group_config)
            security_group_id = sg_response['GroupId']
            print(f"创建安全组: {security_group_id}")
        except ec2.exceptions.ClientError as e:
            if 'already exists' in str(e):
                # Get existing security group
                # 获取现有安全组
                sg_response = ec2.describe_security_groups(
                    GroupNames=[security_group_config['GroupName']]
                )
                security_group_id = sg_response['SecurityGroups'][0]['GroupId']
                print(f"使用现有安全组: {security_group_id}")
            else:
                raise e
        
        # Launch instance
        # 启动实例
        response = ec2.run_instances(
            ImageId=ami_id,
            MinCount=1,
            MaxCount=1,
            InstanceType=instance_type,
            KeyName=key_name,
            SecurityGroupIds=[security_group_id],
            BlockDeviceMappings=[
                {
                    'DeviceName': '/dev/sda1',
                    'Ebs': {
                        'VolumeType': 'gp3',
                        'VolumeSize': 100,  # 100 GB storage
                        'DeleteOnTermination': True
                    }
                }
            ],
            TagSpecifications=[
                {
                    'ResourceType': 'instance',
                    'Tags': [
                        {'Key': 'Name', 'Value': 'Deep Learning Instance'},
                        {'Key': 'Purpose', 'Value': 'ML Training'}
                    ]
                }
            ]
        )
        
        instance_id = response['Instances'][0]['InstanceId']
        print(f"启动实例: {instance_id}")
        
        # Wait for instance to be running
        # 等待实例运行
        print("等待实例启动...")
        waiter = ec2.get_waiter('instance_running')
        waiter.wait(InstanceIds=[instance_id])
        
        # Get instance details
        # 获取实例详情
        instances = ec2.describe_instances(InstanceIds=[instance_id])
        instance = instances['Reservations'][0]['Instances'][0]
        public_ip = instance.get('PublicIpAddress')
        
        print(f"实例启动成功!")
        print(f"实例ID: {instance_id}")
        print(f"公网IP: {public_ip}")
        print(f"SSH连接: ssh -i {key_name}.pem ubuntu@{public_ip}")
        
        return {
            'instance_id': instance_id,
            'public_ip': public_ip,
            'instance_type': instance_type
        }
        
    except Exception as e:
        print(f"启动实例失败: {e}")
        return None

# Example usage
# 使用示例
# instance_info = launch_ec2_instance("g4dn.xlarge", "my-key-pair")
```

#### Step 3: Creating Key Pairs for SSH Access 步骤3：创建用于SSH访问的密钥对

```python
def create_key_pair(key_name="deep-learning-key"):
    """
    Create an EC2 key pair for SSH access
    创建用于SSH访问的EC2密钥对
    """
    ec2 = boto3.client('ec2')
    
    try:
        response = ec2.create_key_pair(KeyName=key_name)
        
        # Save private key to file
        # 将私钥保存到文件
        with open(f"{key_name}.pem", 'w') as key_file:
            key_file.write(response['KeyMaterial'])
        
        # Set proper permissions for the key file
        # 为密钥文件设置适当权限
        import os
        os.chmod(f"{key_name}.pem", 0o400)
        
        print(f"密钥对创建成功: {key_name}")
        print(f"私钥已保存到: {key_name}.pem")
        print("请安全保存此文件，它用于SSH连接到实例")
        
        return key_name
        
    except Exception as e:
        print(f"创建密钥对失败: {e}")
        return None

# SSH connection guide
# SSH连接指南
def print_ssh_guide(public_ip, key_name):
    """
    Print SSH connection instructions
    打印SSH连接说明
    """
    print("\n" + "="*60)
    print("SSH连接指南:")
    print("="*60)
    print(f"1. 确保密钥文件权限正确:")
    print(f"   chmod 400 {key_name}.pem")
    print(f"\n2. 连接到实例:")
    print(f"   ssh -i {key_name}.pem ubuntu@{public_ip}")
    print(f"\n3. 首次连接时，输入 'yes' 接受主机密钥")
    print(f"\n4. 连接成功后，你将进入Ubuntu环境")
    print("="*60)
```

### 20.3.2 Installing CUDA 安装CUDA

CUDA (Compute Unified Device Architecture) is NVIDIA's parallel computing platform that enables GPUs to be used for deep learning. Installing CUDA correctly is crucial for GPU-accelerated training.

CUDA（计算统一设备架构）是NVIDIA的并行计算平台，使GPU能够用于深度学习。正确安装CUDA对于GPU加速训练至关重要。

Think of CUDA as the bridge that allows your deep learning frameworks to communicate with your GPU hardware.

把CUDA想象成允许你的深度学习框架与GPU硬件通信的桥梁。

#### CUDA Installation Script CUDA安装脚本

```bash
#!/bin/bash
# CUDA Installation Script for Ubuntu 20.04
# Ubuntu 20.04的CUDA安装脚本

echo "开始安装CUDA..."

# Update system packages
# 更新系统包
sudo apt update
sudo apt upgrade -y

# Install required dependencies
# 安装必需的依赖项
sudo apt install -y wget software-properties-common

# Remove any existing NVIDIA drivers (if upgrading)
# 删除任何现有的NVIDIA驱动程序（如果升级）
# sudo apt remove --purge nvidia-*
# sudo apt autoremove

# Add NVIDIA package repositories
# 添加NVIDIA包仓库
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update

# Install CUDA Toolkit (version 11.8)
# 安装CUDA工具包（版本11.8）
sudo apt install -y cuda-toolkit-11-8

# Install NVIDIA driver
# 安装NVIDIA驱动程序
sudo apt install -y nvidia-driver-520

# Add CUDA to PATH
# 将CUDA添加到PATH
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

# Reload bash configuration
# 重新加载bash配置
source ~/.bashrc

echo "CUDA安装完成！请重启系统以使NVIDIA驱动程序生效。"
echo "重启后，运行 'nvidia-smi' 验证安装。"
```

#### CUDA Verification Script CUDA验证脚本

```python
# CUDA Verification and Testing
# CUDA验证和测试
import subprocess
import sys

def check_nvidia_driver():
    """
    Check if NVIDIA driver is properly installed
    检查NVIDIA驱动程序是否正确安装
    """
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA驱动程序安装正确")
            print(result.stdout)
            return True
        else:
            print("❌ NVIDIA驱动程序未正确安装")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi命令未找到，请安装NVIDIA驱动程序")
        return False

def check_cuda_installation():
    """
    Check CUDA installation and version
    检查CUDA安装和版本
    """
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ CUDA工具包安装正确")
            print(result.stdout)
            return True
        else:
            print("❌ CUDA工具包未正确安装")
            return False
    except FileNotFoundError:
        print("❌ nvcc命令未找到，请安装CUDA工具包")
        return False

def test_pytorch_cuda():
    """
    Test PyTorch CUDA integration
    测试PyTorch CUDA集成
    """
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"GPU数量: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
            # Test GPU computation
            # 测试GPU计算
            device = torch.device('cuda')
            x = torch.randn(1000, 1000, device=device)
            y = torch.randn(1000, 1000, device=device)
            z = torch.mm(x, y)
            print("✅ GPU计算测试成功")
            
        else:
            print("❌ PyTorch无法访问CUDA")
            
    except ImportError:
        print("❌ PyTorch未安装")

def test_tensorflow_gpu():
    """
    Test TensorFlow GPU integration
    测试TensorFlow GPU集成
    """
    try:
        import tensorflow as tf
        print(f"TensorFlow版本: {tf.__version__}")
        
        gpus = tf.config.list_physical_devices('GPU')
        print(f"检测到GPU数量: {len(gpus)}")
        
        if gpus:
            for i, gpu in enumerate(gpus):
                print(f"GPU {i}: {gpu}")
            
            # Test GPU computation
            # 测试GPU计算
            with tf.device('/GPU:0'):
                a = tf.random.normal([1000, 1000])
                b = tf.random.normal([1000, 1000])
                c = tf.matmul(a, b)
            print("✅ TensorFlow GPU计算测试成功")
        else:
            print("❌ TensorFlow无法检测到GPU")
            
    except ImportError:
        print("❌ TensorFlow未安装")

# Run all checks
# 运行所有检查
def run_cuda_diagnostics():
    """
    Run comprehensive CUDA diagnostics
    运行全面的CUDA诊断
    """
    print("CUDA环境诊断")
    print("="*50)
    
    print("\n1. 检查NVIDIA驱动程序:")
    driver_ok = check_nvidia_driver()
    
    print("\n2. 检查CUDA工具包:")
    cuda_ok = check_cuda_installation()
    
    print("\n3. 测试PyTorch CUDA:")
    test_pytorch_cuda()
    
    print("\n4. 测试TensorFlow GPU:")
    test_tensorflow_gpu()
    
    print("\n" + "="*50)
    if driver_ok and cuda_ok:
        print("✅ CUDA环境配置正确！")
    else:
        print("❌ CUDA环境配置存在问题，请检查安装步骤")

# Run diagnostics
# 运行诊断
# run_cuda_diagnostics()
```

#### Common CUDA Installation Issues 常见CUDA安装问题

```python
# Common CUDA troubleshooting solutions
# 常见CUDA故障排除解决方案

def troubleshoot_cuda():
    """
    Provide solutions for common CUDA issues
    为常见CUDA问题提供解决方案
    """
    print("CUDA故障排除指南")
    print("="*60)
    
    issues_solutions = {
        "nvidia-smi 命令未找到": [
            "sudo apt update",
            "sudo apt install nvidia-driver-520",
            "sudo reboot"
        ],
        
        "CUDA版本不匹配": [
            "检查PyTorch/TensorFlow支持的CUDA版本",
            "卸载当前CUDA: sudo apt remove cuda-*",
            "安装匹配版本的CUDA",
            "重新安装深度学习框架"
        ],
        
        "GPU内存不足": [
            "减少批次大小 (batch_size)",
            "使用梯度累积",
            "启用混合精度训练",
            "使用模型并行化"
        ],
        
        "CUDA out of memory": [
            "torch.cuda.empty_cache() # 清理GPU缓存",
            "减少模型大小或输入大小",
            "使用gradient checkpointing",
            "监控GPU内存使用: nvidia-smi"
        ]
    }
    
    for issue, solutions in issues_solutions.items():
        print(f"\n问题: {issue}")
        print("-" * len(issue))
        for i, solution in enumerate(solutions, 1):
            print(f"{i}. {solution}")

# Display troubleshooting guide
# 显示故障排除指南
# troubleshoot_cuda()
```

### 20.3.3 Installing Libraries for Running the Code 安装运行代码所需的库

After setting up CUDA, you need to install the necessary Python libraries and frameworks for deep learning. This section covers setting up a complete deep learning environment.

设置CUDA后，你需要安装深度学习所需的Python库和框架。本节涵盖设置完整的深度学习环境。

#### Python Environment Setup Python环境设置

```bash
#!/bin/bash
# Python Environment Setup for Deep Learning
# 深度学习Python环境设置

echo "设置Python深度学习环境..."

# Install Python package manager and virtual environment tools
# 安装Python包管理器和虚拟环境工具
sudo apt update
sudo apt install -y python3-pip python3-venv python3-dev

# Install system dependencies
# 安装系统依赖项
sudo apt install -y build-essential cmake git wget curl
sudo apt install -y libopencv-dev libopenblas-dev liblapack-dev
sudo apt install -y libjpeg-dev libpng-dev libtiff-dev
sudo apt install -y libavcodec-dev libavformat-dev libswscale-dev
sudo apt install -y libgtk-3-dev libcanberra-gtk-module libcanberra-gtk3-module

# Create virtual environment
# 创建虚拟环境
python3 -m venv ~/deeplearning_env
source ~/deeplearning_env/bin/activate

# Upgrade pip
# 升级pip
pip install --upgrade pip setuptools wheel

echo "Python环境设置完成！"
```

#### Deep Learning Frameworks Installation 深度学习框架安装

```python
# Deep Learning Libraries Installation Script
# 深度学习库安装脚本
import subprocess
import sys

def install_pytorch():
    """
    Install PyTorch with CUDA support
    安装支持CUDA的PyTorch
    """
    print("安装PyTorch...")
    
    # PyTorch with CUDA 11.8 support
    # 支持CUDA 11.8的PyTorch
    pytorch_packages = [
        "torch==2.0.0+cu118",
        "torchvision==0.15.0+cu118", 
        "torchaudio==2.0.0+cu118"
    ]
    
    for package in pytorch_packages:
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package,
                "--index-url", "https://download.pytorch.org/whl/cu118"
            ])
            print(f"✅ 成功安装: {package}")
        except subprocess.CalledProcessError as e:
            print(f"❌ 安装失败 {package}: {e}")

def install_tensorflow():
    """
    Install TensorFlow with GPU support
    安装支持GPU的TensorFlow
    """
    print("安装TensorFlow...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow[and-cuda]"])
        print("✅ 成功安装TensorFlow GPU版本")
    except subprocess.CalledProcessError as e:
        print(f"❌ TensorFlow安装失败: {e}")

def install_essential_packages():
    """
    Install essential packages for deep learning
    安装深度学习必需包
    """
    print("安装必需的深度学习包...")
    
    essential_packages = [
        # Data manipulation and analysis
        # 数据操作和分析
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        
        # Machine learning
        # 机器学习
        "scikit-learn>=1.0.0",
        "xgboost>=1.5.0",
        
        # Visualization
        # 可视化
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        
        # Image processing
        # 图像处理
        "opencv-python>=4.5.0",
        "Pillow>=8.3.0",
        "imageio>=2.9.0",
        
        # Natural Language Processing
        # 自然语言处理
        "transformers>=4.10.0",
        "datasets>=1.12.0",
        "tokenizers>=0.10.0",
        "nltk>=3.6.0",
        "spacy>=3.4.0",
        
        # Experiment tracking and monitoring
        # 实验跟踪和监控
        "tensorboard>=2.7.0",
        "wandb>=0.12.0",
        "mlflow>=1.20.0",
        
        # Jupyter and development tools
        # Jupyter和开发工具
        "jupyter>=1.0.0",
        "jupyterlab>=3.1.0",
        "ipywidgets>=7.6.0",
        
        # Additional utilities
        # 附加实用工具
        "tqdm>=4.62.0",
        "requests>=2.26.0",
        "PyYAML>=5.4.0",
        "h5py>=3.4.0"
    ]
    
    for package in essential_packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ 成功安装: {package}")
        except subprocess.CalledProcessError as e:
            print(f"❌ 安装失败 {package}: {e}")

def install_specialized_packages():
    """
    Install specialized deep learning packages
    安装专业深度学习包
    """
    print("安装专业深度学习包...")
    
    specialized_packages = [
        # Computer Vision
        # 计算机视觉
        "timm",  # PyTorch Image Models
        "albumentations",  # Image augmentation
        "detectron2 @ git+https://github.com/facebookresearch/detectron2.git",
        
        # Audio processing
        # 音频处理
        "librosa",
        "soundfile",
        
        # Graph Neural Networks
        # 图神经网络
        "torch-geometric",
        "dgl",
        
        # Reinforcement Learning
        # 强化学习
        "gym",
        "stable-baselines3",
        
        # Optimization
        # 优化
        "optuna",
        "hyperopt",
        
        # Model deployment
        # 模型部署
        "onnx",
        "onnxruntime",
        "torchscript"
    ]
    
    for package in specialized_packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ 成功安装: {package}")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ 跳过可选包 {package}: {e}")

def create_requirements_file():
    """
    Create requirements.txt file for environment reproduction
    创建requirements.txt文件以便环境复现
    """
    print("创建requirements.txt文件...")
    
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "freeze"], 
                              capture_output=True, text=True)
        
        with open("requirements.txt", "w") as f:
            f.write(result.stdout)
        
        print("✅ requirements.txt创建成功")
        print("使用以下命令在新环境中安装相同的包:")
        print("pip install -r requirements.txt")
        
    except Exception as e:
        print(f"❌ 创建requirements.txt失败: {e}")

def setup_jupyter_extensions():
    """
    Setup Jupyter extensions for better development experience
    设置Jupyter扩展以获得更好的开发体验
    """
    print("设置Jupyter扩展...")
    
    extensions = [
        "jupyter_contrib_nbextensions",
        "jupyter_nbextensions_configurator",
        "ipywidgets"
    ]
    
    for ext in extensions:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", ext])
            print(f"✅ 成功安装: {ext}")
        except subprocess.CalledProcessError as e:
            print(f"❌ 安装失败 {ext}: {e}")
    
    # Enable extensions
    # 启用扩展
    try:
        subprocess.check_call([sys.executable, "-m", "jupyter", "contrib", "nbextension", "install", "--user"])
        subprocess.check_call([sys.executable, "-m", "jupyter", "nbextensions_configurator", "enable", "--user"])
        print("✅ Jupyter扩展配置完成")
    except subprocess.CalledProcessError as e:
        print(f"❌ Jupyter扩展配置失败: {e}")

# Complete installation function
# 完整安装函数
def complete_environment_setup():
    """
    Set up complete deep learning environment
    设置完整的深度学习环境
    """
    print("开始设置深度学习环境...")
    print("="*60)
    
    # Install core frameworks
    # 安装核心框架
    install_pytorch()
    install_tensorflow()
    
    # Install essential packages
    # 安装必需包
    install_essential_packages()
    
    # Install specialized packages
    # 安装专业包
    install_specialized_packages()
    
    # Setup Jupyter
    # 设置Jupyter
    setup_jupyter_extensions()
    
    # Create requirements file
    # 创建requirements文件
    create_requirements_file()
    
    print("="*60)
    print("✅ 深度学习环境设置完成！")
    print("\n下一步:")
    print("1. 重启终端或运行: source ~/.bashrc")
    print("2. 激活虚拟环境: source ~/deeplearning_env/bin/activate")
    print("3. 启动Jupyter: jupyter lab")
    print("4. 测试安装: python -c 'import torch; print(torch.cuda.is_available())'")

# Run the complete setup
# 运行完整设置
# complete_environment_setup()
```

#### Environment Testing and Validation 环境测试和验证

```python
# Comprehensive environment testing
# 全面环境测试
def test_deep_learning_environment():
    """
    Test the complete deep learning environment setup
    测试完整的深度学习环境设置
    """
    print("深度学习环境测试")
    print("="*60)
    
    # Test core packages
    # 测试核心包
    test_results = {}
    
    # Test PyTorch
    # 测试PyTorch
    try:
        import torch
        import torchvision
        import torchaudio
        
        test_results['PyTorch'] = {
            'installed': True,
            'version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else 'N/A',
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        
        # Test GPU computation
        # 测试GPU计算
        if torch.cuda.is_available():
            device = torch.device('cuda')
            x = torch.randn(100, 100, device=device)
            y = torch.mm(x, x.t())
            test_results['PyTorch']['gpu_test'] = 'Passed'
        else:
            test_results['PyTorch']['gpu_test'] = 'Skipped (No GPU)'
            
    except ImportError as e:
        test_results['PyTorch'] = {'installed': False, 'error': str(e)}
    
    # Test TensorFlow
    # 测试TensorFlow
    try:
        import tensorflow as tf
        
        test_results['TensorFlow'] = {
            'installed': True,
            'version': tf.__version__,
            'gpu_available': len(tf.config.list_physical_devices('GPU')) > 0,
            'gpu_count': len(tf.config.list_physical_devices('GPU'))
        }
        
        # Test GPU computation
        # 测试GPU计算
        if test_results['TensorFlow']['gpu_available']:
            with tf.device('/GPU:0'):
                a = tf.random.normal([100, 100])
                b = tf.matmul(a, a)
            test_results['TensorFlow']['gpu_test'] = 'Passed'
        else:
            test_results['TensorFlow']['gpu_test'] = 'Skipped (No GPU)'
            
    except ImportError as e:
        test_results['TensorFlow'] = {'installed': False, 'error': str(e)}
    
    # Test other essential packages
    # 测试其他必需包
    essential_packages = [
        'numpy', 'pandas', 'matplotlib', 'scikit-learn',
        'opencv-python', 'transformers', 'datasets'
    ]
    
    for package in essential_packages:
        try:
            module = __import__(package.replace('-', '_'))
            version = getattr(module, '__version__', 'Unknown')
            test_results[package] = {'installed': True, 'version': version}
        except ImportError:
            test_results[package] = {'installed': False}
    
    # Display results
    # 显示结果
    print("\n测试结果:")
    print("-"*60)
    
    for package, result in test_results.items():
        if result['installed']:
            status = "✅"
            info = f"版本: {result.get('version', 'Unknown')}"
            
            if 'cuda_available' in result:
                cuda_status = "✅" if result['cuda_available'] else "❌"
                info += f", CUDA: {cuda_status}"
                
            if 'gpu_test' in result:
                gpu_status = "✅" if result['gpu_test'] == 'Passed' else "⚠️"
                info += f", GPU测试: {gpu_status}"
                
        else:
            status = "❌"
            info = f"未安装: {result.get('error', '未知错误')}"
        
        print(f"{status} {package:<15} {info}")
    
    # Summary
    # 总结
    installed_count = sum(1 for r in test_results.values() if r['installed'])
    total_count = len(test_results)
    
    print(f"\n总结: {installed_count}/{total_count} 包已正确安装")
    
    if installed_count == total_count:
        print("🎉 恭喜！您的深度学习环境已完全设置好！")
    else:
        print("⚠️ 某些包未安装，请检查安装日志")

# Run environment test
# 运行环境测试
# test_deep_learning_environment()
``` 