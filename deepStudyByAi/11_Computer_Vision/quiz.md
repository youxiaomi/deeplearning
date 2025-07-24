# 第十一章：计算机视觉 - 测试题

## 选择题

### 1. 图像增强的主要目的是什么？
A. 提高图像分辨率  
B. 增加训练数据的多样性，提高模型泛化能力  
C. 减少计算量  
D. 压缩图像大小  

**答案：B**  
**解析：** 图像增强通过对原始图像进行各种变换（如旋转、翻转、颜色调整等），人工创造更多样化的训练样本，让模型见识到更多变化情况，从而提高泛化能力。就像让学生练习各种类型的题目，而不只是做同一种题。

### 2. 在微调(Fine-tuning)过程中，通常会采取以下哪种策略？
A. 使用比从头训练更大的学习率  
B. 冻结所有层的参数  
C. 使用比从头训练更小的学习率  
D. 随机初始化所有参数  

**答案：C**  
**解析：** 微调使用预训练模型的参数作为起点，这些参数已经相当好了，所以需要用更小的学习率进行"微调"，避免破坏已学到的有用特征。就像调节钢琴，需要小心翼翼地微调，而不是大幅度调整。

### 3. IoU(Intersection over Union)的取值范围是？
A. (-1, 1)  
B. (0, 1)  
C. [0, 1]  
D. (-∞, +∞)  

**答案：C**  
**解析：** IoU = 交集面积 / 并集面积。完全不重叠时IoU=0，完全重叠时IoU=1，所以取值范围是[0,1]。

### 4. 非极大值抑制(NMS)的作用是？
A. 增加检测框的数量  
B. 去除重叠的多余检测框  
C. 提高检测精度  
D. 减少计算量  

**答案：B**  
**解析：** NMS用于去除对同一物体的多个重叠检测框，只保留置信度最高的那个。就像拍合照时每个人可能被多台相机拍到，最后只选择最好的那张照片。

### 5. 转置卷积主要用于？
A. 下采样  
B. 上采样  
C. 特征提取  
D. 降维  

**答案：B**  
**解析：** 转置卷积用于上采样，将低分辨率的特征图恢复到高分辨率，常用于语义分割等需要像素级预测的任务。

## 填空题

### 6. 在目标检测中，边界框通常用 _____ 个数值来表示，分别是 _____、_____、_____、_____。

**答案：** 4，x，y，width，height（或者x1，y1，x2，y2）  
**解析：** 边界框用矩形表示，可以用左上角坐标(x,y)加上宽高(width,height)，或者用左上角坐标(x1,y1)和右下角坐标(x2,y2)来表示。

### 7. R-CNN系列的发展历程是：_____ → _____ → _____ → _____。

**答案：** R-CNN，Fast R-CNN，Faster R-CNN，Mask R-CNN  
**解析：** 这是基于区域的目标检测方法的发展历程，每一代都比前一代更快、更准确。

### 8. 在语义分割中，_____ 分割将相同类别的物体分配相同标签，而 _____ 分割会区分同类别的不同个体。

**答案：** 语义，实例  
**解析：** 语义分割只关心像素属于哪个类别，实例分割还要区分同一类别的不同个体。

## 简答题

### 9. 解释什么是锚框(Anchor Boxes)，以及它们在目标检测中的作用。

**答案：**  
锚框是预先定义的一组边界框模板，具有不同的大小和宽高比。

**作用：**
1. **模板作用**：为不同大小和形状的物体提供检测模板
2. **简化检测**：将目标检测转换为分类和回归问题
3. **多尺度检测**：通过不同尺度的锚框检测不同大小的物体
4. **提高效率**：避免在所有可能位置和尺度上穷举搜索

**类比：** 就像制作衣服时的标准尺码模板（S、M、L、XL），锚框为不同"尺码"的物体提供了检测模板。

### 10. 比较SSD和R-CNN系列方法的主要区别。

**答案：**

**SSD (Single Shot MultiBox Detector):**
- **一阶段方法**：在单次前向传播中同时进行分类和定位
- **速度快**：适合实时应用
- **端到端训练**：整个网络一起训练
- **多尺度检测**：在不同特征图层级进行检测

**R-CNN系列:**
- **两阶段方法**：先生成候选区域，再进行分类
- **精度高**：通常比一阶段方法精度更高
- **计算复杂**：需要更多计算资源
- **逐步优化**：从R-CNN到Faster R-CNN，速度逐步提升

**类比：** SSD像快速诊断的全科医生，一眼看出问题；R-CNN像专科医生，先筛查再细致诊断。

## 编程题

### 11. 编写PyTorch代码实现图像增强的数据变换。

**答案：**
```python
import torch
import torchvision.transforms as transforms

# 定义训练时的图像增强
train_transform = transforms.Compose([
    # 几何变换
    transforms.RandomRotation(degrees=15),        # 随机旋转±15度
    transforms.RandomHorizontalFlip(p=0.5),       # 50%概率水平翻转
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 随机裁剪并缩放
    
    # 颜色变换
    transforms.ColorJitter(
        brightness=0.2,    # 亮度变化±20%
        contrast=0.2,      # 对比度变化±20%
        saturation=0.2,    # 饱和度变化±20%
        hue=0.1           # 色调变化±10%
    ),
    
    # 转换为张量并标准化
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet均值
        std=[0.229, 0.224, 0.225]    # ImageNet标准差
    )
])

# 验证/测试时的变换（通常不使用增强）
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

**解析：** 这段代码展示了常用的图像增强技术，训练时使用增强提高泛化能力，验证时不使用增强确保结果一致性。

### 12. 实现一个简单的微调代码示例。

**答案：**
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

def create_finetuned_model(num_classes=10, freeze_features=True):
    """
    创建微调模型
    
    Args:
        num_classes: 目标任务的类别数
        freeze_features: 是否冻结特征提取层
    """
    # 加载预训练模型
    model = models.resnet18(pretrained=True)
    
    if freeze_features:
        # 冻结特征提取层
        for param in model.parameters():
            param.requires_grad = False
    
    # 替换分类器
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model

# 创建模型
model = create_finetuned_model(num_classes=10)

# 设置优化器（只优化分类器层）
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# 训练循环示例
def train_step(model, data_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    
    for images, labels in data_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(data_loader)

print("微调模型创建完成！")
```

**解析：** 这个例子展示了微调的关键步骤：加载预训练模型、冻结特征层、替换分类器、用小学习率训练。

## 应用题

### 13. 假设你要开发一个智能安防系统，需要检测监控画面中的人和车辆。描述你会选择哪种计算机视觉技术，并说明理由。

**答案：**

**推荐技术：目标检测 + 多类别分类**

**具体方案：**
1. **选择算法**：YOLO或SSD等一阶段检测算法
   - **理由**：实时性要求高，需要快速检测
   
2. **数据准备**：
   - 收集包含人和车辆的监控视频画面
   - 标注边界框和类别标签
   - 使用数据增强增加样本多样性（不同光照、天气）

3. **模型训练**：
   - 使用预训练模型进行微调
   - 重点优化小目标检测（远距离的人和车）
   - 考虑多尺度检测

4. **部署优化**：
   - 模型量化减少计算量
   - 边缘设备部署减少延迟
   - 设置合适的置信度阈值

**挑战和解决方案：**
- **遮挡问题**：使用更先进的检测算法如Mask R-CNN
- **光照变化**：数据增强 + 夜视模式训练
- **实时性**：模型压缩 + 硬件加速

### 14. 某医院想开发一个皮肤病辅助诊断系统，能够从皮肤照片中识别不同类型的皮肤病。设计一个完整的解决方案。

**答案：**

**系统设计方案：**

**1. 数据收集与预处理**
- **数据来源**：医院病例库、公开医学数据集
- **数据标注**：专业皮肤科医生标注
- **预处理**：图像去噪、标准化、ROI提取

**2. 技术选择**
- **主技术**：图像分类 + 语义分割
- **分类网络**：ResNet/EfficientNet等，识别皮肤病类型
- **分割网络**：U-Net等，精确定位病灶区域

**3. 模型训练策略**
```python
# 示例代码框架
def create_skin_diagnosis_model():
    # 使用预训练的图像分类模型
    backbone = models.efficientnet_b0(pretrained=True)
    
    # 替换分类头适应皮肤病分类
    num_classes = 10  # 假设10种皮肤病
    backbone.classifier = nn.Linear(
        backbone.classifier.in_features, 
        num_classes
    )
    
    return backbone

# 数据增强（医学图像特定）
medical_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
```

**4. 系统功能**
- **病灶检测**：定位皮肤异常区域
- **疾病分类**：识别具体皮肤病类型
- **置信度评估**：提供诊断可信度
- **报告生成**：自动生成结构化报告

**5. 质量保证**
- **交叉验证**：确保模型泛化能力
- **专家审核**：医生验证系统判断
- **持续学习**：根据反馈不断改进
- **错误分析**：分析误诊案例并改进

**6. 部署考虑**
- **隐私保护**：患者数据加密
- **监管合规**：符合医疗器械标准
- **用户界面**：医生友好的操作界面
- **性能监控**：实时监控系统表现

**注意事项：**
- 仅作为辅助诊断工具，不替代医生判断
- 需要大量高质量标注数据
- 必须通过严格的医学验证 