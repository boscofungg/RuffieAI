# 使用YOLOv5进行宠物疾病检测

这个项目使用YOLOv5目标检测模型来检测宠物图像中的A2类型疾病（非常/角质/上皮性残环），不仅能识别疾病是否存在，还能定位出疾病在图像中的具体位置。

## 项目特点

- 使用先进的YOLOv5目标检测架构
- 自动将JSON标注转换为YOLOv5兼容格式
- 支持训练和预测两种模式
- 可视化疾病区域并输出置信度
- 自动处理缺失的图像文件（通过创建空白图像）

## 项目结构

```
.
├── images/                # 包含宠物图像和对应的JSON标注
├── pet_disease_yolov5/    # 项目输出目录
│   ├── dataset/           # 处理后的数据集
│   │   ├── images/        # 训练和验证图像
│   │   ├── labels/        # YOLO格式标签
│   │   └── dataset.yaml   # 数据集配置
│   ├── yolov5/            # YOLOv5源代码
│   ├── exp/               # 训练结果
│   └── predict/           # 预测结果
├── pet_disease_detector_yolov5.py  # 主程序脚本
└── README_yolov5.md       # 本文档
```

## 系统要求

- Python 3.7+
- PyTorch 1.7+
- Git (用于克隆YOLOv5仓库)
- CUDA (可选，用于GPU加速)

## 安装

1. 克隆此仓库：
```
git clone <仓库URL>
cd <仓库目录>
```

2. 运行脚本时，所需依赖项会自动安装（包括YOLOv5及其依赖）。

## 使用方法

### 训练模型

运行以下命令来训练模型：

```
python pet_disease_detector_yolov5.py --mode train
```

可选参数：
- `--images` - 图像和标注目录路径（默认为'images'）
- `--output` - 输出目录路径（默认为'pet_disease_yolov5'）
- `--epochs` - 训练轮数（默认为50）
- `--batch-size` - 批次大小（默认为16）
- `--img-size` - 图像大小（默认为640）

例如：
```
python pet_disease_detector_yolov5.py --mode train --epochs 100 --batch-size 8
```

### 预测图像

运行以下命令对新图像进行预测：

```
python pet_disease_detector_yolov5.py --mode predict --image-path path/to/your/image.jpg
```

可选参数：
- `--conf-threshold` - 预测置信度阈值（默认为0.25）

例如：
```
python pet_disease_detector_yolov5.py --mode predict --image-path new_pet_image.jpg --conf-threshold 0.5
```

## 工作流程

1. **数据准备**：
   - 读取JSON标注文件
   - 将标注转换为YOLOv5格式（class_id, x_center, y_center, width, height）
   - 划分训练集和验证集

2. **训练**：
   - 自动下载和配置YOLOv5环境
   - 使用转换后的数据集训练模型
   - 保存训练好的权重和评估结果

3. **预测**：
   - 加载训练好的模型
   - 对输入图像进行疾病检测
   - 输出带有边界框和置信度的结果图像

## 注意事项

- 确保JSON标注文件中的`box`标签包含正确的疾病标签和坐标
- 对于缺失的图像文件，脚本会创建空白图像来避免错误，但这会影响模型训练质量
- YOLOv5训练需要较大的计算资源，推荐使用GPU进行训练

## 模型评估

训练完成后，您可以在`pet_disease_yolov5/exp`目录中找到以下评估结果：

- 精度-召回率曲线
- 混淆矩阵
- F1曲线
- 最佳模型权重

## 预测结果

预测结果保存在`pet_disease_yolov5/predict`目录中，包括：

- 带有边界框的结果图像
- 标签文本文件（包含坐标和置信度）
- 裁剪出的疾病区域图像 