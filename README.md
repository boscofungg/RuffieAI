 # 宠物健康平台

宠物健康平台是一个综合性系统，用于监测和管理宠物健康状况。该系统包括宠物疾病检测、健康数据管理和用户界面等功能。

## 目录

1. [系统要求](#系统要求)
2. [数据库设置](#数据库设置)
3. [模型训练](#模型训练)
4. [后端设置](#后端设置)
5. [前端设置](#前端设置)
6. [系统运行](#系统运行)

## 系统要求

- Python 3.7+
- MySQL 5.7+
- Node.js 14+
- npm 6+
- Git
- CUDA (可选，用于GPU加速模型训练)

## 数据库设置

系统使用MySQL数据库来存储宠物信息、健康记录和用户数据。

### 创建数据库

1. 安装并启动MySQL服务器
2. 使用以下命令创建并导入数据库：

```bash
mysql -u root -p < pet-health-platform/database/pet.sql
```

该SQL文件将创建必要的表结构并导入初始数据，包括：
- 用户表
- 宠物信息表
- 健康记录表
- 疾病诊断表
- 其他相关表

## 模型训练

系统使用YOLOv5目标检测模型来识别宠物图像中的疾病。

### 训练步骤

请参考 [README_yolov5.md](README_yolov5.md) 获取详细训练步骤。概括来说：

1. 准备带有标注的宠物图像数据
2. 运行训练脚本：

```bash
python pet_disease_detector_yolov5.py --mode train
```

3. 可调整参数：
   - `--images` - 图像和标注目录路径（默认为'images'）
   - `--output` - 输出目录路径（默认为'pet_disease_yolov5'）
   - `--epochs` - 训练轮数（默认为50）
   - `--batch-size` - 批次大小（默认为16）
   - `--img-size` - 图像大小（默认为640）

训练完成后，模型权重将保存在 `pet_disease_yolov5/exp` 目录中。

## 后端设置

后端系统基于Python开发，提供API服务和模型推理功能。

### 安装依赖

1. 安装Python依赖：

```bash
pip install -r requirements.txt
```

2. 导航到后端目录：

```bash
cd pet-health-platform/backend
```

3. 配置后端服务：
   - 修改数据库连接配置
   - 设置模型路径
   - 配置服务端口

### 启动后端服务

```bash
cd pet-health-platform/backend
python app.py
```

后端服务将在指定端口（默认为5000）启动，提供API接口。

## 前端设置

前端基于现代Web技术开发，提供直观的用户界面。

### 安装依赖

1. 导航到前端目录：

```bash
cd pet-health-platform/frontend
```

2. 安装前端依赖：

```bash
npm install
```

3. 配置前端：
   - 编辑配置文件，指定后端API地址
   - 根据需要调整UI设置

### 启动前端服务

```bash
cd pet-health-platform/frontend
npm run dev
```

前端开发服务器将启动，通常在 http://localhost:3000 可以访问。

## 系统运行

完成上述设置后，系统的完整运行流程如下：

1. 确保数据库服务正在运行
2. 启动后端服务
3. 启动前端服务
4. 在浏览器中访问前端URL（通常为 http://localhost:3000）

### 主要功能

- 用户注册与登录
- 宠物信息管理
- 上传宠物图像进行疾病检测
- 查看健康记录和诊断结果
- 获取健康建议

### 系统使用建议

- 定期备份数据库
- 监控系统负载，特别是在处理大量图像时
- 定期更新模型权重以提高疾病检测准确性

## 问题排查

如遇到问题，请检查：

- 数据库连接是否正常
- API服务是否正常响应
- 日志文件中是否有错误信息

如需进一步支持，请联系系统管理员或开发团队。