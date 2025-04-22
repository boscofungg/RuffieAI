#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import glob
import shutil
import random
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm
import sys
import subprocess

# 检查和安装必要的包
try:
    import torch
    import yaml
except ImportError:
    print("正在安装必要的依赖...")
    import subprocess
    subprocess.call(['pip', 'install', 'torch', 'pyyaml'])
    import torch
    import yaml

class PetDiseaseDetector:
    """宠物疾病检测器类"""
    
    def __init__(self, images_dir='images', output_dir='pet_disease_yolov5', force_cpu=False):
        """
        初始化检测器
        
        Args:
            images_dir: 包含图像和标注的目录
            output_dir: 输出目录
            force_cpu: 是否强制使用CPU，即使CUDA可用
        """
        self.images_dir = images_dir
        self.output_dir = output_dir
        self.json_files = glob.glob(os.path.join(images_dir, '*.json'))
        self.yolov5_dir = os.path.join(output_dir, 'yolov5')
        self.dataset_dir = os.path.join(output_dir, 'dataset')
        self.force_cpu = force_cpu
        
        if self.force_cpu:
            print("警告：已启用强制CPU模式，即使CUDA可用也将使用CPU")
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.dataset_dir, exist_ok=True)
        
        # 设置数据集子目录
        self.images_train_dir = os.path.join(self.dataset_dir, 'images', 'train')
        self.images_val_dir = os.path.join(self.dataset_dir, 'images', 'val')
        self.labels_train_dir = os.path.join(self.dataset_dir, 'labels', 'train')
        self.labels_val_dir = os.path.join(self.dataset_dir, 'labels', 'val')
        
        # 创建数据集子目录
        os.makedirs(self.images_train_dir, exist_ok=True)
        os.makedirs(self.images_val_dir, exist_ok=True)
        os.makedirs(self.labels_train_dir, exist_ok=True)
        os.makedirs(self.labels_val_dir, exist_ok=True)
        
        # 类别映射 - 使用简单的类别名称，避免非ASCII字符
        self.class_mapping = {'A2': 0}  # A2疾病类别索引为0
        
        print(f"找到 {len(self.json_files)} 个JSON文件")
    
    def prepare_dataset(self, val_ratio=0.2):
        """
        准备数据集，转换标注为YOLOv5格式
        
        Args:
            val_ratio: 验证集比例
        """
        print("准备数据集...")
        
        # 随机划分训练集和验证集
        random.shuffle(self.json_files)
        val_size = int(len(self.json_files) * val_ratio)
        train_files = self.json_files[val_size:]
        val_files = self.json_files[:val_size]
        
        # 处理训练集
        print("处理训练集...")
        for json_file in tqdm(train_files):
            self._process_file(json_file, is_train=True)
        
        # 处理验证集
        print("处理验证集...")
        for json_file in tqdm(val_files):
            self._process_file(json_file, is_train=False)
        
        # 创建数据集配置文件
        self._create_dataset_yaml()
        
        print(f"数据集准备完成。训练集: {len(train_files)}张图像, 验证集: {len(val_files)}张图像")
    
    def _process_file(self, json_file, is_train=True):
        """
        处理单个JSON文件及其对应的图像
        
        Args:
            json_file: JSON文件路径
            is_train: 是否为训练集
        """
        base_name = os.path.basename(json_file)
        image_file = os.path.join(self.images_dir, base_name.replace('.json', '.jpg'))
        
        # 确定目标目录
        target_img_dir = self.images_train_dir if is_train else self.images_val_dir
        target_label_dir = self.labels_train_dir if is_train else self.labels_val_dir
        
        # 复制图像文件，如果存在
        if os.path.exists(image_file):
            shutil.copy(image_file, os.path.join(target_img_dir, base_name.replace('.json', '.jpg')))
        else:
            # 如果图像不存在，创建一个空白图像（仅用于测试）
            img = Image.new('RGB', (640, 640), (0, 0, 0))
            img.save(os.path.join(target_img_dir, base_name.replace('.json', '.jpg')))
            print(f"警告: 找不到图像 {image_file}，创建了一个空白图像")
        
        # 转换标注为YOLO格式
        try:
            # 读取JSON文件
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 获取图像尺寸
            try:
                with Image.open(image_file) as img:
                    img_width, img_height = img.size
            except FileNotFoundError:
                # 默认尺寸
                img_width, img_height = 640, 640
            
            # 创建YOLO格式标签文件
            label_file = os.path.join(target_label_dir, base_name.replace('.json', '.txt'))
            
            with open(label_file, 'w', encoding='utf-8') as f:
                has_disease = False
                # 检查是否有疾病标注信息
                if 'labelingInfo' in data:
                    for item in data['labelingInfo']:
                        if 'box' in item:
                            # 检查是否是A2疾病
                            # 原标注可能有多种形式，如'A2', 'A2_비듬_각질_상피성잔고리'等
                            class_name = item['box']['label']
                            disease_type = data.get('metaData', {}).get('lesions', '')
                            
                            # 如果JSON中的box标签中有A2或者元数据中的lesions是A2，则视为A2疾病
                            if ('A2' in class_name or disease_type == 'A2'):
                                class_idx = 0  # A2疾病索引为0
                                has_disease = True
                                
                                for loc in item['box']['location']:
                                    # 获取边界框坐标
                                    x = loc['x']
                                    y = loc['y']
                                    width = loc['width']
                                    height = loc['height']
                                    
                                    # 转换为YOLO格式 (x_center, y_center, width, height)，归一化为0-1
                                    x_center = (x + width / 2) / img_width
                                    y_center = (y + height / 2) / img_height
                                    w = width / img_width
                                    h = height / img_height
                                    
                                    # 写入标签文件
                                    f.write(f"{class_idx} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
                
                # 如果没有找到边界框但元数据中显示有疾病，则创建一个覆盖整个图像的边界框
                if not has_disease and data.get('metaData', {}).get('lesions', '') == 'A2':
                    class_idx = 0  # A2疾病索引为0
                    # 覆盖整个图像的边界框
                    f.write(f"{class_idx} 0.5 0.5 1.0 1.0\n")
        
        except Exception as e:
            print(f"处理文件 {json_file} 时出错: {str(e)}")
    
    def _create_dataset_yaml(self):
        """创建数据集配置文件"""
        # 使用绝对路径并确保使用正斜杠作为分隔符
        dataset_path = os.path.abspath(self.dataset_dir).replace('\\', '/')
        
        dataset_yaml = {
            'path': dataset_path,
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(self.class_mapping),
            'names': list(self.class_mapping.keys())
        }
        
        yaml_path = os.path.join(self.dataset_dir, 'dataset.yaml')
        print(f"创建数据集配置文件: {yaml_path}")
        print(f"数据集路径: {dataset_path}")
        
        # 写入YAML文件
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(dataset_yaml, f, default_flow_style=False)
        
        # 验证配置文件是否成功创建
        if os.path.exists(yaml_path):
            print(f"数据集配置文件创建成功")
            with open(yaml_path, 'r', encoding='utf-8') as f:
                print(f"数据集配置内容:\n{f.read()}")
        else:
            print(f"警告: 数据集配置文件创建失败")
    
    def setup_yolov5(self):
        """设置YOLOv5环境"""
        print("设置YOLOv5环境...")
        
        if not os.path.exists(self.yolov5_dir):
            print("克隆YOLOv5仓库...")
            import subprocess
            
            # 克隆YOLOv5仓库
            subprocess.call(['git', 'clone', 'https://github.com/ultralytics/yolov5.git', self.yolov5_dir])
            
            # 安装依赖
            subprocess.call(['pip', 'install', '-r', os.path.join(self.yolov5_dir, 'requirements.txt')])
        
        print("YOLOv5环境设置完成")
    
    def train(self, epochs=50, batch_size=16, img_size=640):
        """
        训练YOLOv5模型
        
        Args:
            epochs: 训练轮数
            batch_size: 批次大小
            img_size: 图像大小
        """
        print("开始训练模型...")
        
        # 确保YOLOv5已设置
        self.setup_yolov5()
        
        # 检查CUDA可用性
        cuda_available = torch.cuda.is_available() and not self.force_cpu
        print(f"CUDA是否可用: {cuda_available}")
        
        if cuda_available:
            try:
                cuda_device_count = torch.cuda.device_count()
                cuda_device_name = torch.cuda.get_device_name(0) if cuda_device_count > 0 else "未知"
                cuda_version = torch.version.cuda if hasattr(torch.version, 'cuda') else "未知"
                
                print(f"CUDA设备数量: {cuda_device_count}")
                print(f"CUDA设备名称: {cuda_device_name}")
                print(f"CUDA版本: {cuda_version}")
                device = '0'  # 使用第一个GPU
            except Exception as e:
                print(f"检查CUDA详情时出错: {str(e)}")
                device = 'cpu'
        else:
            if self.force_cpu:
                print("警告: 强制使用CPU，忽略可能存在的GPU")
            else:
                print("警告: 未检测到可用的CUDA设备，将使用CPU进行训练（速度会很慢）")
            device = 'cpu'
            
        # 准备训练命令参数 - 使用绝对路径
        dataset_yaml = os.path.abspath(os.path.join(self.dataset_dir, 'dataset.yaml'))
        # 使用YOLOv5预训练模型而不是空字符串
        weights = 'yolov5s.pt'  # 使用YOLOv5s小型预训练模型
        
        print(f"数据集配置文件路径: {dataset_yaml}")
        if not os.path.exists(dataset_yaml):
            print(f"错误: 数据集配置文件不存在: {dataset_yaml}")
            return
        
        # 训练结果输出目录 - 使用绝对路径
        project_dir = os.path.abspath(os.path.join(self.output_dir))
        
        # 保存当前工作目录
        current_dir = os.getcwd()
        print(f"当前工作目录: {current_dir}")
        print(f"切换到YOLOv5目录: {self.yolov5_dir}")
        os.chdir(self.yolov5_dir)  # 切换到YOLOv5目录
        
        try:
            cmd = [
                sys.executable,  # 当前Python解释器
                'train.py',
                '--data', dataset_yaml,
                '--weights', weights,  # 现在使用有效的预训练权重
                '--epochs', str(epochs),
                '--batch-size', str(batch_size),
                '--img', str(img_size),
                '--project', project_dir,
                '--name', 'exp',
                '--device', device
            ]
            
            print(f"执行命令: {' '.join(cmd)}")
            
            # 使用实时输出方式运行命令，而不是capture_output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1  # 行缓冲
            )
            
            # 实时输出训练进度
            print("训练开始，实时输出：")
            print("-" * 50)
            
            # 读取并打印标准输出
            for line in process.stdout:
                line = line.rstrip()
                print(line)
            
            # 等待进程结束并获取返回码
            return_code = process.wait()
            
            # 检查是否有错误
            if return_code != 0:
                error_output = process.stderr.read()
                print(f"训练过程中出现错误(返回码 {return_code}):\n{error_output}")
            else:
                print("训练成功完成！")
        
        except Exception as e:
            print(f"训练过程中发生异常: {str(e)}")
        
        finally:
            # 恢复原来的工作目录
            print(f"恢复工作目录: {current_dir}")
            os.chdir(current_dir)
        
        print("模型训练完成")
    
    def predict(self, image_path, conf_threshold=0.25):
        """
        预测图像中的疾病
        
        Args:
            image_path: 图像文件路径
            conf_threshold: 置信度阈值
        """
        print(f"预测图像: {image_path}")
        
        # 确保图像存在
        if not os.path.exists(image_path):
            print(f"错误: 找不到图像 {image_path}")
            return None
        
        # 寻找最新的训练模型
        #work_dir = os.path.dirname(os.path.abspath(__file__))   
        #print(f"当前工作目录: {work_dir}")
        exp_dir = os.path.join(self.output_dir, 'exp')
        print(f"exp_dir: {exp_dir}")
        weights = os.path.abspath(os.path.join(exp_dir, 'weights', 'best.pt'))
        print(f"weights: {weights}")
        
        if not os.path.exists(weights):
            print(f"错误: 找不到模型权重 {weights}")
            return None
        
        # 检查CUDA可用性
        cuda_available = torch.cuda.is_available() and not self.force_cpu
        print(f"CUDA是否可用: {cuda_available}")
        
        if cuda_available:
            try:
                cuda_device_count = torch.cuda.device_count()
                print(f"CUDA设备数量: {cuda_device_count}")
                device = '0'  # 使用第一个GPU
            except Exception as e:
                print(f"检查CUDA详情时出错: {str(e)}")
                device = 'cpu'
        else:
            device = 'cpu'
            if self.force_cpu:
                print("警告: 强制使用CPU，忽略可能存在的GPU")
            else:
                print("警告: 未检测到可用的CUDA设备，将使用CPU进行预测")
        
        # 确保输出目录为绝对路径
        output_dir = os.path.abspath(self.output_dir)
        
        # 图像路径也转为绝对路径
        abs_image_path = os.path.abspath(image_path)
        
        # 保存当前工作目录
        current_dir = os.getcwd()
        print(f"当前工作目录: {current_dir}")
        print(f"切换到YOLOv5目录: {self.yolov5_dir}")
        os.chdir(self.yolov5_dir)
        
        try:
            cmd = [
                sys.executable,
                'detect.py',
                '--weights', weights,
                '--source', abs_image_path,
                '--conf-thres', str(conf_threshold),
                '--iou-thres', '0.45',
                '--max-det', '1000',
                '--device', device,
                '--save-txt',
                '--save-conf',
                '--save-crop',
                '--project', output_dir,
                '--name', 'predict',
                '--exist-ok'
            ]
            
            print(f"执行命令: {' '.join(cmd)}")
            
            # 使用实时输出方式运行命令
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1  # 行缓冲
            )
            
            # 实时输出预测进度
            print("预测开始，实时输出：")
            print("-" * 50)
            
            # 读取并打印标准输出
            for line in process.stdout:
                line = line.rstrip()
                print(line)
            
            # 等待进程结束并获取返回码
            return_code = process.wait()
            
            # 检查是否有错误
            if return_code != 0:
                error_output = process.stderr.read()
                print(f"预测过程中出现错误(返回码 {return_code}):\n{error_output}")
            else:
                print("预测成功完成！")
        
        except Exception as e:
            print(f"预测过程中发生异常: {str(e)}")
        
        finally:
            # 恢复原来的工作目录
            print(f"恢复工作目录: {current_dir}")
            os.chdir(current_dir)
        
        print("预测完成，结果保存在 {}/predict 目录".format(output_dir))
        return True

def check_cuda():
    """检查CUDA环境并打印诊断信息"""
    print("\n" + "="*50)
    print("CUDA诊断信息:")
    print("="*50)
    
    # 检查PyTorch是否支持CUDA
    cuda_available = torch.cuda.is_available()
    print(f"PyTorch CUDA是否可用: {cuda_available}")
    
    if not cuda_available:
        print("可能的原因:")
        print("1. CUDA未安装或驱动不兼容")
        print("2. PyTorch未编译支持CUDA")
        print("3. CUDA版本与PyTorch不兼容")
        print("\n检查环境:")
    
    # 输出PyTorch版本
    print(f"PyTorch版本: {torch.__version__}")
    
    # 如果CUDA可用，打印更多信息
    if cuda_available:
        try:
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"cuDNN版本: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else '不可用'}")
            print(f"CUDA设备数量: {torch.cuda.device_count()}")
            
            # 打印所有CUDA设备信息
            for i in range(torch.cuda.device_count()):
                print(f"  设备 {i}: {torch.cuda.get_device_name(i)}")
                print(f"    总内存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            
            # 尝试运行一个简单的CUDA操作
            print("\n测试CUDA运算...")
            x = torch.rand(100, 100).cuda()
            y = torch.rand(100, 100).cuda()
            z = x @ y  # 矩阵乘法
            print(f"CUDA运算测试: {'成功' if z.size() == (100, 100) else '失败'}")
            
        except Exception as e:
            print(f"获取CUDA信息时出错: {str(e)}")
    
    # 输出系统环境信息
    import platform
    print(f"\n操作系统: {platform.system()} {platform.version()}")
    print(f"Python版本: {platform.python_version()}")
    
    # 如果是Windows，尝试获取GPU信息
    if platform.system() == 'Windows':
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                print("\nNVIDIA-SMI输出:")
                print(result.stdout)
            else:
                print("\nnvidia-smi命令执行失败，可能没有安装或不在PATH中")
        except Exception as e:
            print(f"\n执行nvidia-smi时出错: {str(e)}")
    
    print("="*50 + "\n")

def main():
    """主函数"""
    # 检查CUDA环境
    check_cuda()
    
    parser = argparse.ArgumentParser(description='使用YOLOv5训练和预测宠物疾病检测模型')
    parser.add_argument('--mode', choices=['train', 'predict'], default='train', help='操作模式: train或predict')
    parser.add_argument('--images', default='images', help='图像和标注目录')
    parser.add_argument('--output', default='pet_disease_yolov5', help='输出目录')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16, help='批次大小')
    parser.add_argument('--img-size', type=int, default=640, help='图像大小')
    parser.add_argument('--image-path', help='要预测的图像路径')
    parser.add_argument('--conf-threshold', type=float, default=0.25, help='预测置信度阈值')
    parser.add_argument('--force-cpu', action='store_true', help='强制使用CPU，即使CUDA可用')
    
    args = parser.parse_args()
    
    # 创建检测器
    detector = PetDiseaseDetector(args.images, args.output, args.force_cpu)
    
    if args.mode == 'train':
        # 准备数据集
        detector.prepare_dataset()
        
        # 训练模型
        detector.train(epochs=args.epochs, batch_size=args.batch_size, img_size=args.img_size)
    
    elif args.mode == 'predict':
        if not args.image_path:
            print("错误: 预测模式需要指定 --image-path 参数")
            return
        
        # 预测图像
        detector.predict(args.image_path, conf_threshold=args.conf_threshold)

if __name__ == "__main__":
    main() 