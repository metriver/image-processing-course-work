# 2026 图像处理课程大作业

## 项目结构

```
├── exp1_edge_detection.py          # 实验一：边缘检测
├── exp2_unsupervised_vessel.py     # 实验二：无监督血管分割
├── exp3_deep_learning_vessel.py    # 实验三：深度学习血管分割
├── utils/
│   ├── __init__.py
│   ├── datasets.py                 # 数据集加载工具
│   └── metrics.py                  # 评估指标 & 可视化
├── datasets/                       # 数据集目录
│   ├── MBIPED/                     # 实验一数据
│   └── DRIVE/                      # 实验二、三数据
├── models/                         # 训练好的模型权重
├── outputs/                        # 输出结果（图表、可视化）
│   ├── exp1/
│   ├── exp2/
│   └── exp3/
├── requirements.txt
└── README.md
```

## requirements
```
# 图像处理
opencv-python>=4.8.0
scikit-image>=0.21.0
Pillow>=10.0.0

# 数值计算
numpy>=1.24.0
scipy>=1.10.0

# 可视化
matplotlib>=3.7.0

# 深度学习
torch>=2.0.0
torchvision>=0.15.0

# 机器学习 & 评估
scikit-learn>=1.3.0

# 进度条
tqdm>=4.65.0

```
