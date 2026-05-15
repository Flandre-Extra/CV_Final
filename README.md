# CV_Final 运行说明

## 环境要求

- Python 3.8+
- CUDA 12.1（GPU训练，可选——CPU也可运行但训练极慢）
- Windows 10/11（已在Windows 11测试通过）

## 安装步骤

```bash
cd CV_Final
python -m venv venv
.\venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## 数据集结构

数据集已包含在 `dataset/` 目录下：

```
dataset/
├── raw/                        # 原始拍摄照片（320张，四类各80）
│   ├── portrait/
│   ├── indoor/
│   ├── outdoor/
│   └── still/
├── preprocessed/               # 预处理后（512×512，训练/验证/测试划分）
│   ├── train/
│   ├── val/
│   └── test/
└── labels_cartoon/             # 传统卡通化教师标签
    ├── train/
    ├── val/
    └── test/
```

## 运行方式

### 一键全量运行（正式训练）

```bash
python src/run_pipeline.py
```

自动执行：预处理 → 标签生成 → 训练（200轮） → 评估。

### 快速测试（验证管线完整性，约8分钟）

```bash
python src/quick_test.py
```

使用80张图像、5轮训练快速验证所有阶段。

### 分步运行

```bash
python src/preprocess.py          # 步骤1：预处理
python src/generate_labels.py     # 步骤2：生成教师标签
python src/train.py               # 步骤3：训练（可选 --epochs N）
python src/test.py                # 步骤4：评估
python src/export_styles.py       # 额外：导出各风格独立效果图
```

## 产出文件

- `checkpoints/best_model.pth` — 最佳模型权重（28MB）
- `results/training_curves.png` — 训练曲线
- `results/comparisons/` — 传统vs深度学习对比图
- `results/styles/` — 各风格独立效果图
- `results/evaluation_summary.txt` — SSIM/PSNR/速度汇总

## 代码结构

| 文件 | 职责 |
|------|------|
| `src/config.py` | 集中路径、超参数、随机种子 |
| `src/dataset.py` | PyTorch数据加载（预加载策略） |
| `src/preprocess.py` | 清晰度筛选、尺寸统一、去噪、增强、数据集划分 |
| `src/traditional.py` | 卡通化、素描化、水彩化（纯OpenCV） |
| `src/generate_labels.py` | 批量生成教师标签 |
| `src/model.py` | LightUNet + SE注意力模块 |
| `src/loss.py` | 混合损失函数（MSE+TV+ColorStat） |
| `src/train.py` | 训练循环（AMP混合精度+早停） |
| `src/test.py` | 测试集评估对比 |
| `src/export_styles.py` | 导出各风格独立效果图 |
| `src/run_pipeline.py` | 一键全量运行 |
| `src/quick_test.py` | 冒烟测试入口 |
| `src/utils.py` | 图像I/O、可视化、评估指标 |
