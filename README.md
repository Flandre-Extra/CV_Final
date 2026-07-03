# 多风格图像艺术化系统 — 运行说明

## 项目简介

基于传统图像处理与深度学习的双管线图像风格化系统。传统 OpenCV 管线实现卡通化、素描化、水彩化三种风格；深度学习管线采用自建 LightUNet（712 万参数），以传统卡通化输出为教师标签进行知识蒸馏训练。

原始数据集 320 张照片全部由 iPhone 手机自主拍摄（1:1 正方形比例），覆盖 portrait / indoor / outdoor / still 四类场景，每类 80 张。

## 环境要求

- Python 3.8+
- CUDA 12.1
- 显存 ≥ 6 GB（已在 RTX 4050 Laptop 6GB 测试通过）
- Windows 10/11

## 安装步骤

```bash
cd CV_Final
python -m venv venv
.\venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

`requirements.txt` 依赖：torch, opencv-python, numpy, scikit-image, scikit-learn, matplotlib, tqdm, pillow。

## 数据集结构

```
dataset/
├── raw/                        # 原始拍摄照片（320 张，四类各 80）
│   ├── portrait/               # 人像
│   ├── indoor/                 # 室内场景
│   ├── outdoor/                # 室外风光
│   └── still/                  # 静物物品
├── preprocessed/               # 预处理后（512×512，含数据增强）
│   ├── train/                  # 训练集（70%，4× 增强）
│   ├── val/                    # 验证集（15%）
│   └── test/                   # 测试集（15%）
└── labels_cartoon/             # 传统卡通化教师标签
    ├── train/
    ├── val/
    └── test/
```

## 运行方式

### 一键全量运行

```bash
python src/run_pipeline.py
```

按顺序执行四个阶段：预处理 → 标签生成 → 训练（200 轮，早停监控）→ 评估。预计耗时 2.5–3 小时（GPU）。

### 快速冒烟测试（验证管线完整性，约 8 分钟）

```bash
python src/quick_test.py
```

使用 80 张随机图像、5 轮训练快速验证全部四个阶段无报错。

### 分步运行（可按需单独执行）

```bash
python src/preprocess.py              # 步骤 1：清晰度筛选 + 缩放 + 去噪 + 增强 + 数据集划分
python src/preprocess.py --max-total 32   # （可选）限制图像数，快速调试
python src/generate_labels.py         # 步骤 2：传统卡通化生成教师标签
python src/train.py                   # 步骤 3：训练 LightUNet（默认 200 轮）
python src/train.py --epochs 10       # （可选）指定轮数，快速验证
python src/test.py                    # 步骤 4：测试集评估 + 生成对比图
python src/export_styles.py           # 额外：导出各风格独立效果图
```

## 产出文件

- `checkpoints/best_model.pth` — 最佳模型权重（约 85 MB）
- `results/training_curves.png` — Loss 下降 + PSNR 上升训练曲线
- `results/comparisons/` — 传统 vs 深度学习六合一对比图
- `results/styles/` — 11 组 × 5 风格独立效果图（55 张）
- `results/evaluation_summary.txt` — SSIM / PSNR / 推理速度汇总
- `results/training_history.json` — 逐轮 Loss 和 PSNR 数据

## 代码结构

| 文件 | 职责 |
|------|------|
| `src/config.py` | 集中路径、超参数、随机种子、可复现性设置 |
| `src/dataset.py` | PyTorch Dataset，内存预加载策略 |
| `src/preprocess.py` | 拉普拉斯清晰度筛选、缩放、去噪、增强、分层划分 |
| `src/traditional.py` | 卡通化 / 素描化 / 水彩化（纯 OpenCV 手写） |
| `src/generate_labels.py` | 批量运行传统卡通化生成教师标签 |
| `src/model.py` | LightUNet 架构 + SE 通道注意力模块 |
| `src/loss.py` | 混合损失（MSE + TV + 自研 ColorStatLoss） |
| `src/train.py` | 训练循环（AMP 混合精度、早停、`--epochs` 覆盖） |
| `src/test.py` | 测试集评估（SSIM / PSNR / 推理速度对比） |
| `src/export_styles.py` | 导出各风格独立效果图 |
| `src/run_pipeline.py` | 一键全量运行入口 |
| `src/quick_test.py` | 冒烟测试入口 |
| `src/utils.py` | 图像 I/O（含中文路径兼容）、可视化、评估指标 |

## 关键设计

- **双管线对比**：传统 OpenCV（卡通/素描/水彩）→ 教师标签 → LightUNet 知识蒸馏 → 统一测试集对比
- **LightUNet**：4 层编码器-解码器，SE 通道注意力（reduction=4），总参数量 7,120,771
- **ColorStatLoss**：可微逐通道均值/方差匹配，替代不可微直方图距离
- **可复现性**：统一随机种子（seed=42），cudnn.deterministic=True
- **Windows 兼容**：`imread_any` / `imwrite_any` 处理中文路径，num_workers=0
