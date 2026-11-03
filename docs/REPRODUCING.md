# 运行与复现指南

以下命令从仓库根目录执行，需要Python 3.10或更新版本。

## 环境

建议先创建独立环境：

```bash
python -m venv .venv
```

Windows PowerShell激活：`.\.venv\Scripts\Activate.ps1`。macOS/Linux激活：`source .venv/bin/activate`。

先使用[PyTorch官方安装页](https://pytorch.org/get-started/locally/)选择匹配CPU或本机GPU的 `torch` / `torchvision`。原实验使用RTX 5070 Laptop GPU（8GB），具体软件环境和本次检查环境见[验证说明](VALIDATION.md)。

然后安装：

```bash
python -m pip install -e ".[train,viz,notebook]"
galaxy-morphology --help
```

只查看已有结果图不需要Torch：`python -m pip install -e ".[viz]"`。

## 数据

从[Galaxy Zoo — The Galaxy Challenge](https://www.kaggle.com/competitions/galaxy-zoo-the-galaxy-challenge/data)获取数据，并遵循提供方的使用条件。仓库不重新分发数据。

```text
data/
  training_solutions_rev1.csv
  images_training_rev1/
    100008.jpg
    ...
```

CSV需要唯一整数 `GalaxyID`，以及按任务顺序排列的 `Class1.1` 至 `Class11.6` 等37个目标列。程序会检查列名、缺失/重复ID、非有限数值及超出[0,1]的目标值。缺失或损坏的图片会明确报错，不会静默丢弃后继续计算指标。

可以使用 `--data-dir /path/to/data`，或把 `GALAXY_DATA_DIR` 设为数据目录。

## 训练三个模型

默认参数来自 `configs/baseline.json`。每个命令只训练一个模型，顺序执行可以释放前一个进程占用的显存。

```bash
galaxy-morphology train --model vit --data-dir data --output-dir runs/vit
galaxy-morphology train --model swin --data-dir data --output-dir runs/swin
galaxy-morphology train --model efficientnet --data-dir data --output-dir runs/efficientnet
```

每次训练保存：

- `history.json`：当前运行的逐轮训练损失与验证RMSE。
- `best.pt`：最优模型参数、架构、标签顺序、验证ID及训练设置。
- `run.json`：实际设备、epochs、batch size等运行信息。

默认使用可用的CUDA，否则使用CPU。`--no-amp`关闭CUDA混合精度。CPU支持主要用于小规模检查，完整视觉骨干训练可能耗时较长。

## 小规模功能检查

使用数据集前32行、随机初始化权重和一个epoch，检查真实骨干的流程：

```bash
galaxy-morphology train --model efficientnet --data-dir data --output-dir runs/smoke --device cpu --epochs 1 --batch-size 2 --limit 32 --no-pretrained --no-amp
```

这是功能检查，不是原实验复现，结果不应用于性能比较。`--no-pretrained`避免下载预训练权重。

## 评估与预测

三模型集成的检查点顺序要与权重顺序对应：

```bash
galaxy-morphology evaluate --data-dir data --checkpoints runs/vit/best.pt runs/swin/best.pt runs/efficientnet/best.pt --weights 0.55 0.30 0.15 --output runs/ensemble.json
```

也可以给一个检查点。默认使用五视角TTA；加 `--no-tta` 则只用原图。多个模型必须保存相同的有序验证ID，否则程序拒绝合并评估。

```bash
galaxy-morphology predict --image data/images_training_rev1/100008.jpg --checkpoints runs/vit/best.pt --output runs/prediction.json
```

预测JSON包含37个目标列及其连续输出，不把无约束回归值标成校准概率。

## 导入原课程权重

原Notebook保存的是不含元数据的原始state dictionary。CLI检查点需要明确架构与验证样本信息，可按原实验seed重建划分后转换：

```bash
galaxy-morphology import-legacy --weights-file /path/to/galaxy_ViT_tree.pth --model vit --data-dir data --seed 42 --output runs/vit/best.pt
```

Swin与CNN分别选择 `--model swin` 和 `--model efficientnet`。转换采用严格参数匹配，不下载预训练权重。请确认CSV及划分种子与原权重一致；转换不会自动证明原报告中的指标，仍需执行 `evaluate`。

## 自动检查

```bash
python -m unittest discover -s tests -v
python scripts/plot_results.py
```

测试用临时生成的小图片和标签验证数据/训练/评估/预测的衔接，同时对三个真实骨干执行无预训练参数的CPU前向。它不访问Kaggle，不下载预训练权重，也不修改 `results/` 中的历史记录。
