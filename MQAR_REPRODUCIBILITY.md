# MQAR 实验复现指南

## 概述
本文档介绍了如何复现线性注意力（Linear Attention）项目的多查询关联回忆（MQAR）实验。

## 前置条件
- **Python**: 3.8+
- **PyTorch**: 2.0+
- **Lightning**: 2.0+
- **依赖库**：`numpy`, `pandas`, `matplotlib`, `seaborn`, `tqdm`

## 一键复现
要运行全套实验（数据生成 -> 训练 -> 评估），请执行：

```bash
cd ${PROJECT_ROOT}
chmod +x run_mqar_experiment.sh
./run_mqar_experiment.sh
```

**注意**：默认情况下，脚本运行实验的一个子集（SeqLen=256, KeyLen=1）用于演示。要运行完全符合论文的实验（SeqLen 128-1024, KeyLen 1-3），请取消注释 `run_mqar_experiment.sh` 中的相应行。

## 输出结构
结果保存在 `./out/mqar/` 中：
- 每个实验都有自己的文件夹：`{Model}_seq{SeqLen}_key{KeyLen}_seed{Seed}`
- 包含：
  - `results.json`：最终测试指标。
  - `best_model_step_*.pth`：最佳模型检查点。
  - `wandb/`：训练日志。

## 分析
要生成汇总图表和表格：

```bash
python mqar_analysis.py --out_root ./out/mqar --save_dir ./analysis_results
```

或者在 Jupyter 中打开 `mqar_analysis.ipynb`。

## 随机性控制
- **种子**：所有脚本使用默认种子 42。
- **硬件**：实验是在 NVIDIA GPU 上设计的。使用不同的硬件（例如 CPU vs GPU）可能会引入轻微的浮点差异，但整体趋势应保持一致。
