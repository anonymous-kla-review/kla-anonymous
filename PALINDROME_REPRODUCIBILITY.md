# Palindrome 实验复现指南

本文档说明如何在当前代码库中复现 Palindrome（逆序复现）任务，并对比以下三个模型：
- RelaxedKaczmarzQNorm
- GatedDeltaNet
- Mamba2

## 1. 一键运行

在 `GatedDeltaNet` 目录执行：

```bash
chmod +x run_palindrome_experiment.sh
./run_palindrome_experiment.sh
```

默认配置会：
- 训练长度：`seq_len=1024`
- 评测长度：`256,512,1024,2048`
- 训练步数：`20000`
- 词表大小：`vocab_size=128`
- 模型头数：`n_head=2`（`n_embd=256`，对应 `head_dim=128`）
- 模型：`RelaxedKaczmarzQNorm_Palindrome`、`GatedDeltaNet_Palindrome`、`Mamba2_Palindrome`

以上设置与 Kimi Linear 技术报告中 synthetic task 的模型规模更一致（2 层、2 头、head dim 128）。

数据缓存目录会按词表大小区分，例如：
- `./data/palindrome/seq1024_v128_pf0_seed42`

其中 `pf0/pf1` 表示是否开启 `predict_first_token`，用于避免不同任务定义下复用同一缓存。

如果你之前生成过旧目录（如 `seq1024_seed42`，通常对应旧词表设置），新脚本会忽略它并使用带 `v128` 的目录，避免词表不一致导致的 CUDA index 越界。

## 2. 输出目录

- 训练结果：`./out/palindrome/<exp_name>/results.json`
- 日志（W&B 本地）：`./out/palindrome/wandb/`

`results.json` 里会记录：
- `eval_acc_by_seq_len`：最终不同序列长度上的准确率（用于画上图）
- `val_history`：随训练步数变化的验证准确率（用于画下图）
- `test_acc`、`best_val_acc` 等基础指标

最终 `eval_acc_by_seq_len` 现在会对所有长度（包括训练长度）统一使用动态生成的评测集，避免训练长度复用 `val.npz` 带来的评测口径差异。

## 3. 画图与汇总

```bash
python palindrome_analysis.py --out_root ./out/palindrome --save_dir ./analysis_results/palindrome
```

会生成：
- `palindrome_metrics.png`
- `seq_len_metrics_raw.csv` / `seq_len_metrics_mean.csv`
- `step_metrics_raw.csv` / `step_metrics_mean.csv`

## 4. 任务定义说明

当前实现使用因果语言建模格式：
- 输入构造：`source + <SEP> + reverse(source)`
- 监督位置：默认从生成段开始做 next-token 监督（不监督 `<SEP> -> reverse[0]`）

如需监督第一步输出，可在数据生成与训练时加入 `--predict_first_token`。
