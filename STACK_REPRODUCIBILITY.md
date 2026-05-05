# Stack 实验复现指南

本文档说明如何在当前代码库中复现 Stack（多栈状态跟踪）任务，并对比以下三个模型：
- RelaxedKaczmarzQNorm
- GatedDeltaNet
- Mamba2

该实现对齐 Kimi Linear tech report 的 synthetic stack 设置：
- 64 个独立栈
- 操作形式：`<push> stack_id value` 与 `<pop> stack_id ?`
- 训练上限：20000 steps
- 模型规模：2 层、2 头、head dim 128（`n_embd=256`）
- 评测维度：
  - 长度从 256 到 2048 的准确率
  - 固定长度 1024 的收敛曲线

## 1. 一键运行

在 `GatedDeltaNet` 目录执行：

```bash
chmod +x run_stack_experiment.sh
./run_stack_experiment.sh
```

若你当前在 base 环境，建议显式指定解释器：

```bash
PYTHON_BIN=python ./run_stack_experiment.sh
```

默认配置会：
- 训练长度：`seq_len=1024`
- 评测长度：`256,512,1024,2048`
- 训练步数：`20000`
- 数据设置：`num_stacks=64`、`num_values=26`、`vocab_size=128`
- 模型：`RelaxedKaczmarzQNorm_Stack`、`GatedDeltaNet_Stack`、`Mamba2_Stack`

默认不开启 LR sweep，仅使用学习率 `1e-3`。

如需严格按 report 做 LR 网格搜索（`{5e-5, 1e-4, 5e-4, 1e-3}`），可显式开启：

```bash
ENABLE_LR_SWEEP=1 ./run_stack_experiment.sh
```

可与解释器参数一起使用：

```bash
PYTHON_BIN=python ENABLE_LR_SWEEP=1 ./run_stack_experiment.sh
```

## 2. 输出目录

- 训练结果：`./out/stack/<exp_name>/results.json`
- 日志（W&B 本地）：`./out/stack/wandb/`

`results.json` 里会记录：
- `eval_acc_by_seq_len`：最终不同序列长度上的准确率
- `val_history`：随训练步数变化的验证准确率
- `test_acc`、`best_val_acc`

## 3. 画图与汇总

```bash
python stack_analysis.py --out_root ./out/stack --save_dir ./analysis_results/stack
```

会生成：
- `stack_metrics.png`
- `seq_len_metrics_raw.csv` / `seq_len_metrics_mean.csv`
- `step_metrics_raw.csv` / `step_metrics_mean.csv`

## 4. 任务定义说明

当前实现采用因果语言建模格式：
- 每条操作占 3 个 token。
- 对 `<pop> stack_id ?`，仅在 `stack_id` 位置监督下一 token 为弹出值。
- 准确率仅在 `labels != -100` 的 pop 监督位上统计。
