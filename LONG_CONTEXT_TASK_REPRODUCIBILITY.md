# Long-Context Task Benchmark Reproducibility (P2-2)

This document describes how to run P2-2 long-context task benchmarks (RULER-style and similar) **without re-pretraining**.

## 1. Goal

Evaluate long-context generalization on real tasks using existing checkpoints:
- RULER
- MRCR
- RepoQA
- LongBench v2 (optional)

The pipeline follows the same experiment structure used in this repo:
- run script
- benchmark script
- analysis script
- reproducibility doc

## 2. Kimi-Aligned Default Setting (Practical)

Recommended default setup for instruction-untuned base models:
- Zero-shot evaluation.
- Greedy decoding (`temperature=0` behavior via argmax decoding).
- Default max generation tokens: `32`.
- Report both:
  - Overall metrics: EM / Contains / F1
  - Context-bucket metrics: EM / Contains / F1 by prompt token length

Default context buckets (kimi_standard profile):
- `4096,8192,16384,32768,65536,131072`

Quick debug profile (kimi_quick):
- `4096,8192,16384,32768`

## 3. Data Format

Input file is JSONL, one sample per line.

Required fields:
- `prompt`: string
- one of:
  - `answers`: list of strings
  - `answer`: string
  - `target`: string

Optional fields:
- `id`: sample id
- `task`: task name
- `subtask`: sub-task name
- `max_new_tokens`: per-sample override
- `stop_strings`: list of stop strings

Example:

```json
{"id":"1","task":"RULER-SNIAH","prompt":"...","answers":["needle_value"]}
{"id":"2","task":"RepoQA","prompt":"...","answer":"main.py"}
```

## 4. One-Command Run

From the `GatedDeltaNet` directory:

```bash
chmod +x run_long_context_task_benchmark.sh
INPUT_JSONL=./data/long_context_tasks/ruler.jsonl \
BENCHMARK_NAME=RULER \
MODELS=KLA,GDN,Mamba2 \
PROFILE=kimi_standard \
./run_long_context_task_benchmark.sh
```

If you are not in the expected environment, explicitly set python binary:

```bash
PYTHON_BIN=python \
INPUT_JSONL=./data/long_context_tasks/ruler.jsonl \
./run_long_context_task_benchmark.sh
```

## 5. Main Environment Variables

Run-script variables:
- `INPUT_JSONL`: dataset JSONL path (required)
- `BENCHMARK_NAME`: output benchmark name tag (default `RULER`)
- `TASK_NAME`: optional task override
- `MODELS`: model aliases (`KLA,GDN,Mamba2,GLA,DeltaNet,Longhorn`)
- `PROFILE`: `kimi_standard` or `kimi_quick`
- `MAX_SAMPLES`: limit samples for debugging
- `MAX_NEW_TOKENS`: generation length cap
- `LENGTH_BUCKETS`: prompt length bucket upper bounds
- `MAX_PROMPT_TOKENS`: optional prompt length cap
- `PROMPT_TRUNCATION`: `none|left|right` when `MAX_PROMPT_TOKENS > 0`
- `STOP_STRINGS`: comma-separated stop strings
- `ALLOW_OOM_SKIP`: `1` to skip OOM samples
- `LONGBENCH_SAFE_MAX_PROMPT_TOKENS`: auto cap for `DATASET_PRESET=longbench_v2` when `MAX_PROMPT_TOKENS=0` (default `32768`)
- `ALLOW_UNBOUNDED_LONGBENCH_PROMPT`: set `1` to disable the auto cap (may trigger CUDA illegal memory access on very long prompts)
- `SAVE_PREDICTIONS`: `1` to store generated text in JSON/CSV
- `TOKENIZER_NAME`: Hugging Face tokenizer name (default TinyLlama)
- `TOKENIZER_DIR`: optional local tokenizer directory

## 6. Output Artifacts

Raw outputs:
- `./out/long_context_tasks/<BENCHMARK_NAME>/<MODEL>/results.json`
- `./out/long_context_tasks/<BENCHMARK_NAME>/<MODEL>/results.csv`

Aggregated analysis:
- `./analysis_results/long_context_tasks/long_context_task_overall_mean.csv`
- `./analysis_results/long_context_tasks/long_context_task_task_mean.csv`
- `./analysis_results/long_context_tasks/long_context_task_bucket_mean.csv`
- `./analysis_results/long_context_tasks/long_context_tasks_overall.(png|pdf)`
- `./analysis_results/long_context_tasks/long_context_tasks_bucket.(png|pdf)`

## 7. Fair Comparison Checklist

For fair model comparison:
- Use the same input JSONL split for all models.
- Use the same decoding setup (`max_new_tokens`, stop strings).
- Keep evaluation hardware/device/dtype consistent.
- Report both overall and context-bucket metrics.
- Inspect skipped rows (`skip_reason`) in CSV/JSON before drawing conclusions.

## 8. Notes

- This P2-2 pipeline is intentionally evaluation-only and does not require re-pretraining.
- For strict paper comparison with external implementations, keep prompt templates exactly fixed across models.
- LongBench-v2 (`long`) samples can be extremely long. If prompts are unbounded, custom Triton kernels may crash with CUDA illegal memory access.
  - Default run script behavior now auto-applies a safe cap for LongBench-v2.
  - To force full prompts: `ALLOW_UNBOUNDED_LONGBENCH_PROMPT=1`.
