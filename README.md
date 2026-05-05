# Kaczmarz Linear Attention

This repository contains an anonymized implementation for the paper
"Kaczmarz Linear Attention".

## Contents

This repository includes code and scripts for:

- Kaczmarz Linear Attention model components;
- synthetic sequence modeling experiments;
- MQAR experiments;
- stack and palindrome experiments;
- long-context task benchmarks;
- efficiency benchmarks;
- analysis scripts for reproducing the reported results.

## Repository Structure

Main files and directories:

- `lit_gpt/`: model and training components;
- `train_mqar.py`: training script for MQAR experiments;
- `train_stack.py`: training script for stack experiments;
- `train_palindrome.py`: training script for palindrome experiments;
- `pretrain.py`: pretraining entry point;
- `long_context_task_benchmark.py`: long-context task benchmark script;
- `efficiency_prefill_decode_benchmark.py`: efficiency benchmark script;
- `scripts/`: data preparation and auxiliary scripts;
- `*_REPRODUCIBILITY.md`: experiment-specific reproducibility notes.

## Environment

This anonymized release is intended for review and reproducibility checking.
A typical environment requires Python 3.10 or later, PyTorch, NumPy, tqdm,
matplotlib, and the dataset dependencies used by the corresponding scripts.

One possible setup is:

    conda create -n kla python=3.10 -y
    conda activate kla

Install dependencies according to the experiment being run.

## Reproducing Experiments

Example commands:

    bash run_mqar_experiment.sh
    bash run_stack_experiment.sh
    bash run_palindrome_experiment.sh
    bash run_long_context_task_benchmark.sh
    bash run_efficiency_experiment.sh

Additional details are provided in:

- `MQAR_REPRODUCIBILITY.md`
- `STACK_REPRODUCIBILITY.md`
- `PALINDROME_REPRODUCIBILITY.md`
- `LONG_CONTEXT_TASK_REPRODUCIBILITY.md`

## Notes for Reviewers

This repository has been anonymized for double-blind review. Author names,
institutional affiliations, personal paths, and non-anonymous repository links
have been removed.

## Third-party Code and Licenses

This repository may contain components adapted from prior open-source sequence
modeling implementations. Required third-party license notices are preserved
where applicable.
