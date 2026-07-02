# Position Bias Phase 1 Results

This file is the live run log for Phase 1. Scripts append completed run summaries here when they finish.

## Environment

- Target runtime: Google Colab, single NVIDIA T4, fp32 Jacobian measurement.
- Local implementation workspace: `C:\Users\User\Downloads\Lost In The Middle`.
- Data source: `lost-in-the-middle-main/qa_data`, extracted from `lost-in-the-middle-main.zip`.
- QA prompting choice for later tasks: base causal LM prompt, no chat template.

## Phase 1 Step 1-2 Commands

Run from `position_bias/` on Colab after mounting Drive and installing requirements.

```bash
python scripts/run_jacobian.py --model gpt2 --init random --seq-len 1024 --n-seqs 2 --seed 0 --out /content/drive/MyDrive/position_bias_phase1/results/jacobian/gpt2_random_L1024_seed0 --resume --smoke
python scripts/run_jacobian.py --model gpt2 --init random --seq-len 1024 --n-seqs 50 --seed 0 --out /content/drive/MyDrive/position_bias_phase1/results/jacobian/gpt2_random_L1024_seed0 --resume
```

## Runs

No model runs recorded yet.

## Local Verification

- `python -m compileall position_bias` passed on Windows/Python 3.12.
- Direct scaffold checks passed for prompt formatting, metric parity, curve aggregation, checkpoint round trip, and default data discovery.
- `python position_bias/scripts/run_jacobian.py --help` passed.
- `python -m pytest position_bias/tests` was not run locally because `pytest` is not installed in this environment.
- GPT-2 Jacobian smoke/full runs were not run locally because this environment does not have `torch`, `transformers`, or a Colab T4 GPU.
