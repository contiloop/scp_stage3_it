# scp_stage3_it

SFT translation tuning pipeline (Hydra + Unsloth + W&B) for reversed MT datasets.

## Trained Models

This repo is configured for these CPT-stage2 base checkpoints:

- `alwaysgood/QWEN3-4B-CPT-stage2`
- `alwaysgood/QWEN3.5-4B-CPT-half-lr-stage2`
- `alwaysgood/gemma4-CPT-stage2`

## Data

- `alwaysgood/wmt24pp-kr-reversed`
- `alwaysgood/flores-kr-reversed`

## Config Layout (Cleaned)

- Top-level run configs:
  - `configs/config.yaml` (default -> QWEN3-4B SFT)
  - `configs/sft_qwen3_4b_cpt_stage2.yaml`
  - `configs/sft_qwen3_5_4b_cpt_half_lr_stage2.yaml`
  - `configs/sft_gemma4_cpt_stage2.yaml`
- Model presets:
  - `configs/model/qwen3_4b_cpt_stage2.yaml`
  - `configs/model/qwen3_5_4b_cpt_half_lr_stage2.yaml`
  - `configs/model/gemma4_cpt_stage2.yaml`
- Shared components:
  - `configs/data/parallel_wmt24pp_flores_reversed.yaml`
  - `configs/preprocessing/sft_translation.yaml`
  - `configs/prompts/translation_dynamic_5.yaml`
  - `configs/training/gpu96_sft.yaml`
  - `configs/logging/wandb.yaml`

## Setup

```bash
make set
```

`make set` now does:

- install `transformers==5.5.4` and `trl>=0.15.0` with `--no-deps`
- print installed transformers version
- optional installs with status print:
  - `xformers_install_ok=true|false`
  - `weave_install_ok=true|false`

## Run

Default (`configs/config.yaml` -> `sft_qwen3_4b_cpt_stage2`):

```bash
make preprocess
make train
```

Model-specific:

```bash
# QWEN3-4B-CPT-stage2
make preprocess config=sft_qwen3_4b_cpt_stage2
make train config=sft_qwen3_4b_cpt_stage2

# QWEN3.5-4B-CPT-half-lr-stage2
make preprocess config=sft_qwen3_5_4b_cpt_half_lr_stage2
make train config=sft_qwen3_5_4b_cpt_half_lr_stage2

# gemma4-CPT-stage2
make preprocess config=sft_gemma4_cpt_stage2
make train config=sft_gemma4_cpt_stage2
```

## Weights & Biases

- default project: `instruction-tuning`
- Weave tracing is off by default:
  - `logging.weave.enabled=false`

Enable Weave for a run:

```bash
make train config=sft_qwen3_4b_cpt_stage2 ovr="logging.weave.enabled=true"
```

## Notes

- Loss is computed on completion tokens only (`completion_mask` + `completion_only_loss=true`).
- EOS is appended to target by preprocessing.
- Runtime packing is disabled by default for SFT stability.
