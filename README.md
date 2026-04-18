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

## Quick Start

Fresh start:

```bash
git clone https://github.com/contiloop/scp_stage3_it.git
cd scp_stage3_it
make set

# required for pulling private base/data repos and for push-to-hub
python -c "import os; from huggingface_hub import login; login(token=os.environ['HF_TOKEN'])"

# optional
wandb login
```

`make set` now does:

- install `transformers==5.5.4` and `trl>=0.15.0` with `--no-deps`
- print installed transformers version
- optional installs with status print:
  - `xformers_install_ok=true|false`
  - `weave_install_ok=true|false`

Notes:

- Use your own Hugging Face token via `HF_TOKEN`. Do not hardcode tokens in this repo.
- The old stage1 flow used `scp_stage1_cpt`; for this repo, use `scp_stage3_it` and the SFT configs below.

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

If you want to inspect the fully resolved Hydra config before running:

```bash
make show-config config=sft_qwen3_4b_cpt_stage2
```

Cleanup example:

```bash
# gemma4-CPT-stage2 artifacts
rm -rf artifacts/sft_gemma4_cpt_stage2

# shared processed data for all SFT configs in this repo
rm -rf data/processed/sft_parallel_reversed
```

## Weights & Biases

- default project: `instruction-tuning`
- Weave tracing is off by default:
  - `logging.weave.enabled=false`

Enable Weave for a run:

```bash
make train config=sft_qwen3_4b_cpt_stage2 ovr="logging.weave.enabled=true"
```

## Push To HF

After training, upload the final output dir or a checkpoint to Hugging Face Hub:

```bash
# final merged output dir
make push-to-hub \
  config=sft_qwen3_4b_cpt_stage2 \
  HF_REPO=your-name/qwen3-4b-cpt-stage2-it

# latest checkpoint-* under training.output_dir
make push-to-hub \
  config=sft_qwen3_4b_cpt_stage2 \
  HF_REPO=your-name/qwen3-4b-cpt-stage2-it \
  CKPT=latest

# specific checkpoint and private repo
make push-to-hub \
  config=sft_qwen3_4b_cpt_stage2 \
  HF_REPO=your-name/qwen3-4b-cpt-stage2-it \
  CKPT=checkpoint-1000 \
  HF_PRIVATE=true
```

Behavior:

- `HF_REPO` is required, e.g. `your-name/model-name`
- `CKPT=final` is the default
- `CKPT=latest` uploads the newest `checkpoint-*` directory
- matching eval artifacts under `artifacts/.../eval` are uploaded automatically by default
- the repo is created automatically if it does not exist

Useful model-specific examples:

```bash
make push-to-hub config=sft_qwen3_4b_cpt_stage2 HF_REPO=your-name/qwen3-4b-cpt-stage2-it
make push-to-hub config=sft_qwen3_5_4b_cpt_half_lr_stage2 HF_REPO=your-name/qwen3.5-4b-cpt-half-lr-it
make push-to-hub config=sft_gemma4_cpt_stage2 HF_REPO=your-name/gemma4-cpt-stage2-it
```

## Notes

- Loss is computed on completion tokens only (`completion_mask` + `completion_only_loss=true`).
- EOS is appended to target by preprocessing.
- Runtime packing is disabled by default for SFT stability.
