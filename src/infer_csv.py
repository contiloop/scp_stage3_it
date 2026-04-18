"""
Run CSV-based generation for locally trained Stage3 IT models.

Example:
    python -m src.infer_csv \
      --config-name sft_qwen3_4b_cpt_stage2 \
      --config-name sft_qwen3_5_4b_cpt_half_lr_stage2 \
      --input-csv /abs/path/eval_120.csv \
      --output-dir artifacts/eval120_it_predictions
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import re
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from .common import resolve_workspace_path, suppress_noisy_library_logs

DEFAULT_LANG_NAME_MAP = {
    "en": "English",
    "ko": "Korean",
    "ja": "Japanese",
    "zh": "Chinese",
}

DEFAULT_LANG_LOCALE_MAP = {
    "en": "en-US",
    "ko": "ko-KR",
    "ja": "ja-JP",
    "zh": "zh-CN",
}


def compose_cfg(config_path: str, config_name: str) -> DictConfig:
    cfg_dir = resolve_workspace_path(config_path)
    with initialize_config_dir(version_base=None, config_dir=str(cfg_dir)):
        cfg = compose(config_name=config_name, overrides=[])
    OmegaConf.resolve(cfg)
    return cfg


def free_vram() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_model_bundle(
    *,
    model_source: str | Path,
    base_model: str,
    max_seq_length: int,
    cfg: DictConfig,
):
    from .evaluate import _load_with_unsloth, load_model_for_eval

    if isinstance(model_source, Path):
        return load_model_for_eval(
            model_path=model_source,
            base_model=base_model,
            max_seq_length=max_seq_length,
            cfg=cfg,
        )

    model, tokenizer, mode = _load_with_unsloth(
        str(model_source),
        max_seq_length=max_seq_length,
        model_hint=base_model,
        preferred_backend=str(cfg.model.get("backend", "auto")),
        local_files_only=bool(cfg.model.get("local_files_only", False)),
    )
    return model, tokenizer, mode


def slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")


def resolve_model_path(output_dir: Path, checkpoint: str) -> Path:
    checkpoint_text = str(checkpoint).strip().lower()
    if checkpoint_text in {"", "final"}:
        return output_dir

    if checkpoint_text == "latest":
        candidates = [path for path in output_dir.glob("checkpoint-*") if path.is_dir()]
        if not candidates:
            raise FileNotFoundError(f"No checkpoint-* directories found in: {output_dir}")
        return max(
            candidates,
            key=lambda path: int(path.name.split("-")[1]) if "-" in path.name else 0,
        )

    candidate = output_dir / checkpoint
    if not candidate.exists():
        raise FileNotFoundError(f"Checkpoint not found: {candidate}")
    return candidate


def resolve_model_source(
    *,
    cfg: DictConfig,
    checkpoint: str,
    hf_repo: str | None,
) -> tuple[str | Path, str]:
    if hf_repo is not None:
        repo_id = str(hf_repo).strip()
        if not repo_id:
            raise ValueError("Received an empty --hf-repo value.")
        checkpoint_text = str(checkpoint).strip().lower()
        if checkpoint_text not in {"", "final"}:
            raise ValueError(
                "When --hf-repo is used, only --checkpoint final is currently supported."
            )
        return repo_id, repo_id.split("/")[-1]

    training_output_dir = resolve_workspace_path(cfg.training.output_dir)
    model_path = resolve_model_path(training_output_dir, checkpoint=checkpoint)
    return model_path, str(model_path)


def load_eval_rows(input_csv: Path) -> list[dict[str, str]]:
    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        if "source_text" not in fieldnames:
            raise ValueError(f"Missing required column 'source_text'. available={sorted(fieldnames)}")

        if "item_id" in fieldnames:
            item_column = "item_id"
        elif "item_number" in fieldnames:
            item_column = "item_number"
        else:
            raise ValueError("Missing item id column. Need one of ['item_id', 'item_number'].")

        rows: list[dict[str, str]] = []
        for row in reader:
            item_id = str(row.get(item_column, "") or "").strip()
            source_text = str(row.get("source_text", "") or "")
            if not item_id:
                continue
            rows.append({"item_id": item_id, "source_text": source_text})

    if not rows:
        raise RuntimeError(f"No usable rows found in: {input_csv}")
    return rows


def resolve_lang_maps(cfg: DictConfig) -> tuple[dict[str, str], dict[str, str]]:
    sft_cfg = cfg.preprocessing.get("sft")
    if sft_cfg is None:
        return dict(DEFAULT_LANG_NAME_MAP), dict(DEFAULT_LANG_LOCALE_MAP)

    lang_name_map = dict(DEFAULT_LANG_NAME_MAP)
    lang_locale_map = dict(DEFAULT_LANG_LOCALE_MAP)

    custom_name_map = sft_cfg.get("lang_name_map")
    if custom_name_map:
        for key, value in OmegaConf.to_container(custom_name_map, resolve=True).items():
            lang_name_map[str(key).strip().lower()] = str(value)

    custom_locale_map = sft_cfg.get("lang_locale_map")
    if custom_locale_map:
        for key, value in OmegaConf.to_container(custom_locale_map, resolve=True).items():
            lang_locale_map[str(key).strip().lower()] = str(value)

    return lang_name_map, lang_locale_map


def resolve_prompt_templates(cfg: DictConfig, prompt_override: str | None) -> list[str]:
    if prompt_override is not None:
        return [str(prompt_override)]

    prompts_cfg = cfg.get("prompts")
    if prompts_cfg is None or not prompts_cfg.get("templates"):
        raise RuntimeError("Config must include prompts.templates for IT inference.")

    return [str(template) for template in prompts_cfg.templates]


def stable_template_index(sample_key: str, template_count: int, seed: int) -> int:
    if template_count <= 0:
        raise ValueError("template_count must be positive")

    key = f"{seed}|{sample_key}".encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    value = int.from_bytes(digest, byteorder="big", signed=False)
    return value % template_count


def resolve_response_prefix(cfg: DictConfig, response_prefix_override: str | None) -> str:
    if response_prefix_override is not None:
        return str(response_prefix_override)
    sft_cfg = cfg.preprocessing.get("sft")
    if sft_cfg is None:
        return "response: "
    return str(sft_cfg.get("response_prefix", "response: "))


def render_prompt(
    source_text: str,
    prompt_template: str,
    response_prefix: str,
    src_lang_name: str,
    tgt_lang_name: str,
    src_locale: str,
    tgt_locale: str,
) -> str:
    prompt = prompt_template.format(
        src_lang_name=src_lang_name,
        tgt_lang_name=tgt_lang_name,
        src_locale=src_locale,
        tgt_locale=tgt_locale,
        src=source_text,
    )
    return f"{prompt}\n{response_prefix}"


def resolve_template_for_row(
    *,
    prompt_templates: list[str],
    item_id: str,
    template_seed: int,
    fixed_template_index: int | None,
) -> tuple[str, int]:
    if not prompt_templates:
        raise ValueError("prompt_templates must not be empty")

    if fixed_template_index is not None:
        if fixed_template_index < 0 or fixed_template_index >= len(prompt_templates):
            raise ValueError(
                f"template_index={fixed_template_index} is out of range for "
                f"{len(prompt_templates)} prompt templates."
            )
        return prompt_templates[fixed_template_index], fixed_template_index

    template_index = stable_template_index(
        sample_key=item_id,
        template_count=len(prompt_templates),
        seed=template_seed,
    )
    return prompt_templates[template_index], template_index


def get_context_limit(model: Any, tokenizer: Any) -> int | None:
    model_limit = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    if isinstance(model_limit, int) and model_limit > 0:
        return model_limit

    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 1_000_000:
        return tokenizer_limit
    return None


def build_generation_kwargs(
    tokenizer: Any,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
        "num_beams": 1,
    }
    if do_sample:
        kwargs["temperature"] = float(temperature)
    return kwargs


def clean_hypothesis(text: str, response_prefix: str) -> str:
    out = (text or "").strip()
    removable_prefixes = [
        response_prefix.strip(),
        "<KO>",
        "<ko>",
        "번역:",
        "Translation:",
    ]
    for prefix in removable_prefixes:
        if prefix and out.startswith(prefix):
            out = out[len(prefix) :].strip()
    return out


def generate_single(
    *,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    item_id: str,
    source_text: str,
    prompt: str,
    context_limit: int | None,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    response_prefix: str,
) -> dict[str, str]:
    try:
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        effective_max_new = int(max_new_tokens)
        if context_limit is not None:
            prompt_len = int(attention_mask.sum().item())
            allowed = context_limit - prompt_len
            if allowed <= 0:
                raise RuntimeError(
                    f"input exceeds context limit: input={prompt_len} limit={context_limit}"
                )
            effective_max_new = min(effective_max_new, allowed)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **build_generation_kwargs(
                    tokenizer=tokenizer,
                    max_new_tokens=effective_max_new,
                    do_sample=do_sample,
                    temperature=temperature,
                ),
            )

        prompt_len = int(attention_mask.sum().item())
        gen_ids = output_ids[0][prompt_len:]
        hypothesis = clean_hypothesis(
            tokenizer.decode(gen_ids, skip_special_tokens=True),
            response_prefix=response_prefix,
        )
        return {
            "item_id": item_id,
            "source_text": source_text,
            "hypothesis": hypothesis,
            "error": "",
        }
    except Exception as exc:
        return {
            "item_id": item_id,
            "source_text": source_text,
            "hypothesis": "",
            "error": f"{type(exc).__name__}: {exc}",
        }


def generate_rows(
    *,
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, str]],
    prompt_templates: list[str],
    response_prefix: str,
    src_lang_name: str,
    tgt_lang_name: str,
    src_locale: str,
    tgt_locale: str,
    template_seed: int,
    fixed_template_index: int | None,
    batch_size: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    sort_by_input_length: bool,
) -> list[dict[str, str]]:
    device = next(model.parameters()).device
    context_limit = get_context_limit(model, tokenizer)

    indexed_rows = [
        (
            lambda resolved_template, resolved_index: {
                "_orig_pos": index,
                "_template_index": resolved_index,
                "item_id": row["item_id"],
                "source_text": row["source_text"],
                "prompt": render_prompt(
                    source_text=row["source_text"],
                    prompt_template=resolved_template,
                    response_prefix=response_prefix,
                    src_lang_name=src_lang_name,
                    tgt_lang_name=tgt_lang_name,
                    src_locale=src_locale,
                    tgt_locale=tgt_locale,
                ),
            }
        )(
            *resolve_template_for_row(
                prompt_templates=prompt_templates,
                item_id=row["item_id"],
                template_seed=template_seed,
                fixed_template_index=fixed_template_index,
            )
        )
        for index, row in enumerate(rows)
    ]

    if sort_by_input_length:
        indexed_rows.sort(key=lambda row: (len(row["source_text"]), row["_orig_pos"]))

    results: list[dict[str, str]] = []
    total = len(indexed_rows)
    progress = tqdm(range(0, total, max(1, batch_size)), desc="generate", leave=False)
    for start in progress:
        chunk = indexed_rows[start : start + max(1, batch_size)]
        prompts = [row["prompt"] for row in chunk]
        item_ids = [row["item_id"] for row in chunk]
        source_texts = [row["source_text"] for row in chunk]
        original_positions = [row["_orig_pos"] for row in chunk]
        template_indices = [row["_template_index"] for row in chunk]

        try:
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                add_special_tokens=False,
                padding=True,
                truncation=False,
                pad_to_multiple_of=8,
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            effective_max_new = int(max_new_tokens)
            if context_limit is not None:
                prompt_lengths = attention_mask.sum(dim=1)
                allowed = int((context_limit - prompt_lengths).min().item())
                if allowed <= 0:
                    raise RuntimeError(
                        "input exceeds context limit in batch: "
                        f"max_prompt={int(prompt_lengths.max().item())} limit={context_limit}"
                    )
                effective_max_new = min(effective_max_new, allowed)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **build_generation_kwargs(
                        tokenizer=tokenizer,
                        max_new_tokens=effective_max_new,
                        do_sample=do_sample,
                        temperature=temperature,
                    ),
                )

            for index, item_id in enumerate(item_ids):
                prompt_len = int(attention_mask[index].sum().item())
                gen_ids = output_ids[index][prompt_len:]
                hypothesis = clean_hypothesis(
                    tokenizer.decode(gen_ids, skip_special_tokens=True),
                    response_prefix=response_prefix,
                )
                results.append(
                    {
                        "_orig_pos": original_positions[index],
                        "template_index": str(template_indices[index]),
                        "item_id": item_id,
                        "source_text": source_texts[index],
                        "hypothesis": hypothesis,
                        "error": "",
                    }
                )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and len(chunk) > 1:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                for row in chunk:
                    single = generate_single(
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        item_id=row["item_id"],
                        source_text=row["source_text"],
                        prompt=row["prompt"],
                        context_limit=context_limit,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        temperature=temperature,
                        response_prefix=response_prefix,
                    )
                    single["_orig_pos"] = row["_orig_pos"]
                    single["template_index"] = str(row["_template_index"])
                    results.append(single)
            else:
                error_text = f"{type(exc).__name__}: {exc}"
                for row in chunk:
                    results.append(
                        {
                            "_orig_pos": row["_orig_pos"],
                            "template_index": str(row["_template_index"]),
                            "item_id": row["item_id"],
                            "source_text": row["source_text"],
                            "hypothesis": "",
                            "error": error_text,
                        }
                    )
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            for row in chunk:
                results.append(
                    {
                        "_orig_pos": row["_orig_pos"],
                        "template_index": str(row["_template_index"]),
                        "item_id": row["item_id"],
                        "source_text": row["source_text"],
                        "hypothesis": "",
                        "error": error_text,
                    }
                )

    results.sort(key=lambda row: int(row.get("_orig_pos", 0)))
    for row in results:
        row.pop("_orig_pos", None)
    return results


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage3 IT inference on a CSV file.")
    parser.add_argument(
        "--config-path",
        default="configs",
        help="Hydra config directory path relative to the repo root.",
    )
    parser.add_argument(
        "--config-name",
        action="append",
        required=True,
        help="Repeatable stage3 config name, e.g. sft_qwen3_4b_cpt_stage2.",
    )
    parser.add_argument("--input-csv", required=True, help="CSV path with source_text and item_id/item_number.")
    parser.add_argument(
        "--hf-repo",
        action="append",
        default=None,
        help=(
            "Optional HF repo IDs aligned with --config-name order. "
            "When provided, the corresponding model is loaded from Hub instead of local training.output_dir."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/csv_inference",
        help="Output directory for per-model and merged CSV files.",
    )
    parser.add_argument(
        "--checkpoint",
        default="final",
        help="final | latest | checkpoint-XXXX under each config's training.output_dir.",
    )
    parser.add_argument(
        "--template-index",
        type=int,
        default=None,
        help=(
            "Optional fixed prompt template index. "
            "If omitted, templates are assigned per row with the same stable-hash policy used in training."
        ),
    )
    parser.add_argument(
        "--prompt-template",
        default=None,
        help="Optional raw prompt template override. Must include {src}.",
    )
    parser.add_argument(
        "--response-prefix",
        default=None,
        help="Optional response prefix override. Defaults to preprocessing.sft.response_prefix.",
    )
    parser.add_argument("--source-lang-iso", default="en")
    parser.add_argument("--target-lang-iso", default="ko")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--no-sort-by-input-length", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suppress_noisy_library_logs()

    hf_repos = list(args.hf_repo or [])
    if hf_repos and len(hf_repos) != len(args.config_name):
        raise ValueError(
            "--hf-repo must be provided either zero times or exactly once per --config-name."
        )

    input_csv = resolve_workspace_path(args.input_csv)
    output_dir = resolve_workspace_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_eval_rows(input_csv)
    print(f"input_csv: {input_csv}")
    print(f"rows: {len(rows)}")
    print(f"output_dir: {output_dir}")

    merged_rows: list[dict[str, str]] = []
    seen_labels: set[str] = set()

    for config_index, config_name in enumerate(args.config_name):
        cfg = compose_cfg(config_path=args.config_path, config_name=config_name)
        hf_repo = hf_repos[config_index] if hf_repos else None

        model_source, source_label = resolve_model_source(
            cfg=cfg,
            checkpoint=args.checkpoint,
            hf_repo=hf_repo,
        )
        model_label = slugify(source_label if hf_repo else str(cfg.experiment.get("name", config_name)) or config_name)
        if model_label in seen_labels:
            model_label = slugify(f"{model_label}_{config_name}")
        seen_labels.add(model_label)

        print("=" * 80)
        print(f"config_name: {config_name}")
        print(f"model_source: {model_source}")
        print("=" * 80)

        prompt_templates = resolve_prompt_templates(cfg=cfg, prompt_override=args.prompt_template)
        response_prefix = resolve_response_prefix(
            cfg=cfg,
            response_prefix_override=args.response_prefix,
        )
        template_seed = int(cfg.preprocessing.sft.get("template_seed", cfg.seed))

        lang_name_map, lang_locale_map = resolve_lang_maps(cfg)
        source_lang_iso = str(args.source_lang_iso).strip().lower()
        target_lang_iso = str(args.target_lang_iso).strip().lower()
        src_lang_name = lang_name_map.get(source_lang_iso, source_lang_iso.upper())
        tgt_lang_name = lang_name_map.get(target_lang_iso, target_lang_iso.upper())
        src_locale = lang_locale_map.get(source_lang_iso, source_lang_iso)
        tgt_locale = lang_locale_map.get(target_lang_iso, target_lang_iso)

        model = tokenizer = None
        try:
            model, tokenizer, loader_mode = load_model_bundle(
                model_source=model_source,
                base_model=str(cfg.model.pretrained_model_name_or_path),
                max_seq_length=int(cfg.model.max_seq_length),
                cfg=cfg,
            )
            print(f"loader: {loader_mode}")

            base_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
            if base_tokenizer.pad_token_id is None and base_tokenizer.eos_token_id is not None:
                base_tokenizer.pad_token = base_tokenizer.eos_token
            base_tokenizer.padding_side = "left"
            if hasattr(model, "generation_config"):
                model.generation_config.use_cache = True

            model_rows = generate_rows(
                model=model,
                tokenizer=base_tokenizer,
                rows=rows,
                prompt_templates=prompt_templates,
                response_prefix=response_prefix,
                src_lang_name=src_lang_name,
                tgt_lang_name=tgt_lang_name,
                src_locale=src_locale,
                tgt_locale=tgt_locale,
                template_seed=template_seed,
                fixed_template_index=args.template_index,
                batch_size=max(1, int(args.batch_size)),
                max_new_tokens=max(1, int(args.max_new_tokens)),
                do_sample=bool(args.do_sample),
                temperature=float(args.temperature),
                sort_by_input_length=not bool(args.no_sort_by_input_length),
            )

            per_model_path = output_dir / f"{model_label}__{slugify(config_name)}.csv"
            write_csv(
                path=per_model_path,
                rows=model_rows,
                fieldnames=["item_id", "source_text", "template_index", "hypothesis", "error"],
            )
            print(f"saved: {per_model_path}")

            merged_rows.extend(
                [
                    {
                        "model_label": model_label,
                        "config_name": config_name,
                        "item_id": row["item_id"],
                        "source_text": row["source_text"],
                        "template_index": row.get("template_index", ""),
                        "hypothesis": row["hypothesis"],
                        "error": row["error"],
                    }
                    for row in model_rows
                ]
            )
        finally:
            del model
            del tokenizer
            free_vram()

    merged_path = output_dir / "merged_predictions.csv"
    write_csv(
        path=merged_path,
        rows=merged_rows,
        fieldnames=[
            "model_label",
            "config_name",
            "item_id",
            "source_text",
            "template_index",
            "hypothesis",
            "error",
        ],
    )
    print(f"saved: {merged_path}")


if __name__ == "__main__":
    main()
