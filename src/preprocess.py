"""
Hydra-based SFT translation preprocessing.

Builds translation SFT records from parallel datasets with:
- dynamic prompt rendering based on source/target language per row
- response-only supervision metadata via `completion_mask`
- optional overlength drop policy
- train/validation save_to_disk outputs
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import hydra
from datasets import Dataset, IterableDataset, load_dataset
from datasets.exceptions import DatasetGenerationCastError
from omegaconf import DictConfig, OmegaConf, open_dict
from transformers import AutoTokenizer

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


def resolve_dataset_cfgs(cfg: DictConfig) -> list[DictConfig]:
    datasets = cfg.data.get("datasets")
    if datasets is not None:
        return [dataset_cfg for dataset_cfg in datasets]

    dataset = cfg.data.get("dataset")
    if dataset is not None:
        return [dataset]

    raise RuntimeError("Config must include either data.dataset or data.datasets.")


def resolve_train_dataset_cfgs(cfg: DictConfig) -> list[DictConfig]:
    train_datasets = cfg.data.get("train_datasets")
    if train_datasets is not None:
        return [dataset_cfg for dataset_cfg in train_datasets]
    return resolve_dataset_cfgs(cfg)


def resolve_validation_dataset_cfgs(cfg: DictConfig) -> list[DictConfig]:
    validation_datasets = cfg.data.get("validation_datasets")
    if validation_datasets is not None:
        return [dataset_cfg for dataset_cfg in validation_datasets]
    return []


def resolve_local_dataset_snapshot(repo_id: str) -> Path | None:
    """
    Return latest local HF dataset snapshot path if available.
    """
    repo_cache = repo_id.replace("/", "--")
    snapshots_dir = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / f"datasets--{repo_cache}"
        / "snapshots"
    )
    if not snapshots_dir.exists():
        return None

    snapshots = [p for p in snapshots_dir.iterdir() if p.is_dir()]
    if not snapshots:
        return None
    return max(snapshots, key=lambda p: p.stat().st_mtime)


def iter_rows_from_local_snapshot(dataset_cfg: DictConfig):
    """
    Iterate rows from local snapshot JSONL files first.
    Returns None if local snapshot is unavailable or unusable.
    """
    repo_id = str(dataset_cfg.path)
    snapshot = resolve_local_dataset_snapshot(repo_id)
    if snapshot is None:
        return None

    jsonl_files = sorted(snapshot.rglob("*.jsonl"))
    if not jsonl_files:
        return None

    max_rows = dataset_cfg.get("max_rows")
    max_rows = int(max_rows) if max_rows is not None else None
    emitted = 0

    print(
        f"[INFO] Loading dataset from local snapshot: {snapshot} "
        f"({len(jsonl_files)} jsonl files)"
    )

    for data_file in jsonl_files:
        ds = load_dataset("json", data_files=str(data_file), split="train")
        for row in ds:
            if max_rows is not None and emitted >= max_rows:
                return
            yield emitted, row
            emitted += 1


def normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def iter_dataset_rows(dataset_cfg: DictConfig):
    if bool(dataset_cfg.get("prefer_local_snapshot", True)):
        local_iter = iter_rows_from_local_snapshot(dataset_cfg)
        if local_iter is not None:
            yield from local_iter
            return

    load_kwargs = {
        "path": dataset_cfg.path,
        "name": dataset_cfg.name,
        "split": dataset_cfg.split,
        "streaming": bool(dataset_cfg.streaming),
        "trust_remote_code": bool(dataset_cfg.trust_remote_code),
    }

    try:
        dataset = load_dataset(**load_kwargs)
    except DatasetGenerationCastError:
        if bool(dataset_cfg.streaming):
            raise
        # Some JSONL datasets on HF include mixed schemas across files.
        # Fallback to streaming mode, which avoids strict Arrow schema casting.
        print(
            "[WARN] Dataset schema cast failed with non-streaming load. "
            "Retrying with streaming=True."
        )
        dataset = load_dataset(
            path=dataset_cfg.path,
            name=dataset_cfg.name,
            split=dataset_cfg.split,
            streaming=True,
            trust_remote_code=bool(dataset_cfg.trust_remote_code),
        )

    max_rows = dataset_cfg.get("max_rows")
    if max_rows is not None:
        max_rows = int(max_rows)

    if isinstance(dataset, IterableDataset):
        for idx, row in enumerate(dataset):
            if max_rows is not None and idx >= max_rows:
                break
            yield idx, row
        return

    total = len(dataset)
    end = min(total, max_rows) if max_rows is not None else total
    for idx in range(end):
        yield idx, dataset[idx]


def resolve_prompt_templates(cfg: DictConfig) -> list[str]:
    prompts_cfg = cfg.get("prompts")
    if prompts_cfg is None:
        raise RuntimeError("SFT mode requires `prompts.templates` in config.")

    templates = prompts_cfg.get("templates")
    if not templates:
        raise RuntimeError("SFT mode requires a non-empty `prompts.templates` list.")

    return [str(template) for template in templates]


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


def stable_template_index(sample_key: str, template_count: int, seed: int) -> int:
    if template_count <= 0:
        raise ValueError("template_count must be positive")

    key = f"{seed}|{sample_key}".encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    value = int.from_bytes(digest, byteorder="big", signed=False)
    return value % template_count


def parse_pair_fallback(pair_value: Any) -> tuple[str | None, str | None]:
    if pair_value is None:
        return None, None
    pair_text = str(pair_value).strip().lower()
    if "->" not in pair_text:
        return None, None
    src, tgt = pair_text.split("->", 1)
    src = src.strip()
    tgt = tgt.strip()
    return (src or None), (tgt or None)


def resolve_lang_descriptors(
    src_lang_iso: str | None,
    tgt_lang_iso: str | None,
    lang_name_map: dict[str, str],
    lang_locale_map: dict[str, str],
) -> tuple[str, str, str, str]:
    src_iso = str(src_lang_iso or "source").strip().lower()
    tgt_iso = str(tgt_lang_iso or "target").strip().lower()

    src_lang_name = lang_name_map.get(src_iso, src_iso.upper())
    tgt_lang_name = lang_name_map.get(tgt_iso, tgt_iso.upper())
    src_locale = lang_locale_map.get(src_iso, src_iso)
    tgt_locale = lang_locale_map.get(tgt_iso, tgt_iso)

    return src_lang_name, tgt_lang_name, src_locale, tgt_locale


def build_sft_translation_records(
    cfg: DictConfig,
    tokenizer,
    dataset_cfgs: list[DictConfig],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise RuntimeError("Tokenizer has no eos_token_id")

    sft_cfg = cfg.preprocessing.get("sft")
    if sft_cfg is None:
        raise RuntimeError("SFT mode requires preprocessing.sft config.")

    max_length = int(sft_cfg.get("max_length", cfg.model.max_seq_length))
    if max_length <= 0:
        raise ValueError("preprocessing.sft.max_length must be positive.")

    response_prefix = str(sft_cfg.get("response_prefix", "response: "))
    response_prefix_with_newline = f"\n{response_prefix}"
    drop_overlength = bool(sft_cfg.get("drop_overlength", True))
    add_eos_to_target = bool(sft_cfg.get("add_eos_to_target", True))
    template_seed = int(sft_cfg.get("template_seed", cfg.seed))

    source_text_field = str(sft_cfg.get("source_text_field", "source_text"))
    target_text_field = str(sft_cfg.get("target_text_field", "target_text"))
    source_lang_field = str(sft_cfg.get("source_lang_field", "source_lang_iso"))
    target_lang_field = str(sft_cfg.get("target_lang_field", "target_lang_iso"))
    pair_field = str(sft_cfg.get("pair_field", "pair"))

    templates = resolve_prompt_templates(cfg)
    lang_name_map, lang_locale_map = resolve_lang_maps(cfg)

    min_chars = int(cfg.preprocessing.cleaning.min_chars)
    id_column_default = cfg.data.get("id_column")
    source_column_default = cfg.data.get("source_column")

    records: list[dict[str, Any]] = []
    template_counts = [0 for _ in templates]
    rows_seen = 0
    rows_kept = 0
    skipped_missing_fields = 0
    skipped_short = 0
    skipped_overlength = 0

    for dataset_idx, dataset_cfg in enumerate(dataset_cfgs):
        id_column = dataset_cfg.get("id_column", id_column_default)
        source_column = dataset_cfg.get("source_column", source_column_default)
        default_source = str(dataset_cfg.path)

        for row_idx, row in iter_dataset_rows(dataset_cfg):
            rows_seen += 1

            source_value = row.get(source_text_field)
            target_value = row.get(target_text_field)
            if source_value is None or target_value is None:
                skipped_missing_fields += 1
                continue

            source_text = normalize_text(str(source_value))
            target_text = normalize_text(str(target_value))
            if not source_text or not target_text:
                skipped_missing_fields += 1
                continue

            if len(source_text) < min_chars or len(target_text) < min_chars:
                skipped_short += 1
                continue

            src_lang_iso = row.get(source_lang_field)
            tgt_lang_iso = row.get(target_lang_field)
            if src_lang_iso is None or tgt_lang_iso is None:
                src_from_pair, tgt_from_pair = parse_pair_fallback(row.get(pair_field))
                src_lang_iso = src_lang_iso if src_lang_iso is not None else src_from_pair
                tgt_lang_iso = tgt_lang_iso if tgt_lang_iso is not None else tgt_from_pair

            src_lang_name, tgt_lang_name, src_locale, tgt_locale = resolve_lang_descriptors(
                src_lang_iso=src_lang_iso,
                tgt_lang_iso=tgt_lang_iso,
                lang_name_map=lang_name_map,
                lang_locale_map=lang_locale_map,
            )

            sample_id = (
                str(row.get(id_column, f"row:{dataset_idx}:{row_idx}"))
                if id_column
                else f"row:{dataset_idx}:{row_idx}"
            )
            source_name = str(row.get(source_column, default_source)) if source_column else default_source

            template_idx = stable_template_index(
                sample_key=f"{dataset_idx}:{sample_id}",
                template_count=len(templates),
                seed=template_seed,
            )
            template_counts[template_idx] += 1

            prompt_template = templates[template_idx]
            prompt = prompt_template.format(
                src_lang_name=src_lang_name,
                tgt_lang_name=tgt_lang_name,
                src_locale=src_locale,
                tgt_locale=tgt_locale,
                src=source_text,
            )

            prompt_prefix = f"{prompt}{response_prefix_with_newline}"
            prompt_prefix_ids = tokenizer.encode(prompt_prefix, add_special_tokens=False)
            target_ids = tokenizer.encode(target_text, add_special_tokens=False)

            input_ids = list(prompt_prefix_ids)
            completion_mask = [0] * len(prompt_prefix_ids)
            input_ids.extend(target_ids)
            completion_mask.extend([1] * len(target_ids))

            if add_eos_to_target and (not input_ids or input_ids[-1] != eos_id):
                input_ids.append(eos_id)
                completion_mask.append(1)

            if len(input_ids) > max_length:
                if drop_overlength:
                    skipped_overlength += 1
                    continue
                input_ids = input_ids[:max_length]
                completion_mask = completion_mask[:max_length]

            if 1 not in completion_mask:
                skipped_missing_fields += 1
                continue

            rows_kept += 1
            records.append(
                {
                    "sample_id": sample_id,
                    "source": source_name,
                    "pair": str(row.get(pair_field, "")),
                    "source_lang_iso": str(src_lang_iso or "").lower(),
                    "target_lang_iso": str(tgt_lang_iso or "").lower(),
                    "prompt_template_id": template_idx + 1,
                    "prompt": prompt,
                    "response_prefix": response_prefix,
                    "target_text": target_text,
                    "text": f"{prompt_prefix}{target_text}",
                    "input_ids": input_ids,
                    "completion_mask": completion_mask,
                    "seq_lengths": [len(input_ids)],
                }
            )

    stats = {
        "rows_seen": rows_seen,
        "rows_kept": rows_kept,
        "rows_skipped_missing_fields": skipped_missing_fields,
        "rows_skipped_short": skipped_short,
        "rows_skipped_overlength": skipped_overlength,
        "records_total": len(records),
        "template_counts": {f"prompt_{idx + 1}": count for idx, count in enumerate(template_counts)},
        "max_length": max_length,
        "drop_overlength": drop_overlength,
        "response_prefix": response_prefix,
    }
    return records, stats


def log_split_drop_summary(split_name: str, stats: dict[str, Any]) -> None:
    skipped_missing = int(stats.get("rows_skipped_missing_fields", 0))
    skipped_short = int(stats.get("rows_skipped_short", 0))
    skipped_overlength = int(stats.get("rows_skipped_overlength", 0))
    skipped_total = skipped_missing + skipped_short + skipped_overlength
    print(
        f"[{split_name}] drop_detected={'yes' if skipped_total > 0 else 'no'} "
        f"(overlength_drop={'yes' if skipped_overlength > 0 else 'no'})"
    )


def build_sft_records_with_snapshot_fallback(
    cfg: DictConfig,
    tokenizer,
    dataset_cfgs: list[DictConfig],
    split_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records, stats = build_sft_translation_records(
        cfg=cfg,
        tokenizer=tokenizer,
        dataset_cfgs=dataset_cfgs,
    )
    if records:
        return records, stats

    if not any(bool(ds.get("prefer_local_snapshot", True)) for ds in dataset_cfgs):
        return records, stats

    print(
        f"[WARN] No SFT records built from local snapshot path for {split_name}. "
        "Retrying with HF Hub streaming."
    )
    with open_dict(cfg):
        for dataset_cfg in dataset_cfgs:
            dataset_cfg.prefer_local_snapshot = False
            dataset_cfg.streaming = True

    return build_sft_translation_records(
        cfg=cfg,
        tokenizer=tokenizer,
        dataset_cfgs=dataset_cfgs,
    )


def split_dataset(dataset: Dataset, cfg: DictConfig) -> tuple[Dataset, Dataset | None]:
    if len(dataset) < 2:
        return dataset, None

    val_ratio = float(cfg.preprocessing.split.val_ratio)
    if val_ratio <= 0.0:
        return dataset, None

    val_size = max(1, int(len(dataset) * val_ratio))
    if val_size >= len(dataset):
        val_size = len(dataset) - 1

    split = dataset.train_test_split(
        test_size=val_size,
        seed=int(cfg.preprocessing.split.seed),
        shuffle=True,
    )
    return split["train"], split["test"]


def prepare_save_path(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"{path} already exists. Set preprocessing.overwrite_output=true to replace it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def run_sft_translation_preprocess(cfg: DictConfig, tokenizer) -> None:
    train_dataset_cfgs = resolve_train_dataset_cfgs(cfg)
    val_dataset_cfgs = resolve_validation_dataset_cfgs(cfg)

    prompt_templates = resolve_prompt_templates(cfg)

    train_records, train_stats = build_sft_records_with_snapshot_fallback(
        cfg=cfg,
        tokenizer=tokenizer,
        dataset_cfgs=train_dataset_cfgs,
        split_name="train",
    )
    if not train_records:
        raise RuntimeError(
            "No SFT train records were built. "
            f"rows_seen={train_stats.get('rows_seen', 0)}, "
            f"rows_skipped_missing_fields={train_stats.get('rows_skipped_missing_fields', 0)}, "
            f"rows_skipped_short={train_stats.get('rows_skipped_short', 0)}, "
            f"rows_skipped_overlength={train_stats.get('rows_skipped_overlength', 0)}."
        )

    unpacked_train_ds = Dataset.from_list(train_records)
    train_ds = unpacked_train_ds

    unpacked_val_ds: Dataset | None = None
    val_ds: Dataset | None = None
    val_stats: dict[str, Any] | None = None

    if val_dataset_cfgs:
        val_records, val_stats = build_sft_records_with_snapshot_fallback(
            cfg=cfg,
            tokenizer=tokenizer,
            dataset_cfgs=val_dataset_cfgs,
            split_name="validation",
        )
        if not val_records:
            raise RuntimeError(
                "validation_datasets are configured but no SFT validation records were built. "
                f"rows_seen={val_stats.get('rows_seen', 0) if val_stats else 0}, "
                f"rows_skipped_missing_fields={val_stats.get('rows_skipped_missing_fields', 0) if val_stats else 0}, "
                f"rows_skipped_short={val_stats.get('rows_skipped_short', 0) if val_stats else 0}, "
                f"rows_skipped_overlength={val_stats.get('rows_skipped_overlength', 0) if val_stats else 0}."
            )
        unpacked_val_ds = Dataset.from_list(val_records)
        val_ds = unpacked_val_ds
    else:
        train_ds, val_ds = split_dataset(train_ds, cfg)

    output_dir = resolve_workspace_path(cfg.preprocessing.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    meta_path = output_dir / "metadata.json"

    overwrite = bool(cfg.preprocessing.overwrite_output)
    prepare_save_path(train_dir, overwrite=overwrite)
    train_ds.save_to_disk(str(train_dir))

    if val_ds is not None:
        prepare_save_path(val_dir, overwrite=overwrite)
        val_ds.save_to_disk(str(val_dir))
    elif overwrite and val_dir.exists():
        shutil.rmtree(val_dir)

    train_source_paths = [str(dataset_cfg.path) for dataset_cfg in train_dataset_cfgs]
    train_source_splits = [str(dataset_cfg.split) for dataset_cfg in train_dataset_cfgs]

    metadata = {
        "task": "sft_translation",
        "source": train_source_paths[0] if len(train_source_paths) == 1 else train_source_paths,
        "split": train_source_splits[0] if len(train_source_splits) == 1 else train_source_splits,
        "validation_mode": "explicit_datasets" if val_dataset_cfgs else "random_split",
        "prompts": {
            "count": len(prompt_templates),
            "templates": prompt_templates,
        },
        "sft": {
            "source_text_field": str(cfg.preprocessing.sft.get("source_text_field", "source_text")),
            "target_text_field": str(cfg.preprocessing.sft.get("target_text_field", "target_text")),
            "source_lang_field": str(cfg.preprocessing.sft.get("source_lang_field", "source_lang_iso")),
            "target_lang_field": str(cfg.preprocessing.sft.get("target_lang_field", "target_lang_iso")),
            "pair_field": str(cfg.preprocessing.sft.get("pair_field", "pair")),
            "response_prefix": str(cfg.preprocessing.sft.get("response_prefix", "response: ")),
            "max_length": int(cfg.preprocessing.sft.get("max_length", cfg.model.max_seq_length)),
            "drop_overlength": bool(cfg.preprocessing.sft.get("drop_overlength", True)),
            "add_eos_to_target": bool(cfg.preprocessing.sft.get("add_eos_to_target", True)),
        },
        "stats": {
            "train_rows_seen": train_stats["rows_seen"],
            "train_rows_kept": train_stats["rows_kept"],
            "train_rows_skipped_missing_fields": train_stats["rows_skipped_missing_fields"],
            "train_rows_skipped_short": train_stats["rows_skipped_short"],
            "train_rows_skipped_overlength": train_stats["rows_skipped_overlength"],
            "train_records_total": train_stats["records_total"],
            "train_template_counts": train_stats["template_counts"],
            "unpacked_train_rows": len(unpacked_train_ds),
            "train_rows": len(train_ds),
            "unpacked_val_rows": len(unpacked_val_ds) if unpacked_val_ds is not None else 0,
            "val_rows": len(val_ds) if val_ds is not None else 0,
        },
        "config": OmegaConf.to_container(cfg, resolve=True),
    }

    if val_dataset_cfgs:
        val_source_paths = [str(dataset_cfg.path) for dataset_cfg in val_dataset_cfgs]
        val_source_splits = [str(dataset_cfg.split) for dataset_cfg in val_dataset_cfgs]
        metadata["validation_source"] = (
            val_source_paths[0] if len(val_source_paths) == 1 else val_source_paths
        )
        metadata["validation_split"] = (
            val_source_splits[0] if len(val_source_splits) == 1 else val_source_splits
        )

    if val_stats is not None:
        metadata["stats"]["validation_rows_seen"] = val_stats["rows_seen"]
        metadata["stats"]["validation_rows_kept"] = val_stats["rows_kept"]
        metadata["stats"]["validation_rows_skipped_missing_fields"] = val_stats[
            "rows_skipped_missing_fields"
        ]
        metadata["stats"]["validation_rows_skipped_short"] = val_stats["rows_skipped_short"]
        metadata["stats"]["validation_rows_skipped_overlength"] = val_stats[
            "rows_skipped_overlength"
        ]
        metadata["stats"]["validation_records_total"] = val_stats["records_total"]
        metadata["stats"]["validation_template_counts"] = val_stats["template_counts"]

    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 80)
    print("SFT Translation Preprocess Complete")
    print("=" * 80)
    print(f"unpacked train rows: {len(unpacked_train_ds):,}")
    if unpacked_val_ds is not None:
        print(f"unpacked val rows  : {len(unpacked_val_ds):,}")
    print(f"train rows   : {len(train_ds):,}")
    print(f"val rows     : {len(val_ds):,}" if val_ds is not None else "val rows     : 0")
    print("-" * 80)
    log_split_drop_summary("train", train_stats)
    if val_stats is not None:
        log_split_drop_summary("validation", val_stats)
    print(f"saved to     : {output_dir}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    suppress_noisy_library_logs()

    print("=" * 80)
    print("Preprocess Config")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg, resolve=True))

    preprocess_task = str(cfg.preprocessing.get("task", "sft_translation")).strip().lower()
    if preprocess_task != "sft_translation":
        raise ValueError(
            f"Unsupported preprocessing task in this module: {preprocess_task}. "
            "This preprocess.py now supports only `sft_translation`."
        )

    tokenizer_name = cfg.model.tokenizer_name_or_path or cfg.model.pretrained_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=bool(cfg.model.trust_remote_code),
        local_files_only=bool(cfg.model.get("local_files_only", False)),
    )
    if tokenizer.eos_token_id is None:
        raise RuntimeError(f"Tokenizer '{tokenizer_name}' has no eos token id")

    run_sft_translation_preprocess(cfg=cfg, tokenizer=tokenizer)


if __name__ == "__main__":
    main()
