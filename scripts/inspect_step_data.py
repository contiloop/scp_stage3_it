"""
Step 데이터 역산 스크립트 — 특정 training step에 투입된 데이터를 역추적한다.

재현 원리:
  1. 전처리된 train dataset을 load_from_disk으로 로드 (원본 레코드)
  2. TRL pack_dataset으로 동일한 packing 재현
  3. packed row ↔ 원본 record 역매핑 구축
  4. HuggingFace Trainer의 SeedableRandomSampler 셔플 순서 재현
     → torch.randperm(n, generator=Generator().manual_seed(seed + epoch))
  5. step 번호 → shuffled index → packed row → 원본 record 추적

사용법:
  python scripts/inspect_step_data.py --config full_96gb --steps 960
  python scripts/inspect_step_data.py --config full_96gb --steps 950-970 --output result.json
  python scripts/inspect_step_data.py --config full_96gb --steps 960 --anomaly-only
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset, load_from_disk
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from trl import pack_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect training data at a given step")
    p.add_argument("--config", required=True, help="Hydra config name (e.g. full_96gb)")
    p.add_argument(
        "--steps",
        required=True,
        help="Step(s) to inspect: single (960), range (950-970), or comma-separated (950,955,960)",
    )
    p.add_argument("--output", default=None, help="Optional JSON output path")
    p.add_argument("--no-packing", action="store_true", help="Skip packing (use original records)")
    p.add_argument("--anomaly-only", action="store_true", help="Only show flagged samples")
    p.add_argument("--verbose", action="store_true", help="Show full text (default: truncated)")
    p.add_argument(
        "--text-preview-len",
        type=int,
        default=200,
        help="Max chars for text preview (default: 200)",
    )
    return p.parse_args()


def parse_steps(steps_str: str) -> list[int]:
    """Parse step specification into sorted list of ints."""
    results: set[int] = set()
    for part in steps_str.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            results.update(range(int(lo), int(hi) + 1))
        else:
            results.add(int(part))
    return sorted(results)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(config_name: str) -> Any:
    config_dir = str(PROJECT_ROOT / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name)
    return cfg


# ---------------------------------------------------------------------------
# Reverse index: packed row -> original records
# ---------------------------------------------------------------------------

PREFIX_LEN = 32  # tokens used as fingerprint


def build_prefix_index(
    original_ds: Dataset, max_length: int
) -> dict[tuple[int, ...], list[tuple[int, int]]]:
    """Map token prefix → [(original_row_idx, offset_in_original), ...]"""
    index: dict[tuple[int, ...], list[tuple[int, int]]] = defaultdict(list)
    for i in range(len(original_ds)):
        ids = original_ds[i]["input_ids"]
        # bfd_split may split long sequences — index each chunk boundary
        if len(ids) > max_length:
            for offset in range(0, len(ids), max_length):
                chunk = ids[offset : offset + max_length]
                if len(chunk) >= PREFIX_LEN:
                    key = tuple(chunk[:PREFIX_LEN])
                    index[key].append((i, offset))
        else:
            if len(ids) >= PREFIX_LEN:
                key = tuple(ids[:PREFIX_LEN])
                index[key].append((i, 0))
            elif ids:
                key = tuple(ids)
                index[key].append((i, 0))
    return index


def build_reverse_map(
    original_ds: Dataset,
    packed_ds: Dataset,
    max_length: int,
    prefix_index: dict[tuple[int, ...], list[tuple[int, int]]],
) -> dict[int, list[dict[str, Any]]]:
    """For each packed row, find the original record(s) it contains."""
    reverse: dict[int, list[dict[str, Any]]] = {}

    for packed_idx in range(len(packed_ds)):
        row = packed_ds[packed_idx]
        all_ids = row["input_ids"]
        seq_lens = row.get("seq_lengths", [len(all_ids)])

        segments: list[dict[str, Any]] = []
        pos = 0
        for seg_len in seq_lens:
            seg_ids = all_ids[pos : pos + seg_len]
            matched = _match_segment(seg_ids, original_ds, prefix_index)
            if matched is not None:
                segments.append(matched | {"segment_tokens": seg_len})
            else:
                segments.append(
                    {
                        "original_row_idx": None,
                        "offset": 0,
                        "segment_tokens": seg_len,
                        "warning": "UNMATCHED",
                    }
                )
            pos += seg_len

        reverse[packed_idx] = segments

    return reverse


def _match_segment(
    seg_ids: list[int],
    original_ds: Dataset,
    prefix_index: dict[tuple[int, ...], list[tuple[int, int]]],
) -> dict[str, Any] | None:
    if len(seg_ids) >= PREFIX_LEN:
        key = tuple(seg_ids[:PREFIX_LEN])
    else:
        key = tuple(seg_ids)

    candidates = prefix_index.get(key, [])
    for orig_idx, offset in candidates:
        orig_ids = original_ds[orig_idx]["input_ids"]
        chunk = orig_ids[offset : offset + len(seg_ids)]
        if chunk == seg_ids:
            return {"original_row_idx": orig_idx, "offset": offset}
    return None


# ---------------------------------------------------------------------------
# Shuffle reproduction
# ---------------------------------------------------------------------------


def reproduce_shuffle(n: int, seed: int, epoch: int = 0) -> list[int]:
    """
    HuggingFace Trainer의 SeedableRandomSampler 동작을 재현.
    accelerate의 SeedableRandomSampler:
      generator.manual_seed(initial_seed + epoch)
      yield from torch.randperm(n, generator=generator)
    initial_seed = torch.random.initial_seed() after set_seed(seed)
    """
    g = torch.Generator()
    g.manual_seed(seed + epoch)
    return torch.randperm(n, generator=g).tolist()


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------


def compute_dataset_stats(original_ds: Dataset) -> dict[str, float]:
    lengths = [len(original_ds[i]["input_ids"]) for i in range(len(original_ds))]
    arr = np.array(lengths)
    return {
        "p5_tokens": float(np.percentile(arr, 5)),
        "p50_tokens": float(np.percentile(arr, 50)),
        "p95_tokens": float(np.percentile(arr, 95)),
        "mean_tokens": float(arr.mean()),
    }


def detect_anomalies(text: str, token_count: int, stats: dict[str, float]) -> list[str]:
    flags: list[str] = []

    # Length anomalies
    if token_count > stats["p95_tokens"]:
        flags.append(f"very_long ({token_count} tok, p95={stats['p95_tokens']:.0f})")
    if token_count < 32:
        flags.append(f"very_short ({token_count} tok)")

    # Repetition check (bigram unique ratio)
    words = text.split()
    if len(words) > 20:
        bigrams = [tuple(words[i : i + 2]) for i in range(len(words) - 1)]
        unique_ratio = len(set(bigrams)) / len(bigrams) if bigrams else 1.0
        if unique_ratio < 0.3:
            flags.append(f"highly_repetitive (bigram_uniq={unique_ratio:.2f})")

    # Encoding issues — non-printable chars (excluding common whitespace)
    non_printable = sum(1 for c in text if not c.isprintable() and c not in "\n\t\r")
    if len(text) > 0 and non_printable / len(text) > 0.01:
        flags.append(f"encoding_issues ({non_printable} non-printable/{len(text)} chars)")

    # Excessive special character ratio
    if len(text) > 50:
        alpha_ratio = sum(1 for c in text if c.isalpha()) / len(text)
        if alpha_ratio < 0.2:
            flags.append(f"low_alpha_ratio ({alpha_ratio:.2f})")

    return flags


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    steps = parse_steps(args.steps)

    # 1. Load config
    print(f"[1/5] Loading config: {args.config}")
    cfg = load_config(args.config)

    seed = int(cfg.seed)
    batch_size = int(cfg.training.per_device_train_batch_size)
    grad_accum = int(cfg.training.gradient_accumulation_steps)
    samples_per_step = batch_size * grad_accum
    max_length = int(cfg.model.max_seq_length)

    runtime_packing_cfg = cfg.training.get("runtime_packing")
    use_packing = bool(runtime_packing_cfg and runtime_packing_cfg.get("enabled", False))
    if args.no_packing:
        use_packing = False
    packing_strategy = str(runtime_packing_cfg.get("strategy", "bfd_split")) if use_packing else None

    dataset_dir = Path(str(cfg.preprocessing.output_dir))
    if not dataset_dir.is_absolute():
        dataset_dir = PROJECT_ROOT / dataset_dir
    train_dir = dataset_dir / "train"

    print(f"  seed={seed}, batch={batch_size}, grad_accum={grad_accum}")
    print(f"  samples_per_step={samples_per_step}, max_length={max_length}")
    print(f"  packing={use_packing}, strategy={packing_strategy}")
    print(f"  dataset: {train_dir}")

    # 2. Load preprocessed train dataset
    print(f"\n[2/5] Loading preprocessed train dataset...")
    if not train_dir.exists():
        print(f"ERROR: {train_dir} not found. Run preprocessing first:")
        print(f"  make preprocess config={args.config}")
        sys.exit(1)

    original_ds = load_from_disk(str(train_dir))
    print(f"  Original records: {len(original_ds)}")

    # 3. Reproduce packing
    if use_packing:
        print(f"\n[3/5] Reproducing TRL packing (strategy={packing_strategy})...")
        packable = original_ds.select_columns(["input_ids"])
        packed_ds = pack_dataset(packable, seq_length=max_length, strategy=packing_strategy)
        print(f"  Packed rows: {len(packed_ds)} (from {len(original_ds)} originals)")
    else:
        print(f"\n[3/5] No runtime packing — using original dataset directly")
        packed_ds = original_ds

    # Validate step range
    max_possible_step = len(packed_ds) // samples_per_step
    for s in steps:
        if s >= max_possible_step:
            print(
                f"WARNING: step {s} exceeds max possible step {max_possible_step - 1} "
                f"(packed_rows={len(packed_ds)}, samples_per_step={samples_per_step})"
            )

    # 4. Build reverse index
    if use_packing:
        print(f"\n[4/5] Building reverse index (packed → original)...")
        prefix_index = build_prefix_index(original_ds, max_length)
        reverse_map = build_reverse_map(original_ds, packed_ds, max_length, prefix_index)
        unmatched = sum(
            1
            for segs in reverse_map.values()
            for s in segs
            if s.get("original_row_idx") is None
        )
        if unmatched:
            print(f"  WARNING: {unmatched} segments could not be matched to originals")
        print(f"  Reverse index built for {len(reverse_map)} packed rows")
    else:
        reverse_map = {
            i: [{"original_row_idx": i, "offset": 0, "segment_tokens": len(original_ds[i]["input_ids"])}]
            for i in range(len(original_ds))
        }
        print(f"\n[4/5] No packing — identity mapping")

    # 5. Reproduce shuffle & extract data
    print(f"\n[5/5] Reproducing shuffle order (seed={seed})...")
    perm = reproduce_shuffle(len(packed_ds), seed=seed, epoch=0)
    print(f"  Permutation length: {len(perm)}")

    # Compute stats for anomaly detection
    stats = compute_dataset_stats(original_ds)
    print(f"  Dataset stats: p5={stats['p5_tokens']:.0f}, p50={stats['p50_tokens']:.0f}, "
          f"p95={stats['p95_tokens']:.0f}, mean={stats['mean_tokens']:.0f}")

    # Process each step
    all_results: dict[str, Any] = {
        "config": args.config,
        "seed": seed,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "samples_per_step": samples_per_step,
        "max_length": max_length,
        "packing": use_packing,
        "packing_strategy": packing_strategy,
        "original_dataset_size": len(original_ds),
        "packed_dataset_size": len(packed_ds),
        "dataset_stats": stats,
        "steps": {},
    }

    for step in steps:
        step_result = process_step(
            step=step,
            perm=perm,
            samples_per_step=samples_per_step,
            batch_size=batch_size,
            packed_ds=packed_ds,
            original_ds=original_ds,
            reverse_map=reverse_map,
            stats=stats,
            args=args,
        )
        all_results["steps"][str(step)] = step_result

    # Output
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {out_path}")
    else:
        print()  # extra newline before results


def process_step(
    *,
    step: int,
    perm: list[int],
    samples_per_step: int,
    batch_size: int,
    packed_ds: Dataset,
    original_ds: Dataset,
    reverse_map: dict[int, list[dict[str, Any]]],
    stats: dict[str, float],
    args: argparse.Namespace,
) -> dict[str, Any]:
    start_idx = step * samples_per_step
    end_idx = start_idx + samples_per_step
    packed_indices = perm[start_idx:end_idx]

    print(f"\n{'='*70}")
    print(f"  STEP {step}  (shuffled positions {start_idx}-{end_idx - 1})")
    print(f"{'='*70}")

    step_data: dict[str, Any] = {
        "packed_row_indices": packed_indices,
        "micro_batches": [],
        "summary": {},
    }

    all_anomalies: list[dict[str, Any]] = []
    all_token_counts: list[int] = []
    unique_docs: set[str] = set()
    total_segments = 0

    grad_accum = samples_per_step // batch_size

    for mb_idx in range(grad_accum):
        mb_start = mb_idx * batch_size
        mb_end = mb_start + batch_size
        mb_indices = packed_indices[mb_start:mb_end]

        mb_data: dict[str, Any] = {"micro_batch_idx": mb_idx, "samples": []}

        if not args.anomaly_only:
            print(f"\n  Micro-batch {mb_idx} (packed rows: {mb_indices})")

        for local_idx, packed_idx in enumerate(mb_indices):
            packed_row = packed_ds[packed_idx]
            packed_tokens = len(packed_row["input_ids"])
            all_token_counts.append(packed_tokens)

            segments = reverse_map.get(packed_idx, [])
            total_segments += len(segments)

            sample_data: dict[str, Any] = {
                "packed_row_idx": packed_idx,
                "total_tokens": packed_tokens,
                "num_segments": len(segments),
                "sequences": [],
            }

            sample_has_anomaly = False

            for seg in segments:
                orig_idx = seg.get("original_row_idx")
                seg_tokens = seg.get("segment_tokens", 0)

                if orig_idx is not None:
                    orig_row = original_ds[orig_idx]
                    doc_id = orig_row.get("doc_id", "?")
                    source = orig_row.get("source", "?")
                    chunk_id = orig_row.get("chunk_id", "?")
                    text = orig_row.get("text", "")
                    unique_docs.add(str(doc_id))

                    anomalies = detect_anomalies(text, seg_tokens, stats)
                    if anomalies:
                        sample_has_anomaly = True
                        all_anomalies.append(
                            {
                                "step": step,
                                "packed_idx": packed_idx,
                                "original_idx": orig_idx,
                                "doc_id": doc_id,
                                "flags": anomalies,
                            }
                        )

                    text_preview = text if args.verbose else text[: args.text_preview_len]
                    if not args.verbose and len(text) > args.text_preview_len:
                        text_preview += "..."

                    seq_data = {
                        "original_row_idx": orig_idx,
                        "doc_id": doc_id,
                        "source": source,
                        "chunk_id": chunk_id,
                        "segment_tokens": seg_tokens,
                        "offset_in_original": seg.get("offset", 0),
                        "anomalies": anomalies,
                        "text_preview": text_preview,
                    }
                else:
                    seq_data = {
                        "original_row_idx": None,
                        "segment_tokens": seg_tokens,
                        "warning": "UNMATCHED",
                        "anomalies": ["unmatched_segment"],
                    }
                    sample_has_anomaly = True

                sample_data["sequences"].append(seq_data)

            mb_data["samples"].append(sample_data)

            # Console output
            should_print = not args.anomaly_only or sample_has_anomaly
            if should_print:
                _print_sample(sample_data, args)

        step_data["micro_batches"].append(mb_data)

    # Summary
    summary = {
        "total_unique_docs": len(unique_docs),
        "total_segments": total_segments,
        "anomalous_count": len(all_anomalies),
        "token_stats": {
            "min": min(all_token_counts) if all_token_counts else 0,
            "max": max(all_token_counts) if all_token_counts else 0,
            "mean": sum(all_token_counts) / len(all_token_counts) if all_token_counts else 0,
        },
        "anomalies": all_anomalies,
    }
    step_data["summary"] = summary

    print(f"\n  --- Step {step} Summary ---")
    print(f"  Unique documents: {len(unique_docs)}")
    print(f"  Total segments: {total_segments}")
    print(f"  Anomalous samples: {len(all_anomalies)}")
    if all_token_counts:
        print(
            f"  Packed tokens: min={min(all_token_counts)}, "
            f"max={max(all_token_counts)}, "
            f"mean={sum(all_token_counts)/len(all_token_counts):.0f}"
        )
    if all_anomalies:
        print(f"\n  *** ANOMALIES DETECTED ***")
        for a in all_anomalies:
            print(f"    doc_id={a['doc_id']}, orig_idx={a['original_idx']}, flags={a['flags']}")

    return step_data


def _print_sample(sample_data: dict[str, Any], args: argparse.Namespace) -> None:
    packed_idx = sample_data["packed_row_idx"]
    total_tok = sample_data["total_tokens"]
    n_seg = sample_data["num_segments"]
    print(f"    [Packed#{packed_idx}] {n_seg} seq(s), {total_tok} tokens")

    for seq in sample_data["sequences"]:
        orig_idx = seq.get("original_row_idx")
        seg_tok = seq.get("segment_tokens", "?")

        if orig_idx is not None:
            doc_id = seq.get("doc_id", "?")
            chunk_id = seq.get("chunk_id", "?")
            anomalies = seq.get("anomalies", [])
            flag_str = f"  *** {anomalies}" if anomalies else ""
            print(f"      -> orig[{orig_idx}] doc={doc_id} chunk={chunk_id} ({seg_tok} tok){flag_str}")
            if args.verbose:
                preview = seq.get("text_preview", "")
                for line in preview[:500].split("\n")[:5]:
                    print(f"         | {line}")
        else:
            print(f"      -> UNMATCHED ({seg_tok} tok)")


if __name__ == "__main__":
    main()
