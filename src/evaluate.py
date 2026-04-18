"""
Evaluate Stage1 CPT checkpoints.

Features:
- Perplexity on preprocessed validation set
- Optional lm-eval-harness benchmarks (CPT only or base vs CPT)
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import unsloth  # must be imported before transformers/lm_eval
import torch
from datasets import load_from_disk
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tqdm import tqdm

from .common import resolve_workspace_path, suppress_noisy_library_logs

BENCHMARK_TASKS = "mmlu,hellaswag,arc_easy,arc_challenge,winogrande"
KOREAN_BENCHMARK_TASKS = "kmmlu,kobest_boolq,kobest_copa,kobest_hellaswag"


def compose_cfg(config_path: str, config_name: str):
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


def _load_with_unsloth(
    path_or_repo: str,
    max_seq_length: int,
    model_hint: str,
    full_finetuning: bool = True,
):
    from unsloth import FastLanguageModel, FastVisionModel

    errors: list[str] = []
    for mode, model_class in (
        ("FastVisionModel", FastVisionModel),
        ("FastLanguageModel", FastLanguageModel),
    ):
        try:
            model, tokenizer = model_class.from_pretrained(
                model_name=path_or_repo,
                max_seq_length=max_seq_length,
                dtype=None,
                load_in_4bit=False,
                **(
                    {"full_finetuning": full_finetuning}
                    if "full_finetuning" in inspect.signature(model_class.from_pretrained).parameters
                    else {}
                ),
            )
            return model, tokenizer, mode
        except Exception as exc:
            errors.append(f"{mode}: {type(exc).__name__}: {exc}")

    joined = "\n".join(errors)
    raise RuntimeError(
        "Failed to load model with both vision and language backends.\n"
        f"model={path_or_repo}\n"
        f"hint={model_hint}\n"
        f"errors:\n{joined}"
    )


def compute_ppl(model, dataset, batch_size: int = 4, max_batches: int | None = None) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_rows = len(dataset)
    n = min(total_rows, max_batches * batch_size) if max_batches else total_rows
    device = next(model.parameters()).device

    with torch.no_grad():
        for start in tqdm(range(0, n, batch_size), desc="ppl"):
            batch = dataset[start : min(start + batch_size, n)]
            ids_list = [list(ids) for ids in batch["input_ids"]]
            if "labels" in batch:
                lbl_list = [list(lbl) for lbl in batch["labels"]]
            elif "completion_mask" in batch:
                lbl_list = []
                for ids, mask in zip(ids_list, batch["completion_mask"], strict=True):
                    mask_list = list(mask)
                    if len(mask_list) < len(ids):
                        mask_list = mask_list + [0] * (len(ids) - len(mask_list))
                    elif len(mask_list) > len(ids):
                        mask_list = mask_list[: len(ids)]
                    lbl_list.append(
                        [token_id if int(is_completion) == 1 else -100 for token_id, is_completion in zip(ids, mask_list, strict=True)]
                    )
            else:
                lbl_list = [list(ids) for ids in ids_list]

            max_len = max(
                max(len(ids) for ids in ids_list),
                max(len(lbl) for lbl in lbl_list),
            )
            pad_token_id = getattr(getattr(model, "config", None), "pad_token_id", None)
            if pad_token_id is None:
                pad_token_id = 0

            pad_ids = [ids + [pad_token_id] * (max_len - len(ids)) for ids in ids_list]
            pad_lbl = [lbl + [-100] * (max_len - len(lbl)) for lbl in lbl_list]
            attn = [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in ids_list]

            input_ids = torch.tensor(pad_ids, dtype=torch.long, device=device)
            labels = torch.tensor(pad_lbl, dtype=torch.long, device=device)
            attention_mask = torch.tensor(attn, dtype=torch.long, device=device)

            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
            tokens = (labels != -100).sum().item()
            total_loss += float(loss.item()) * tokens
            total_tokens += tokens

    avg_loss = total_loss / max(total_tokens, 1)
    return {
        "loss": avg_loss,
        "ppl": math.exp(avg_loss),
        "tokens": total_tokens,
        "rows": n,
    }


def _run_streaming(cmd: list[str]) -> tuple[int, str]:
    """
    Stream child process logs to the current terminal while keeping a copy for error summaries.
    """
    captured_chunks: list[str] = []
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for chunk in process.stdout:
        captured_chunks.append(chunk)
        print(chunk, end="", flush=True)
    return_code = process.wait()
    return return_code, "".join(captured_chunks)


def run_lm_eval(model_path: str, tasks: str, batch_size: int, limit: int, output_path: Path) -> tuple[dict[str, Any], str]:
    cmd = [
        sys.executable,
        "-m",
        "src.lm_eval_with_unsloth",
        "--model",
        "hf",
        "--model_args",
        f"pretrained={model_path},trust_remote_code=True",
        "--tasks",
        tasks,
        "--verbosity",
        "WARNING",
        "--batch_size",
        str(batch_size),
        "--limit",
        str(limit),
        "--output_path",
        str(output_path),
    ]

    return_code, captured = _run_streaming(cmd)
    if return_code != 0:
        stderr_tail = captured[-1200:] if captured else "no stderr"
        raise RuntimeError(f"lm-eval failed: {stderr_tail}")

    results = {}
    for candidate in sorted(output_path.rglob("results_*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        results = json.loads(candidate.read_text(encoding="utf-8"))
        break
    return results, captured


def _model_args_for_lm_eval(model_path: Path, base_model: str) -> str:
    if (model_path / "adapter_config.json").exists():
        return f"pretrained={base_model},peft={model_path},trust_remote_code=True"
    return f"pretrained={model_path},trust_remote_code=True"


def run_lm_eval_with_model_args(model_args: str, tasks: str, batch_size: int, limit: int, output_path: Path) -> tuple[dict[str, Any], str]:
    cmd = [
        sys.executable,
        "-m",
        "src.lm_eval_with_unsloth",
        "--model",
        "hf",
        "--model_args",
        model_args,
        "--tasks",
        tasks,
        "--verbosity",
        "WARNING",
        "--batch_size",
        str(batch_size),
        "--limit",
        str(limit),
        "--output_path",
        str(output_path),
    ]

    return_code, captured = _run_streaming(cmd)
    if return_code != 0:
        stderr_tail = captured[-1200:] if captured else "no stderr"
        raise RuntimeError(f"lm-eval failed: {stderr_tail}")

    results = {}
    for candidate in sorted(output_path.rglob("results_*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        results = json.loads(candidate.read_text(encoding="utf-8"))
        break
    return results, captured


def resolve_tasks(args) -> str:
    tasks = args.benchmark_tasks
    if args.include_korean_benchmarks:
        tasks = f"{tasks},{KOREAN_BENCHMARK_TASKS}"
    return tasks


def resolve_model_paths(args, cfg) -> list[Path]:
    if args.model_path:
        return [resolve_workspace_path(path) for path in args.model_path]

    default_dir = resolve_workspace_path(cfg.training.output_dir)
    if args.all_checkpoints:
        checkpoints = sorted(
            default_dir.glob("checkpoint-*"),
            key=lambda path: int(path.name.split("-")[1]) if "-" in path.name else 0,
        )
        if (default_dir / "config.json").exists() or (default_dir / "adapter_config.json").exists():
            checkpoints.append(default_dir)
        return checkpoints

    return [default_dir]


def load_model_for_eval(model_path: Path, base_model: str, max_seq_length: int):
    adapter_cfg = model_path / "adapter_config.json"

    if adapter_cfg.exists():
        from peft import PeftModel

        base, tokenizer, mode = _load_with_unsloth(
            base_model,
            max_seq_length=max_seq_length,
            model_hint=base_model,
            full_finetuning=False,
        )
        model = PeftModel.from_pretrained(base, str(model_path))
        return model, tokenizer, f"{mode}+PEFT"

    model, tokenizer, mode = _load_with_unsloth(
        str(model_path),
        max_seq_length=max_seq_length,
        model_hint=base_model,
    )
    return model, tokenizer, mode


def compute_base_ppl(val_ds, args, cfg) -> dict[str, Any]:
    base_model = args.base_model or str(cfg.model.pretrained_model_name_or_path)
    max_seq_length = int(cfg.model.max_seq_length)

    print("=" * 80)
    print(f"Evaluating Base PPL: {base_model}")
    print("=" * 80)

    model, _tokenizer, mode = _load_with_unsloth(
        base_model,
        max_seq_length=max_seq_length,
        model_hint=base_model,
    )
    print(f"loader(base): {mode}")
    ppl_metrics = compute_ppl(model, val_ds, batch_size=args.batch_size, max_batches=args.max_batches)
    print(f"base ppl: {ppl_metrics['ppl']:.4f} | loss: {ppl_metrics['loss']:.6f}")

    del model
    free_vram()
    return ppl_metrics


def evaluate_single(model_path: Path, val_ds, args, cfg, eval_out_dir: Path, base_ppl_metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    base_model = args.base_model or str(cfg.model.pretrained_model_name_or_path)
    max_seq_length = int(cfg.model.max_seq_length)

    print("=" * 80)
    print(f"Evaluating: {model_path}")
    print("=" * 80)

    ppl_metrics = None
    if not args.benchmarks_only:
        model, _tokenizer, mode = load_model_for_eval(model_path, base_model=base_model, max_seq_length=max_seq_length)
        print(f"loader: {mode}")

        ppl_metrics = compute_ppl(model, val_ds, batch_size=args.batch_size, max_batches=args.max_batches)
        print(f"ppl: {ppl_metrics['ppl']:.4f} | loss: {ppl_metrics['loss']:.6f}")

        del model
        free_vram()

    results = {
        "model_path": str(model_path),
        "ppl": ppl_metrics,
        "base_ppl": base_ppl_metrics,
        "benchmarks": {},
    }

    if not args.skip_benchmarks:
        tasks = resolve_tasks(args)
        bench_root = eval_out_dir / "lm_eval" / model_path.name
        bench_root.mkdir(parents=True, exist_ok=True)

        run_cpt = args.bench_target in {"cpt", "both"}
        run_base = args.bench_target in {"base", "both"} and not args.skip_base_benchmarks

        if run_cpt:
            cpt_args = _model_args_for_lm_eval(model_path, base_model=base_model)
            cpt_out = bench_root / "cpt"
            cpt_out.mkdir(parents=True, exist_ok=True)
            cpt_results, cpt_stdout = run_lm_eval_with_model_args(
                model_args=cpt_args,
                tasks=tasks,
                batch_size=args.batch_size,
                limit=args.limit,
                output_path=cpt_out,
            )
            results["benchmarks"]["cpt"] = cpt_results
            (cpt_out / "stdout.txt").write_text(cpt_stdout, encoding="utf-8")

        if run_base:
            base_out = bench_root / "base"
            base_out.mkdir(parents=True, exist_ok=True)
            base_results, base_stdout = run_lm_eval_with_model_args(
                model_args=f"pretrained={base_model},trust_remote_code=True",
                tasks=tasks,
                batch_size=args.batch_size,
                limit=args.limit,
                output_path=base_out,
            )
            results["benchmarks"]["base"] = base_results
            (base_out / "stdout.txt").write_text(base_stdout, encoding="utf-8")

    label = model_path.name if model_path.name.startswith("checkpoint-") else "final"
    result_path = eval_out_dir / f"eval_results_{label}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {result_path}")

    return {
        "label": label,
        "ppl": None if ppl_metrics is None else ppl_metrics["ppl"],
        "base_ppl": None if base_ppl_metrics is None else base_ppl_metrics["ppl"],
        "path": str(model_path),
    }


def evaluate_base_only(args, cfg, eval_out_dir: Path) -> None:
    base_model = args.base_model or str(cfg.model.pretrained_model_name_or_path)
    tasks = resolve_tasks(args)

    bench_root = eval_out_dir / "lm_eval" / "base_only"
    bench_root.mkdir(parents=True, exist_ok=True)

    base_results, base_stdout = run_lm_eval_with_model_args(
        model_args=f"pretrained={base_model},trust_remote_code=True",
        tasks=tasks,
        batch_size=args.batch_size,
        limit=args.limit,
        output_path=bench_root,
    )
    (bench_root / "stdout.txt").write_text(base_stdout, encoding="utf-8")

    out_path = eval_out_dir / "eval_results_base_only.json"
    out_path.write_text(
        json.dumps(
            {
                "base_model": base_model,
                "tasks": tasks,
                "benchmarks": {"base": base_results},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved: {out_path}")


def main():
    suppress_noisy_library_logs()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", default="configs")
    parser.add_argument("--config-name", default="config")
    parser.add_argument("--model_path", nargs="+", default=None, help="Checkpoint/model path(s)")
    parser.add_argument("--all_checkpoints", action="store_true")
    parser.add_argument("--base_model", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--skip_benchmarks", action="store_true")
    parser.add_argument("--benchmarks_only", action="store_true")
    parser.add_argument("--benchmark_tasks", default=BENCHMARK_TASKS)
    parser.add_argument("--include_korean_benchmarks", dest="include_korean_benchmarks", action="store_true", default=True)
    parser.add_argument("--no_korean_benchmarks", dest="include_korean_benchmarks", action="store_false")
    parser.add_argument("--skip_base_benchmarks", action="store_true")
    parser.add_argument("--bench_target", choices=["cpt", "base", "both"], default="cpt")
    parser.add_argument("--limit", type=int, default=400)
    args = parser.parse_args()

    cfg = compose_cfg(args.config_path, args.config_name)

    if args.batch_size is None:
        if "evaluation" in cfg and "batch_size" in cfg.evaluation:
            args.batch_size = int(cfg.evaluation.batch_size)
        else:
            args.batch_size = int(cfg.training.per_device_eval_batch_size)
    else:
        args.batch_size = int(args.batch_size)

    print(f"eval batch_size: {args.batch_size}")
    model_paths = resolve_model_paths(args, cfg)

    if not model_paths:
        raise RuntimeError("No model paths found to evaluate.")

    eval_out_dir = resolve_workspace_path(cfg.experiment.output_root) / "eval"
    eval_out_dir.mkdir(parents=True, exist_ok=True)

    if args.benchmarks_only:
        args.skip_benchmarks = False

    if args.benchmarks_only and args.bench_target == "base":
        evaluate_base_only(args, cfg, eval_out_dir=eval_out_dir)
        return

    val_ds = None
    base_ppl_metrics = None
    if not args.benchmarks_only:
        dataset_dir = resolve_workspace_path(cfg.training.dataset_dir)
        val_path = dataset_dir / "val"
        if not val_path.exists():
            raise FileNotFoundError(f"Validation dataset not found: {val_path}")
        val_ds = load_from_disk(str(val_path))
        print(f"validation rows: {len(val_ds)}")
        try:
            base_ppl_metrics = compute_base_ppl(val_ds, args, cfg)
        except Exception as exc:
            print(f"[WARN] failed to compute base ppl: {exc}")
            free_vram()

    summaries = []
    for model_path in model_paths:
        model_path = model_path.resolve()
        if not model_path.exists():
            print(f"[WARN] skip missing model path: {model_path}")
            continue
        try:
            summary = evaluate_single(
                model_path,
                val_ds,
                args,
                cfg,
                eval_out_dir=eval_out_dir,
                base_ppl_metrics=base_ppl_metrics,
            )
            summaries.append(summary)
        except Exception as exc:
            print(f"[ERROR] failed on {model_path.name}: {exc}")
            free_vram()

    if summaries:
        summary_path = eval_out_dir / "summary.json"
        summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
        print("=" * 80)
        print("PPL Summary")
        print("=" * 80)
        if base_ppl_metrics is not None:
            print(f"{'base':<20} ppl={base_ppl_metrics['ppl']:.4f}")
        for item in summaries:
            if item["ppl"] is None:
                print(f"{item['label']:<20} ppl=skip  path={item['path']}")
            else:
                print(f"{item['label']:<20} ppl={item['ppl']:.4f}  path={item['path']}")
        print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
