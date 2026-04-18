from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
import shutil

from huggingface_hub import HfApi
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from .common import resolve_workspace_path, suppress_noisy_library_logs


def compose_cfg(config_path: str, config_name: str):
    cfg_dir = resolve_workspace_path(config_path)
    with initialize_config_dir(version_base=None, config_dir=str(cfg_dir)):
        cfg = compose(config_name=config_name, overrides=[])
    OmegaConf.resolve(cfg)
    return cfg


def _checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.split("-", 1)[1])
    except Exception:
        return -1


def resolve_upload_folder(output_dir: Path, checkpoint: str) -> Path:
    ckpt = checkpoint.strip()
    if ckpt.lower() in {"", "final", "none", "null"}:
        return output_dir

    if ckpt.lower() == "latest":
        candidates = [p for p in output_dir.glob("checkpoint-*") if p.is_dir()]
        if not candidates:
            raise FileNotFoundError(f"No checkpoint-* directories found in: {output_dir}")
        return max(candidates, key=_checkpoint_step)

    as_path = Path(ckpt)
    if as_path.is_absolute():
        if not as_path.exists():
            raise FileNotFoundError(f"Checkpoint path not found: {as_path}")
        return as_path

    candidate = output_dir / ckpt
    if not candidate.exists():
        raise FileNotFoundError(f"Checkpoint not found: {candidate}")
    return candidate


def _link_or_copy_file(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def stage_upload_folder(upload_folder: Path, output_dir: Path) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if upload_folder != output_dir:
        return upload_folder, None

    tmpdir = tempfile.TemporaryDirectory(
        prefix="hf_model_upload_",
        dir=str(output_dir.parent),
    )
    staged_root = Path(tmpdir.name) / output_dir.name
    staged_root.mkdir(parents=True, exist_ok=True)

    print(
        "[INFO] Preparing final-model upload without nested checkpoint-* directories "
        "or duplicate tokenizer/ folder."
    )

    for child in output_dir.iterdir():
        if child.name.startswith("checkpoint-"):
            continue
        if child.name == "tokenizer":
            continue

        target = staged_root / child.name
        if child.is_dir():
            shutil.copytree(child, target, copy_function=_link_or_copy_file)
        else:
            _link_or_copy_file(child, target)

    return staged_root, tmpdir


def resolve_delete_patterns(upload_folder: Path, output_dir: Path) -> list[str] | None:
    if upload_folder != output_dir:
        return None
    return [
        "checkpoint-*",
        "checkpoint-*/**",
        "tokenizer",
        "tokenizer/**",
    ]


def maybe_upload_eval_artifacts(
    api: HfApi,
    repo_id: str,
    upload_folder: Path,
    experiment_root: Path,
    commit_message: str,
) -> None:
    eval_root = experiment_root / "eval"
    if not eval_root.exists():
        print("[INFO] No eval directory found. Skipping eval upload.")
        return

    model_eval_dir = eval_root / "lm_eval" / upload_folder.name
    base_only_eval_dir = eval_root / "lm_eval" / "base_only"
    summary_path = eval_root / "summary.json"
    base_only_summary_path = eval_root / "eval_results_base_only.json"

    if (
        not model_eval_dir.exists()
        and not base_only_eval_dir.exists()
        and not summary_path.exists()
        and not base_only_summary_path.exists()
    ):
        print(f"[INFO] No eval artifacts for '{upload_folder.name}'. Skipping eval upload.")
        return

    with tempfile.TemporaryDirectory(prefix="hf_eval_upload_") as tmpdir:
        staging_root = Path(tmpdir)
        target_eval_root = staging_root / "eval"
        target_eval_root.mkdir(parents=True, exist_ok=True)

        if summary_path.exists():
            shutil.copy2(summary_path, target_eval_root / "summary.json")

        if base_only_summary_path.exists():
            shutil.copy2(base_only_summary_path, target_eval_root / "eval_results_base_only.json")

        if model_eval_dir.exists():
            shutil.copytree(model_eval_dir, target_eval_root / upload_folder.name, dirs_exist_ok=True)

        if base_only_eval_dir.exists():
            target_lm_eval_root = target_eval_root / "lm_eval"
            target_lm_eval_root.mkdir(parents=True, exist_ok=True)
            shutil.copytree(
                base_only_eval_dir,
                target_lm_eval_root / "base_only",
                dirs_exist_ok=True,
            )

        api.upload_folder(
            repo_id=repo_id,
            folder_path=str(target_eval_root),
            repo_type="model",
            path_in_repo="eval",
            commit_message=f"{commit_message} (eval artifacts)",
        )
    print(f"[INFO] Uploaded eval artifacts for: {upload_folder.name}")


def main() -> None:
    suppress_noisy_library_logs()

    parser = argparse.ArgumentParser(description="Upload trained model/checkpoint to Hugging Face Hub")
    parser.add_argument("--config-path", default="../configs")
    parser.add_argument("--config-name", default="config")
    parser.add_argument("--repo", required=True, help="HF repo id, e.g. your-name/your-model")
    parser.add_argument(
        "--checkpoint",
        default="final",
        help="final | latest | checkpoint-XXXX | absolute path",
    )
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--commit-message", default=None)
    parser.add_argument(
        "--include-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Upload matching eval artifacts (summary + lm_eval/<checkpoint_or_output_dir_name>) to repo/eval",
    )
    args = parser.parse_args()

    cfg = compose_cfg(args.config_path, args.config_name)
    output_dir = resolve_workspace_path(cfg.training.output_dir)
    experiment_root = resolve_workspace_path(cfg.experiment.output_root)
    if not output_dir.exists():
        raise FileNotFoundError(f"Training output dir not found: {output_dir}")

    upload_folder = resolve_upload_folder(output_dir, args.checkpoint)
    upload_source, upload_tmpdir = stage_upload_folder(upload_folder, output_dir)
    delete_patterns = resolve_delete_patterns(upload_folder, output_dir)

    api = HfApi()
    api.create_repo(repo_id=args.repo, repo_type="model", private=bool(args.private), exist_ok=True)

    message = args.commit_message
    if not message:
        if upload_folder == output_dir:
            message = f"Upload final model from {cfg.experiment.name}"
        else:
            message = f"Upload checkpoint {upload_folder.name} from {cfg.experiment.name}"

    print("=" * 80)
    print("Push To Hub")
    print("=" * 80)
    print(f"repo: {args.repo}")
    print(f"source: {upload_source}")
    print(f"private: {bool(args.private)}")
    print(f"commit: {message}")

    try:
        api.upload_folder(
            repo_id=args.repo,
            folder_path=str(upload_source),
            repo_type="model",
            commit_message=message,
            delete_patterns=delete_patterns,
        )

        if bool(args.include_eval):
            maybe_upload_eval_artifacts(
                api=api,
                repo_id=args.repo,
                upload_folder=upload_folder,
                experiment_root=experiment_root,
                commit_message=message,
            )
    finally:
        if upload_tmpdir is not None:
            upload_tmpdir.cleanup()

    print("=" * 80)
    print("Upload Complete")
    print("=" * 80)
    print(f"https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
