from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, ListConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_workspace_path(path_like: str | Path) -> Path:
    path = Path(str(path_like))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def to_report_to_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, ListConfig)):
        return [str(v) for v in value if str(v).strip()]
    text = str(value).strip()
    if not text or text.lower() == "none":
        return []
    return [text]


def resolve_torch_dtype(name: str | None):
    if name is None:
        return None
    key = str(name).strip().lower()
    if key in {"", "auto"}:
        return None
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if key not in mapping:
        raise ValueError(f"Unsupported dtype '{name}'")
    return mapping[key]


def setup_wandb_env(
    logging_cfg: DictConfig,
    experiment_name: str | None = None,
    tags_override: list[str] | None = None,
) -> None:
    report_to = to_report_to_list(logging_cfg.get("report_to"))
    if "wandb" not in report_to:
        return

    wandb_cfg = logging_cfg.get("wandb") or {}

    project = wandb_cfg.get("project")
    if project:
        os.environ.setdefault("WANDB_PROJECT", str(project))

    entity = wandb_cfg.get("entity")
    if entity:
        os.environ.setdefault("WANDB_ENTITY", str(entity))

    tags = tags_override if tags_override is not None else wandb_cfg.get("tags")
    if tags:
        tag_values = [str(tag) for tag in tags if str(tag).strip()]
        if tag_values:
            os.environ["WANDB_TAGS"] = ",".join(tag_values)

    notes = wandb_cfg.get("notes")
    if notes:
        os.environ.setdefault("WANDB_NOTES", str(notes))

    if experiment_name:
        os.environ.setdefault("WANDB_NAME", str(experiment_name))


def init_weave_if_enabled(logging_cfg: DictConfig) -> bool:
    """
    Initialize Weave tracing when explicitly enabled in logging config.
    """
    weave_cfg = logging_cfg.get("weave") or {}
    if not bool(weave_cfg.get("enabled", False)):
        return False

    wandb_cfg = logging_cfg.get("wandb") or {}
    weave_project = weave_cfg.get("project") or wandb_cfg.get("project")
    if not weave_project:
        raise ValueError(
            "logging.weave.enabled=true requires `logging.weave.project` or `logging.wandb.project`."
        )

    entity = wandb_cfg.get("entity")
    weave_project_text = str(weave_project).strip()
    if entity and "/" not in weave_project_text:
        weave_project_text = f"{entity}/{weave_project_text}"

    settings: dict[str, Any] = {}
    if weave_cfg.get("print_call_link") is not None:
        settings["print_call_link"] = bool(weave_cfg.get("print_call_link"))
    if weave_cfg.get("implicitly_patch_integrations") is not None:
        settings["implicitly_patch_integrations"] = bool(
            weave_cfg.get("implicitly_patch_integrations")
        )

    try:
        import weave
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "Weave is enabled but could not be imported (%s). Continuing without Weave.",
            exc,
        )
        return False

    if settings:
        weave.init(weave_project_text, settings=settings)
    else:
        weave.init(weave_project_text)

    print(f"[Weave] enabled: project={weave_project_text}")
    return True


def suppress_noisy_library_logs() -> None:
    """
    Reduce noisy INFO logs from HF/http clients during train/preprocess/eval.
    """
    for logger_name in (
        "httpx",
        "httpcore",
        "huggingface_hub",
        "transformers",
        "datasets",
    ):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
