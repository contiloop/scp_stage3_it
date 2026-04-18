from __future__ import annotations

import json
import os
import logging
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, ListConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UNSLOTH_BACKEND_AUTO = "auto"
UNSLOTH_BACKEND_VISION = "vision"
UNSLOTH_BACKEND_LANGUAGE = "language"
UNSLOTH_BACKENDS = {
    UNSLOTH_BACKEND_AUTO,
    UNSLOTH_BACKEND_VISION,
    UNSLOTH_BACKEND_LANGUAGE,
}

_MULTIMODAL_MODEL_TYPES = {
    "gemma3",
    "gemma3n",
    "gemma4",
    "idefics",
    "idefics2",
    "idefics3",
    "llava",
    "mllama",
    "paligemma",
    "qwen2vl",
    "qwen25vl",
    "qwen3vl",
}

def resolve_workspace_path(path_like: str | Path) -> Path:
    path = Path(str(path_like))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def normalize_unsloth_backend(value: Any) -> str:
    text = str(value).strip().lower() if value is not None else UNSLOTH_BACKEND_AUTO
    aliases = {
        "": UNSLOTH_BACKEND_AUTO,
        "auto": UNSLOTH_BACKEND_AUTO,
        "text": UNSLOTH_BACKEND_LANGUAGE,
        "language": UNSLOTH_BACKEND_LANGUAGE,
        "llm": UNSLOTH_BACKEND_LANGUAGE,
        "vision": UNSLOTH_BACKEND_VISION,
        "vlm": UNSLOTH_BACKEND_VISION,
    }
    normalized = aliases.get(text, text)
    if normalized not in UNSLOTH_BACKENDS:
        raise ValueError(
            f"Unsupported model backend '{value}'. Expected one of: "
            f"{sorted(UNSLOTH_BACKENDS)}"
        )
    return normalized


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _normalize_model_type(model_type: Any) -> str:
    return str(model_type).strip().lower().replace("-", "").replace("_", "").replace(".", "")


def _is_multimodal_config(config: dict[str, Any] | None) -> bool:
    if not isinstance(config, dict):
        return False

    if _normalize_model_type(config.get("model_type")) in _MULTIMODAL_MODEL_TYPES:
        return True

    multimodal_keys = (
        "audio_config",
        "audio_seq_length",
        "feature_extractor",
        "image_seq_length",
        "image_token_index",
        "mm_projector_type",
        "video_config",
        "video_seq_length",
        "vision_config",
        "vision_feature_layer",
        "vision_tower",
    )
    if any(key in config for key in multimodal_keys):
        return True

    architectures = " ".join(str(name) for name in config.get("architectures") or [])
    if "ImageTextToText" in architectures or "ConditionalGeneration" in architectures:
        model_type = _normalize_model_type(config.get("model_type"))
        if model_type.startswith("gemma") or model_type in _MULTIMODAL_MODEL_TYPES:
            return True

    return False


def _is_multimodal_processor_config(config: dict[str, Any] | None) -> bool:
    if not isinstance(config, dict):
        return False

    if any(key in config for key in ("image_processor", "video_processor", "feature_extractor")):
        return True

    processor_class = str(config.get("processor_class", "")).strip().lower()
    if processor_class and processor_class.endswith("processor"):
        return True

    return False


def _detect_backend_from_repo_files(
    repo_id: str,
    local_files_only: bool,
) -> tuple[str, str] | None:
    try:
        from huggingface_hub import hf_hub_download
    except Exception:
        return None

    config_data = None
    processor_data = None
    preprocessor_data = None

    for filename, target in (
        ("config.json", "config"),
        ("processor_config.json", "processor"),
        ("preprocessor_config.json", "preprocessor"),
    ):
        try:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="model",
                local_files_only=bool(local_files_only),
            )
        except Exception:
            continue

        loaded = _load_json_if_exists(Path(downloaded))
        if target == "config":
            config_data = loaded
        elif target == "processor":
            processor_data = loaded
        else:
            preprocessor_data = loaded

    if _is_multimodal_config(config_data):
        return UNSLOTH_BACKEND_VISION, "remote config indicates multimodal model"
    if _is_multimodal_processor_config(processor_data):
        return UNSLOTH_BACKEND_VISION, "remote processor_config indicates multimodal model"
    if _is_multimodal_processor_config(preprocessor_data):
        return UNSLOTH_BACKEND_VISION, "remote preprocessor_config indicates multimodal model"

    if any(item is not None for item in (config_data, processor_data, preprocessor_data)):
        return UNSLOTH_BACKEND_LANGUAGE, "remote metadata indicates text-only model"

    return None


def resolve_unsloth_backend(
    path_or_repo: str | Path,
    preferred_backend: Any = None,
    local_files_only: bool = False,
) -> tuple[str, str]:
    requested = normalize_unsloth_backend(preferred_backend)
    if requested != UNSLOTH_BACKEND_AUTO:
        return requested, f"config backend={requested}"

    candidate = Path(str(path_or_repo))
    if candidate.exists():
        local_dir = candidate if candidate.is_dir() else candidate.parent
        config_data = _load_json_if_exists(local_dir / "config.json")
        if _is_multimodal_config(config_data):
            return UNSLOTH_BACKEND_VISION, "local config indicates multimodal model"

        processor_data = _load_json_if_exists(local_dir / "processor_config.json")
        if _is_multimodal_processor_config(processor_data):
            return UNSLOTH_BACKEND_VISION, "local processor_config indicates multimodal model"

        preprocessor_data = _load_json_if_exists(local_dir / "preprocessor_config.json")
        if _is_multimodal_processor_config(preprocessor_data):
            return UNSLOTH_BACKEND_VISION, "local preprocessor_config indicates multimodal model"

    remote_detected = _detect_backend_from_repo_files(
        repo_id=str(path_or_repo),
        local_files_only=bool(local_files_only),
    )
    if remote_detected is not None:
        return remote_detected

    return UNSLOTH_BACKEND_LANGUAGE, "metadata unavailable; default text backend"


def resolve_unsloth_backend_order(
    path_or_repo: str | Path,
    preferred_backend: Any = None,
    local_files_only: bool = False,
) -> tuple[list[str], str]:
    primary, reason = resolve_unsloth_backend(
        path_or_repo=path_or_repo,
        preferred_backend=preferred_backend,
        local_files_only=local_files_only,
    )
    fallback = (
        UNSLOTH_BACKEND_LANGUAGE
        if primary == UNSLOTH_BACKEND_VISION
        else UNSLOTH_BACKEND_VISION
    )
    return [primary, fallback], reason


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
