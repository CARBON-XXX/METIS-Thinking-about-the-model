"""
METIS Pipeline — YAML Configuration System

Replaces hardcoded dataclass defaults with YAML files for easy management
of different experiment profiles (1.5B dev, 7B staging, 72B production).

Usage:
    from metis.pipeline.yaml_config import load_config, save_config

    # Load from YAML (with dataclass defaults as fallback)
    config = load_config("configs/experiment_70b.yaml")

    # Save current config for reproducibility
    save_config(config, "output/experiment_config.yaml")

    # Override specific fields from CLI
    config = load_config("configs/base.yaml", overrides={"model_name": "meta-llama/..."})

Preset profiles:
    configs/dev.yaml      — 1.5B model, 20 prompts, fast iteration
    configs/staging.yaml  — 7B model, 100 prompts, validation
    configs/prod.yaml     — 72B model, 300 prompts, full experiment
"""
from __future__ import annotations

import logging
import os
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _try_import_yaml() -> Any:
    """Import PyYAML with graceful fallback."""
    try:
        import yaml
        return yaml
    except ImportError:
        logger.warning(
            "PyYAML not installed. YAML config support disabled. "
            "Install with: pip install pyyaml"
        )
        return None


def load_config(
    path: Union[str, Path],
    config_class: Optional[Type[T]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> T:
    """
    Load configuration from a YAML file into a dataclass.

    Args:
        path: Path to YAML config file
        config_class: Target dataclass type (default: ExperimentConfig)
        overrides: Dict of field overrides applied after loading YAML

    Returns:
        Populated config dataclass instance

    Falls back to defaults if YAML file doesn't exist or PyYAML is missing.
    """
    if config_class is None:
        from metis.pipeline.config import ExperimentConfig
        config_class = ExperimentConfig

    yaml = _try_import_yaml()
    raw: Dict[str, Any] = {}

    path = Path(path)
    if yaml and path.exists():
        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        if isinstance(loaded, dict):
            raw = loaded
            logger.info(f"[Config] Loaded {len(raw)} fields from {path}")
        else:
            logger.warning(f"[Config] {path} did not contain a mapping, using defaults")
    elif not path.exists():
        logger.info(f"[Config] {path} not found, using defaults")

    # Apply CLI overrides
    if overrides:
        raw.update(overrides)

    # Filter to only known fields
    valid_fields = {f.name for f in fields(config_class)}
    filtered = {k: v for k, v in raw.items() if k in valid_fields}
    unknown = set(raw.keys()) - valid_fields
    if unknown:
        logger.warning(f"[Config] Unknown fields ignored: {unknown}")

    return config_class(**filtered)


def save_config(config: Any, path: Union[str, Path]) -> None:
    """
    Save a dataclass config to YAML for reproducibility.

    Args:
        config: Dataclass instance to serialize
        path: Output YAML file path
    """
    yaml = _try_import_yaml()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = asdict(config)

    if yaml:
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        logger.info(f"[Config] Saved config to {path}")
    else:
        # Fallback: write as simple key=value
        import json
        with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"[Config] Saved config as JSON to {path.with_suffix('.json')} (PyYAML not available)")


# ─────────────────────────────────────────────────────
# Preset Profiles
# ─────────────────────────────────────────────────────

_PRESETS: Dict[str, Dict[str, Any]] = {
    "dev": {
        "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "n_train_prompts": 20,
        "n_eval_prompts": 10,
        "n_samples_per_prompt": 4,
        "max_new_tokens": 512,
        "dpo_epochs": 1,
        "dpo_batch_size": 4,
        "lora_r": 16,
        "lora_alpha": 32,
        "run_benchmarks": False,
        "truthfulqa_questions": 50,
        "mmlu_subjects": 3,
        "mmlu_per_subject": 10,
    },
    "staging": {
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "n_train_prompts": 100,
        "n_eval_prompts": 30,
        "n_samples_per_prompt": 8,
        "max_new_tokens": 1024,
        "dpo_epochs": 3,
        "dpo_batch_size": 8,
        "lora_r": 32,
        "lora_alpha": 64,
        "run_benchmarks": True,
        "truthfulqa_questions": 100,
        "mmlu_subjects": 5,
        "mmlu_per_subject": 20,
    },
    "prod": {
        "model_name": "Qwen/Qwen2.5-72B-Instruct",
        "n_train_prompts": 300,
        "n_eval_prompts": 50,
        "n_samples_per_prompt": 16,
        "max_new_tokens": 1024,
        "dpo_epochs": 3,
        "dpo_batch_size": 8,
        "dpo_gradient_accumulation": 4,
        "lora_r": 64,
        "lora_alpha": 128,
        "run_benchmarks": True,
        "truthfulqa_questions": 200,
        "mmlu_subjects": 10,
        "mmlu_per_subject": 30,
    },
    "dgx_full": {
        "model_name": "Qwen/Qwen2.5-72B-Instruct",
        "n_train_prompts": 500,
        "n_eval_prompts": 100,
        "n_samples_per_prompt": 32,
        "max_new_tokens": 2048,
        "dpo_epochs": 5,
        "dpo_batch_size": 16,
        "dpo_gradient_accumulation": 2,
        "dpo_max_length": 4096,
        "lora_r": 128,
        "lora_alpha": 256,
        "gradient_checkpointing": False,
        "run_benchmarks": True,
        "truthfulqa_questions": 400,
        "mmlu_subjects": 20,
        "mmlu_per_subject": 50,
    },
}


def load_preset(name: str, config_class: Optional[Type[T]] = None, **overrides: Any) -> T:
    """
    Load a named preset profile.

    Args:
        name: Preset name ("dev", "staging", "prod", "dgx_70b")
        config_class: Target dataclass type
        **overrides: Additional field overrides

    Returns:
        Config instance with preset + overrides applied
    """
    if name not in _PRESETS:
        available = ", ".join(_PRESETS.keys())
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")

    if config_class is None:
        from metis.pipeline.config import ExperimentConfig
        config_class = ExperimentConfig

    merged = {**_PRESETS[name], **overrides}
    valid_fields = {f.name for f in fields(config_class)}
    filtered = {k: v for k, v in merged.items() if k in valid_fields}

    logger.info(f"[Config] Loaded preset '{name}' ({len(filtered)} fields)")
    return config_class(**filtered)


def generate_preset_files(output_dir: Union[str, Path] = "configs") -> None:
    """Generate YAML preset files for all built-in profiles."""
    yaml = _try_import_yaml()
    if not yaml:
        logger.error("PyYAML required. Install with: pip install pyyaml")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, preset in _PRESETS.items():
        path = output_dir / f"{name}.yaml"
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(preset, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        logger.info(f"[Config] Generated {path}")
