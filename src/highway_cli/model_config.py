from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping


ALLOWED_ADAPTERS = {"openai-compatible"}
ALLOWED_RESPONSE_FORMATS = {"json_object", "none"}
ALLOWED_REASONING_EFFORTS = {
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
}


def _load_document(path: Path) -> Dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        value = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as error:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "YAML configuration requires PyYAML; use JSON or install the package"
            ) from error
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be an object: {path}")
    return value


def load_model_config(path: Path) -> Dict[str, Any]:
    """Load and validate a secret-free model configuration.

    API keys are deliberately excluded from the file format.  A configuration
    names an environment variable instead so configs and manifests remain safe
    to publish and exchange.
    """

    value = _load_document(Path(path))
    model = value["model"] if isinstance(value.get("model"), dict) else value
    if not isinstance(model, dict):
        raise ValueError("The model configuration must be an object")
    return validate_model_config(model)


def validate_model_config(value: Mapping[str, Any]) -> Dict[str, Any]:
    unknown_secret_fields = {"api_key", "token", "secret"}.intersection(value)
    if unknown_secret_fields:
        raise ValueError(
            "Do not store credentials in config files; use api_key_env instead"
        )

    adapter = str(value.get("adapter", "openai-compatible"))
    if adapter not in ALLOWED_ADAPTERS:
        raise ValueError(f"Unsupported model adapter: {adapter}")
    model = str(value.get("model", "")).strip()
    if not model:
        raise ValueError("Model configuration requires a non-empty model identifier")
    base_url = str(value.get("base_url", "")).strip()
    if not base_url:
        raise ValueError("Model configuration requires base_url")
    api_key_required = bool(value.get("api_key_required", True))
    api_key_env = str(value.get("api_key_env", "OPENROUTER_API_KEY")).strip()
    if api_key_required and not api_key_env:
        raise ValueError("api_key_env must not be empty")

    response_format = str(value.get("response_format", "json_object"))
    if response_format not in ALLOWED_RESPONSE_FORMATS:
        raise ValueError(
            f"response_format must be one of {sorted(ALLOWED_RESPONSE_FORMATS)}"
        )
    retries = int(value.get("format_retries", 1))
    if retries < 0 or retries > 5:
        raise ValueError("format_retries must be between 0 and 5")
    timeout_s = float(value.get("timeout_s", 30.0))
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    max_output_tokens = int(value.get("max_output_tokens", 128))
    if max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be positive")

    reasoning_effort = str(value.get("reasoning_effort", "none"))
    if reasoning_effort not in ALLOWED_REASONING_EFFORTS:
        raise ValueError(
            f"reasoning_effort must be one of {sorted(ALLOWED_REASONING_EFFORTS)}"
        )

    return {
        "adapter": adapter,
        "model": model,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "api_key_required": api_key_required,
        "timeout_s": timeout_s,
        "temperature": float(value.get("temperature", 0.0)),
        "max_output_tokens": max_output_tokens,
        "reasoning_effort": reasoning_effort,
        "format_retries": retries,
        "response_format": response_format,
    }


def load_conditions(path: Path) -> Any:
    value = _load_document(Path(path))
    return value.get("conditions", value)
