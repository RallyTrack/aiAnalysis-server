"""Analysis-mode settings that are safe to validate without loading ML models."""

from __future__ import annotations

from typing import Any


ANALYSIS_MODE_PROFILES: dict[str, dict[str, Any]] = {
    "pro": {
        "tracknet_batch_size": 4,
        "pose_conf_threshold": 0.3,
        "pose_model_conf": 0.3,
        "run_near_miss_rescue": False,
    },
    "amateur": {
        "tracknet_batch_size": 1,
        "pose_conf_threshold": 0.2,
        "pose_model_conf": 0.2,
        "run_near_miss_rescue": True,
    },
}


def get_analysis_mode_profile(mode: str | None) -> tuple[str, dict[str, Any]]:
    """Return a normalized mode and an isolated copy of its runtime settings."""
    normalized = mode if mode in ANALYSIS_MODE_PROFILES else "pro"
    return normalized, ANALYSIS_MODE_PROFILES[normalized].copy()
