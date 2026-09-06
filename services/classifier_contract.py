"""Helpers for reporting the classifier contract without importing ML libraries."""

from __future__ import annotations

from typing import Any, Iterable


def describe_classifier(mode: str, model: Any, vit_available: bool) -> dict[str, Any]:
    """Describe the labels a loaded feature model can actually predict."""
    if model is None:
        feature_model = None
        feature_scheme = None
        trained_labels: list[str] = []
    else:
        feature_model = getattr(model, "model_name", "unknown")
        feature_scheme = getattr(model, "label_scheme", "unknown")
        raw_class_names = getattr(model, "class_names", None)
        raw_class_indices = getattr(model, "classes_", None)
        class_names = list(raw_class_names) if raw_class_names is not None else []
        class_indices = (
            [int(index) for index in list(raw_class_indices)]
            if raw_class_indices is not None
            else []
        )
        trained_labels = [
            class_names[index]
            for index in class_indices
            if 0 <= index < len(class_names)
        ]

    return {
        "analysis_mode": mode,
        "feature_model": feature_model,
        "feature_scheme": feature_scheme,
        "trained_label_count": len(trained_labels),
        "trained_labels": trained_labels,
        "vit_fallback_scheme": "9class" if vit_available else None,
    }


def format_classifier_contract(contract: dict[str, Any]) -> str:
    labels = ",".join(contract["trained_labels"]) or "none"
    return (
        "[StrokeContract] "
        f"mode={contract['analysis_mode']} "
        f"feature_model={contract['feature_model'] or 'missing'} "
        f"scheme={contract['feature_scheme'] or 'none'} "
        f"trained_labels={contract['trained_label_count']}[{labels}] "
        f"vit_fallback={contract['vit_fallback_scheme'] or 'missing'}"
    )


def collect_stroke_class_schemes(hits_data: Iterable[dict[str, Any]]) -> list[str]:
    """Collect the actual classifier schemas represented in API hit results."""
    schemes: set[str] = set()
    for hit in hits_data:
        scheme = hit.get("stroke_class_scheme")
        if isinstance(scheme, str) and scheme:
            schemes.add(scheme)
        elif hit.get("stroke_source") == "vit_only":
            schemes.add("9class")
    return sorted(schemes)
