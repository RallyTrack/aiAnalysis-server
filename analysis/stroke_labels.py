"""Stroke classification canonical label schema.

이 모듈은 RallyTrack 스트로크 분류의 **유일한 진실의 원천(single source of truth)** 으로
다음 세 가지를 제공한다.

  1) ``CANONICAL_COLUMNS`` — ``data/labels/stroke_labels.csv`` 표 스키마.
  2) ``CLASS_NAMES_9`` / ``CLASS_NAMES_6_PRO`` / ``CLASS_NAMES_5_AMATEUR``
     세 가지 분류 체계의 클래스 순서.
  3) ``map_9_to_6_pro()`` / ``map_9_to_5_amateur()`` / ``map_kaggle_to_9()``
     9-class 라벨을 coarser 체계로 매핑하는 함수.

설계 원칙
---------
- **canonical = 9-class**: 모든 데이터(VideoBadminton, AICUP, Pro 방송, Kaggle)는 9-class
  로 저장된다. 더 거친 체계는 학습/평가 시 매핑 함수로만 파생.
- **Defense는 학습 제외**: ``Defense`` 는 "수비 상황" 이라는 모호한 상태 라벨이라
  매핑 시 ``None`` 으로 반환된다 (sample 자체를 학습에서 빼라는 신호).
- **Kaggle 6-class 호환**: Kaggle 의 ``forehand/backhand × stroke`` 6-class 는
  ``map_kaggle_to_9()`` 로 canonical 9-class 에 흡수된다. handedness 축은 별도 컬럼.
"""
from __future__ import annotations

from typing import Optional


# ──────────────────────────────────────────────────────────────────────
# 1) Canonical CSV 스키마 — data/labels/stroke_labels.csv
# ──────────────────────────────────────────────────────────────────────
# 각 row = 한 ImpactEvent. video_id / rally_id / hit_number 가 GroupKFold 시
# group 단위 후보가 된다 (video_id 우선, 부족하면 rally_id 로 fallback).
CANONICAL_COLUMNS: list[str] = [
    "sample_id",          # 전역 유일 식별자 (video_id + frame 또는 clip_stem 등)
    "video_id",           # GroupKFold 최우선 그룹 키 (촬영 세션 / 경기 단위)
    "rally_id",           # 더 세밀한 그룹 (랠리 단위). 없으면 video_id 와 동일.
    "hit_number",         # 랠리 내 타점 순서 (0-indexed). VideoBadminton 처럼 클립이
                          # 단일 스트로크면 0.
    "video_path",         # 영상 파일 경로 (REPO_ROOT 기준 상대경로)
    "frame",              # 영상 내 hit frame index (0-indexed)
    "time_sec",           # hit 시각 (frame / fps). 메타 충분치 않으면 -1.
    "owner",              # "top" / "bottom" / "" (미상)
    "label_9class",       # canonical 9-class 라벨 (Serve..Push). 매핑 전 원본.
    "label_source",       # videobadminton | aicup | manual_pro | kaggle | vit_teacher | rule_prior
    "verified",           # bool — 사람이 직접 검증했는지 (true/false)
    "confidence_label",   # 라벨러 자신감 (1=low, 2=mid, 3=high). 미상이면 빈 값.
    "handedness",         # forehand | backhand | "" (Kaggle 외엔 대부분 미상)
    "notes",              # 자유 텍스트
    "vb_class",           # VideoBadminton 원본 클래스 (provenance). 그 외엔 빈 값.
]


# ──────────────────────────────────────────────────────────────────────
# 2) 클래스 체계 정의
# ──────────────────────────────────────────────────────────────────────
# wish44165 / AICUP 와 동일 순서. argmax(0-indexed) → label(1-indexed)
CLASS_NAMES_9: list[str] = [
    "Serve",     # 0
    "Defense",   # 1  — 학습 제외 권장
    "Lob",       # 2
    "Smash",     # 3
    "Drop",      # 4
    "Drive",     # 5
    "Net",       # 6
    "Clear",     # 7
    "Push",      # 8
]

# 프로 방송 영상 분류용 6-class. Lob/Push 는 Clear/Drive 로 흡수, Defense 는 제외.
CLASS_NAMES_6_PRO: list[str] = [
    "Serve",
    "Smash",
    "Clear",
    "Drive",
    "Drop",
    "Net",
]

# 아마추어 5-class. Drop+Net 통합 ("짧은 타구"), Clear+Lob 통합 ("길게 올림"),
# Push 는 Drive 로 흡수. Serve 는 유지 (시작 시점 식별).
# 통합 라벨명은 대표 클래스 이름으로 통일 ("Net" 이 Drop+Net 의 라벨).
CLASS_NAMES_5_AMATEUR: list[str] = [
    "Serve",
    "Smash",
    "Clear",
    "Drive",
    "Net",
]

# 아마추어 4-class (현재 채택). 5-class 에서 Net 폐기 → 모든 "느리게 띄우는" 류는
# Clear 로 흡수, 모든 "빠르게 보내는" 류는 Drive 로 흡수.
# 사용자 정의 (2026-05-16): "로브, 하이클리어, 네트 → Clear; 드롭 → Drive".
CLASS_NAMES_4_AMATEUR: list[str] = [
    "Serve",
    "Smash",
    "Clear",
    "Drive",
]


# ──────────────────────────────────────────────────────────────────────
# 3) 매핑 함수
# ──────────────────────────────────────────────────────────────────────
# 9-class → 6-class (프로). 'Defense' 는 None 반환 (학습에서 제외 신호).
_MAP_9_TO_6_PRO: dict[str, Optional[str]] = {
    "Serve": "Serve",
    "Defense": None,   # 제외
    "Lob": "Clear",
    "Smash": "Smash",
    "Drop": "Drop",
    "Drive": "Drive",
    "Net": "Net",
    "Clear": "Clear",
    "Push": "Drive",
}

# 9-class → 5-class (아마추어). Defense 제외 + Net/Drop 통합.
_MAP_9_TO_5_AMATEUR: dict[str, Optional[str]] = {
    "Serve": "Serve",
    "Defense": None,
    "Lob": "Clear",
    "Smash": "Smash",
    "Drop": "Net",     # 짧은 타구 통합
    "Drive": "Drive",
    "Net": "Net",
    "Clear": "Clear",
    "Push": "Drive",
}

# 9-class → 4-class (아마추어, 채택본). Lob/Net 모두 Clear 로, Drop 은 Drive 로.
_MAP_9_TO_4_AMATEUR: dict[str, Optional[str]] = {
    "Serve": "Serve",
    "Defense": None,
    "Lob": "Clear",     # 로브 → 길게 띄움
    "Smash": "Smash",
    "Drop": "Drive",    # 드랍 → 가공된 결과는 Drive 와 비슷
    "Drive": "Drive",
    "Net": "Clear",     # 네트 → 일단 "어딘가 띄움" 카테고리
    "Clear": "Clear",
    "Push": "Drive",
}

# Kaggle 6-class → canonical 9-class. handedness 는 별도 처리.
_MAP_KAGGLE_TO_9: dict[str, str] = {
    "forehand_drive": "Drive",
    "backhand_drive": "Drive",
    "forehand_clear": "Clear",
    "forehand_lift": "Clear",      # underhand lift = Clear 의 underhand 버전
    "forehand_net_shot": "Net",
    "backhand_net_shot": "Net",
}


def map_9_to_6_pro(label_9: str) -> Optional[str]:
    """9-class 라벨을 프로 6-class 라벨로 매핑.

    Args:
        label_9: ``CLASS_NAMES_9`` 중 하나.

    Returns:
        ``CLASS_NAMES_6_PRO`` 중 하나, 또는 학습 제외 신호인 경우 ``None``.

    Raises:
        KeyError: 알 수 없는 9-class 라벨인 경우. 데이터 오류 신호이므로 raise.
    """
    if label_9 not in _MAP_9_TO_6_PRO:
        raise KeyError(f"unknown 9-class label: {label_9!r}")
    return _MAP_9_TO_6_PRO[label_9]


def map_9_to_5_amateur(label_9: str) -> Optional[str]:
    """9-class 라벨을 아마추어 5-class 라벨로 매핑."""
    if label_9 not in _MAP_9_TO_5_AMATEUR:
        raise KeyError(f"unknown 9-class label: {label_9!r}")
    return _MAP_9_TO_5_AMATEUR[label_9]


def map_9_to_4_amateur(label_9: str) -> Optional[str]:
    """9-class 라벨을 아마추어 4-class 라벨로 매핑 (현재 채택본)."""
    if label_9 not in _MAP_9_TO_4_AMATEUR:
        raise KeyError(f"unknown 9-class label: {label_9!r}")
    return _MAP_9_TO_4_AMATEUR[label_9]


def map_kaggle_to_9(kaggle_class: str) -> str:
    """Kaggle ``forehand_*/backhand_*`` 클래스명을 canonical 9-class 로 매핑.

    handedness 정보는 손실되므로 호출자가 별도로 ``handedness`` 컬럼에 저장해야 한다.
    """
    if kaggle_class not in _MAP_KAGGLE_TO_9:
        raise KeyError(f"unknown Kaggle class: {kaggle_class!r}")
    return _MAP_KAGGLE_TO_9[kaggle_class]


def kaggle_handedness(kaggle_class: str) -> str:
    """Kaggle 클래스명에서 handedness (forehand/backhand) 추출."""
    if kaggle_class.startswith("forehand_"):
        return "forehand"
    if kaggle_class.startswith("backhand_"):
        return "backhand"
    raise ValueError(f"cannot infer handedness from {kaggle_class!r}")
