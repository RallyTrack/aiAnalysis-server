"""Stroke 분류용 feature extractor.

각 샘플(``ImpactEvent`` 한 개) 에 대해 4개 카테고리 feature 를 dict 로 반환한다.

  ``traj_*``   shuttle trajectory feature (TrackNet CSV 필요)
  ``court_*``  코트/선수 컨텍스트 feature (court 호모그래피 필요)
  ``pose_*``   YOLO pose feature (yolov8n-pose 가중치 필요)
  ``vit_*``    ViT teacher ensemble feature (5-fold 가중치 + GPU)

각 블록은 *입력 캐시가 없으면 NaN + missing indicator* 로 부드럽게 격하된다.
이 설계로 작은 데이터에서도 일부 feature 만으로 baseline 학습을 시작하고,
이후 trajectory/pose 캐시가 쌓이면 추가 학습으로 lift 측정이 가능하다.

Window 정책
------------
타점 frame t 기준,

  - pre window:  ``[t - pre_offset, t - 1]`` (기본 -15..-1, 즉 직전 15 frame)
  - post window: ``[t + 1, t + post_offset]``

pre 가 비는 경우(예: VideoBadminton 처럼 t=0) 는 NaN + ``has_pre=0`` 로 표시.

호출 패턴
---------
파이프라인은 보통 다음과 같이 캐시를 미리 만들어두고 feature 만 합치는 식:

    1. ``scripts/compute_vit_teacher_probs.py``  →  ``data/features/vit_teacher.parquet``
    2. (옵션) TrackNet 캐시  →  ``data/cache/tracknet/<video_stem>_ball.csv``
    3. (옵션) Pose 캐시       →  ``data/cache/pose/<video_stem>.parquet``
    4. ``scripts/build_stroke_dataset.py`` 가 row 별로 캐시 lookup → ``compute_features()``
       호출 → 결과 누적해 parquet 저장.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────
# 공통 상수 / 유틸
# ──────────────────────────────────────────────────────────────────────
EPS = 1e-6  # division-by-zero 방지 작은 값

# YOLOv8 COCO keypoint 인덱스. wrist/shoulder/elbow 만 사용.
_KP_LEFT_SHOULDER = 5
_KP_RIGHT_SHOULDER = 6
_KP_LEFT_ELBOW = 7
_KP_RIGHT_ELBOW = 8
_KP_LEFT_WRIST = 9
_KP_RIGHT_WRIST = 10


def _nan_dict(keys: list[str]) -> dict[str, float]:
    """주어진 키에 NaN 만 매핑된 dict 반환 (graceful fallback)."""
    return {k: float("nan") for k in keys}


def _safe_atan2(dy: float, dx: float) -> float:
    """0/0 도 NaN 처리."""
    if abs(dy) < EPS and abs(dx) < EPS:
        return float("nan")
    return float(np.arctan2(dy, dx))


def _entropy(probs: np.ndarray) -> float:
    """natural log 엔트로피. 9-class 기준 max = log(9) ≈ 2.197."""
    p = np.clip(probs, EPS, 1.0)
    return float(-np.sum(p * np.log(p)))


# ──────────────────────────────────────────────────────────────────────
# 1) Trajectory features
# ──────────────────────────────────────────────────────────────────────
TRAJ_KEYS: list[str] = [
    "traj_has_pre",
    "traj_has_post",
    "traj_pre_speed_mean",
    "traj_pre_speed_max",
    "traj_post_speed_mean",
    "traj_post_speed_max",
    "traj_speed_ratio_post_pre",
    "traj_pre_angle",
    "traj_post_angle",
    "traj_angle_change",
    "traj_pre_dx",
    "traj_pre_dy",
    "traj_post_dx",
    "traj_post_dy",
    "traj_post_vertical_dir",   # +1: 아래로, -1: 위로
    "traj_post_horizontal_dir", # +1: 오른쪽, -1: 왼쪽
    "traj_post_flight_distance",
    "traj_y_range_pre",
    "traj_y_range_post",
    "traj_net_cross_frames",    # net 통과까지 frame 수 (없으면 NaN)
    "traj_visibility_ratio",
]


def compute_trajectory_features(
    ball_df: Optional[pd.DataFrame],
    hit_frame: int,
    pre_window: int = 10,
    post_window: int = 10,
    pre_offset: int = 15,
    post_offset: int = 20,
    net_y: Optional[float] = None,
) -> dict[str, float]:
    """TrackNet output 으로 trajectory feature 계산.

    Args:
        ball_df: 컬럼 [Frame, X, Y, Visibility] (``src.models.tracknet.TrackingResult.df``).
            None 이면 모든 feature NaN + has_pre/has_post = 0.
        hit_frame: 임팩트 frame index.
        pre_window/post_window: hit 직전/직후로 *연속* 수집할 frame 수
            (실제 사용 window 는 hit-1..hit-pre_window / hit+1..hit+post_window).
        pre_offset/post_offset: ``ball_df`` 에서 검색할 최대 범위. window 안에 visibility
            가 부족할 때 더 넓게 보고 보간 / 평균.
        net_y: 네트의 y 픽셀 좌표 (homography 알면 전달). 없으면 net_cross 계산 skip.

    Returns:
        ``TRAJ_KEYS`` 키 dict.
    """
    out = _nan_dict(TRAJ_KEYS)
    out["traj_has_pre"] = 0.0
    out["traj_has_post"] = 0.0

    if ball_df is None or len(ball_df) == 0:
        return out

    # visibility 1 인 좌표만 사용. Frame index 별 lookup table 만들기.
    visible = ball_df[ball_df["Visibility"] == 1].copy()
    if visible.empty:
        return out

    # hit 주변 범위로 미리 자르기.
    lo = hit_frame - pre_offset
    hi = hit_frame + post_offset
    around = visible[(visible["Frame"] >= lo) & (visible["Frame"] <= hi)]
    if around.empty:
        return out

    # pre / post 분리.
    pre = around[around["Frame"] < hit_frame].sort_values("Frame").tail(pre_window)
    post = around[around["Frame"] > hit_frame].sort_values("Frame").head(post_window)

    out["traj_visibility_ratio"] = float(len(around)) / max(pre_offset + post_offset + 1, 1)

    # pre block.
    if len(pre) >= 2:
        out["traj_has_pre"] = 1.0
        dx_pre = np.diff(pre["X"].to_numpy())
        dy_pre = np.diff(pre["Y"].to_numpy())
        speed_pre = np.sqrt(dx_pre ** 2 + dy_pre ** 2)
        out["traj_pre_speed_mean"] = float(np.mean(speed_pre))
        out["traj_pre_speed_max"] = float(np.max(speed_pre))
        out["traj_pre_dx"] = float(pre["X"].iloc[-1] - pre["X"].iloc[0])
        out["traj_pre_dy"] = float(pre["Y"].iloc[-1] - pre["Y"].iloc[0])
        out["traj_pre_angle"] = _safe_atan2(out["traj_pre_dy"], out["traj_pre_dx"])
        out["traj_y_range_pre"] = float(pre["Y"].max() - pre["Y"].min())

    # post block.
    if len(post) >= 2:
        out["traj_has_post"] = 1.0
        dx_post = np.diff(post["X"].to_numpy())
        dy_post = np.diff(post["Y"].to_numpy())
        speed_post = np.sqrt(dx_post ** 2 + dy_post ** 2)
        out["traj_post_speed_mean"] = float(np.mean(speed_post))
        out["traj_post_speed_max"] = float(np.max(speed_post))
        out["traj_post_dx"] = float(post["X"].iloc[-1] - post["X"].iloc[0])
        out["traj_post_dy"] = float(post["Y"].iloc[-1] - post["Y"].iloc[0])
        out["traj_post_angle"] = _safe_atan2(out["traj_post_dy"], out["traj_post_dx"])
        out["traj_y_range_post"] = float(post["Y"].max() - post["Y"].min())
        out["traj_post_flight_distance"] = float(
            np.sqrt(out["traj_post_dx"] ** 2 + out["traj_post_dy"] ** 2),
        )
        out["traj_post_vertical_dir"] = 1.0 if out["traj_post_dy"] > 0 else -1.0
        out["traj_post_horizontal_dir"] = 1.0 if out["traj_post_dx"] > 0 else -1.0

    # 결합 feature.
    if out["traj_has_pre"] and out["traj_has_post"]:
        out["traj_speed_ratio_post_pre"] = (
            out["traj_post_speed_mean"] / (out["traj_pre_speed_mean"] + EPS)
        )
        ang_pre = out["traj_pre_angle"]
        ang_post = out["traj_post_angle"]
        if not (np.isnan(ang_pre) or np.isnan(ang_post)):
            diff = ang_post - ang_pre
            # [-pi, pi] 로 wrap
            diff = (diff + np.pi) % (2 * np.pi) - np.pi
            out["traj_angle_change"] = float(diff)

    # net crossing — post 좌표 시퀀스가 net_y 를 가로지르는 지점 찾기.
    if net_y is not None and len(post) >= 2:
        ys = post["Y"].to_numpy()
        sign = np.sign(ys - net_y)
        cross_idx = np.where(np.diff(sign) != 0)[0]
        if len(cross_idx) > 0:
            # post 의 cross_idx 첫 번째가 hit 후 몇 frame 째인지.
            first_cross_frame = int(post["Frame"].iloc[cross_idx[0] + 1])
            out["traj_net_cross_frames"] = float(first_cross_frame - hit_frame)

    return out


# ──────────────────────────────────────────────────────────────────────
# 2) Court/player context features
# ──────────────────────────────────────────────────────────────────────
COURT_KEYS: list[str] = [
    "court_has_homography",
    "court_impact_x",
    "court_impact_y",
    "court_top_half",
    "court_dist_to_net",
    "court_dist_to_top_baseline",
    "court_dist_to_bottom_baseline",
    "court_dist_to_left_sideline",
    "court_dist_to_right_sideline",
    "court_serve_candidate",     # 랠리 첫 hit 인지 (hit_number == 0)
    "court_rally_first_hit",
    "court_prev_hit_gap_sec",
    "court_next_hit_gap_sec",
]


def compute_court_features(
    impact_xy_pixel: Optional[tuple[float, float]],
    homography_matrix: Optional[np.ndarray],
    hit_number: int,
    prev_hit_gap_sec: Optional[float] = None,
    next_hit_gap_sec: Optional[float] = None,
    court_dims_m: tuple[float, float] = (6.1, 13.4),  # BWF 단식 너비, 길이
    net_y_m: float = 6.7,                              # 길이 중심
) -> dict[str, float]:
    """코트/선수 컨텍스트 feature.

    homography 가 있으면 픽셀 → 코트 좌표 (meters) 로 변환 후 거리 계산.
    homography 가 없으면 모든 위치 feature 는 NaN, ``court_has_homography=0``.

    Args:
        impact_xy_pixel: 임팩트 시점 셔틀 픽셀 좌표 (X, Y).
        homography_matrix: 3x3 픽셀→코트(m) 변환. ``None`` 이면 좌표 변환 skip.
        hit_number: 랠리 내 hit 순서 (0-indexed).
        prev_hit_gap_sec/next_hit_gap_sec: 이전/다음 hit 까지 시간 간격 (sec).
        court_dims_m: (width, length) BWF 단식 코트.
        net_y_m: 코트 중앙 net y 좌표.
    """
    out = _nan_dict(COURT_KEYS)
    out["court_has_homography"] = 0.0
    out["court_rally_first_hit"] = 1.0 if hit_number == 0 else 0.0
    # serve candidate: rally 첫 hit 이고 prev_hit_gap_sec 이 크거나 None (즉 새 랠리 시작).
    out["court_serve_candidate"] = 1.0 if hit_number == 0 else 0.0

    if prev_hit_gap_sec is not None:
        out["court_prev_hit_gap_sec"] = float(prev_hit_gap_sec)
    if next_hit_gap_sec is not None:
        out["court_next_hit_gap_sec"] = float(next_hit_gap_sec)

    if homography_matrix is None or impact_xy_pixel is None:
        return out

    # 픽셀 → 코트 좌표 (m).
    px = np.array([[impact_xy_pixel[0], impact_xy_pixel[1]]], dtype=np.float32)
    px_h = np.array([[px[0, 0], px[0, 1], 1.0]], dtype=np.float64)
    mapped = (homography_matrix @ px_h.T).T
    w = mapped[0, 2]
    if abs(w) < EPS:
        return out
    cx = float(mapped[0, 0] / w)
    cy = float(mapped[0, 1] / w)

    width_m, length_m = court_dims_m
    out["court_has_homography"] = 1.0
    out["court_impact_x"] = cx
    out["court_impact_y"] = cy
    out["court_top_half"] = 1.0 if cy > net_y_m else 0.0
    out["court_dist_to_net"] = abs(cy - net_y_m)
    out["court_dist_to_top_baseline"] = abs(cy - length_m)
    out["court_dist_to_bottom_baseline"] = abs(cy - 0.0)
    out["court_dist_to_left_sideline"] = abs(cx - 0.0)
    out["court_dist_to_right_sideline"] = abs(cx - width_m)
    return out


# ──────────────────────────────────────────────────────────────────────
# 3) Pose features
# ──────────────────────────────────────────────────────────────────────
POSE_KEYS: list[str] = [
    "pose_has_hitter",
    "pose_kp_conf_mean",
    "pose_left_wrist_x",
    "pose_left_wrist_y",
    "pose_right_wrist_x",
    "pose_right_wrist_y",
    "pose_left_wrist_dist_shuttle",
    "pose_right_wrist_dist_shuttle",
    "pose_closest_wrist_side",   # 0=left, 1=right
    "pose_left_elbow_angle",
    "pose_right_elbow_angle",
    "pose_shoulder_width",
    "pose_wrist_velocity_pre",
    "pose_wrist_velocity_post",
]


def _kp_xy(kp: np.ndarray, idx: int, conf_thresh: float = 0.3) -> Optional[tuple[float, float]]:
    """keypoint (17,3) 에서 idx 의 (x,y) 추출. conf < thresh → None."""
    if kp.shape[0] <= idx:
        return None
    x, y, c = kp[idx]
    if c < conf_thresh:
        return None
    return float(x), float(y)


def _angle_at(b: tuple[float, float], a: tuple[float, float], c: tuple[float, float]) -> float:
    """각 BAC 라디안 (B가 중심)."""
    v1 = np.array([a[0] - b[0], a[1] - b[1]])
    v2 = np.array([c[0] - b[0], c[1] - b[1]])
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < EPS or n2 < EPS:
        return float("nan")
    cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.arccos(cos))


def compute_pose_features(
    hitter_kp_at_hit: Optional[np.ndarray],          # (17, 3): YOLO 17-kp xyconf
    shuttle_xy_at_hit: Optional[tuple[float, float]],
    hitter_kp_pre: Optional[np.ndarray] = None,      # hit 보다 몇 프레임 전
    hitter_kp_post: Optional[np.ndarray] = None,     # hit 보다 몇 프레임 후
    conf_thresh: float = 0.3,
) -> dict[str, float]:
    """선수 keypoint 기반 feature.

    선수 한 명(hitter)의 17개 COCO keypoint 만 사용한다. owner 판정과 hitter
    선택은 상위 파이프라인 책임이고, 본 함수는 *이미 골라진 hitter* 의 kp 만 받는다.

    Args:
        hitter_kp_at_hit: (17,3) 또는 None (검출 실패).
        shuttle_xy_at_hit: 셔틀 픽셀 좌표 (X, Y) 또는 None.
        hitter_kp_pre/post: wrist velocity 계산용 인접 프레임 kp.
    """
    out = _nan_dict(POSE_KEYS)
    out["pose_has_hitter"] = 0.0

    if hitter_kp_at_hit is None:
        return out
    if hitter_kp_at_hit.shape != (17, 3):
        return out

    out["pose_has_hitter"] = 1.0
    out["pose_kp_conf_mean"] = float(hitter_kp_at_hit[:, 2].mean())

    lw = _kp_xy(hitter_kp_at_hit, _KP_LEFT_WRIST, conf_thresh)
    rw = _kp_xy(hitter_kp_at_hit, _KP_RIGHT_WRIST, conf_thresh)
    ls = _kp_xy(hitter_kp_at_hit, _KP_LEFT_SHOULDER, conf_thresh)
    rs = _kp_xy(hitter_kp_at_hit, _KP_RIGHT_SHOULDER, conf_thresh)
    le = _kp_xy(hitter_kp_at_hit, _KP_LEFT_ELBOW, conf_thresh)
    re_ = _kp_xy(hitter_kp_at_hit, _KP_RIGHT_ELBOW, conf_thresh)

    if lw is not None:
        out["pose_left_wrist_x"] = lw[0]
        out["pose_left_wrist_y"] = lw[1]
        if shuttle_xy_at_hit is not None:
            out["pose_left_wrist_dist_shuttle"] = float(
                np.hypot(lw[0] - shuttle_xy_at_hit[0], lw[1] - shuttle_xy_at_hit[1]),
            )
    if rw is not None:
        out["pose_right_wrist_x"] = rw[0]
        out["pose_right_wrist_y"] = rw[1]
        if shuttle_xy_at_hit is not None:
            out["pose_right_wrist_dist_shuttle"] = float(
                np.hypot(rw[0] - shuttle_xy_at_hit[0], rw[1] - shuttle_xy_at_hit[1]),
            )

    # closest wrist side: 셔틀에 더 가까운 손목 식별.
    ld = out["pose_left_wrist_dist_shuttle"]
    rd = out["pose_right_wrist_dist_shuttle"]
    if not (np.isnan(ld) or np.isnan(rd)):
        out["pose_closest_wrist_side"] = 0.0 if ld < rd else 1.0

    if ls is not None and le is not None and lw is not None:
        out["pose_left_elbow_angle"] = _angle_at(le, ls, lw)
    if rs is not None and re_ is not None and rw is not None:
        out["pose_right_elbow_angle"] = _angle_at(re_, rs, rw)
    if ls is not None and rs is not None:
        out["pose_shoulder_width"] = float(np.hypot(ls[0] - rs[0], ls[1] - rs[1]))

    # wrist velocity — closest wrist 기준.
    if not np.isnan(out["pose_closest_wrist_side"]):
        target_idx = _KP_LEFT_WRIST if out["pose_closest_wrist_side"] == 0.0 else _KP_RIGHT_WRIST
        cur = _kp_xy(hitter_kp_at_hit, target_idx, conf_thresh)
        if cur is not None and hitter_kp_pre is not None:
            pre_p = _kp_xy(hitter_kp_pre, target_idx, conf_thresh)
            if pre_p is not None:
                out["pose_wrist_velocity_pre"] = float(
                    np.hypot(cur[0] - pre_p[0], cur[1] - pre_p[1]),
                )
        if cur is not None and hitter_kp_post is not None:
            post_p = _kp_xy(hitter_kp_post, target_idx, conf_thresh)
            if post_p is not None:
                out["pose_wrist_velocity_post"] = float(
                    np.hypot(post_p[0] - cur[0], post_p[1] - cur[1]),
                )

    return out


# ──────────────────────────────────────────────────────────────────────
# 4) ViT teacher features
# ──────────────────────────────────────────────────────────────────────
# 9-class probs at offsets [-2, -1, 0, +1, +2].
# raw: 5 offsets × 9 class = 45 (보존)
# pooled: mean(9), max(9), entropy, top1_idx, top1_conf = 21
# 호출자는 이미 ensemble 평균된 9-class probs at 각 offset 을 전달.
VIT_OFFSETS: list[int] = [-2, -1, 0, 1, 2]
_VIT_RAW_KEYS = [
    f"vit_prob_off{o:+d}_c{c}" for o in VIT_OFFSETS for c in range(9)
]
_VIT_POOLED_KEYS = (
    [f"vit_mean_c{c}" for c in range(9)]
    + [f"vit_max_c{c}" for c in range(9)]
    + ["vit_entropy_mean", "vit_top1_idx", "vit_top1_conf"]
)
VIT_KEYS: list[str] = ["vit_has_probs"] + _VIT_RAW_KEYS + _VIT_POOLED_KEYS


def compute_vit_features(
    probs_by_offset: Optional[dict[int, np.ndarray]],
) -> dict[str, float]:
    """ViT teacher feature 추출.

    Args:
        probs_by_offset: ``{offset: probs9}`` dict. 누락 offset 은 NaN 처리.
            ``probs9`` 는 (9,) softmax 확률 (5-fold ensemble 평균 권장).

    Returns:
        ``VIT_KEYS`` 키 dict.
    """
    out = _nan_dict(VIT_KEYS)
    out["vit_has_probs"] = 0.0

    if probs_by_offset is None or len(probs_by_offset) == 0:
        return out

    # raw probs 채우기.
    available: list[np.ndarray] = []
    for o in VIT_OFFSETS:
        if o in probs_by_offset:
            p = np.asarray(probs_by_offset[o], dtype=np.float64)
            if p.shape == (9,):
                available.append(p)
                for c in range(9):
                    out[f"vit_prob_off{o:+d}_c{c}"] = float(p[c])

    if not available:
        return out
    out["vit_has_probs"] = 1.0

    # pooled aggregates.
    stack = np.stack(available, axis=0)  # (N_avail, 9)
    mean = stack.mean(axis=0)
    mx = stack.max(axis=0)
    for c in range(9):
        out[f"vit_mean_c{c}"] = float(mean[c])
        out[f"vit_max_c{c}"] = float(mx[c])
    out["vit_entropy_mean"] = _entropy(mean)
    top1 = int(np.argmax(mean))
    out["vit_top1_idx"] = float(top1)
    out["vit_top1_conf"] = float(mean[top1])
    return out


# ──────────────────────────────────────────────────────────────────────
# 5) Frame meta features (FPS, duration, hit position 등)
# ──────────────────────────────────────────────────────────────────────
META_KEYS: list[str] = [
    "meta_fps",
    "meta_n_frames",
    "meta_hit_frame",
    "meta_hit_rel_pos",   # hit_frame / n_frames (트리밍 클립에서 0 ≈ 시작)
    "meta_duration_sec",
    "meta_has_video_meta",
]


def compute_meta_features(video_path: Path, hit_frame: int) -> dict[str, float]:
    """영상 메타 (FPS, frame count, duration) + hit 위치 상대값."""
    out = _nan_dict(META_KEYS)
    out["meta_has_video_meta"] = 0.0
    out["meta_hit_frame"] = float(hit_frame)
    if not video_path.exists():
        return out
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return out
        fps = cap.get(cv2.CAP_PROP_FPS)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if n <= 0 or fps <= 0:
        return out
    out["meta_has_video_meta"] = 1.0
    out["meta_fps"] = float(fps)
    out["meta_n_frames"] = float(n)
    out["meta_duration_sec"] = float(n / fps)
    out["meta_hit_rel_pos"] = float(hit_frame) / float(n)
    return out


# ──────────────────────────────────────────────────────────────────────
# Top-level compose
# ──────────────────────────────────────────────────────────────────────
@dataclass
class FeatureInputs:
    """``compute_features()`` 호출 시 받는 모든 입력 묶음.

    각 캐시는 옵션이며 ``None`` 이면 해당 블록의 feature 는 NaN.
    """
    video_path: Path
    hit_frame: int
    hit_number: int = 0

    ball_df: Optional[pd.DataFrame] = None
    net_y: Optional[float] = None

    impact_xy_pixel: Optional[tuple[float, float]] = None
    homography_matrix: Optional[np.ndarray] = None
    prev_hit_gap_sec: Optional[float] = None
    next_hit_gap_sec: Optional[float] = None

    hitter_kp_at_hit: Optional[np.ndarray] = None
    hitter_kp_pre: Optional[np.ndarray] = None
    hitter_kp_post: Optional[np.ndarray] = None
    shuttle_xy_at_hit: Optional[tuple[float, float]] = None

    vit_probs_by_offset: Optional[dict[int, np.ndarray]] = None

    extra: dict = field(default_factory=dict)


ALL_FEATURE_KEYS: list[str] = META_KEYS + TRAJ_KEYS + COURT_KEYS + POSE_KEYS + VIT_KEYS


def compute_features(inputs: FeatureInputs) -> dict[str, float]:
    """5개 카테고리 feature 를 합쳐서 단일 dict 반환.

    누락 캐시는 NaN + missing indicator 로 격하된다.
    """
    feats: dict[str, float] = {}
    feats.update(compute_meta_features(inputs.video_path, inputs.hit_frame))
    feats.update(
        compute_trajectory_features(
            inputs.ball_df, inputs.hit_frame, net_y=inputs.net_y,
        ),
    )
    feats.update(
        compute_court_features(
            inputs.impact_xy_pixel,
            inputs.homography_matrix,
            inputs.hit_number,
            inputs.prev_hit_gap_sec,
            inputs.next_hit_gap_sec,
        ),
    )
    feats.update(
        compute_pose_features(
            inputs.hitter_kp_at_hit,
            inputs.shuttle_xy_at_hit,
            inputs.hitter_kp_pre,
            inputs.hitter_kp_post,
        ),
    )
    feats.update(compute_vit_features(inputs.vit_probs_by_offset))
    return feats
