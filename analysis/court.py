"""
RallyTrack - 코트 기하학 / 호모그래피 모듈

역할:
  - 영상 좌표(픽셀)를 미니맵 좌표(픽셀)로 변환하는 호모그래피 행렬 계산
  - 2D 미니맵 코트 그리기
  - 정규화 좌표 계산 (히트맵 생성용)

[수정 사항]
  - compute_homographies()에 user_corners 파라미터 추가
    · 제공 시: 사용자 지정 4점 기반 단식(Singles) 규격 변환
    · 미제공 시: COURT_CORNERS 비율 기반 자동 계산 (기존 동작)
  - _singles_dst_pts() 헬퍼 추가
    · 단식 코트 실제 규격(5.18 m × 13.4 m) 비율을 미니맵에 투영
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import COURT_CORNERS, COURT_LINES, MINIMAP_CONFIG

# ────────────────────────────────────────────────────────────
# 배드민턴 단식 코트 실제 치수 (국제 규격 BWF)
# ────────────────────────────────────────────────────────────
SINGLES_WIDTH_M  = 5.18   # 단식 코트 폭
SINGLES_LENGTH_M = 13.40  # 코트 전장


# ────────────────────────────────────────────────────────────
# 내부 헬퍼: 단식 비율을 반영한 미니맵 목적지 좌표
# ────────────────────────────────────────────────────────────

def _singles_dst_pts(
    minimap_w: int = MINIMAP_CONFIG["width"],
    minimap_h: int = MINIMAP_CONFIG["height"],
    pad: int       = MINIMAP_CONFIG["padding"],
) -> np.ndarray:
    """
    단식 코트(5.18 × 13.4 m) 비율을 유지하면서 미니맵 캔버스에
    수평 중앙 정렬된 목적지 꼭짓점 4개를 반환한다.

    반환 순서: [좌상, 우상, 우하, 좌하]
    """
    avail_w = minimap_w - 2 * pad  # 320 (기본값)
    avail_h = minimap_h - 2 * pad  # 560 (기본값)

    aspect = SINGLES_WIDTH_M / SINGLES_LENGTH_M  # ≈ 0.3866

    # 세로를 최대한 사용하고 가로를 비율에 맞춤
    court_h = avail_h
    court_w = int(court_h * aspect)

    # 가로 방향 중앙 정렬
    x_offset = pad + (avail_w - court_w) // 2
    y_offset = pad  # 세로는 패딩 그대로

    return np.array([
        [x_offset,           y_offset],           # 좌상
        [x_offset + court_w, y_offset],           # 우상
        [x_offset + court_w, y_offset + court_h], # 우하
        [x_offset,           y_offset + court_h], # 좌하
    ], dtype=np.float32)


# ────────────────────────────────────────────────────────────
# 호모그래피 계산
# ────────────────────────────────────────────────────────────

def compute_homographies(
    frame_w: int,
    frame_h: int,
    minimap_w: int = MINIMAP_CONFIG["width"],
    minimap_h: int = MINIMAP_CONFIG["height"],
    user_corners: Optional[np.ndarray] = None,
) -> dict:
    """
    영상 해상도를 기반으로 두 가지 호모그래피 행렬을 계산합니다.

    Args:
        frame_w, frame_h : 원본 영상 해상도
        minimap_w/h      : 미니맵 캔버스 크기
        user_corners     : shape (4,2) float32, 순서 [좌상, 우상, 우하, 좌하].
                           사용자가 지정한 단식 코트 꼭짓점(픽셀 좌표).
                           None이면 COURT_CORNERS 비율로 자동 계산.

    반환값 (dict):
        src_pts          : 영상 내 코트 코너 좌표 (4×2 float32)
        to_minimap       : 영상 → 미니맵 변환 행렬 (3×3)
        to_normalized    : 영상 → [0,1]² 정규화 행렬 (3×3)  ← 히트맵용
        minimap_pts      : 미니맵 내 코트 코너 좌표 (4×2 float32)
        net_y_minimap    : 미니맵 내 네트 Y 좌표 (코트 정중앙)
        coordinate_mode  : "user_singles" | "auto_ratio"  (디버그용)
    """
    pad = MINIMAP_CONFIG["padding"]

    # ── 소스 좌표 결정 ────────────────────────────────────────
    if user_corners is not None:
        src_pts = np.asarray(user_corners, dtype=np.float32).reshape(4, 2)
        coordinate_mode = "user_singles"
        print(f"[court] 사용자 지정 코너 사용: {src_pts.tolist()}")
    else:
        # COURT_CORNERS 비율 기반 자동 계산 (기존 동작)
        cc = COURT_CORNERS
        src_pts = np.array([
            [frame_w * cc["top_left"][0],     frame_h * cc["top_left"][1]],
            [frame_w * cc["top_right"][0],    frame_h * cc["top_right"][1]],
            [frame_w * cc["bottom_right"][0], frame_h * cc["bottom_right"][1]],
            [frame_w * cc["bottom_left"][0],  frame_h * cc["bottom_left"][1]],
        ], dtype=np.float32)
        coordinate_mode = "auto_ratio"

    # ── 목적지 좌표 결정 ──────────────────────────────────────
    if user_corners is not None:
        # 단식 코트 실제 규격 비율을 미니맵에 반영
        dst_pts = _singles_dst_pts(minimap_w, minimap_h, pad)
    else:
        # 기존 직사각형 패딩 방식
        dst_pts = np.array([
            [pad,             pad],
            [minimap_w - pad, pad],
            [minimap_w - pad, minimap_h - pad],
            [pad,             minimap_h - pad],
        ], dtype=np.float32)

    # ── 정규화 목적지 (항상 [0,1]²) ─────────────────────────
    norm_pts = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ], dtype=np.float32)

    to_minimap    = cv2.getPerspectiveTransform(src_pts, dst_pts)
    to_normalized = cv2.getPerspectiveTransform(src_pts, norm_pts)

    # 네트 Y = 미니맵 코트 수직 정중앙
    net_y_minimap = float((dst_pts[0][1] + dst_pts[3][1]) / 2)

    return {
        "src_pts":          src_pts,
        "to_minimap":       to_minimap,
        "to_normalized":    to_normalized,
        "minimap_pts":      dst_pts,
        "net_y_minimap":    net_y_minimap,
        "coordinate_mode":  coordinate_mode,
    }


# ────────────────────────────────────────────────────────────
# 좌표 변환 헬퍼
# ────────────────────────────────────────────────────────────

def frame_to_minimap(
    point: Tuple[float, float],
    matrix: np.ndarray,
) -> Tuple[int, int]:
    pt     = np.array([[[point[0], point[1]]]], dtype=np.float32)
    result = cv2.perspectiveTransform(pt, matrix)[0][0]
    return int(result[0]), int(result[1])


def frame_to_normalized(
    point: Tuple[float, float],
    matrix: np.ndarray,
) -> Tuple[float, float]:
    pt     = np.array([[[point[0], point[1]]]], dtype=np.float32)
    result = cv2.perspectiveTransform(pt, matrix)[0][0]
    return float(result[0]), float(result[1])


def normalized_to_minimap(
    norm_x: float,
    norm_y: float,
    minimap_w: int = MINIMAP_CONFIG["width"],
    minimap_h: int = MINIMAP_CONFIG["height"],
    pad: int = MINIMAP_CONFIG["padding"],
) -> Tuple[int, int]:
    usable_w = minimap_w - 2 * pad
    usable_h = minimap_h - 2 * pad
    mx = int(np.clip(norm_x * usable_w + pad, 0, minimap_w - 1))
    my = int(np.clip(norm_y * usable_h + pad, 0, minimap_h - 1))
    return mx, my


def is_inside_court(
    minimap_pt: Tuple[float, float],
    minimap_pts: np.ndarray,
) -> bool:
    polygon = minimap_pts.astype(np.int32)
    return cv2.pointPolygonTest(
        polygon, (float(minimap_pt[0]), float(minimap_pt[1])), False
    ) >= 0


# ────────────────────────────────────────────────────────────
# 미니맵 코트 그리기
# ────────────────────────────────────────────────────────────

def draw_minimap_court(
    canvas: np.ndarray,
    minimap_w: int = MINIMAP_CONFIG["width"],
    minimap_h: int = MINIMAP_CONFIG["height"],
    line_color: Tuple = MINIMAP_CONFIG["line_color"],
    net_color:  Tuple = MINIMAP_CONFIG["net_color"],
) -> np.ndarray:
    pad = MINIMAP_CONFIG["padding"]
    cl  = COURT_LINES
    cw  = minimap_w - 2 * pad
    ch  = minimap_h - 2 * pad
    lw  = 1

    cv2.rectangle(canvas, (pad, pad), (minimap_w - pad, minimap_h - pad), line_color, 2)

    net_y = pad + ch // 2
    cv2.line(canvas, (pad, net_y), (minimap_w - pad, net_y), net_color, 3)

    for ratio in [cl["back_ratio"], cl["service_ratio"],
                  1.0 - cl["service_ratio"], 1.0 - cl["back_ratio"]]:
        y = pad + int(ch * ratio)
        cv2.line(canvas, (pad, y), (minimap_w - pad, y), line_color, lw)

    for ratio in [cl["side_ratio"], 1.0 - cl["side_ratio"]]:
        x = pad + int(cw * ratio)
        cv2.line(canvas, (x, pad), (x, minimap_h - pad), line_color, lw)

    x_mid   = minimap_w // 2
    y_svc_t = pad + int(ch * cl["service_ratio"])
    y_svc_b = pad + int(ch * (1.0 - cl["service_ratio"]))
    cv2.line(canvas, (x_mid, pad),     (x_mid, y_svc_t), line_color, lw)
    cv2.line(canvas, (x_mid, y_svc_b), (x_mid, minimap_h - pad), line_color, lw)

    return canvas


def create_minimap_canvas(
    bg_color: Tuple[int, int, int] = (40, 100, 55),
) -> np.ndarray:
    w      = MINIMAP_CONFIG["width"]
    h      = MINIMAP_CONFIG["height"]
    canvas = np.full((h, w, 3), bg_color, dtype=np.uint8)
    draw_minimap_court(canvas, w, h)
    return canvas