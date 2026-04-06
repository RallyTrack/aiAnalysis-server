"""
RallyTrack - 코트 기하학 / 호모그래피 모듈

역할:
  - 영상 좌표(픽셀)를 미니맵 좌표(픽셀)로 변환하는 호모그래피 행렬 계산
  - 2D 미니맵 코트 그리기
  - 정규화 좌표 계산 (히트맵 생성용)
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Tuple, Optional

from .config import COURT_CORNERS, COURT_LINES, MINIMAP_CONFIG


# ────────────────────────────────────────────────────────────
# 호모그래피 계산
# ────────────────────────────────────────────────────────────

def compute_homographies(
    frame_w: int,
    frame_h: int,
    court_corners=None,
    minimap_w: int = MINIMAP_CONFIG["width"],
    minimap_h: int = MINIMAP_CONFIG["height"],
) -> dict:
    """
    영상 해상도를 기반으로 두 가지 호모그래피 행렬을 계산합니다.

    반환값 (dict):
        src_pts          : 영상 내 코트 코너 좌표 (4×2 float32)
        to_minimap       : 영상 → 미니맵 변환 행렬 (3×3)
        to_normalized    : 영상 → [0,1]² 정규화 행렬 (3×3)  ← 히트맵용
        minimap_pts      : 미니맵 내 코트 코너 좌표 (4×2 float32)
        net_y_minimap    : 미니맵 내 네트 Y 좌표 (코트 정중앙)
    """
    pad = MINIMAP_CONFIG["padding"]

    if court_corners is not None:
        src_pts = np.array([
            [court_corners.topLeft.x, court_corners.topLeft.y],
            [court_corners.topRight.x, court_corners.topRight.y],
            [court_corners.bottomRight.x, court_corners.bottomRight.y],
            [court_corners.bottomLeft.x, court_corners.bottomLeft.y],
        ], dtype=np.float32)
    else:
        # fallback: config.py 하드코딩 값 사용
        cc = COURT_CORNERS
        src_pts = np.array([
            [frame_w * cc["top_left"][0], frame_h * cc["top_left"][1]],
            [frame_w * cc["top_right"][0], frame_h * cc["top_right"][1]],
            [frame_w * cc["bottom_right"][0], frame_h * cc["bottom_right"][1]],
            [frame_w * cc["bottom_left"][0], frame_h * cc["bottom_left"][1]],
        ], dtype=np.float32)

    # 미니맵 내 목적지 코너: 패딩을 제외한 직사각형
    dst_pts = np.array([
        [pad,             pad],
        [minimap_w - pad, pad],
        [minimap_w - pad, minimap_h - pad],
        [pad,             minimap_h - pad],
    ], dtype=np.float32)

    # [0,1]² 정규화 목적지
    norm_pts = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ], dtype=np.float32)

    to_minimap    = cv2.getPerspectiveTransform(src_pts, dst_pts)
    to_normalized = cv2.getPerspectiveTransform(src_pts, norm_pts)

    net_y_minimap = pad + (minimap_h - 2 * pad) / 2  # 올바른 정중앙 값

    return {
        "src_pts":       src_pts,
        "to_minimap":    to_minimap,
        "to_normalized": to_normalized,
        "minimap_pts":   dst_pts,
        "net_y_minimap": net_y_minimap,
    }


# ────────────────────────────────────────────────────────────
# 좌표 변환 헬퍼
# ────────────────────────────────────────────────────────────

def frame_to_minimap(
    point: Tuple[float, float],
    matrix: np.ndarray,
) -> Tuple[int, int]:
    pt = np.array([[[point[0], point[1]]]], dtype=np.float32)
    result = cv2.perspectiveTransform(pt, matrix)[0][0]
    return int(result[0]), int(result[1])


def frame_to_normalized(
    point: Tuple[float, float],
    matrix: np.ndarray,
) -> Tuple[float, float]:
    pt = np.array([[[point[0], point[1]]]], dtype=np.float32)
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
    return cv2.pointPolygonTest(polygon, (float(minimap_pt[0]), float(minimap_pt[1])), False) >= 0


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
    pad    = MINIMAP_CONFIG["padding"]
    cl     = COURT_LINES
    cw     = minimap_w - 2 * pad
    ch     = minimap_h - 2 * pad
    lw     = 1

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
    w = MINIMAP_CONFIG["width"]
    h = MINIMAP_CONFIG["height"]
    canvas = np.full((h, w, 3), bg_color, dtype=np.uint8)
    draw_minimap_court(canvas, w, h)
    return canvas
