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
        net_y_minimap    : 미니맵 내 네트 Y 좌표
    """
    cc = COURT_CORNERS
    pad = MINIMAP_CONFIG["padding"]

    # 영상 내 코트 코너 [좌상, 우상, 우하, 좌하]
    src_pts = np.array([
        [frame_w * cc["top_left"][0],     frame_h * cc["top_left"][1]],
        [frame_w * cc["top_right"][0],    frame_h * cc["top_right"][1]],
        [frame_w * cc["bottom_right"][0], frame_h * cc["bottom_right"][1]],
        [frame_w * cc["bottom_left"][0],  frame_h * cc["bottom_left"][1]],
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

    # 네트 Y 좌표: 미니맵 상단 두 점의 Y 평균 + 코트 높이의 절반
    net_y_minimap = (dst_pts[0][1] + dst_pts[3][1]) / 2 + (minimap_h - 2 * pad) / 2

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
    """
    영상 좌표 (x, y) → 미니맵 픽셀 좌표.

    Args:
        point  : (x, y) 영상 픽셀 좌표
        matrix : compute_homographies()의 to_minimap 행렬

    Returns:
        (mx, my) 정수 픽셀 좌표
    """
    pt = np.array([[[point[0], point[1]]]], dtype=np.float32)
    result = cv2.perspectiveTransform(pt, matrix)[0][0]
    return int(result[0]), int(result[1])


def frame_to_normalized(
    point: Tuple[float, float],
    matrix: np.ndarray,
) -> Tuple[float, float]:
    """
    영상 좌표 → [0,1]² 정규화 좌표 (히트맵 셀 계산용).
    """
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
    """
    [0,1]² 정규화 좌표 → 미니맵 픽셀 좌표.
    히트맵 및 타점 마커 위치 계산에 사용됩니다.
    """
    usable_w = minimap_w - 2 * pad
    usable_h = minimap_h - 2 * pad
    mx = int(np.clip(norm_x * usable_w + pad, 0, minimap_w - 1))
    my = int(np.clip(norm_y * usable_h + pad, 0, minimap_h - 1))
    return mx, my


def is_inside_court(
    minimap_pt: Tuple[float, float],
    minimap_pts: np.ndarray,
) -> bool:
    """
    미니맵 좌표가 코트 폴리곤 내부에 있는지 확인합니다.

    [수정 이유]
    원본 코드에서는 영상 src_pts 기준으로 pointPolygonTest를 했는데,
    미니맵 좌표와 src_pts 좌표계가 달라 부정확했습니다.
    미니맵 dst_pts를 기준으로 판단하도록 수정합니다.
    """
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
    """
    배드민턴 코트 라인을 캔버스에 그립니다.

    국제 규격 비율:
      - 서비스 라인: 네트에서 1.98m (코트 길이 13.4m의 약 14.8%)
      - 백 바운더리: 후방 0.76m (약 5.7%)
      - 사이드 라인: 양쪽 0.46m (코트 폭 6.1m의 약 7.5%)

    Args:
        canvas: 그릴 대상 numpy 배열 (height × width × 3, uint8)
    Returns:
        그려진 canvas
    """
    pad    = MINIMAP_CONFIG["padding"]
    cl     = COURT_LINES
    cw     = minimap_w - 2 * pad   # usable width
    ch     = minimap_h - 2 * pad   # usable height
    lw     = 1                      # line width

    # 외곽 경계선
    cv2.rectangle(canvas, (pad, pad), (minimap_w - pad, minimap_h - pad), line_color, 2)

    # 네트 (중앙 가로선)
    net_y = pad + ch // 2
    cv2.line(canvas, (pad, net_y), (minimap_w - pad, net_y), net_color, 3)

    # 가로 라인: 서비스 라인 & 롱 서비스 라인
    for ratio in [cl["back_ratio"], cl["service_ratio"],
                  1.0 - cl["service_ratio"], 1.0 - cl["back_ratio"]]:
        y = pad + int(ch * ratio)
        cv2.line(canvas, (pad, y), (minimap_w - pad, y), line_color, lw)

    # 세로 라인: 사이드 라인 (복식 경계)
    for ratio in [cl["side_ratio"], 1.0 - cl["side_ratio"]]:
        x = pad + int(cw * ratio)
        cv2.line(canvas, (x, pad), (x, minimap_h - pad), line_color, lw)

    # 중앙 서비스 라인 (상·하 각각 절반까지만)
    x_mid   = minimap_w // 2
    y_svc_t = pad + int(ch * cl["service_ratio"])
    y_svc_b = pad + int(ch * (1.0 - cl["service_ratio"]))
    cv2.line(canvas, (x_mid, pad),   (x_mid, y_svc_t), line_color, lw)
    cv2.line(canvas, (x_mid, y_svc_b), (x_mid, minimap_h - pad), line_color, lw)

    return canvas


def create_minimap_canvas(
    bg_color: Tuple[int, int, int] = (40, 100, 55),
) -> np.ndarray:
    """
    배경 색과 코트 라인이 그려진 빈 미니맵 캔버스를 생성합니다.
    """
    w = MINIMAP_CONFIG["width"]
    h = MINIMAP_CONFIG["height"]
    canvas = np.full((h, w, 3), bg_color, dtype=np.uint8)
    draw_minimap_court(canvas, w, h)
    return canvas
