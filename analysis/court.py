"""
RallyTrack - 코트 기하학 / 호모그래피 모듈 (v3)

[v3 변경 사항]
  1. 하드코딩 제거
     - 이전: COURT_CORNERS dict(수동 좌표)에서 src_pts 생성
     - 변경: court_detector.CourtCornerDetector가 자동으로 검출한
             CourtCorners 인스턴스를 받아 처리합니다.

  2. compute_homographies() 시그니처 변경
     - 이전: compute_homographies(frame_w, frame_h)
             → config.COURT_CORNERS에서 좌표 읽음
     - 변경: compute_homographies(frame_w, frame_h, corners)
             → CourtCorners 인스턴스 전달 (또는 None → 자동 폴백)

  3. 미니맵 코트 라인 BWF 정규 규격으로 전면 재작성
     - 실제 치수 기반 비율 계산 (COURT_RATIOS 사용)
     - 코트 배경: 딥 그린 → 더 밝은 필드 그린 그라데이션 효과
     - 라인: 외곽(4mm 폭), 내부 경계선, 서비스 라인, 센터 라인
     - 복식/단식 구분선 정확히 표현
     - 네트 및 네트 기둥 시각화
     - 각 코트 구역 레이블(선택)

역할:
  - 영상 좌표(픽셀) → 미니맵 좌표(픽셀) 변환 호모그래피 행렬 계산
  - 2D Top-Down 배드민턴 코트 미니맵 그리기
  - 정규화 좌표 계산 (히트맵 생성용)
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Tuple, Optional, TYPE_CHECKING

from .config import COURT_RATIOS, MINIMAP_CONFIG, BADMINTON_COURT

if TYPE_CHECKING:
    from .court_detector import CourtCorners


# ────────────────────────────────────────────────────────────
# 호모그래피 계산
# ────────────────────────────────────────────────────────────

def compute_homographies(
    frame_w:  int,
    frame_h:  int,
    corners:  "Optional[CourtCorners]" = None,
    minimap_w: int = MINIMAP_CONFIG["width"],
    minimap_h: int = MINIMAP_CONFIG["height"],
) -> dict:
    """
    코트 코너를 기반으로 두 가지 호모그래피 행렬을 계산합니다.

    Args:
        frame_w:   영상 가로 해상도 (픽셀)
        frame_h:   영상 세로 해상도 (픽셀)
        corners:   CourtCornerDetector가 반환한 CourtCorners 인스턴스.
                   None이면 내부에서 자동 폴백 추정값 사용.
        minimap_w: 미니맵 캔버스 가로 크기
        minimap_h: 미니맵 캔버스 세로 크기

    반환값 (dict):
        src_pts        : 영상 내 코트 코너 좌표 (4×2 float32) [TL,TR,BR,BL]
        to_minimap     : 영상 → 미니맵 변환 행렬 (3×3)
        to_normalized  : 영상 → [0,1]² 정규화 행렬 (3×3) ← 히트맵용
        minimap_pts    : 미니맵 내 코트 코너 좌표 (4×2 float32)
        net_y_minimap  : 미니맵 내 네트 Y 좌표 (코트 정중앙)
        corners_info   : 검출된 CourtCorners 메타데이터 dict
    """
    if corners is None:
        # 런타임 import (순환 참조 방지)
        from .court_detector import _fallback_corners
        corners = _fallback_corners(frame_w, frame_h)

    pad = MINIMAP_CONFIG["padding"]

    # 영상 내 코트 코너 [좌상, 우상, 우하, 좌하]
    src_pts = corners.to_pixel_array()  # (4, 2) float32

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

    # 네트 Y 좌표: 코트 상단 Y(=pad) + usable_height * 0.5 (정중앙)
    net_y_minimap = pad + (minimap_h - 2 * pad) / 2

    return {
        "src_pts":       src_pts,
        "to_minimap":    to_minimap,
        "to_normalized": to_normalized,
        "minimap_pts":   dst_pts,
        "net_y_minimap": net_y_minimap,
        "corners_info":  corners.to_dict(),
    }


# ────────────────────────────────────────────────────────────
# 좌표 변환 헬퍼
# ────────────────────────────────────────────────────────────

def frame_to_minimap(
    point:  Tuple[float, float],
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
    pt     = np.array([[[point[0], point[1]]]], dtype=np.float32)
    result = cv2.perspectiveTransform(pt, matrix)[0][0]
    return int(result[0]), int(result[1])


def frame_to_normalized(
    point:  Tuple[float, float],
    matrix: np.ndarray,
) -> Tuple[float, float]:
    """영상 좌표 → [0,1]² 정규화 좌표 (히트맵 셀 계산용)."""
    pt     = np.array([[[point[0], point[1]]]], dtype=np.float32)
    result = cv2.perspectiveTransform(pt, matrix)[0][0]
    return float(result[0]), float(result[1])


def normalized_to_minimap(
    norm_x:   float,
    norm_y:   float,
    minimap_w: int = MINIMAP_CONFIG["width"],
    minimap_h: int = MINIMAP_CONFIG["height"],
    pad:       int = MINIMAP_CONFIG["padding"],
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
    minimap_pt:  Tuple[float, float],
    minimap_pts: np.ndarray,
) -> bool:
    """
    미니맵 좌표가 코트 폴리곤 내부에 있는지 확인합니다.
    미니맵 dst_pts 좌표계 기준으로 판단합니다.
    """
    polygon = minimap_pts.astype(np.int32)
    return cv2.pointPolygonTest(polygon, (float(minimap_pt[0]), float(minimap_pt[1])), False) >= 0


# ────────────────────────────────────────────────────────────
# 미니맵 코트 그리기 (BWF 정규 규격)
# ────────────────────────────────────────────────────────────

def draw_minimap_court(
    canvas:     np.ndarray,
    minimap_w:  int   = MINIMAP_CONFIG["width"],
    minimap_h:  int   = MINIMAP_CONFIG["height"],
    line_color: Tuple = MINIMAP_CONFIG["line_color"],
    net_color:  Tuple = MINIMAP_CONFIG["net_color"],
) -> np.ndarray:
    """
    BWF 표준 배드민턴 코트 라인을 캔버스에 그립니다.

    ┌──────────────────────────────────────────────────┐
    │           상단 경계선 (Back Boundary Line)         │
    │  ┌────────────────────────────────────────────┐  │
    │  │   복식 롱 서비스 라인 (Long Service Line)    │  │
    │  │  ┌──────────────────────────────────────┐  │  │
    │  │  │                                      │  │  │
    │  │  │   숏 서비스 라인 (Short Service Line)  │  │  │
    │══════════════ 네트 (Net) ═══════════════════════│
    │  │  │   숏 서비스 라인                       │  │  │
    │  │  │                                      │  │  │
    │  │  └──────────────────────────────────────┘  │  │
    │  │   복식 롱 서비스 라인                        │  │
    │  └────────────────────────────────────────────┘  │
    │           하단 경계선                              │
    └──────────────────────────────────────────────────┘

    단식 사이드라인: 복식 경계선 안쪽 0.46m (각 쪽)
    센터 라인: 숏 서비스 라인 ~ 숏 서비스 라인 (좌우 대칭)
    """
    pad = MINIMAP_CONFIG["padding"]
    r   = COURT_RATIOS

    # 사용 가능 영역
    cw  = minimap_w - 2 * pad   # usable width
    ch  = minimap_h - 2 * pad   # usable height

    # 코트 꼭짓점 (복식 외곽)
    x0 = pad           # 좌측 경계
    x1 = minimap_w - pad   # 우측 경계
    y0 = pad           # 상단 경계
    y1 = minimap_h - pad   # 하단 경계

    # ── 1. 외곽 경계선 (복식 아웃 라인, 굵게) ──────────────
    cv2.rectangle(canvas, (x0, y0), (x1, y1), line_color, 2)

    # ── 2. 단식 사이드라인 ─────────────────────────────────
    xs_l = x0 + int(cw * r["singles_left"])
    xs_r = x0 + int(cw * r["singles_right"])
    cv2.line(canvas, (xs_l, y0), (xs_l, y1), line_color, 1)
    cv2.line(canvas, (xs_r, y0), (xs_r, y1), line_color, 1)

    # ── 3. 길이 방향 가로 라인들 ──────────────────────────
    # 3a. 상단 롱 서비스 라인 (복식)
    y_back_top = y0 + int(ch * r["back_service_top"])
    cv2.line(canvas, (x0, y_back_top), (x1, y_back_top), line_color, 1)

    # 3b. 상단 숏 서비스 라인
    y_short_top = y0 + int(ch * r["short_service_top"])
    cv2.line(canvas, (x0, y_short_top), (x1, y_short_top), line_color, 1)

    # 3c. 하단 숏 서비스 라인
    y_short_bot = y0 + int(ch * r["short_service_bot"])
    cv2.line(canvas, (x0, y_short_bot), (x1, y_short_bot), line_color, 1)

    # 3d. 하단 롱 서비스 라인 (복식)
    y_back_bot = y0 + int(ch * r["back_service_bot"])
    cv2.line(canvas, (x0, y_back_bot), (x1, y_back_bot), line_color, 1)

    # ── 4. 센터 서비스 라인 (숏 서비스 라인 사이만) ──────────
    # 실제 규격: 숏 서비스 라인에서 네트 방향으로만 그려짐
    # → 상단: y_back_top ~ y_short_top (상단 코트 전체)
    # → 하단: y_short_bot ~ y_back_bot (하단 코트 전체)
    # 실제로는 back boundary ~ short service line 구간에 센터 라인 있음
    x_center = x0 + int(cw * r["center"])

    # 상단 코트 센터 라인 (상단 경계 ~ 상단 숏 서비스 라인)
    cv2.line(canvas, (x_center, y0),        (x_center, y_short_top), line_color, 1)
    # 하단 코트 센터 라인 (하단 숏 서비스 라인 ~ 하단 경계)
    cv2.line(canvas, (x_center, y_short_bot), (x_center, y1),        line_color, 1)

    # ── 5. 네트 (중앙 가로선, 굵게) ─────────────────────────
    net_y = y0 + int(ch * r["net"])
    cv2.line(canvas, (x0, net_y), (x1, net_y), net_color, 3)

    # 네트 기둥 (단식 사이드라인 바깥쪽 작은 표시)
    pole_len = 6
    for px in [x0, x1]:
        cv2.line(canvas, (px, net_y - pole_len), (px, net_y + pole_len), net_color, 4)

    return canvas


def create_minimap_canvas(
    bg_color: Tuple[int, int, int] = (34, 85, 34),
) -> np.ndarray:
    """
    배경 색과 코트 라인이 그려진 빈 미니맵 캔버스를 생성합니다.

    배경: 딥 그린 (실제 배드민턴 코트 바닥 색상)
    """
    w = MINIMAP_CONFIG["width"]
    h = MINIMAP_CONFIG["height"]

    canvas = np.full((h, w, 3), bg_color, dtype=np.uint8)

    # 코트 내부 영역을 더 밝은 녹색으로 채워 필드와 아웃 구역 구분
    pad = MINIMAP_CONFIG["padding"]
    cv2.rectangle(
        canvas,
        (pad, pad),
        (w - pad, h - pad),
        (45, 110, 45),   # 약간 밝은 필드 그린
        -1,              # 채우기
    )

    draw_minimap_court(canvas, w, h)
    return canvas
