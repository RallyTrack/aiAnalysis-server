"""
RallyTrack - 코트 기하학 / 호모그래피 모듈

역할:
  - 영상 좌표(픽셀)를 미니맵 좌표(픽셀)로 변환하는 호모그래피 행렬 계산
  - BWF 표준 규격에 따른 2D 미니맵 코트 그리기
  - 정규화 좌표 계산 (히트맵 생성용)

[수정 사항 - BWF 표준 좌표계 적용]
  - BWF 코트 실제 치수(m) 상수 정의
  - compute_homographies(): user_corners 제공 시 BWF 단식 코트 기준 변환
      · user_corners 순서: [TL, TR, BR, BL] (영상 기준)
      · 매핑: BL→(0.46, 0), BR→(5.64, 0), TR→(5.64, 13.40), TL→(0.46, 13.40)
      · 미니맵 Y: near end(Y=0)→캔버스 하단, far end(Y=13.40)→캔버스 상단
  - draw_minimap_court(): BWF 실제 좌표 기반으로 모든 라인 정확히 그리기
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import COURT_CORNERS, MINIMAP_CONFIG


# ────────────────────────────────────────────────────────────
# BWF 표준 코트 규격 (단위: m)
# ────────────────────────────────────────────────────────────

BWF_COURT_WIDTH  = 6.10   # 더블 코트 폭
BWF_COURT_LENGTH = 13.40  # 코트 전장

# X축 라인 위치 (m) — 더블 0~6.10, 단식 0.46~5.64
BWF_X = {
    "doubles_left":  0.00,
    "singles_left":  0.46,
    "center":        3.05,
    "singles_right": 5.64,
    "doubles_right": 6.10,
}

# Y축 라인 위치 (m) — Y=0: 가까운 엔드(near), Y=13.40: 먼 엔드(far)
BWF_Y = {
    "end_near":           0.00,
    "back_service_near":  0.76,   # 더블 롱 서비스 라인 (near)
    "short_service_near": 4.72,   # 숏 서비스 라인 (near) = 네트에서 1.98m
    "net":                6.70,   # 네트
    "short_service_far":  8.68,   # 숏 서비스 라인 (far)
    "back_service_far":   12.64,  # 더블 롱 서비스 라인 (far)
    "end_far":            13.40,
}


# ────────────────────────────────────────────────────────────
# 내부 헬퍼: BWF 실제 좌표(m) → 캔버스 픽셀 좌표
# ────────────────────────────────────────────────────────────

def _bwf_to_canvas(
    x_m: float,
    y_m: float,
    minimap_w: int,
    minimap_h: int,
    pad: int,
) -> Tuple[float, float]:
    """
    BWF 실제 좌표(m)를 미니맵 캔버스 픽셀 좌표로 변환한다.

    Y축 방향:
      - Y=0 (near end, 카메라 쪽) → 캔버스 하단 (large py)
      - Y=13.40 (far end)         → 캔버스 상단 (small py)

    이 방향은 영상에서 near end가 아래, far end가 위에 보이는
    표준 카메라 앵글과 미니맵 방향을 일치시킨다.
    """
    aw = minimap_w - 2 * pad
    ah = minimap_h - 2 * pad
    px = pad + (x_m / BWF_COURT_WIDTH) * aw
    py = (minimap_h - pad) - (y_m / BWF_COURT_LENGTH) * ah
    return (px, py)


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
        user_corners     : shape (4,2) float32, 순서 [TL, TR, BR, BL] (영상 픽셀).
                           사용자가 영상에서 클릭한 단식 코트 네 모서리.
                           None이면 COURT_CORNERS 비율로 자동 계산.

    BWF 단식 코트 매핑 (user_corners 제공 시):
        TL (영상 좌상단) → BWF (0.46m, 13.40m) → 캔버스 좌상단 영역
        TR (영상 우상단) → BWF (5.64m, 13.40m) → 캔버스 우상단 영역
        BR (영상 우하단) → BWF (5.64m,  0.00m) → 캔버스 우하단 영역
        BL (영상 좌하단) → BWF (0.46m,  0.00m) → 캔버스 좌하단 영역

    반환값 (dict):
        src_pts          : 영상 내 코트 코너 좌표 (4×2 float32)
        to_minimap       : 영상 → 미니맵 변환 행렬 (3×3)
        to_normalized    : 영상 → [0,1]² 정규화 행렬 (3×3)  ← 히트맵용
        minimap_pts      : 미니맵 내 단식 코트 코너 좌표 (4×2 float32)
        net_y_minimap    : 미니맵 내 네트 Y 좌표 (픽셀)
        coordinate_mode  : "user_singles" | "auto_ratio"  (디버그용)
    """
    pad = MINIMAP_CONFIG["padding"]

    # ── 소스 좌표 결정 ────────────────────────────────────────
    if user_corners is not None:
        src_pts = np.asarray(user_corners, dtype=np.float32).reshape(4, 2)
        coordinate_mode = "user_singles"
        print(f"[court] 사용자 지정 코너 사용 (BWF 단식 매핑): {src_pts.tolist()}")
    else:
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
        # BWF 단식 코트 모서리를 캔버스 픽셀 위치로 변환
        # user_corners 순서: [TL, TR, BR, BL]
        #   TL → BWF(0.46, 13.40), TR → BWF(5.64, 13.40)
        #   BR → BWF(5.64,  0.00), BL → BWF(0.46,  0.00)
        dst_pts = np.array([
            _bwf_to_canvas(BWF_X["singles_left"],  BWF_Y["end_far"],  minimap_w, minimap_h, pad),  # TL
            _bwf_to_canvas(BWF_X["singles_right"], BWF_Y["end_far"],  minimap_w, minimap_h, pad),  # TR
            _bwf_to_canvas(BWF_X["singles_right"], BWF_Y["end_near"], minimap_w, minimap_h, pad),  # BR
            _bwf_to_canvas(BWF_X["singles_left"],  BWF_Y["end_near"], minimap_w, minimap_h, pad),  # BL
        ], dtype=np.float32)

        # 네트 Y = BWF 6.70m 위치
        net_y_minimap = _bwf_to_canvas(0, BWF_Y["net"], minimap_w, minimap_h, pad)[1]

    else:
        # 기존 직사각형 패딩 방식 (자동 비율)
        dst_pts = np.array([
            [pad,             pad],
            [minimap_w - pad, pad],
            [minimap_w - pad, minimap_h - pad],
            [pad,             minimap_h - pad],
        ], dtype=np.float32)

        # 네트 Y = 미니맵 코트 수직 정중앙
        net_y_minimap = float((dst_pts[0][1] + dst_pts[3][1]) / 2)

    # ── 정규화 목적지 (항상 [0,1]²) ─────────────────────────
    norm_pts = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ], dtype=np.float32)

    to_minimap    = cv2.getPerspectiveTransform(src_pts, dst_pts)
    to_normalized = cv2.getPerspectiveTransform(src_pts, norm_pts)

    return {
        "src_pts":         src_pts,
        "to_minimap":      to_minimap,
        "to_normalized":   to_normalized,
        "minimap_pts":     dst_pts,
        "net_y_minimap":   net_y_minimap,
        "coordinate_mode": coordinate_mode,
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
# 미니맵 코트 그리기 (BWF 표준 좌표계)
# ────────────────────────────────────────────────────────────

def draw_minimap_court(
    canvas: np.ndarray,
    minimap_w: int = MINIMAP_CONFIG["width"],
    minimap_h: int = MINIMAP_CONFIG["height"],
    line_color: Tuple = MINIMAP_CONFIG["line_color"],
    net_color:  Tuple = MINIMAP_CONFIG["net_color"],
) -> np.ndarray:
    """
    BWF 표준 규격에 따라 미니맵 캔버스에 코트를 그립니다.

    바닥 채색 (Two-tone Green):
      - 더블 코트 외곽 전체: Dark Green (#064E3B) — 단식 아웃 영역(복식 앨리) 포함
      - 단식 유효 영역:       Bright Green (#4ADE80) — 단식 인 영역

    Y축: near end(Y=0) → 캔버스 하단, far end(Y=13.40) → 캔버스 상단
    """
    pad        = MINIMAP_CONFIG["padding"]
    c_singles  = MINIMAP_CONFIG["court_singles_color"]   # Bright Green
    c_doubles  = MINIMAP_CONFIG["court_doubles_color"]   # Dark  Green

    def xp(x_m: float) -> int:
        return int(_bwf_to_canvas(x_m, 0, minimap_w, minimap_h, pad)[0])

    def yp(y_m: float) -> int:
        return int(_bwf_to_canvas(0, y_m, minimap_w, minimap_h, pad)[1])

    lw = 1  # 일반 라인 굵기

    # ── 바닥 채색: 더블 영역(앨리 포함) → Dark Green ────────
    cv2.rectangle(
        canvas,
        (xp(BWF_X["doubles_left"]),  yp(BWF_Y["end_far"])),
        (xp(BWF_X["doubles_right"]), yp(BWF_Y["end_near"])),
        c_doubles, -1,
    )

    # ── 바닥 채색: 단식 유효 영역 → Bright Green ────────────
    cv2.rectangle(
        canvas,
        (xp(BWF_X["singles_left"]),  yp(BWF_Y["end_far"])),
        (xp(BWF_X["singles_right"]), yp(BWF_Y["end_near"])),
        c_singles, -1,
    )

    # ── 더블 코트 외곽선 ──────────────────────────────────────
    cv2.rectangle(
        canvas,
        (xp(BWF_X["doubles_left"]),  yp(BWF_Y["end_far"])),
        (xp(BWF_X["doubles_right"]), yp(BWF_Y["end_near"])),
        line_color, 2,
    )

    # ── 단식 사이드 라인 ─────────────────────────────────────
    cv2.line(canvas,
             (xp(BWF_X["singles_left"]),  yp(BWF_Y["end_far"])),
             (xp(BWF_X["singles_left"]),  yp(BWF_Y["end_near"])),
             line_color, lw)
    cv2.line(canvas,
             (xp(BWF_X["singles_right"]), yp(BWF_Y["end_far"])),
             (xp(BWF_X["singles_right"]), yp(BWF_Y["end_near"])),
             line_color, lw)

    # ── 더블 롱 서비스 라인 (엔드 바운더리 안쪽) ────────────
    cv2.line(canvas,
             (xp(BWF_X["doubles_left"]),  yp(BWF_Y["back_service_near"])),
             (xp(BWF_X["doubles_right"]), yp(BWF_Y["back_service_near"])),
             line_color, lw)
    cv2.line(canvas,
             (xp(BWF_X["doubles_left"]),  yp(BWF_Y["back_service_far"])),
             (xp(BWF_X["doubles_right"]), yp(BWF_Y["back_service_far"])),
             line_color, lw)

    # ── 숏 서비스 라인 (BWF 규격: 복식 사이드라인 전체 폭) ──
    cv2.line(canvas,
             (xp(BWF_X["doubles_left"]),  yp(BWF_Y["short_service_near"])),
             (xp(BWF_X["doubles_right"]), yp(BWF_Y["short_service_near"])),
             line_color, lw)
    cv2.line(canvas,
             (xp(BWF_X["doubles_left"]),  yp(BWF_Y["short_service_far"])),
             (xp(BWF_X["doubles_right"]), yp(BWF_Y["short_service_far"])),
             line_color, lw)

    # ── 센터 라인 (BWF 규격: 숏 서비스 라인 → 엔드라인, 네트 구간 제외) ─
    cv2.line(canvas,
             (xp(BWF_X["center"]), yp(BWF_Y["end_far"])),
             (xp(BWF_X["center"]), yp(BWF_Y["short_service_far"])),
             line_color, lw)
    cv2.line(canvas,
             (xp(BWF_X["center"]), yp(BWF_Y["short_service_near"])),
             (xp(BWF_X["center"]), yp(BWF_Y["end_near"])),
             line_color, lw)

    # ── 네트 ──────────────────────────────────────────────────
    cv2.line(canvas,
             (xp(BWF_X["doubles_left"]),  yp(BWF_Y["net"])),
             (xp(BWF_X["doubles_right"]), yp(BWF_Y["net"])),
             net_color, 3)

    return canvas


def create_minimap_canvas() -> np.ndarray:
    """
    미니맵 캔버스 생성.
    배경은 코트 외곽을 위한 중립 다크 컬러로, 코트 바닥은 draw_minimap_court에서 채색.
    """
    w      = MINIMAP_CONFIG["width"]
    h      = MINIMAP_CONFIG["height"]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)   # 코트 바깥 = 검정
    draw_minimap_court(canvas, w, h)
    return canvas
