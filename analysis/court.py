"""
RallyTrack - 코트 기하학 / 호모그래피 모듈 (v4)

[v4 변경 사항]
  1. 미니맵 코트 라인 BWF 규격 완전 재작성
     ─ COURT_RATIOS 기반으로 모든 라인 좌표 계산
     ─ 복식 외곽 / 단식 사이드라인 / 더블 롱서비스 / 숏서비스 / 센터라인
     ─ 구역별 배경색 구분 (아웃존, 단식인코트, 서비스존)
     ─ 네트 + 포스트 시각화

  2. 좌표계 일치 버그 수정
     ─ 문제: 호모그래피 변환 후 선수 위치와 코트 라인이 어긋남
       원인: 코트 코너 src_pts 순서 [TL,TR,BR,BL]와
             미니맵 dst_pts가 암묵적으로 '위=top, 아래=bottom'으로
             가정하지만, 영상에 따라 코너가 반전될 수 있었음.
     ─ 수정: dst_pts를 명확히 [top-left, top-right, bottom-right, bottom-left]
             순서로 고정하고, net_y_minimap도 정확한 비율로 계산.

  3. compute_homographies()에 net_y_minimap을 COURT_RATIOS["net"]으로 계산
     ─ 이전: pad + (h - 2*pad) / 2  (단순 중앙값 → 패딩 때문에 오차)
     ─ 수정: pad + usable_h * COURT_RATIOS["net"]  (비율 기반 정확한 값)
             COURT_RATIOS["net"] = 0.5 이므로 결과 동일하지만
             비율이 바뀌어도 자동으로 맞게 됨.

역할:
  - 영상 좌표(픽셀) → 미니맵 좌표(픽셀) 변환 호모그래피 행렬 계산
  - 2D Top-Down 배드민턴 코트 미니맵 그리기 (BWF 정규 규격)
  - 정규화 좌표 계산 (히트맵 / 웹 연동용)
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Tuple, Optional, TYPE_CHECKING

from .config import COURT_RATIOS, MINIMAP_CONFIG, BADMINTON_COURT, COURT_LINES_SPEC

if TYPE_CHECKING:
    from .court_detector import CourtCorners


# ────────────────────────────────────────────────────────────
# 호모그래피 계산
# ────────────────────────────────────────────────────────────

def compute_homographies(
    frame_w:   int,
    frame_h:   int,
    corners:   "Optional[CourtCorners]" = None,
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
        src_pts       : 영상 내 코트 코너 좌표 (4×2 float32) [TL,TR,BR,BL]
        to_minimap    : 영상 → 미니맵 변환 행렬 (3×3)
        to_normalized : 영상 → [0,1]² 정규화 행렬 (3×3)
        minimap_pts   : 미니맵 내 코트 코너 좌표 (4×2 float32)
        net_y_minimap : 미니맵 내 네트 Y 픽셀 좌표
        corners_info  : 검출된 CourtCorners 메타데이터 dict
    """
    if corners is None:
        from .court_detector import _fallback_corners
        corners = _fallback_corners(frame_w, frame_h)

    pad      = MINIMAP_CONFIG["padding"]
    usable_w = minimap_w - 2 * pad
    usable_h = minimap_h - 2 * pad

    # ── 영상 내 코트 코너 [TL, TR, BR, BL] ─────────────────
    src_pts = corners.to_pixel_array()   # (4, 2) float32

    # ── 미니맵 목적지 코너 ───────────────────────────────────
    # 미니맵 좌표계: 상단(top) = 패딩, 하단(bottom) = h-패딩
    # 코트 코너 순서와 정확히 대응: [TL, TR, BR, BL]
    dst_pts = np.array([
        [pad,             pad],              # TL
        [pad + usable_w,  pad],              # TR
        [pad + usable_w,  pad + usable_h],   # BR
        [pad,             pad + usable_h],   # BL
    ], dtype=np.float32)

    # ── [0,1]² 정규화 목적지 ────────────────────────────────
    # (0,0) = TL, (1,1) = BR
    norm_pts = np.array([
        [0.0, 0.0],  # TL
        [1.0, 0.0],  # TR
        [1.0, 1.0],  # BR
        [0.0, 1.0],  # BL
    ], dtype=np.float32)

    to_minimap    = cv2.getPerspectiveTransform(src_pts, dst_pts)
    to_normalized = cv2.getPerspectiveTransform(src_pts, norm_pts)

    # ── 네트 Y 좌표 (COURT_RATIOS 기반, 정확한 비율) ─────────
    # COURT_RATIOS["net"] = 0.5 (코트 정중앙)
    # 미니맵 상단=0 기준이므로: pad + usable_h * net_ratio
    net_y_minimap = pad + usable_h * COURT_RATIOS["net"]   # = pad + usable_h * 0.5

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
    """영상 좌표 (x, y) → 미니맵 픽셀 좌표."""
    pt     = np.array([[[point[0], point[1]]]], dtype=np.float32)
    result = cv2.perspectiveTransform(pt, matrix)[0][0]
    return int(result[0]), int(result[1])


def frame_to_normalized(
    point:  Tuple[float, float],
    matrix: np.ndarray,
) -> Tuple[float, float]:
    """영상 좌표 → [0,1]² 정규화 좌표."""
    pt     = np.array([[[point[0], point[1]]]], dtype=np.float32)
    result = cv2.perspectiveTransform(pt, matrix)[0][0]
    return float(result[0]), float(result[1])


def normalized_to_minimap(
    norm_x:    float,
    norm_y:    float,
    minimap_w: int = MINIMAP_CONFIG["width"],
    minimap_h: int = MINIMAP_CONFIG["height"],
    pad:       int = MINIMAP_CONFIG["padding"],
) -> Tuple[int, int]:
    """[0,1]² 정규화 좌표 → 미니맵 픽셀 좌표."""
    usable_w = minimap_w - 2 * pad
    usable_h = minimap_h - 2 * pad
    mx = int(np.clip(norm_x * usable_w + pad, 0, minimap_w - 1))
    my = int(np.clip(norm_y * usable_h + pad, 0, minimap_h - 1))
    return mx, my


def is_inside_court(
    minimap_pt:  Tuple[float, float],
    minimap_pts: np.ndarray,
) -> bool:
    """미니맵 좌표가 코트 폴리곤 내부인지 확인."""
    polygon = minimap_pts.astype(np.int32)
    return cv2.pointPolygonTest(polygon, (float(minimap_pt[0]), float(minimap_pt[1])), False) >= 0


# ────────────────────────────────────────────────────────────
# 미니맵 코트 그리기 (BWF 정규 규격 v4)
# ────────────────────────────────────────────────────────────

def _ratio_to_px(ratio: float, start: int, span: int) -> int:
    """0~1 비율 → 픽셀 좌표."""
    return start + int(span * ratio)


def draw_minimap_court(
    canvas:     np.ndarray,
    minimap_w:  int   = MINIMAP_CONFIG["width"],
    minimap_h:  int   = MINIMAP_CONFIG["height"],
    line_color: Tuple = MINIMAP_CONFIG["line_color"],
    net_color:  Tuple = MINIMAP_CONFIG["net_color"],
) -> np.ndarray:
    """
    BWF 표준 배드민턴 코트 라인을 캔버스에 그립니다.

    ┌────────────────────────── y_top(0.0) ──── 위쪽 엔드라인 ──┐
    │░░░░░░░░░░░░░░░░░░ 아웃 구역 (복식 외곽) ░░░░░░░░░░░░░░░░░│
    ├──── 위 더블 롱서비스라인 (y≈0.057) ──────────────────────┤
    │     위쪽 서비스 구역 (더블 서브 유효범위)                   │
    ├──── 위 숏서비스라인 (y≈0.352) ───────────────────────────┤
    │     │             │             │  ← 단식폭 | 센터 | 단식폭│
    │     │  위 왼쪽     │  위 오른쪽   │                       │
    │     │  서비스코트  │  서비스코트  │                       │
    ╠══════════════ 네트 (y=0.5) ═══════════════════════════════╣
    │     │  아래 왼쪽   │  아래 오른쪽  │                      │
    │     │  서비스코트  │  서비스코트   │                      │
    ├──── 아래 숏서비스라인 (y≈0.648) ─────────────────────────┤
    │     아래쪽 서비스 구역                                      │
    ├──── 아래 더블 롱서비스라인 (y≈0.943) ────────────────────┤
    │░░░░░░░░░░░░░░░░░░ 아웃 구역 ░░░░░░░░░░░░░░░░░░░░░░░░░░░│
    └────────────────────────── y_bot(1.0) ──── 아래쪽 엔드라인 ┘

    좌표계: 미니맵 상단 = y_ratio 0.0 (실제 코트 위쪽 엔드라인)
            미니맵 하단 = y_ratio 1.0 (실제 코트 아래쪽 엔드라인)
    """
    pad = MINIMAP_CONFIG["padding"]
    r   = COURT_RATIOS

    cw  = minimap_w - 2 * pad   # usable width  (복식 폭에 대응)
    ch  = minimap_h - 2 * pad   # usable height (코트 길이에 대응)

    # 경계 픽셀
    x0 = pad              # 왼쪽 더블 경계
    x1 = pad + cw         # 오른쪽 더블 경계
    y0 = pad              # 위 엔드라인 (미니맵 상단)
    y1 = pad + ch         # 아래 엔드라인 (미니맵 하단)

    # ── 주요 좌표 픽셀값 사전 계산 ──────────────────────────
    y_long_top   = _ratio_to_px(r["long_svc_top"],   pad, ch)   # 위 더블 롱서비스
    y_short_top  = _ratio_to_px(r["short_svc_top"],  pad, ch)   # 위 숏서비스
    y_net        = _ratio_to_px(r["net"],             pad, ch)   # 네트
    y_short_bot  = _ratio_to_px(r["short_svc_bot"],  pad, ch)   # 아래 숏서비스
    y_long_bot   = _ratio_to_px(r["long_svc_bot"],   pad, ch)   # 아래 더블 롱서비스

    x_sgl_l      = _ratio_to_px(r["singles_left"],   pad, cw)   # 왼쪽 단식 사이드
    x_center     = _ratio_to_px(r["center"],          pad, cw)   # 센터라인
    x_sgl_r      = _ratio_to_px(r["singles_right"],  pad, cw)   # 오른쪽 단식 사이드

    # ── 1. 배경 구역 색상 채우기 ────────────────────────────
    # 단식 이너 필드 (서비스 구역 전체) - 조금 더 밝은 색
    FIELD_INNER  = (50, 120, 50)   # 서비스 코트 안쪽
    FIELD_OUTER  = (38,  95, 38)   # 롱서비스~엔드라인 구역
    FIELD_SIDE   = (30,  75, 30)   # 복식 사이드 구역 (단식 바깥)

    # 단식 안쪽 전체 (y0 ~ y1, x_sgl_l ~ x_sgl_r)
    cv2.rectangle(canvas, (x_sgl_l, y0), (x_sgl_r, y1), FIELD_INNER, -1)

    # 복식 사이드 구역 (양쪽, 단식 라인 바깥)
    cv2.rectangle(canvas, (x0,     y0), (x_sgl_l, y1), FIELD_SIDE, -1)
    cv2.rectangle(canvas, (x_sgl_r, y0), (x1,     y1), FIELD_SIDE, -1)

    # 더블 롱서비스 바깥쪽 (위/아래 엔드라인 ~ 롱서비스라인 사이, 복식 안쪽)
    cv2.rectangle(canvas, (x0, y0),        (x1, y_long_top), FIELD_OUTER, -1)
    cv2.rectangle(canvas, (x0, y_long_bot), (x1, y1),         FIELD_OUTER, -1)

    # ── 2. 외곽 경계선 (복식 경계, 굵기 2) ──────────────────
    cv2.rectangle(canvas, (x0, y0), (x1, y1), line_color, 2)

    # ── 3. 단식 사이드라인 (전체 길이) ──────────────────────
    cv2.line(canvas, (x_sgl_l, y0), (x_sgl_l, y1), line_color, 1)
    cv2.line(canvas, (x_sgl_r, y0), (x_sgl_r, y1), line_color, 1)

    # ── 4. 더블 롱서비스라인 (위/아래) ──────────────────────
    # 복식 전체 폭에 그림
    cv2.line(canvas, (x0, y_long_top), (x1, y_long_top), line_color, 1)
    cv2.line(canvas, (x0, y_long_bot), (x1, y_long_bot), line_color, 1)

    # ── 5. 숏서비스라인 (위/아래) ────────────────────────────
    # 복식 전체 폭에 그림
    cv2.line(canvas, (x0, y_short_top), (x1, y_short_top), line_color, 1)
    cv2.line(canvas, (x0, y_short_bot), (x1, y_short_bot), line_color, 1)

    # ── 6. 센터 서비스 라인 ──────────────────────────────────
    # BWF 규격: 위쪽 코트는 위 엔드라인 ~ 위 숏서비스라인
    #           아래 코트는 아래 숏서비스라인 ~ 아래 엔드라인
    cv2.line(canvas, (x_center, y0),        (x_center, y_short_top), line_color, 1)
    cv2.line(canvas, (x_center, y_short_bot), (x_center, y1),        line_color, 1)

    # ── 7. 네트 (중앙, 굵게) ────────────────────────────────
    cv2.line(canvas, (x0, y_net), (x1, y_net), net_color, 3)

    # 네트 포스트 (외곽 경계선 양끝에 짧은 수직 표시)
    _pole = 5
    cv2.line(canvas, (x0, y_net - _pole), (x0, y_net + _pole), net_color, 4)
    cv2.line(canvas, (x1, y_net - _pole), (x1, y_net + _pole), net_color, 4)

    return canvas


def create_minimap_canvas(
    bg_color: Tuple[int, int, int] = (28, 70, 28),
) -> np.ndarray:
    """
    배경 색과 코트 라인이 그려진 빈 미니맵 캔버스를 생성합니다.
    배경(아웃존): 어두운 녹색  /  필드: 밝은 녹색으로 draw_minimap_court()가 채움
    """
    w = MINIMAP_CONFIG["width"]
    h = MINIMAP_CONFIG["height"]
    canvas = np.full((h, w, 3), bg_color, dtype=np.uint8)
    draw_minimap_court(canvas, w, h)
    return canvas
