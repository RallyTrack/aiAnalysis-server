"""
RallyTrack - 미니맵 렌더러 모듈 (v5 - 주 선수 2인 집중 + 궤적 단축)

[수정 사항]
  1. 주 선수 2인(Top/Bottom) 선정 로직 도입
     - 네트를 기준으로 상단/하단 구역에서 코트 가로 중앙에 가장 가까운 인물 1명씩만 선정
     - 선정된 2명 외 제3자(옆 코트 선수·심판·관중 등)는 렌더링에서 완전 제외
     - track_id 기반 dict 제거 → Top/Bottom 슬롯(_RoleState) 방식으로 교체

  2. 이동 궤적 표시 길이 30프레임으로 제한
     - _RoleState.path: deque(maxlen=TRAIL_LEN=30)
     - draw_paths() 에서 해당 deque를 그대로 사용 → 최근 30개 점만 연결

색상 기준:
  Top    선수 → 블루 (config top_color    = #3B82F6)
  Bottom 선수 → 골드 (config bottom_color = #F59E0B)

렌더링 레이어 순서:
  코트 배경 → 선수 경로 라인 → 타격 번호 마커 → 선수 현재 위치 도트
"""

from __future__ import annotations

import cv2
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple

from .config import MINIMAP_CONFIG, POSE_CONFIG
from .court import (
    create_minimap_canvas,
    frame_to_minimap,
    _bwf_to_canvas,
    BWF_X,
    BWF_Y,
)
from .impact import ImpactEvent, build_hit_lookup


# ────────────────────────────────────────────────────────────
# 상수
# ────────────────────────────────────────────────────────────

_TOP_COLOR  = MINIMAP_CONFIG["top_color"]
_BOT_COLOR  = MINIMAP_CONFIG["bottom_color"]
_HIT_RADIUS = MINIMAP_CONFIG["hit_radius"]
_KCONF      = POSE_CONFIG["keypoint_conf"]

# 미니맵에 표시할 궤적 길이 (프레임 수)
TRAIL_LEN   = 30

# 코트 가로 중앙 픽셀 (주 선수 선정 기준)
# BWF 단식 중앙 = 3.05/6.10 = 50% → 캔버스 가로 정중앙과 일치
_CENTER_X   = MINIMAP_CONFIG["width"] / 2


# ────────────────────────────────────────────────────────────
# 슬롯별 선수 상태 (Top / Bottom)
# ────────────────────────────────────────────────────────────

class _RoleState:
    """Top 또는 Bottom 슬롯의 최근 궤적을 저장한다."""

    def __init__(self) -> None:
        self.path:      deque[Tuple[int, int]]         = deque(maxlen=TRAIL_LEN)
        self.full_path: List[Tuple[int, int, int]]     = []   # (frame_idx, mx, my) — no length cap

    def add(self, frame_idx: int, mx: int, my: int) -> None:
        self.path.append((mx, my))
        self.full_path.append((frame_idx, mx, my))

    @property
    def last_pos(self) -> Optional[Tuple[int, int]]:
        return self.path[-1] if self.path else None


# ────────────────────────────────────────────────────────────
# 선수 트래커 (네트 기준 2인 집중)
# ────────────────────────────────────────────────────────────

class PlayerTracker:
    """
    매 프레임마다 네트 기준 상단/하단 구역에서 각 1명씩 선수를 선정하고
    그 위치만 저장한다.

    선정 기준: 코트 가로 중앙(_CENTER_X)과 가장 가까운 인물
    → 옆 코트 선수나 심판처럼 사이드에 있는 사람은 자동으로 제외된다.
    """

    def __init__(self, net_y_minimap: float) -> None:
        self._net_y  = net_y_minimap
        self._top    = _RoleState()
        self._bottom = _RoleState()

    # ── 업데이트 ─────────────────────────────────────────────

    def update(
        self,
        frame_idx:         int,
        keypoints_list:    list,
        to_minimap_matrix: np.ndarray,
        minimap_pts:       np.ndarray,
    ) -> None:
        """
        keypoints_list에서 상단·하단 주 선수를 각 1명 선정하고 위치를 기록한다.
        선정되지 않은 인물은 완전히 무시한다.
        """
        # 프레임당 최적 후보: (코트 중앙까지 X 거리, mx, my)
        top_candidate:    Optional[Tuple[float, int, int]] = None
        bottom_candidate: Optional[Tuple[float, int, int]] = None

        for person in keypoints_list:
            _, kpts = _parse_person(person)
            fx, fy  = _foot_center(kpts)
            if fx <= 0 or fy <= 0:
                continue

            mx, my = frame_to_minimap((fx, fy), to_minimap_matrix)

            if not _in_court_relaxed(mx, my, minimap_pts, margin=30):
                continue

            dist = abs(mx - _CENTER_X)

            if my < self._net_y:   # 상단 구역
                if top_candidate is None or dist < top_candidate[0]:
                    top_candidate = (dist, mx, my)
            else:                  # 하단 구역
                if bottom_candidate is None or dist < bottom_candidate[0]:
                    bottom_candidate = (dist, mx, my)

        # 선정된 후보만 해당 슬롯에 추가
        if top_candidate is not None:
            self._top.add(frame_idx, top_candidate[1], top_candidate[2])
        if bottom_candidate is not None:
            self._bottom.add(frame_idx, bottom_candidate[1], bottom_candidate[2])

    # ── 위치 조회 (HitMarkerRenderer 용) ──────────────────────

    def get_last_pos(self, side: str) -> Optional[Tuple[int, int]]:
        if side == "top":
            return self._top.last_pos
        return self._bottom.last_pos

    def get_full_path(self, side: str) -> List[Tuple[int, int, int]]:
        """Return the complete frame-level position history as (frame_idx, mx, my) tuples."""
        return self._top.full_path if side == "top" else self._bottom.full_path

    # ── 렌더링 ───────────────────────────────────────────────

    def draw_paths(self, canvas: np.ndarray) -> None:
        """최근 TRAIL_LEN 프레임 궤적만 그린다."""
        _draw_path(canvas, self._top.path,    _TOP_COLOR)
        _draw_path(canvas, self._bottom.path, _BOT_COLOR)

    def draw_dots(self, canvas: np.ndarray) -> None:
        if self._top.last_pos is not None:
            _draw_dot(canvas, self._top.last_pos,    _TOP_COLOR)
        if self._bottom.last_pos is not None:
            _draw_dot(canvas, self._bottom.last_pos, _BOT_COLOR)


# ────────────────────────────────────────────────────────────
# 타격 마커 렌더러
# ────────────────────────────────────────────────────────────

class HitMarkerRenderer:
    def __init__(self) -> None:
        self._markers: Dict[int, Tuple[int, int, ImpactEvent, Tuple]] = {}

    def register(
        self,
        event:          ImpactEvent,
        player_tracker: PlayerTracker,
    ) -> None:
        color = _TOP_COLOR if event.owner == "top" else _BOT_COLOR
        pos   = player_tracker.get_last_pos(event.owner)

        if pos is None:
            return

        self._markers[event.frame] = (pos[0], pos[1], event, color)

    def draw(self, canvas: np.ndarray) -> None:
        for mx, my, event, color in self._markers.values():
            r = _HIT_RADIUS
            cv2.circle(canvas, (mx, my), r, color,           -1)
            cv2.circle(canvas, (mx, my), r, (255, 255, 255),  2)
            text   = str(event.hit_number)
            font   = cv2.FONT_HERSHEY_SIMPLEX
            scale  = 0.45
            thick  = 1
            tw, th = cv2.getTextSize(text, font, scale, thick)[0]
            cv2.putText(
                canvas, text,
                (mx - tw // 2, my + th // 2),
                font, scale, (255, 255, 255), thick, cv2.LINE_AA,
            )


# ────────────────────────────────────────────────────────────
# 미니맵 통합 렌더러
# ────────────────────────────────────────────────────────────

class MinimapRenderer:
    def __init__(
        self,
        homographies: dict,
        hit_events:   List[ImpactEvent],
    ) -> None:
        self._hg         = homographies
        self._hit_lookup = build_hit_lookup(hit_events)
        self._player     = PlayerTracker(homographies["net_y_minimap"])
        self._hit_marker = HitMarkerRenderer()

    def render_frame(
        self,
        frame_idx:      int,
        keypoints_list: list,
    ) -> np.ndarray:
        hg     = self._hg
        canvas = create_minimap_canvas()

        parsed = _to_dict_format(keypoints_list)
        self._player.update(frame_idx, parsed, hg["to_minimap"], hg["minimap_pts"])

        if frame_idx in self._hit_lookup:
            self._hit_marker.register(self._hit_lookup[frame_idx], self._player)

        self._player.draw_paths(canvas)
        self._hit_marker.draw(canvas)
        self._player.draw_dots(canvas)

        return canvas


# ────────────────────────────────────────────────────────────
# 홈존 계산 (기동력 메트릭용)
# ────────────────────────────────────────────────────────────

def compute_home_zone_minimap(
    side: str,
    minimap_w: int,
    minimap_h: int,
    pad: int,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Compute the home-zone bounding box in minimap pixel coordinates.

    Home zone is defined in BWF physical coordinates (metres) and converted
    via _bwf_to_canvas() so it adapts to any court input layout.

    Singles court centre line: x = 3.05m.
    Home zone x-width: ±1.3m from centre  → (1.75, 4.35)
    Short service line: near=4.72m, far=8.68m.

    Bottom player home: short_service_far(8.68m) ±1.0m  → (7.68, 9.68)
    Top    player home: short_service_near(4.72m) ±1.0m → (3.72, 5.72)

    Returns:
        ( (x_min_px, y_min_px), (x_max_px, y_max_px) )  — minimap pixel bbox
    """
    x_lo_m = BWF_X["center"] - 1.3   # 1.75m
    x_hi_m = BWF_X["center"] + 1.3   # 4.35m

    if side == "bottom":
        # 숏 서비스 라인(8.68m) 기준 ±1.0m
        y_lo_m = BWF_Y["short_service_far"] - 1.0   # 7.68m
        y_hi_m = BWF_Y["short_service_far"] + 1.0   # 9.68m
    else:
        # 숏 서비스 라인(4.72m) 기준 ±1.0m
        y_lo_m = BWF_Y["short_service_near"] - 1.0  # 3.72m
        y_hi_m = BWF_Y["short_service_near"] + 1.0  # 5.72m

    corners_bwf = [
        (x_lo_m, y_lo_m),
        (x_hi_m, y_lo_m),
        (x_hi_m, y_hi_m),
        (x_lo_m, y_hi_m),
    ]
    px_pts = [
        _bwf_to_canvas(x, y, minimap_w, minimap_h, pad)
        for x, y in corners_bwf
    ]
    xs = [int(p[0]) for p in px_pts]
    ys = [int(p[1]) for p in px_pts]
    return (min(xs), min(ys)), (max(xs), max(ys))


def point_in_zone(
    pt: Tuple[int, int],
    zone_min: Tuple[int, int],
    zone_max: Tuple[int, int],
) -> bool:
    return zone_min[0] <= pt[0] <= zone_max[0] and zone_min[1] <= pt[1] <= zone_max[1]


# ────────────────────────────────────────────────────────────
# 모듈 수준 순수 함수
# ────────────────────────────────────────────────────────────

def _parse_person(person_data) -> Tuple[int, list]:
    if isinstance(person_data, dict):
        return person_data.get("track_id", -1), person_data.get("keypoints", [])
    kpts = person_data.tolist() if hasattr(person_data, "tolist") else list(person_data)
    return -999, kpts


def _foot_center(kpts: list) -> Tuple[float, float]:
    if len(kpts) < 17:
        return 0.0, 0.0
    try:
        lx, ly, lc = float(kpts[15][0]), float(kpts[15][1]), float(kpts[15][2])
        rx, ry, rc = float(kpts[16][0]), float(kpts[16][1]), float(kpts[16][2])
    except (IndexError, TypeError):
        return 0.0, 0.0

    if lc >= _KCONF and rc >= _KCONF:
        return (lx + rx) / 2, (ly + ry) / 2
    elif lc >= _KCONF:
        return lx, ly
    elif rc >= _KCONF:
        return rx, ry
    return 0.0, 0.0


def _in_court_relaxed(mx: int, my: int, minimap_pts: np.ndarray, margin: int = 30) -> bool:
    polygon = minimap_pts.astype(np.int32)
    dist    = cv2.pointPolygonTest(polygon, (float(mx), float(my)), True)
    return dist >= -margin


def _draw_path(canvas: np.ndarray, path: deque, color: Tuple) -> None:
    if len(path) < 2:
        return
    pts = np.array(list(path), dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=2)


def _draw_dot(canvas: np.ndarray, pos: Tuple[int, int], color: Tuple) -> None:
    cv2.circle(canvas, pos, 9, color,           -1)
    cv2.circle(canvas, pos, 9, (255, 255, 255),  2)
    cv2.circle(canvas, pos, 4, (255, 255, 255), -1)


def _to_dict_format(keypoints_list: list) -> list:
    if not keypoints_list:
        return []
    if isinstance(keypoints_list[0], dict):
        return keypoints_list
    result = []
    for i, person in enumerate(keypoints_list):
        kpts = person.tolist() if hasattr(person, "tolist") else list(person)
        result.append({"track_id": -(i + 1), "keypoints": kpts})
    return result
