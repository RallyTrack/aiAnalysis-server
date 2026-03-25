"""
RallyTrack - 미니맵 렌더러 모듈 (v4 - 색상 버그 수정)

[수정 사항]
  1. _PlayerState에서 side / color 제거
     - 초기 분류(my < net_y)가 잘못될 경우 색상이 영구 고정되는 버그 수정
     - court.py의 net_y_minimap 버그(580→300 수정)와 함께 이중 보호

  2. PlayerTracker._get_ranked() 도입
     - 호출 시점마다 누적 경로 평균 Y로 top/bottom 동적 결정
     - draw_paths / draw_dots 에서 동일 순위를 참조 → 경로·점·마커 색상 일관

  3. HitMarkerRenderer fallback 제거
     - 기존: bottom이 None이면 top 위치로 대체 → 반대편에 마커 찍힘
     - 수정: None이면 조용히 스킵 (잘못된 마커 방지)

  4. update()에서 _top_rep / _bot_rep 잔재 코드 제거

색상 기준 (config.py와 동일):
  top    선수 → 핑크  (255, 105, 180)
  bottom 선수 → 라임  ( 50, 205,  50)

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
)
from .impact import ImpactEvent, build_hit_lookup


# ────────────────────────────────────────────────────────────
# 상수
# ────────────────────────────────────────────────────────────

_TOP_COLOR    = MINIMAP_CONFIG["top_color"]    # 핑크: (255, 105, 180)
_BOT_COLOR    = MINIMAP_CONFIG["bottom_color"] # 라임: ( 50, 205,  50)
_HIT_RADIUS   = MINIMAP_CONFIG["hit_radius"]
_MAX_PATH_LEN = 250
_KCONF        = POSE_CONFIG["keypoint_conf"]


# ────────────────────────────────────────────────────────────
# 단일 선수 상태  (side/color 제거 → 동적 결정)
# ────────────────────────────────────────────────────────────

class _PlayerState:
    """
    track_id 하나에 대응하는 선수 상태.
    색상/side는 보관하지 않고 PlayerTracker._get_ranked()에서 동적으로 결정합니다.
    """

    def __init__(self, track_id: int):
        self.track_id = track_id
        self.path: deque[Tuple[int, int]] = deque(maxlen=_MAX_PATH_LEN)
        self.last_pos: Optional[Tuple[int, int]] = None

    def add(self, mx: int, my: int) -> None:
        self.path.append((mx, my))
        self.last_pos = (mx, my)

    @property
    def avg_y(self) -> float:
        """경로 누적 평균 Y (정렬 기준)."""
        if not self.path:
            return float("inf")
        return sum(p[1] for p in self.path) / len(self.path)


# ────────────────────────────────────────────────────────────
# 선수 트래커
# ────────────────────────────────────────────────────────────

class PlayerTracker:
    """
    YOLO track_id 기반 선수 추적기.

    핵심 설계 원칙:
      - 초기 등장 시 임시 분류(net 기준)는 하지 않음
      - 호출 시점의 누적 평균 Y로 top/bottom 동적 결정
      - 모든 색상·위치 결정은 _get_ranked()를 공유해 일관성 유지
    """

    def __init__(self, net_y_minimap: float):
        self._net_y   = net_y_minimap  # 참고용 (현재 로직에서 미사용, 추후 확장 대비)
        self._players: Dict[int, _PlayerState] = {}

    # ── 업데이트 ──────────────────────────────────────────────

    def update(
        self,
        keypoints_list:    list,
        to_minimap_matrix: np.ndarray,
        minimap_pts:       np.ndarray,
    ) -> None:
        for person in keypoints_list:
            track_id, kpts = _parse_person(person)
            fx, fy = _foot_center(kpts)
            if fx <= 0 or fy <= 0:
                continue

            mx, my = frame_to_minimap((fx, fy), to_minimap_matrix)

            # 코트 바깥 30px 여백까지 허용
            if not _in_court_relaxed(mx, my, minimap_pts, margin=30):
                continue

            if track_id not in self._players:
                self._players[track_id] = _PlayerState(track_id)

            self._players[track_id].add(mx, my)

    # ── 핵심: 동적 순위 결정 ─────────────────────────────────

    def _get_ranked(self) -> List[_PlayerState]:
        """
        경로 평균 Y 기준 오름차순 정렬된 활성 선수 목록을 반환합니다.
          ranked[0]  → 평균 Y 가장 작음 = 화면 상단 = top  (핑크)
          ranked[-1] → 평균 Y 가장 큼   = 화면 하단 = bottom (라임)
        """
        active = [s for s in self._players.values() if s.last_pos is not None]
        return sorted(active, key=lambda s: s.avg_y)

    def _color_for_rank(self, rank: int) -> Tuple:
        """rank 0 → top(핑크), rank 1+ → bottom(라임)."""
        return _TOP_COLOR if rank == 0 else _BOT_COLOR

    # ── 위치 조회 (HitMarkerRenderer에서 사용) ───────────────

    def get_last_pos(self, side: str) -> Optional[Tuple[int, int]]:
        """
        'top' 또는 'bottom' 선수의 마지막 미니맵 위치를 반환합니다.
        누적 평균 Y 기반으로 동적 결정합니다.
        """
        ranked = self._get_ranked()
        if not ranked:
            return None
        if side == "top":
            return ranked[0].last_pos
        # bottom: 선수가 2명 이상 감지된 경우에만 반환
        return ranked[-1].last_pos if len(ranked) >= 2 else None

    # ── 렌더링 ────────────────────────────────────────────────

    def draw_paths(self, canvas: np.ndarray) -> None:
        """경로 라인 그리기. rank 기반으로 색상 결정."""
        for rank, state in enumerate(self._get_ranked()):
            color = self._color_for_rank(rank)
            _draw_path(canvas, state.path, color)

    def draw_dots(self, canvas: np.ndarray) -> None:
        """현재 위치 도트 그리기. rank 기반으로 색상 결정."""
        for rank, state in enumerate(self._get_ranked()):
            if state.last_pos is None:
                continue
            color = self._color_for_rank(rank)
            _draw_dot(canvas, state.last_pos, color)


# ────────────────────────────────────────────────────────────
# 타격 마커 렌더러
# ────────────────────────────────────────────────────────────

class HitMarkerRenderer:
    """
    타격 이벤트를 번호 마커로 미니맵에 영구 표시합니다.

    - event.owner("top"/"bottom")에 해당하는 선수의 미니맵 위치 사용
    - 선수 위치를 알 수 없으면 마커 등록 생략 (잘못된 위치 방지)
    """

    def __init__(self):
        self._markers: Dict[int, Tuple[int, int, ImpactEvent, Tuple]] = {}

    def register(
        self,
        event:          ImpactEvent,
        player_tracker: PlayerTracker,
    ) -> None:
        color = _TOP_COLOR if event.owner == "top" else _BOT_COLOR
        pos   = player_tracker.get_last_pos(event.owner)

        if pos is None:
            # 해당 선수 위치 정보 없음 → 반대 선수 위치로 대체하지 않고 스킵
            # (잘못된 쪽에 마커가 찍히는 것 방지)
            return

        self._markers[event.frame] = (pos[0], pos[1], event, color)

    def draw(self, canvas: np.ndarray) -> None:
        for mx, my, event, color in self._markers.values():
            r = _HIT_RADIUS
            # 배경 원
            cv2.circle(canvas, (mx, my), r,     color,           -1)
            cv2.circle(canvas, (mx, my), r,     (255, 255, 255),  2)
            # 번호 텍스트
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
    """
    미니맵 통합 렌더러.

    keypoints_list 형식 (두 가지 모두 지원):
      A) dict 래핑 형식 (권장):
           [{"track_id": 1, "keypoints": [[x,y,c],...]}, ...]
      B) raw ndarray/list 형식 (track_id 없음):
           [[[x,y,c], ...], ...]
         → 인덱스 기반 fake track_id 부여 (추적 불안정 가능)
    """

    def __init__(
        self,
        homographies: dict,
        hit_events:   List[ImpactEvent],
    ):
        self._hg         = homographies
        self._hit_lookup = build_hit_lookup(hit_events)
        self._player     = PlayerTracker(homographies["net_y_minimap"])
        self._hit_marker = HitMarkerRenderer()

    def render_frame(
        self,
        frame_idx:      int,
        shuttle_x:      float,
        shuttle_y:      float,
        keypoints_list: list,
    ) -> np.ndarray:
        hg     = self._hg
        canvas = create_minimap_canvas()

        parsed = _to_dict_format(keypoints_list)
        self._player.update(parsed, hg["to_minimap"], hg["minimap_pts"])

        if frame_idx in self._hit_lookup:
            self._hit_marker.register(self._hit_lookup[frame_idx], self._player)

        # 렌더링 순서: 경로 → 마커 → 도트 (도트가 최상단)
        self._player.draw_paths(canvas)
        self._hit_marker.draw(canvas)
        self._player.draw_dots(canvas)

        return canvas


# ────────────────────────────────────────────────────────────
# 모듈 수준 순수 함수
# ────────────────────────────────────────────────────────────

def _parse_person(person_data) -> Tuple[int, list]:
    """dict 형식과 raw ndarray/list 형식 모두 처리."""
    if isinstance(person_data, dict):
        return person_data.get("track_id", -1), person_data.get("keypoints", [])
    kpts = person_data.tolist() if hasattr(person_data, "tolist") else list(person_data)
    return -999, kpts


def _foot_center(kpts: list) -> Tuple[float, float]:
    """키포인트 목록에서 양발 중심 좌표를 반환합니다 (COCO 인덱스 15, 16)."""
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
    """코트 폴리곤 내부 또는 margin 픽셀 이내이면 True."""
    polygon = minimap_pts.astype(np.int32)
    dist    = cv2.pointPolygonTest(polygon, (float(mx), float(my)), True)
    return dist >= -margin


def _draw_path(canvas: np.ndarray, path: deque, color: Tuple) -> None:
    """경로 폴리라인을 캔버스에 그립니다."""
    if len(path) < 2:
        return
    pts = np.array(list(path), dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=2)


def _draw_dot(canvas: np.ndarray, pos: Tuple[int, int], color: Tuple) -> None:
    """선수 현재 위치 도트를 캔버스에 그립니다."""
    cv2.circle(canvas, pos, 9, color,           -1)
    cv2.circle(canvas, pos, 9, (255, 255, 255),  2)
    cv2.circle(canvas, pos, 4, (255, 255, 255), -1)


def _to_dict_format(keypoints_list: list) -> list:
    """raw 형식을 dict 형식으로 통일합니다."""
    if not keypoints_list:
        return []
    if isinstance(keypoints_list[0], dict):
        return keypoints_list
    result = []
    for i, person in enumerate(keypoints_list):
        kpts = person.tolist() if hasattr(person, "tolist") else list(person)
        result.append({"track_id": -(i + 1), "keypoints": kpts})
    return result