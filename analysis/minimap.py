"""
RallyTrack - 미니맵 렌더러 모듈 (v3 - track_id 기반 선수 추적)

핵심 변경:
  1. YOLO track_id를 이용해 선수를 프레임 간 고정 추적
     → Y좌표 기반 매 프레임 재분류 제거 → 경로 왔다갔다 버그 해결
  2. 셔틀콕 궤적/경로 표시 제거
  3. 선수 위치 도트 + 경로 라인만 표시
  4. 타격 마커: 타격 시점의 해당 선수(owner side) 위치에 정확히 표시
  5. is_inside_court 체크 완화 (30px 여백 허용)

렌더링 레이어 순서:
  코트 배경 → 선수 경로 라인 → 타격 번호 마커 → 선수 현재 위치 도트

main.py / test_minimap.py에서 extract_keypoints를 아래처럼 수정하면
track_id가 자동으로 연결됩니다:

    def extract_keypoints(pose_result) -> list:
        if pose_result.keypoints is None:
            return []
        kpts  = pose_result.keypoints.data.cpu().numpy()
        boxes = pose_result.boxes
        ids   = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else []
        result = []
        for i, k in enumerate(kpts):
            tid = int(ids[i]) if i < len(ids) else -(i + 1)
            result.append({"track_id": tid, "keypoints": k.tolist()})
        return result
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

_TOP_COLOR    = MINIMAP_CONFIG["top_color"]
_BOT_COLOR    = MINIMAP_CONFIG["bottom_color"]
_HIT_RADIUS   = MINIMAP_CONFIG["hit_radius"]
_MAX_PATH_LEN = 250
_KCONF        = POSE_CONFIG["keypoint_conf"]


# ────────────────────────────────────────────────────────────
# 단일 선수 상태
# ────────────────────────────────────────────────────────────

class _PlayerState:
    """track_id 하나에 대응하는 선수 상태."""

    def __init__(self, track_id: int, side: str):
        self.track_id = track_id
        self.side     = side
        self.color    = _TOP_COLOR if side == "top" else _BOT_COLOR
        self.path: deque[Tuple[int, int]] = deque(maxlen=_MAX_PATH_LEN)
        self.last_pos: Optional[Tuple[int, int]] = None

    def add(self, mx: int, my: int) -> None:
        self.path.append((mx, my))
        self.last_pos = (mx, my)


# ────────────────────────────────────────────────────────────
# 선수 트래커
# ────────────────────────────────────────────────────────────

class PlayerTracker:
    """
    YOLO track_id 기반 선수 추적기.

    - track_id 최초 등장 시 미니맵 Y좌표(net 기준)로 top/bottom 결정
    - 이후 프레임에서는 track_id로 동일인 식별 → 경로 안정
    """

    def __init__(self, net_y_minimap: float):
        self._net_y   = net_y_minimap
        self._players: Dict[int, _PlayerState] = {}
        # side 별 대표 (가장 최근 업데이트된 선수)
        self._top_rep: Optional[_PlayerState] = None
        self._bot_rep: Optional[_PlayerState] = None

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
                side = "top" if my < self._net_y else "bottom"
                self._players[track_id] = _PlayerState(track_id, side)

            state = self._players[track_id]
            state.add(mx, my)

            if state.side == "top":
                self._top_rep = state
            else:
                self._bot_rep = state

    def get_last_pos(self, side: str) -> Optional[Tuple[int, int]]:
        rep = self._top_rep if side == "top" else self._bot_rep
        return rep.last_pos if rep else None

    def draw_paths(self, canvas: np.ndarray) -> None:
        for state in self._players.values():
            _draw_path(canvas, state.path, state.color)

    def draw_dots(self, canvas: np.ndarray) -> None:
        for state in self._players.values():
            if state.last_pos:
                _draw_dot(canvas, state.last_pos, state.color)


# ────────────────────────────────────────────────────────────
# 타격 마커 렌더러
# ────────────────────────────────────────────────────────────

class HitMarkerRenderer:
    """
    타격 이벤트를 번호 마커로 미니맵에 영구 표시.

    - event.owner("top"/"bottom")에 해당하는 선수의 미니맵 위치를 사용
    - 선수 위치를 알 수 없으면 반대편 선수 위치로 폴백
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
            other = "bottom" if event.owner == "top" else "top"
            pos   = player_tracker.get_last_pos(other)
        if pos is None:
            return
        self._markers[event.frame] = (pos[0], pos[1], event, color)

    def draw(self, canvas: np.ndarray) -> None:
        for mx, my, event, color in self._markers.values():
            r = _HIT_RADIUS
            cv2.circle(canvas, (mx, my), r,     color,          -1)
            cv2.circle(canvas, (mx, my), r,     (255, 255, 255), 2)
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
        shuttle_x:      float,   # 셔틀콕 표시 제거 → 미사용 (호환 유지)
        shuttle_y:      float,
        keypoints_list: list,
    ) -> np.ndarray:
        hg     = self._hg
        canvas = create_minimap_canvas()

        parsed = _to_dict_format(keypoints_list)
        self._player.update(parsed, hg["to_minimap"], hg["minimap_pts"])

        if frame_idx in self._hit_lookup:
            self._hit_marker.register(self._hit_lookup[frame_idx], self._player)

        # 렌더링 순서
        self._player.draw_paths(canvas)      # 1. 경로 라인
        self._hit_marker.draw(canvas)        # 2. 타격 번호 마커
        self._player.draw_dots(canvas)       # 3. 현재 위치 도트 (최상단)

        return canvas


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
    cv2.circle(canvas, pos, 9, color,          -1)
    cv2.circle(canvas, pos, 9, (255, 255, 255), 2)
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