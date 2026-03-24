"""
RallyTrack - 미니맵 렌더러 모듈 (개선판)

변경 사항:
  1. 스켈레톤 렌더러 제거 (Top-Down 뷰에 스켈레톤은 부적합)
  2. 셔틀콕 궤적 유지 (선택적 표시)
  3. 선수 경로 + 현재 위치 도트만 표시 (깔끔)
  4. 타점 마커: 번호 + 색상 (선수 구분)
  5. 선수 위치 히스토리를 PlayerTracker에서 관리
     → 타점 선수 판별을 impact.py의 owner 기반으로만 받아서 사용
"""

from __future__ import annotations

import cv2
import numpy as np
from collections import deque
from typing import List, Optional, Tuple, Dict

from .config import MINIMAP_CONFIG, POSE_CONFIG, SKELETON_EDGES
from .court import (
    create_minimap_canvas,
    frame_to_minimap,
    frame_to_normalized,
    normalized_to_minimap,
    is_inside_court,
)
from .impact import ImpactEvent, build_hit_lookup


# ────────────────────────────────────────────────────────────
# 셔틀콕 궤적 렌더러
# ────────────────────────────────────────────────────────────

class ShuttleTrailRenderer:
    """셔틀콕 궤적을 미니맵에 페이드-인 점으로 그립니다."""

    def __init__(self, trail_length: int = MINIMAP_CONFIG["trail_length"]):
        self._trail: deque[Tuple[int, int]] = deque(maxlen=trail_length)
        self._color = MINIMAP_CONFIG["shuttle_color"]

    def update(self, x: float, y: float, to_minimap_matrix: np.ndarray) -> None:
        if x > 0 and y > 0:
            mx, my = frame_to_minimap((x, y), to_minimap_matrix)
            self._trail.append((mx, my))

    def draw(self, canvas: np.ndarray) -> None:
        n = len(self._trail)
        if n == 0:
            return
        for i, (mx, my) in enumerate(self._trail):
            alpha  = (i + 1) / n
            radius = max(2, int(3 + 5 * alpha))
            color  = tuple(int(c * alpha) for c in self._color)
            cv2.circle(canvas, (mx, my), radius, color, -1)


# ────────────────────────────────────────────────────────────
# 선수 위치 트래커
# ────────────────────────────────────────────────────────────

class PlayerTracker:
    """
    선수 발 위치를 추적하고 Top/Bottom을 분류합니다.

    - 미니맵 좌표계에서 네트 Y 기준으로 분류
    - 경로(polyline) + 현재 위치 도트 렌더링
    - 타점 마커를 위한 마지막 위치 제공
    """

    MAX_PATH_LEN = 300

    def __init__(self):
        self._top_path:    deque[Tuple[int, int]] = deque(maxlen=self.MAX_PATH_LEN)
        self._bottom_path: deque[Tuple[int, int]] = deque(maxlen=self.MAX_PATH_LEN)
        self._top_color    = MINIMAP_CONFIG["top_color"]
        self._bottom_color = MINIMAP_CONFIG["bottom_color"]

        self.last_top_pos:    Optional[Tuple[int, int]] = None
        self.last_bottom_pos: Optional[Tuple[int, int]] = None

    def update(
        self,
        keypoints_list:    list,
        to_minimap_matrix: np.ndarray,
        minimap_pts:       np.ndarray,
        net_y_minimap:     float,
    ) -> None:
        """
        키포인트에서 발 위치를 추출하여 경로에 추가합니다.
        코트 내부에 있는 선수만 추적합니다.
        """
        for person in keypoints_list:
            foot_x, foot_y = self._get_foot_center(person)
            if foot_x <= 0 or foot_y <= 0:
                continue

            mx, my = frame_to_minimap((foot_x, foot_y), to_minimap_matrix)

            if not is_inside_court((mx, my), minimap_pts):
                continue

            if my < net_y_minimap:
                self._top_path.append((mx, my))
                self.last_top_pos = (mx, my)
            else:
                self._bottom_path.append((mx, my))
                self.last_bottom_pos = (mx, my)

    def draw(self, canvas: np.ndarray) -> None:
        """경로 라인과 현재 위치 마커를 그립니다."""
        self._draw_path(canvas, self._top_path,    self._top_color)
        self._draw_path(canvas, self._bottom_path, self._bottom_color)

        if self.last_top_pos:
            self._draw_player_dot(canvas, self.last_top_pos, self._top_color)
        if self.last_bottom_pos:
            self._draw_player_dot(canvas, self.last_bottom_pos, self._bottom_color)

    @staticmethod
    def _get_foot_center(person: list) -> Tuple[float, float]:
        """양 발목(15, 16) 평균 → 선수 위치."""
        if len(person) < 17:
            return 0.0, 0.0
        l_x, l_y, l_c = person[15][0], person[15][1], person[15][2]
        r_x, r_y, r_c = person[16][0], person[16][1], person[16][2]
        kconf = POSE_CONFIG["keypoint_conf"]

        if l_c >= kconf and r_c >= kconf:
            return (l_x + r_x) / 2, (l_y + r_y) / 2
        elif l_c >= kconf:
            return float(l_x), float(l_y)
        elif r_c >= kconf:
            return float(r_x), float(r_y)
        return 0.0, 0.0

    @staticmethod
    def _draw_path(canvas: np.ndarray, path: deque, color: Tuple) -> None:
        if len(path) < 2:
            return
        pts = np.array(list(path), dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=2)

    @staticmethod
    def _draw_player_dot(canvas: np.ndarray, pos: Tuple[int, int], color: Tuple) -> None:
        cv2.circle(canvas, pos, 9,  color,        -1)
        cv2.circle(canvas, pos, 9,  (255,255,255), 2)
        cv2.circle(canvas, pos, 4,  (255,255,255), -1)


# ────────────────────────────────────────────────────────────
# 타점 마커 렌더러
# ────────────────────────────────────────────────────────────

class HitMarkerRenderer:
    """
    타점 이벤트를 미니맵에 번호 마커로 영구 표시합니다.

    [개선]
    - 선수 위치(발 중심)를 우선 사용, 없으면 셔틀콕 위치 폴백
    - 마커 크기/폰트 최적화
    """

    def __init__(self):
        # frame → (mx, my, event, color)
        self._markers: Dict[int, Tuple[int, int, ImpactEvent, Tuple]] = {}
        self._radius = MINIMAP_CONFIG["hit_radius"]

    def register(
        self,
        event:          ImpactEvent,
        shuttle_x:      float,
        shuttle_y:      float,
        player_tracker: PlayerTracker,
        to_minimap_matrix: np.ndarray,
    ) -> None:
        """
        타점 이벤트 등록.
        event.owner("top"/"bottom") 기반으로 선수 위치를 결정합니다.
        """
        color = (
            MINIMAP_CONFIG["top_color"]
            if event.owner == "top"
            else MINIMAP_CONFIG["bottom_color"]
        )

        # 선수 발 위치 우선
        if event.owner == "top" and player_tracker.last_top_pos:
            mx, my = player_tracker.last_top_pos
        elif event.owner == "bottom" and player_tracker.last_bottom_pos:
            mx, my = player_tracker.last_bottom_pos
        elif shuttle_x > 0 and shuttle_y > 0:
            # 셔틀콕 미니맵 좌표 폴백
            mx, my = frame_to_minimap((shuttle_x, shuttle_y), to_minimap_matrix)
        else:
            return

        self._markers[event.frame] = (mx, my, event, color)

    def draw(self, canvas: np.ndarray) -> None:
        for mx, my, event, color in self._markers.values():
            r = self._radius
            # 외곽 원
            cv2.circle(canvas, (mx, my), r,     color,        -1)
            cv2.circle(canvas, (mx, my), r,     (255,255,255), 2)

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
    미니맵 요소를 통합 관리하는 파사드(Facade) 클래스.

    [변경 사항]
    - SkeletonMinimapRenderer 제거 (Top-Down 뷰에 스켈레톤 부적합)
    - PlayerPathRenderer → PlayerTracker로 리팩터링
    - ShuttleTrailRenderer 유지 (선택적)
    - HitMarkerRenderer: owner 기반 선수 분류 사용

    Usage:
        renderer = MinimapRenderer(homographies, hit_events)
        for frame_idx, frame in enumerate(frames):
            canvas = renderer.render_frame(frame_idx, sx, sy, keypoints_list)
    """

    def __init__(
        self,
        homographies: dict,
        hit_events:   List[ImpactEvent],
    ):
        self._hg         = homographies
        self._hit_lookup = build_hit_lookup(hit_events)

        self._shuttle    = ShuttleTrailRenderer()
        self._player     = PlayerTracker()
        self._hit_marker = HitMarkerRenderer()

    def render_frame(
        self,
        frame_idx:      int,
        shuttle_x:      float,
        shuttle_y:      float,
        keypoints_list: list,
    ) -> np.ndarray:
        """
        한 프레임의 미니맵을 렌더링하여 반환합니다.

        렌더링 순서:
          1. 코트 배경 (create_minimap_canvas)
          2. 셔틀콕 궤적
          3. 선수 경로 라인
          4. 타점 번호 마커
          5. 선수 현재 위치 도트 (가장 위에 표시)
        """
        hg     = self._hg
        canvas = create_minimap_canvas()

        # ── 업데이트 ──────────────────────────────────────
        # 선수 위치 추적 (키포인트 → 미니맵 좌표 변환)
        self._player.update(
            keypoints_list,
            hg["to_minimap"],
            hg["minimap_pts"],
            hg["net_y_minimap"],
        )

        # 셔틀콕 궤적 추적
        self._shuttle.update(shuttle_x, shuttle_y, hg["to_minimap"])

        # 타점 이벤트 등록 (해당 프레임이면)
        if frame_idx in self._hit_lookup:
            event = self._hit_lookup[frame_idx]
            self._hit_marker.register(
                event,
                shuttle_x,
                shuttle_y,
                self._player,
                hg["to_minimap"],
            )

        # ── 렌더링 순서 ───────────────────────────────────
        # 1. 셔틀콕 궤적 (배경 바로 위)
        self._shuttle.draw(canvas)

        # 2. 선수 경로 라인
        self._draw_paths_only(canvas)

        # 3. 타점 번호 마커 (경로 위에)
        self._hit_marker.draw(canvas)

        # 4. 선수 현재 위치 도트 (최상단 레이어)
        self._draw_player_dots(canvas)

        return canvas

    def _draw_paths_only(self, canvas: np.ndarray) -> None:
        """선수 경로 라인만 그립니다 (현재 위치 도트 제외)."""
        PlayerTracker._draw_path(
            canvas, self._player._top_path, self._player._top_color
        )
        PlayerTracker._draw_path(
            canvas, self._player._bottom_path, self._player._bottom_color
        )

    def _draw_player_dots(self, canvas: np.ndarray) -> None:
        """선수 현재 위치 도트만 그립니다 (마커 위에 표시)."""
        if self._player.last_top_pos:
            PlayerTracker._draw_player_dot(
                canvas, self._player.last_top_pos, self._player._top_color
            )
        if self._player.last_bottom_pos:
            PlayerTracker._draw_player_dot(
                canvas, self._player.last_bottom_pos, self._player._bottom_color
            )