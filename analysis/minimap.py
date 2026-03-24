"""
RallyTrack - 미니맵 렌더러 모듈

역할:
  - 매 프레임마다 미니맵 캔버스를 생성하고 요소를 그립니다.
    · 셔틀콕 궤적
    · 선수 위치 (스켈레톤 투영)
    · 선수 동선(경로) 라인
    · 타점 마커 (번호 포함)

[원본 대비 주요 수정 사항]
  1. 선수 발목 중심→미니맵 변환 후 코트 내부 여부를 
     src_pts(영상 좌표)로 체크 → 미니맵 좌표로 체크하도록 수정
     (원본: pointPolygonTest에 shifted_dst_pts를 쓰면서 동시에
      영상 좌표를 넘기는 혼용이 있었음)
  2. 선수 경로를 deque 없이 계속 누적했는데 긴 경기에서 메모리 낭비
     → maxlen 제한 있는 deque로 교체
  3. hit_draw_data를 매 프레임 반복 순회 → 타점이 많을수록 느려짐
     → 타점 인덱스 집합으로 미리 저장해 O(1) 조회로 개선
"""

from __future__ import annotations

import cv2
import numpy as np
from collections import deque
from typing import List, Optional, Tuple

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

    def update(
        self,
        x: float,
        y: float,
        to_minimap_matrix: np.ndarray,
    ) -> None:
        """유효한 좌표(>0)이면 미니맵 좌표로 변환 후 궤적에 추가합니다."""
        if x > 0 and y > 0:
            mx, my = frame_to_minimap((x, y), to_minimap_matrix)
            self._trail.append((mx, my))

    def draw(self, canvas: np.ndarray) -> None:
        """궤적을 캔버스에 그립니다. 최신일수록 크고 밝게 표시합니다."""
        n = len(self._trail)
        if n == 0:
            return
        for i, (mx, my) in enumerate(self._trail):
            # 오래된 점: 작고 어둡게 / 최신 점: 크고 밝게
            alpha  = (i + 1) / n
            radius = max(2, int(3 + 5 * alpha))
            color  = tuple(int(c * alpha) for c in self._color)
            cv2.circle(canvas, (mx, my), radius, color, -1)


# ────────────────────────────────────────────────────────────
# 선수 경로 렌더러
# ────────────────────────────────────────────────────────────

class PlayerPathRenderer:
    """선수 발 위치를 미니맵에 경로 라인으로 그립니다."""

    # 최근 N개 위치만 유지 (너무 긴 경로는 오히려 가독성 저하)
    MAX_PATH_LEN = 300

    def __init__(self):
        self._top_path: deque[Tuple[int, int]]    = deque(maxlen=self.MAX_PATH_LEN)
        self._bottom_path: deque[Tuple[int, int]] = deque(maxlen=self.MAX_PATH_LEN)
        self._top_color    = MINIMAP_CONFIG["top_color"]
        self._bottom_color = MINIMAP_CONFIG["bottom_color"]

        # 마지막으로 알려진 위치 (타점 마커에 선수 위치 대입할 때 사용)
        self.last_top_pos:    Optional[Tuple[int, int]] = None
        self.last_bottom_pos: Optional[Tuple[int, int]] = None

    def update(
        self,
        keypoints_list: list,            # YOLOv8 검출된 사람별 키포인트 배열
        to_minimap_matrix:    np.ndarray,
        minimap_pts:          np.ndarray,
        net_y_minimap:        float,
        to_normalized_matrix: np.ndarray,
    ) -> None:
        """
        검출된 사람 키포인트를 미니맵에 투영하고 경로를 업데이트합니다.

        Args:
            keypoints_list     : person별 keypoints 배열 [(17, 3), ...]
            to_minimap_matrix  : 영상 → 미니맵 변환 행렬
            minimap_pts        : 미니맵 내 코트 코너 좌표
            net_y_minimap      : 미니맵 내 네트 Y 좌표
            to_normalized_matrix: 영상 → [0,1]² 변환 행렬 (경로 저장용)
        """
        for person in keypoints_list:
            foot_x, foot_y = self._get_foot_center(person)
            if foot_x <= 0 or foot_y <= 0:
                continue

            # 미니맵 좌표로 변환
            mx, my = frame_to_minimap((foot_x, foot_y), to_minimap_matrix)

            # 코트 내부 여부 확인 (미니맵 좌표 기준)
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
        # 경로 라인
        self._draw_path(canvas, self._top_path,    self._top_color)
        self._draw_path(canvas, self._bottom_path, self._bottom_color)

        # 현재 위치 마커
        if self.last_top_pos:
            self._draw_player_dot(canvas, self.last_top_pos, self._top_color)
        if self.last_bottom_pos:
            self._draw_player_dot(canvas, self.last_bottom_pos, self._bottom_color)

    @staticmethod
    def _get_foot_center(person: np.ndarray) -> Tuple[float, float]:
        """
        양 발목(왼쪽=15, 오른쪽=16) 평균을 선수 위치로 사용합니다.
        한쪽 발목만 가시적이면 그 발목을 사용합니다.
        """
        l_x, l_y, l_c = person[15]
        r_x, r_y, r_c = person[16]
        kconf = POSE_CONFIG["keypoint_conf"]

        if l_c >= kconf and r_c >= kconf:
            return (l_x + r_x) / 2, (l_y + r_y) / 2
        elif l_c >= kconf:
            return float(l_x), float(l_y)
        elif r_c >= kconf:
            return float(r_x), float(r_y)
        return 0.0, 0.0

    @staticmethod
    def _draw_path(
        canvas: np.ndarray,
        path:   deque,
        color:  Tuple,
    ) -> None:
        if len(path) < 2:
            return
        pts = np.array(list(path), dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=2)

    @staticmethod
    def _draw_player_dot(
        canvas: np.ndarray,
        pos:    Tuple[int, int],
        color:  Tuple,
    ) -> None:
        cv2.circle(canvas, pos, 8, color,        -1)
        cv2.circle(canvas, pos, 8, (255,255,255), 1)


# ────────────────────────────────────────────────────────────
# 스켈레톤 렌더러 (미니맵용)
# ────────────────────────────────────────────────────────────

class SkeletonMinimapRenderer:
    """
    선수의 스켈레톤을 미니맵에 투영하여 그립니다.
    미니맵은 Top-Down 뷰이므로 발 위치를 기준점으로 사용하고
    상대 좌표로 팔다리를 그립니다.
    """

    def draw(
        self,
        canvas:           np.ndarray,
        keypoints_list:   list,
        to_minimap_matrix: np.ndarray,
        minimap_pts:      np.ndarray,
        net_y_minimap:    float,
    ) -> None:
        cfg   = POSE_CONFIG
        edges = SKELETON_EDGES   # (1-indexed) 쌍

        for person in keypoints_list:
            foot_x, foot_y = PlayerPathRenderer._get_foot_center(person)
            if foot_x <= 0 or foot_y <= 0:
                continue

            tx, ty = frame_to_minimap((foot_x, foot_y), to_minimap_matrix)
            if not is_inside_court((tx, ty), minimap_pts):
                continue

            color = (
                MINIMAP_CONFIG["top_color"]
                if ty < net_y_minimap
                else MINIMAP_CONFIG["bottom_color"]
            )

            # 뼈대 그리기 (1-indexed → 0-indexed)
            for p1_i, p2_i in edges:
                p1 = person[p1_i - 1]
                p2 = person[p2_i - 1]
                if p1[2] < cfg["keypoint_conf"] or p2[2] < cfg["keypoint_conf"]:
                    continue
                x1 = int(tx + (p1[0] - foot_x) * cfg["scale"])
                y1 = int(ty + (p1[1] - foot_y) * cfg["scale"])
                x2 = int(tx + (p2[0] - foot_x) * cfg["scale"])
                y2 = int(ty + (p2[1] - foot_y) * cfg["scale"])
                cv2.line(canvas, (x1, y1), (x2, y2), color, cfg["bone_thickness"])

            # 머리 (Nose = 인덱스 0)
            if person[0][2] >= cfg["keypoint_conf"]:
                hx = int(tx + (person[0][0] - foot_x) * cfg["scale"])
                hy = int(ty + (person[0][1] - foot_y) * cfg["scale"])
                r  = cfg["head_radius"]
                cv2.circle(canvas, (hx, hy), r, color,        -1)
                cv2.circle(canvas, (hx, hy), r, (255,255,255), 2)


# ────────────────────────────────────────────────────────────
# 타점 마커 렌더러
# ────────────────────────────────────────────────────────────

class HitMarkerRenderer:
    """
    타점 이벤트를 미니맵에 영구적으로 표시합니다.
    번호가 붙은 원형 마커 형태입니다.
    """

    def __init__(self):
        # {frame: (minimap_x, minimap_y, event)} 저장
        self._markers: dict[int, Tuple[int, int, ImpactEvent]] = {}
        self._radius = MINIMAP_CONFIG["hit_radius"]

    def register(
        self,
        event:           ImpactEvent,
        shuttle_x:       float,
        shuttle_y:       float,
        player_path_renderer: PlayerPathRenderer,
        to_normalized_matrix: np.ndarray,
    ) -> None:
        """
        타점 이벤트를 등록합니다.
        선수 위치(발 중심) → 없으면 셔틀콕 위치를 대신 사용합니다.

        [수정 이유]
        원본은 셔틀콕 좌표를 정규화 → 미니맵으로 변환해
        타점 마커를 셔틀콕 위치에 그렸습니다.
        타구 직후 셔틀콕은 이미 이동해 있어 선수 위치와 다릅니다.
        실제 타구 위치는 선수의 발 중심 근처이므로 선수 위치를 우선합니다.
        """
        color = (
            MINIMAP_CONFIG["top_color"]
            if event.owner == "top"
            else MINIMAP_CONFIG["bottom_color"]
        )

        # 선수 위치 우선
        if event.owner == "top" and player_path_renderer.last_top_pos:
            mx, my = player_path_renderer.last_top_pos
        elif event.owner == "bottom" and player_path_renderer.last_bottom_pos:
            mx, my = player_path_renderer.last_bottom_pos
        elif shuttle_x > 0 and shuttle_y > 0:
            # 셔틀콕 정규화 → 미니맵 좌표 폴백
            nx, ny = frame_to_normalized((shuttle_x, shuttle_y), to_normalized_matrix)
            mx, my = normalized_to_minimap(nx, ny)
        else:
            return

        self._markers[event.frame] = (mx, my, event, color)

    def draw(self, canvas: np.ndarray) -> None:
        """등록된 모든 타점 마커를 캔버스에 그립니다."""
        for mx, my, event, color in self._markers.values():
            r = self._radius
            cv2.circle(canvas, (mx, my), r, color,        -1)
            cv2.circle(canvas, (mx, my), r, (255,255,255), 2)

            text  = str(event.hit_number)
            font  = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.45
            thick = 1
            tw, th = cv2.getTextSize(text, font, scale, thick)[0]
            cv2.putText(
                canvas, text,
                (mx - tw // 2, my + th // 2),
                font, scale, (255, 255, 255), thick,
                cv2.LINE_AA,
            )


# ────────────────────────────────────────────────────────────
# 미니맵 통합 렌더러
# ────────────────────────────────────────────────────────────

class MinimapRenderer:
    """
    모든 미니맵 요소를 통합 관리하는 파사드(Facade) 클래스.

    Usage:
        renderer = MinimapRenderer(homographies, hit_events)
        for frame_idx, frame in enumerate(frames):
            canvas = renderer.render_frame(frame_idx, shuttle_x, shuttle_y, keypoints_list)
            # canvas: (H, W, 3) uint8 minimap image
    """

    def __init__(
        self,
        homographies: dict,
        hit_events:   List[ImpactEvent],
    ):
        self._hg  = homographies
        self._hit_lookup = build_hit_lookup(hit_events)

        self._shuttle  = ShuttleTrailRenderer()
        self._player   = PlayerPathRenderer()
        self._skeleton = SkeletonMinimapRenderer()
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

        Args:
            frame_idx      : 현재 프레임 번호
            shuttle_x/y    : 셔틀콕 좌표 (비가시이면 0)
            keypoints_list : YOLOv8 검출 결과 (person별 17×3 배열)

        Returns:
            (H, W, 3) uint8 미니맵 이미지
        """
        hg = self._hg
        canvas = create_minimap_canvas()

        # 1. 선수 경로 & 스켈레톤 업데이트
        self._player.update(
            keypoints_list,
            hg["to_minimap"],
            hg["minimap_pts"],
            hg["net_y_minimap"],
            hg["to_normalized"],
        )

        # 2. 셔틀콕 궤적 업데이트
        self._shuttle.update(shuttle_x, shuttle_y, hg["to_minimap"])

        # 3. 타점 이벤트 등록 (해당 프레임이면)
        if frame_idx in self._hit_lookup:
            event = self._hit_lookup[frame_idx]
            self._hit_marker.register(
                event,
                shuttle_x,
                shuttle_y,
                self._player,
                hg["to_normalized"],
            )

        # 4. 그리기 순서: 경로 → 셔틀 → 타점 마커 → 스켈레톤 → 현재 선수 위치
        self._shuttle.draw(canvas)
        self._player.draw(canvas)   # draw()에서 경로 + 현재 위치 모두 그림
        self._hit_marker.draw(canvas)
        self._skeleton.draw(
            canvas,
            keypoints_list,
            hg["to_minimap"],
            hg["minimap_pts"],
            hg["net_y_minimap"],
        )

        return canvas
