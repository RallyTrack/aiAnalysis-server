"""
RallyTrack - 네트 판정 모듈 (NetJudge)

[역할]
  사용자가 지정한 네트 상단 좌우 좌표(net_coords)를 기반으로 셔틀콕
  궤적을 분석하여 '네트 걸림(NET_FAULT)' 이벤트를 감지한다.

[좌표 규칙]
  - net_coords: [[left_x, left_y], [right_x, right_y]]  영상 픽셀 좌표
  - 영상 좌표계에서 y는 아래로 갈수록 증가
    → shuttle_y > net_y_at_x  이면 셔틀콕이 네트보다 낮음 (걸림 가능)

[알고리즘]
  1. 네트 상단 선분: 두 점을 잇는 직선 (y = slope·x + intercept)
  2. 셔틀콕 x 좌표가 네트 중심 x를 좌→우 또는 우→좌로 통과할 때를
     '네트 통과 시도'로 판단
  3. 통과 순간 셔틀콕의 보간 y와 해당 x에서의 네트 y를 비교
     - shuttle_y_at_crossing > net_top_y  →  NET_FAULT
  4. 같은 통과 이벤트가 연속 프레임에서 중복 감지되지 않도록
     cooldown(기본 30프레임) 적용
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# ────────────────────────────────────────────────────────────
# 데이터 클래스
# ────────────────────────────────────────────────────────────

@dataclass
class NetFaultEvent:
    """네트 걸림 이벤트 하나를 표현한다."""
    frame:           int
    time_sec:        float
    crossing_x:      float   # 네트 중심 x
    shuttle_y:       float   # 통과 순간 셔틀콕 y (보간)
    net_top_y_at_x:  float   # 해당 x에서 네트 상단 y
    clearance_px:    float   # 양수 = 통과, 음수 = 걸림

    def to_timeline_dict(self) -> dict:
        """백엔드 AnalysisCompleteRequest.TimelineEventRequest 형식으로 변환."""
        mm, ss = divmod(int(self.time_sec), 60)
        return {
            "timestamp":        int(self.time_sec),
            "displayTime":      f"{mm:02d}:{ss:02d}",
            "eventType":        "NET_FAULT",
            "eventTitle":       "네트 걸림",
            "eventDescription": (
                f"셔틀콕이 네트 상단보다 {abs(self.clearance_px):.1f}px 낮은 "
                f"위치를 통과하려 했습니다."
            ),
            "eventScore":       None,
        }

    def to_dict(self) -> dict:
        return {
            "frame":          self.frame,
            "time_sec":       round(self.time_sec, 3),
            "crossing_x":     round(self.crossing_x, 1),
            "shuttle_y":      round(self.shuttle_y, 1),
            "net_top_y_at_x": round(self.net_top_y_at_x, 1),
            "clearance_px":   round(self.clearance_px, 1),
        }


# ────────────────────────────────────────────────────────────
# 메인 클래스
# ────────────────────────────────────────────────────────────

class NetJudge:
    """
    네트 판정기.

    Usage::

        judge = NetJudge(net_coords=[[300, 690], [980, 690]], fps=30.0)
        faults = judge.detect_faults(x_arr, y_arr)
    """

    def __init__(
        self,
        net_coords: List[List[float]],
        fps:        float,
        cooldown_frames: int = 30,
    ):
        """
        Args:
            net_coords      : [[left_x, left_y], [right_x, right_y]] 픽셀 좌표
            fps             : 영상 FPS
            cooldown_frames : 동일 이벤트 중복 감지 방지 프레임 수
        """
        if len(net_coords) != 2:
            raise ValueError("net_coords는 [[좌], [우]] 형식의 2점이어야 합니다.")

        self.left:  Tuple[float, float] = (float(net_coords[0][0]), float(net_coords[0][1]))
        self.right: Tuple[float, float] = (float(net_coords[1][0]), float(net_coords[1][1]))
        self.fps             = fps
        self.cooldown_frames = cooldown_frames

        # 네트 상단 직선: y = slope·x + intercept
        dx = self.right[0] - self.left[0]
        dy = self.right[1] - self.left[1]
        if abs(dx) > 1e-6:
            self._slope     = dy / dx
            self._intercept = self.left[1] - self._slope * self.left[0]
        else:
            # 수직 네트(거의 없음)는 평균 y로 수평 처리
            self._slope     = 0.0
            self._intercept = (self.left[1] + self.right[1]) / 2.0

        # 네트가 커버하는 x 범위
        self._net_x_min = min(self.left[0], self.right[0])
        self._net_x_max = max(self.left[0], self.right[0])

        # 네트 중심 x (crossing 판단 기준)
        self._net_cx = (self.left[0] + self.right[0]) / 2.0

    # ── 공개 메서드 ──────────────────────────────────────────

    def net_y_at(self, x: float) -> float:
        """주어진 x에서 네트 상단의 y 좌표를 반환한다."""
        return self._slope * x + self._intercept

    def detect_faults(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> List[NetFaultEvent]:
        """
        셔틀콕 x/y 궤적을 분석해 NetFaultEvent 리스트를 반환한다.

        Args:
            x, y : TrackNet CSV에서 읽은 프레임별 셔틀콕 좌표 배열
                   미감지 프레임은 0 또는 NaN.

        Returns:
            감지된 NET_FAULT 이벤트 리스트 (시간 순 정렬)
        """
        events: List[NetFaultEvent] = []
        n = len(x)

        # 유효 프레임 마스크
        valid = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)

        prev_side:       Optional[str] = None  # "left" | "right"
        prev_valid_idx:  int           = -1
        last_fault_frame: int          = -self.cooldown_frames  # cooldown 초기화

        for i in range(n):
            if not valid[i]:
                continue

            curr_x = float(x[i])
            curr_y = float(y[i])

            # 네트 x 범위 외에 있을 때는 사이드만 추적
            curr_side = "left" if curr_x < self._net_cx else "right"

            if prev_side is not None and curr_side != prev_side:
                # ── 네트 중심선 통과 감지 ─────────────────────────────
                if prev_valid_idx >= 0 and i - last_fault_frame > self.cooldown_frames:
                    fault = self._check_crossing(
                        prev_valid_idx, i,
                        x, y,
                    )
                    if fault is not None:
                        events.append(fault)
                        last_fault_frame = fault.frame

            prev_side      = curr_side
            prev_valid_idx = i

        return events

    # ── 내부 메서드 ──────────────────────────────────────────

    def _check_crossing(
        self,
        i_prev: int,
        i_curr: int,
        x: np.ndarray,
        y: np.ndarray,
    ) -> Optional[NetFaultEvent]:
        """
        두 유효 프레임 사이에서 네트 중심선 통과 시 셔틀콕 높이를 검사한다.

        선형 보간으로 통과 순간의 y를 추정하고 net_y_at(net_cx)와 비교.

        Returns:
            NET_FAULT이면 NetFaultEvent, 아니면 None
        """
        x0, y0 = float(x[i_prev]), float(y[i_prev])
        x1, y1 = float(x[i_curr]), float(y[i_curr])

        dx = x1 - x0
        if abs(dx) < 1e-6:
            return None  # x가 거의 변하지 않음 → 통과 아님

        # 네트 중심 x에서의 보간 t
        t = (self._net_cx - x0) / dx
        if not (0.0 <= t <= 1.0):
            return None  # 보간 범위 외

        shuttle_y_at_crossing = y0 + t * (y1 - y0)
        crossing_frame        = int(round(i_prev + t * (i_curr - i_prev)))
        net_top_y             = self.net_y_at(self._net_cx)

        # clearance: 양수 = 네트 위(통과), 음수 = 네트 아래(걸림)
        # 이미지 좌표에서 y가 작을수록 위이므로 부호 반전
        clearance = net_top_y - shuttle_y_at_crossing  # > 0 이면 셔틀콕이 네트 위

        if clearance >= 0:
            return None  # 정상 통과

        # NET_FAULT
        return NetFaultEvent(
            frame           = crossing_frame,
            time_sec        = crossing_frame / self.fps,
            crossing_x      = self._net_cx,
            shuttle_y       = shuttle_y_at_crossing,
            net_top_y_at_x  = net_top_y,
            clearance_px    = clearance,  # 음수 값 그대로 (얼마나 아래인지)
        )