"""
RallyTrack - 네트 판정 모듈 (NetJudge) v2

[카메라 규칙 — 탑뷰/오버헤드]
  - 선수가 화면 위(Top)·아래(Bottom)에 위치
  - 네트는 Y ≈ frame_height * 0.46 의 수평선
  - 셔틀콕 Y가 net_y_at(x)를 교차할 때 '네트 통과'

[알고리즘]
  1. 네트 상단 직선 y = slope·x + intercept 정의
  2. 연속 유효 프레임 쌍에서 셔틀콕 Y가 net_y를 교차하는 시점을 감지 (Y축 기준)
  3. 선형 보간으로 정확한 교차 프레임·좌표 계산
  4. 교차 방향(top→bottom / bottom→top) 및 네트 걸림 여부 판정
  5. cooldown으로 연속 중복 감지 방지

[주의]
  - NET_FAULT: 탑뷰에서 셔틀콕 Z 높이를 직접 알 수 없으므로
    '교차 시 보간 y ≈ net_y_at(x)인데 실제로 네트 아래로 지나간 경우'를
    궤적 연속성으로 추정. 정밀도는 카메라 앵글에 따라 다름.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# ────────────────────────────────────────────────────────────
# 데이터 클래스
# ────────────────────────────────────────────────────────────

@dataclass
class NetCrossingEvent:
    """
    네트 Y선 교차 이벤트 (정상 통과 + 네트 걸림 모두 포함).

    ImpactDetector의 윈도우 경계로 사용된다.
    """
    frame:         int
    time_sec:      float
    direction:     str    # "top_to_bottom" | "bottom_to_top"
    is_fault:      bool   # True = 네트 걸림
    clearance_px:  float  # + = 네트 위(정상), - = 네트 아래(걸림)

    @property
    def hitter_side(self) -> str:
        """이 교차를 유발한 선수 사이드 ("top" | "bottom")."""
        # top_to_bottom: 위 선수가 쳐서 아래로 넘어감
        return "top" if self.direction == "top_to_bottom" else "bottom"

    @property
    def receiver_side(self) -> str:
        """이 교차 이후 공을 받는 선수 사이드."""
        return "bottom" if self.direction == "top_to_bottom" else "top"


@dataclass
class NetFaultEvent:
    """네트 걸림 이벤트 (하위 호환용)."""
    frame:           int
    time_sec:        float
    crossing_x:      float
    shuttle_y:       float
    net_top_y_at_x:  float
    clearance_px:    float

    def to_timeline_dict(self) -> dict:
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
    네트 판정기 (탑뷰/오버헤드 카메라 기준).

    Usage::

        judge = NetJudge(net_coords=[[300, 330], [980, 330]], fps=30.0)
        crossings = judge.detect_crossings(x_arr, y_arr)   # 윈도우용
        faults    = judge.detect_faults(x_arr, y_arr)      # 네트 걸림 이벤트
    """

    def __init__(
        self,
        net_coords:      List[List[float]],
        fps:             float,
        cooldown_frames: int = 15,
    ):
        """
        Args:
            net_coords      : [[left_x, left_y], [right_x, right_y]] 픽셀 좌표
            fps             : 영상 FPS
            cooldown_frames : 동일 이벤트 중복 감지 방지 프레임 수
                              (아마추어 영상은 15 이하 권장)
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
            self._slope     = 0.0
            self._intercept = (self.left[1] + self.right[1]) / 2.0

        # 네트 중심 좌표 (참조용)
        self._net_cx = (self.left[0] + self.right[0]) / 2.0
        self._net_cy = self.net_y_at(self._net_cx)

    # ── 공개 메서드 ──────────────────────────────────────────

    def net_y_at(self, x: float) -> float:
        """주어진 x에서 네트 상단의 y 좌표를 반환한다."""
        return self._slope * x + self._intercept

    def detect_crossings(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> List[NetCrossingEvent]:
        """
        셔틀콕 궤적에서 네트 Y선 교차 이벤트를 모두 검출한다.
        (정상 통과 + 네트 걸림 포함, ImpactDetector 윈도우 입력용)

        Args:
            x, y : 프레임별 셔틀콕 좌표 배열. 미감지는 0 또는 NaN.

        Returns:
            시간 순 정렬된 NetCrossingEvent 리스트
        """
        crossings: List[NetCrossingEvent] = []
        n = len(x)
        valid = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)

        # "셔틀콕이 네트보다 위에 있는가" 상태 추적
        # 탑뷰에서 y 작을수록 위 → above = shuttle_y < net_y_at(shuttle_x)
        prev_above:     Optional[bool] = None
        prev_valid_idx: int            = -1
        last_cross_frame: int          = -self.cooldown_frames

        for i in range(n):
            if not valid[i]:
                continue

            curr_x     = float(x[i])
            curr_y     = float(y[i])
            curr_above = curr_y < self.net_y_at(curr_x)

            if prev_above is not None and curr_above != prev_above:
                # ── Y축 교차 감지 ─────────────────────────────
                if i - last_cross_frame > self.cooldown_frames:
                    event = self._compute_crossing_event(
                        prev_valid_idx, i,
                        prev_above,
                        x, y,
                    )
                    if event is not None:
                        crossings.append(event)
                        last_cross_frame = event.frame

            prev_above     = curr_above
            prev_valid_idx = i

        return crossings

    def detect_faults(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> List[NetFaultEvent]:
        """
        네트 걸림 이벤트만 반환한다 (하위 호환 메서드).

        내부적으로 detect_crossings()를 호출하고 is_fault=True 항목만 필터링.
        """
        crossings = self.detect_crossings(x, y)
        faults    = []
        for ev in crossings:
            if ev.is_fault:
                cross_x = self._net_cx
                faults.append(NetFaultEvent(
                    frame           = ev.frame,
                    time_sec        = ev.time_sec,
                    crossing_x      = cross_x,
                    shuttle_y       = float(y[min(ev.frame, len(y) - 1)]),
                    net_top_y_at_x  = self.net_y_at(cross_x),
                    clearance_px    = ev.clearance_px,
                ))
        return faults

    # ── 내부 메서드 ──────────────────────────────────────────

    def _compute_crossing_event(
        self,
        i_prev:     int,
        i_curr:     int,
        prev_above: bool,
        x:          np.ndarray,
        y:          np.ndarray,
    ) -> Optional[NetCrossingEvent]:
        """
        두 유효 프레임 사이의 Y축 교차 지점을 선형 보간으로 계산한다.

        셔틀콕 경로: (x0,y0) → (x1,y1)
        네트 직선:   y = slope·X + intercept

        교차 파라미터 t 풀이:
          y0 + t·dy = slope·(x0 + t·dx) + intercept
          t = (net_y_at(x0) - y0) / (dy - slope·dx)
        """
        x0, y0 = float(x[i_prev]), float(y[i_prev])
        x1, y1 = float(x[i_curr]), float(y[i_curr])

        dy          = y1 - y0
        dx          = x1 - x0
        denominator = dy - self._slope * dx

        if abs(denominator) < 1e-6:
            # 궤적이 네트 직선과 거의 평행 → 중간점 사용
            t = 0.5
        else:
            t = (self.net_y_at(x0) - y0) / denominator

        t = float(np.clip(t, 0.0, 1.0))

        crossing_frame = int(round(i_prev + t * (i_curr - i_prev)))
        cross_x        = x0 + t * dx
        cross_y        = y0 + t * dy        # 보간 셔틀콕 y
        net_top_y      = self.net_y_at(cross_x)

        # clearance: 양수 = 셔틀콕이 네트 위(정상), 음수 = 아래(걸림)
        # 이미지 좌표에서 y 작을수록 위 → net_y - shuttle_y > 0 이면 위
        clearance = net_top_y - cross_y

        # 교차 방향
        # prev_above=True  (이전 = 네트 위) → 현재 아래 → 아래로 내려감 → "top_to_bottom"
        # prev_above=False (이전 = 네트 아래) → 현재 위 → 위로 올라감 → "bottom_to_top"
        direction = "top_to_bottom" if prev_above else "bottom_to_top"

        # clearance < -FAULT_TOLERANCE_PX 이면 네트 걸림
        # 0 근처 값은 측정 노이즈로 간주하여 정상 통과로 처리
        FAULT_TOLERANCE_PX = 2.0

        return NetCrossingEvent(
            frame        = crossing_frame,
            time_sec     = crossing_frame / self.fps,
            direction    = direction,
            is_fault     = clearance < -FAULT_TOLERANCE_PX,
            clearance_px = clearance,
        )
