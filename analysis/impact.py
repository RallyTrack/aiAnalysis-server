"""
RallyTrack - 타점(Impact) 감지 모듈 (개선판)

[개선 사항]
  1. 타격 선수 판별 개선
     - 기존: 타격 순간 Y좌표 기반으로만 top/bottom 판별
       → 코트 가장자리에서 오판 발생 (예: top→top→bottom 연속 같은 선수)
     - 개선: "물리적으로 배드민턴은 두 선수가 반드시 교대로 타격"
       원칙을 활용한 교대 강제 + 위치 신뢰도 가중치 결합

     구체적 방법:
       a) 1차: Y좌표 기반 소속 판별 (기존 방식)
       b) 2차: 직전 타격자와 같은 owner면 → "교대 우선" 원칙 적용
          - 현재 y vs net_y 거리를 신뢰도 점수로 사용
          - net_y 중앙에 가까울수록 신뢰도 낮음 (모호 구간)
          - 신뢰도가 낮고(net 근처) 직전 owner와 같으면 교대
       c) 결과: 그룹핑 없이 개별 타점 그대로 유지하면서 owner만 교정

  2. detect_ironclad_physics_hits(Separation.ipynb)와 동일 로직 유지
     - 기존 impulse 계산, skip-vector check, 비행거리 방어 모두 유지
     - 그룹핑(_group_by_player) 제거 → 개별 타점 보존

  3. 모호 구간(net 근처) 임계값: net_y ± 15% 높이
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

from scipy.signal import find_peaks

from .config import IMPACT_CONFIG


# ────────────────────────────────────────────────────────────
# 데이터 클래스
# ────────────────────────────────────────────────────────────

@dataclass
class ImpactEvent:
    """단일 타점 이벤트."""
    frame:      int
    time_sec:   float
    score:      float
    owner:      str   # "top" | "bottom"
    player:     str   # "pink_top" | "green_bottom"
    hit_number: int = 0

    def to_dict(self) -> dict:
        return {
            "hit_number": self.hit_number,
            "frame":      self.frame,
            "time_sec":   round(self.time_sec, 3),
            "player":     self.player,
        }


# ────────────────────────────────────────────────────────────
# 핵심 감지 로직
# ────────────────────────────────────────────────────────────

class ImpactDetector:
    """
    셔틀콕 궤적에서 타점을 감지하는 클래스.

    알고리즘:
      1. impulse 점수 계산 (Separation.ipynb의 detect_ironclad_physics_hits 기반)
      2. 피크 검출
      3. owner 교대 교정 (그룹핑 없이 개별 타점 보존)
    """

    def __init__(
        self,
        fps:          float,
        net_y_ratio:  float = 0.46,
        frame_height: int   = 720,
    ):
        self.fps          = fps
        self.net_y        = frame_height * net_y_ratio
        self.frame_height = frame_height
        self.cfg          = IMPACT_CONFIG

        # 모호 구간: net_y ± ambiguity_ratio * frame_height
        # 이 구간 안에서 직전 owner와 같으면 교대 적용
        self._ambiguity_band = frame_height * 0.12

    # ── 공개 메서드 ──────────────────────────────────────────

    def detect(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> List[ImpactEvent]:
        scores     = self._compute_impulse_scores(x, y)
        raw_frames = self._find_peaks(scores)
        candidates = self._build_candidates(raw_frames, x, y, scores)
        candidates = self._correct_owner_alternation(candidates, y)

        for idx, event in enumerate(candidates):
            event.hit_number = idx + 1

        return candidates

    # ── impulse 점수 계산 ────────────────────────────────────

    def _compute_impulse_scores(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """
        Separation.ipynb의 detect_ironclad_physics_hits 기반.
        k-2/k+2 점프 인덱스로 안정적인 속도 벡터를 계산합니다.
        """
        scores    = np.zeros(len(x))
        valid_idx = np.where((x > 0) & (y > 0))[0]
        cfg       = self.cfg

        valid_y     = y[valid_idx]
        y_floor_lim = np.percentile(valid_y, 85) if len(valid_y) > 0 else 9999

        for k in range(2, len(valid_idx) - 2):
            i_curr = valid_idx[k]
            i_pre  = valid_idx[k - 2]
            i_post = valid_idx[k + 2]

            dt_in  = i_curr - i_pre
            dt_out = i_post - i_curr

            if dt_in > cfg["max_frame_gap"] or dt_out > cfg["max_frame_gap"]:
                continue

            v_in  = np.array([x[i_curr] - x[i_pre],  y[i_curr] - y[i_pre]])  / dt_in
            v_out = np.array([x[i_post] - x[i_curr],  y[i_post] - y[i_curr]]) / dt_out

            speed_in  = np.linalg.norm(v_in)
            speed_out = np.linalg.norm(v_out)

            if speed_in < cfg["min_speed"] and speed_out < cfg["min_speed"]:
                continue

            # Skip-Vector Check: 직진 노이즈 제거
            v_skip    = np.array([x[i_post] - x[i_pre], y[i_post] - y[i_pre]])
            norm_skip = np.linalg.norm(v_skip)
            norm_in   = np.linalg.norm(v_in)

            if norm_skip > 1e-6 and norm_in > 1e-6:
                cos_skip   = np.dot(v_in, v_skip) / (norm_in * norm_skip)
                angle_skip = np.degrees(np.arccos(np.clip(cos_skip, -1.0, 1.0)))
                if angle_skip < cfg["skip_angle_thresh"]:
                    continue

            # 타점 후 최소 비행 거리
            future_idx = valid_idx[valid_idx > i_curr]
            if len(future_idx) < 3:
                continue

            final_x  = x[future_idx[-1]]
            final_y  = y[future_idx[-1]]
            flight_d = np.hypot(final_x - x[i_curr], final_y - y[i_curr])
            if flight_d < cfg["min_flight_dist"]:
                continue

            # 바닥 오작동 방지
            if y[i_curr] > y_floor_lim:
                near_future = [idx for idx in future_idx if idx <= i_curr + 15]
                if near_future:
                    if min(y[idx] for idx in near_future) > y[i_curr] - 20:
                        continue

            impulse = np.linalg.norm(v_out - v_in)

            if v_out[1] - v_in[1] > 10:
                impulse *= 1.5

            scores[i_curr] = impulse

        return scores

    def _find_peaks(self, scores: np.ndarray) -> np.ndarray:
        cfg       = self.cfg
        max_score = np.max(scores)
        if max_score <= 0:
            return np.array([], dtype=int)

        threshold = max_score * cfg["peak_threshold_ratio"]
        peaks, _  = find_peaks(
            scores,
            distance=cfg["peak_distance"],
            prominence=1.0,
            height=threshold,
        )
        return peaks

    def _build_candidates(
        self,
        frames: np.ndarray,
        x:      np.ndarray,
        y:      np.ndarray,
        scores: np.ndarray,
    ) -> List[ImpactEvent]:
        """피크 프레임을 ImpactEvent 객체로 변환합니다."""
        events = []
        for f in frames:
            frame_adj = max(0, int(f) - 1)
            owner     = "top" if y[f] < self.net_y else "bottom"
            player    = "pink_top" if owner == "top" else "green_bottom"
            events.append(ImpactEvent(
                frame    = frame_adj,
                time_sec = frame_adj / self.fps,
                score    = float(scores[f]),
                owner    = owner,
                player   = player,
            ))
        return events

    # ── owner 교대 교정 ──────────────────────────────────────

    def _correct_owner_alternation(
        self,
        candidates: List[ImpactEvent],
        y:          np.ndarray,
    ) -> List[ImpactEvent]:
        """
        배드민턴은 두 선수가 반드시 교대로 타격합니다.
        연속으로 같은 owner가 나오면 아래 기준으로 교정합니다.

        교정 전략:
          1. 타점의 Y좌표와 net_y의 거리를 '위치 신뢰도'로 계산
             - net에서 멀수록(코트 안쪽) 신뢰도 높음
             - net 근처(±ambiguity_band)이면 신뢰도 낮음
          2. 연속 같은 owner 중 신뢰도가 낮은 타점의 owner를 반전
          3. 신뢰도가 비슷하면 두 번째 타점을 반전 (먼저 온 타점 우선)

        그룹핑 없이 개별 타점을 모두 보존합니다.
        """
        if len(candidates) < 2:
            return candidates

        corrected = list(candidates)

        for i in range(1, len(corrected)):
            prev = corrected[i - 1]
            curr = corrected[i]

            if prev.owner != curr.owner:
                # 정상 교대 → 수정 불필요
                continue

            # 연속 같은 owner → 둘 중 하나를 교정
            prev_conf = self._position_confidence(
                y[min(prev.frame, len(y) - 1)]
            )
            curr_conf = self._position_confidence(
                y[min(curr.frame, len(y) - 1)]
            )

            # 신뢰도가 낮은 쪽을 반전 (동점이면 현재 타점 반전)
            if curr_conf <= prev_conf:
                flipped_owner  = "bottom" if curr.owner == "top" else "top"
                flipped_player = "pink_top" if flipped_owner == "top" else "green_bottom"
                corrected[i] = ImpactEvent(
                    frame      = curr.frame,
                    time_sec   = curr.time_sec,
                    score      = curr.score,
                    owner      = flipped_owner,
                    player     = flipped_player,
                    hit_number = curr.hit_number,
                )
            else:
                flipped_owner  = "bottom" if prev.owner == "top" else "top"
                flipped_player = "pink_top" if flipped_owner == "top" else "green_bottom"
                corrected[i - 1] = ImpactEvent(
                    frame      = prev.frame,
                    time_sec   = prev.time_sec,
                    score      = prev.score,
                    owner      = flipped_owner,
                    player     = flipped_player,
                    hit_number = prev.hit_number,
                )

        return corrected

    def _position_confidence(self, ball_y: float) -> float:
        """
        타점 Y좌표의 신뢰도를 반환합니다 (0.0 ~ 1.0).
        net에서 멀수록 1.0, net 바로 위/아래는 0.0에 가깝습니다.

        배드민턴 특성상:
          - 서브/스매시: 코트 안쪽 (신뢰도 높음)
          - 네트 플레이: net 근처 (신뢰도 낮음 → 교대 교정 허용)
        """
        dist = abs(ball_y - self.net_y)
        # ambiguity_band 이내면 선형으로 신뢰도 낮아짐
        if dist >= self._ambiguity_band:
            return 1.0
        return dist / self._ambiguity_band


# ────────────────────────────────────────────────────────────
# 편의 함수
# ────────────────────────────────────────────────────────────

def build_hit_lookup(events: List[ImpactEvent]) -> dict:
    """frame → ImpactEvent 딕셔너리 반환."""
    return {e.frame: e for e in events}


def to_api_json(events: List[ImpactEvent], fps: float) -> dict:
    """백엔드 API 전송용 JSON 구조로 변환합니다."""
    return {
        "video_fps":   round(fps, 2),
        "total_hits":  len(events),
        "hits_data":   [e.to_dict() for e in events],
    }