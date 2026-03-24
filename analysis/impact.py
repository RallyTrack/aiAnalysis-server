"""
RallyTrack - 타점(Impact) 감지 모듈

역할:
  - 셔틀콕 궤적 CSV를 분석하여 실제 타구 시점을 자동 탐지
  - Skip-Vector Check 알고리즘으로 코트 기둥/네트 노이즈 제거
  - 연속된 같은 선수 타점을 그룹핑하여 중복 제거

[원본 대비 주요 수정 사항]
  1. 중첩된 v_in/v_out 계산 인덱스가 잘못되어 있었음
     → k-2/k+2 점프가 실제로는 dt 검증 없이 무조건 적용되었음
     → 명시적으로 dt_in/dt_out 계산 후 조건부 적용으로 수정
  2. skip-vector 검증이 is_smash 플래그와 얽혀 논리가 불명확했음
     → 모든 타점에 동일하게 적용하되 각도 임계값으로만 판단
  3. 미래 비행거리 체크가 frame 5개 고정이었음
     → future_idx 슬라이딩 윈도우로 더 유연하게 처리
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple

from scipy.signal import find_peaks

from .config import IMPACT_CONFIG


# ────────────────────────────────────────────────────────────
# 데이터 클래스
# ────────────────────────────────────────────────────────────

@dataclass
class ImpactEvent:
    """단일 타점 이벤트."""
    frame:   int
    time_sec: float
    score:   float
    owner:   str          # "top" | "bottom"
    player:  str          # "pink_top" | "green_bottom"
    hit_number: int = 0   # 그룹핑 후 순서 부여

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
    Skip-Vector 물리 기반 타점 감지기.

    Usage:
        detector = ImpactDetector(fps=30.0, net_y_ratio=0.46)
        events   = detector.detect(x_arr, y_arr)
    """

    def __init__(
        self,
        fps:          float,
        net_y_ratio:  float = 0.46,   # 영상 높이 기준 네트 Y 위치 비율
        frame_height: int   = 720,
    ):
        self.fps          = fps
        self.net_y        = frame_height * net_y_ratio
        self.cfg          = IMPACT_CONFIG

    # ── 공개 메서드 ──────────────────────────────────────────

    def detect(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> List[ImpactEvent]:
        """
        x, y 궤적 배열을 분석하여 ImpactEvent 목록을 반환합니다.

        Args:
            x: 셔틀콕 X 좌표 배열 (0이면 비가시 프레임)
            y: 셔틀콕 Y 좌표 배열

        Returns:
            그룹핑·순서 번호가 부여된 ImpactEvent 리스트
        """
        scores     = self._compute_impulse_scores(x, y)
        raw_frames = self._find_peaks(scores)
        candidates = self._build_candidates(raw_frames, x, y, scores)
        grouped    = self._group_by_player(candidates)
        return grouped

    # ── 내부 메서드 ─────────────────────────────────────────

    def _compute_impulse_scores(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """
        프레임별 충격량(impulse) 점수를 계산합니다.

        HIt2.ipynb의 detect_ironclad_physics_hits 알고리즘을 기반으로 합니다.
        k-2/k+2 점프 인덱스를 사용해 단일 프레임 노이즈에 둔감한
        안정적인 속도 벡터를 계산합니다.
        """
        scores    = np.zeros(len(x))
        valid_idx = np.where((x > 0) & (y > 0))[0]
        cfg       = self.cfg

        # 바닥(고Y값) 오작동 방지용 임계값: 상위 85% Y값
        valid_y     = y[valid_idx]
        y_floor_lim = np.percentile(valid_y, 85) if len(valid_y) > 0 else 9999

        for k in range(2, len(valid_idx) - 2):
            i_curr = valid_idx[k]
            i_pre  = valid_idx[k - 2]   # 2프레임 전 → 단일 프레임 노이즈에 둔감
            i_post = valid_idx[k + 2]   # 2프레임 후

            dt_in  = i_curr - i_pre
            dt_out = i_post - i_curr

            # 화면 밖 긴 구간(체공) 무시
            if dt_in > cfg["max_frame_gap"] or dt_out > cfg["max_frame_gap"]:
                continue

            v_in  = np.array([x[i_curr] - x[i_pre],  y[i_curr] - y[i_pre]])  / dt_in
            v_out = np.array([x[i_post] - x[i_curr],  y[i_post] - y[i_curr]]) / dt_out

            speed_in  = np.linalg.norm(v_in)
            speed_out = np.linalg.norm(v_out)

            # 미세한 트래킹 떨림 무시
            if speed_in < cfg["min_speed"] and speed_out < cfg["min_speed"]:
                continue

            # ── [방어 1] Skip-Vector Check ──────────────────
            # i_curr 을 건너뛰고 i_pre→i_post 직선을 그렸을 때
            # 원래 오던 방향과 거의 같으면 → 기둥/네트 노이즈
            v_skip    = np.array([x[i_post] - x[i_pre], y[i_post] - y[i_pre]])
            norm_skip = np.linalg.norm(v_skip)
            norm_in   = np.linalg.norm(v_in)

            if norm_skip > 1e-6 and norm_in > 1e-6:
                cos_skip   = np.dot(v_in, v_skip) / (norm_in * norm_skip)
                angle_skip = np.degrees(np.arccos(np.clip(cos_skip, -1.0, 1.0)))
                if angle_skip < cfg["skip_angle_thresh"]:
                    continue

            # ── [방어 2] 타점 후 최소 비행 거리 ────────────
            future_idx = valid_idx[valid_idx > i_curr]
            if len(future_idx) < 3:
                continue
            # 앞으로 최대 10프레임 뒤 위치까지 확인 (고정 [4] 인덱스 문제 해결)
            check_end = min(len(future_idx), 10)
            final_x   = x[future_idx[check_end - 1]]
            final_y   = y[future_idx[check_end - 1]]
            flight_d  = np.hypot(final_x - x[i_curr], final_y - y[i_curr])
            if flight_d < cfg["min_flight_dist"]:
                continue

            # ── [방어 3] 바닥 오작동 방지 ───────────────────
            # 공이 바닥(고Y값) 근처에서 튀어 오르지 않으면 무시
            if y[i_curr] > y_floor_lim:
                near_future = [idx for idx in future_idx if idx <= i_curr + 15]
                if near_future:
                    if min(y[idx] for idx in near_future) > y[i_curr] - 20:
                        continue

            # 충격량: 속도 벡터 변화량
            impulse = np.linalg.norm(v_out - v_in)

            # 아래 방향으로 빠르게 내려오는 스매시 가중치
            if v_out[1] - v_in[1] > 10:
                impulse *= 1.5

            scores[i_curr] = impulse

        return scores

    def _find_peaks(self, scores: np.ndarray) -> np.ndarray:
        """충격량 점수에서 피크(타점 후보)를 추출합니다."""
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
        frames:  np.ndarray,
        x:       np.ndarray,
        y:       np.ndarray,
        scores:  np.ndarray,
    ) -> List[ImpactEvent]:
        """피크 프레임을 ImpactEvent 객체로 변환합니다."""
        events = []
        for f in frames:
            # 타이밍 오프셋(-1): 실제 타격은 속도 변화 직전 프레임
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

    def _group_by_player(
        self,
        candidates: List[ImpactEvent],
    ) -> List[ImpactEvent]:
        """
        연속으로 같은 선수가 치는 후보들 중 점수가 가장 높은 것만 유지합니다.
        (실제 랠리는 두 선수가 번갈아 침)
        """
        if not candidates:
            return []

        grouped: List[ImpactEvent] = []
        group   = [candidates[0]]

        for i in range(1, len(candidates)):
            if candidates[i].owner == group[-1].owner:
                group.append(candidates[i])
            else:
                best = max(group, key=lambda e: e.score)
                grouped.append(best)
                group = [candidates[i]]

        best = max(group, key=lambda e: e.score)
        grouped.append(best)

        # 순서 번호 부여
        for idx, event in enumerate(grouped):
            event.hit_number = idx + 1

        return grouped


# ────────────────────────────────────────────────────────────
# 편의 함수
# ────────────────────────────────────────────────────────────

def build_hit_lookup(events: List[ImpactEvent]) -> dict:
    """
    frame → ImpactEvent 딕셔너리를 반환합니다.
    프레임 단위 빠른 조회에 사용합니다.
    """
    return {e.frame: e for e in events}


def to_api_json(events: List[ImpactEvent], fps: float) -> dict:
    """백엔드 API 전송용 JSON 구조로 변환합니다."""
    return {
        "video_fps":   round(fps, 2),
        "total_hits":  len(events),
        "hits_data":   [e.to_dict() for e in events],
    }
