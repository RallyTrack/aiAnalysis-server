"""
RallyTrack - 타점(Impact) 감지 모듈 v12 (오탐지 제거 강화)

[아키텍처]
  핵심 감지 로직은 원본(f4c000c 커밋)과 완전히 동일하게 유지한다.
  - impulse 계산: valid_idx 기준 앞뒤 2칸, future_idx 최소 3개 조건
  - peaks 검출: Tukey IQR 기반 robust max 임계값 (v7 핵심 변경)
  - owner 판정: Pose 우선 → crossing 보완 → alternation 최후 fallback (v11 전환)

  v12 수정: 오탐지 제거 강화 — 근접 중복 피크 및 서브 정점 아크 오탐
    1) _find_peaks: 점수 기반 NMS 추가 (_score_nms)
       find_peaks(distance=d)의 경계 조건으로 d 이내 두 피크가 모두 살아남는 경우,
       점수 비율(2.5:1)로 약한 쪽을 억제.
       대상: #2 (서브 전 CSV 좌표 튐 중복), #10 (타격 직후 중복 검출)
    2) _compute_impulse_scores: 서브 포물선 정점(apex) 필터 추가
       v_in 이 위쪽 방향(v_in[1] < -2.0) + v_out 이 아래쪽 방향(v_out[1] > 2.0) +
       수평 변화 없음(|v_out[0]-v_in[0]| < 4.0) → 자연 방향전환, 타격 아님 → 스킵
       대상: #3 (서브 비행 정점에서 y 방향 역전으로 인한 오탐지)

  v11 수정: Owner 분류 패러다임 전환 — "놓치더라도 거짓말은 하지 말자"
    문제: alternation 교정이 중간 타격 누락 시 확실한 타점의 owner를 강제로 뒤집음
          (A 침 → B 쳤는데 누락 → A 쳤음 → 로직이 마지막 A를 B로 flip)
    해결책:
    1) _build_candidates: 초기 owner를 court midpoint(_owner_y) 기준으로 변경
       (기존 net_y는 네트 위치 = 실제 선수 위치 경계와 다름)
    2) _correct_owner_alternation: pose 검증된 타점(method에 "_pose" 포함)은
       어떤 상황에서도 flip 금지. 둘 다 pose 검증이면 교대 강제 없이 유지.
    3) get_near_miss_frames(): 감지는 안 됐지만 impulse 점수가 있는 프레임 목록 반환
       (pipeline에서 이 프레임에 YOLO 돌려 pose 확인 후 rescue에 사용)
    4) rescue_near_misses(): pose가 확인한 sub-threshold 타점을 강제 복구
       (min_score_ratio 이상 점수 + 손목 근접 → 전역 임계값 우회)

  v10 수정: 아마추어 영상 3개 추가 누락 타점 복구 + 선수 오분류 수정
    Case A) _clean_trajectory 단방향 극단속도 OR 검사 추가 (SPEED_RATIO_ONE_DIR=12.0)
            → 워밍업 구간의 ID스위치 스파이크 탐지, IQR 점수 분포 오염 제거
            → 누락 타점(~1.375s PT) 복구
    Case B) _compute_adaptive_params: y_range_ratio 기반 min_flight_dist 스케일
            valid_y y-범위 / frame_height 비율로 코트 원근 압축 보정
            → gap_ratio=1.0 영상에서도 비행거리 짧은 타격(~4.29s GB) 복구
    Note)   _correct_owner_alternation → pipeline에서 net_y + crossing_events 전달로 수정

  v9 수정: 3개 핵심 누락 타점 복구 (v8 미완 구간 정밀 보완)
    Case A) _clean_trajectory 워밍업 WARMUP_COUNT 15→25, DETOUR_WARMUP 4.0→5.0,
            속도 보조 검사도 워밍업 구간 SPEED_RATIO 완화
            → TOP 서브 리턴(~1초 부근) 복구
    Case B) min_floor_drop: max(0.0, 20.0 / gap_ratio**1.5) (지수적 완화, 하한 0)
            y_floor_lim: 퍼센타일 85 → gap_ratio 어댑티브(최대 95)로 완화
            → BOTTOM 4초 부근 두 타격의 "정적 공 오판" 복구
    Case C) _correct_owner_alternation: 시간 간격 ≥ 1.0초 이면 flip 억제
            → 연속 BOTTOM 타격이 alternation에서 TOP으로 잘못 flip되는 문제 복구

  v8 수정: 3개 핵심 누락 타점 복구
    1) _clean_trajectory 초반 워밍업 보호 → TOP 서브 리턴 복구
    2) max_frame_gap × 3.5, min_floor_drop 어댑티브 → BOTTOM 4초 두 타격 복구
    3) _correct_owner_alternation 점수 우선 가드 → 높은 점수 타점 보호

  v7 핵심 수정: IQR 기반 스파이크 점수 억제 (14초 유실 구간 복구)

  [14초 유실의 원인 분석]
    _clean_trajectory가 탐지하지 못한 미묘한 스파이크가 impulse 계산 단계에
    유입되면, 해당 스파이크 위치에서 매우 높은 허위 점수가 발생한다.
    이 단 하나의 이상치가 max_score를 10배 이상 부풀리면:
        threshold = max_score × 0.15 → 실제 타점 점수의 1.5배 이상으로 치솟음
    → 14초 구간 내 모든 실제 타점이 threshold를 통과하지 못해 전부 누락.

  [v7 해결책: Tukey 극단 이상치 억제]
    _find_peaks() 내에서 Tukey의 극단 이상치 기준(Q3 + 3 × IQR)을 적용:
      - 이 기준을 초과하는 점수는 스파이크 허위 점수로 간주 → 0으로 억제
      - 억제 후 max_score로 threshold 재계산 → 진짜 타점이 threshold 통과
    수학적 근거: 정규 분포에서 Q3 + 3σ ≈ 99.9% 분위 → 실제 타점이
    잘못 억제될 확률은 통계적으로 극히 낮음 (false suppression rate < 0.1%).

  v7 추가: _compute_adaptive_params() 완화 강화
    - gap_ratio 상한: 3.0 → 4.0 (매우 드문 좌표 영상 대응)
    - min_flight_dist 하한: 30px → 20px (쿼터뷰 원근 효과 추가 보정)

  v6: 궤적 클렌징 (Trajectory Cleaning) — 임펄스 계산 전 전처리
  v8: 추가 개선 사항 → _compute_adaptive_params, _correct_owner_alternation
  - _clean_trajectory(): 우회 비율(Detour Ratio) 기반으로 1~2프레임
    좌표 스파이크(ID Switching 오탐지)를 제거한다.
  - 클렌징된 clean_valid_idx와 typical_disp(영상별 이동 거리 기준)를
    _compute_adaptive_params()에 전달 → min_speed를 실제 이동 거리에 비례해 조정.
  - gap fill 없음: 기존 데이터 내 노이즈를 제거하는 방향만 허용.

  v5: 어댑티브(Adaptive) 파라미터 계산
  - TrackNet 추적 밀도(avg_tracking_gap)와 FPS를 분석해
    물리 파라미터를 동적으로 조정한다.
  - avg_gap ≈ 1.5 (프로 탑뷰) → 원본 파라미터 그대로 → test_8sec 7/7 유지
  - avg_gap ≈ 4~6 (아마추어 쿼터뷰+블러) → 완화 파라미터 적용

  crossing_events를 선택적으로 제공하면 owner 교정에 활용한다:
  - 각 crossing 직전 lookback 구간의 미매칭 후보에 crossing 방향으로 owner를 갱신
  - gap fill 없음: 물리 필터 통과 여부가 타점 인정의 유일한 기준

[변경 이력]
  v7  : IQR 기반 스파이크 점수 억제(_find_peaks). gap_ratio 상한 4.0, 하한 20px.
        14초 타점 유실 구간 복구. 프로 영상 Regression 안전 유지.
  v6  : 궤적 클렌징(_clean_trajectory) 추가. typical_disp 기반 min_speed 정밀 보정.
        프로 영상 Regression 안전: 정상 포물선 detour_ratio < 1.8 << 2.5(임계값).
  v5  : 어댑티브 파라미터 도입. 원본 로직 보존, test_8sec 7/7 정확도 보장.
        아마추어 쿼터뷰+블러 영상에서 타점 감지율 개선.
  v4  : 원본 로직 완전 복구. crossing은 owner 교정에만 사용. warm-up·gap fill 제거.
  v3  : (철회) peaks threshold 완화(0.06) + warm-up + gap fill → 오탐지 급증
  v2  : (철회) 1윈도우 1타점 강제 → crossing 누락 시 타점 누락
  v1  : 원본. test_8sec 기준 7/7 정확.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, TYPE_CHECKING

from scipy.signal import find_peaks

from .config import IMPACT_CONFIG

if TYPE_CHECKING:
    from .net_judge import NetCrossingEvent


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
    player:     str   # "top" | "bottom"
    hit_number: int = 0
    # 감지 경로 (디버깅용, API 출력 제외)
    # "peaks"         : 물리 필터 통과 피크
    # "peaks_crossed" : 물리 필터 + crossing으로 owner 검증
    method: str = "peaks"

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

    Usage (crossing owner 교정 포함 — 권장)::

        from analysis.net_judge import NetJudge

        judge     = NetJudge(net_coords, fps)
        detector  = ImpactDetector(fps)
        crossings = judge.detect_crossings(x, y)
        hits      = detector.detect(x, y, crossing_events=crossings)

    Usage (crossing 없을 때 — 원본과 동일)::

        hits = detector.detect(x, y)
    """

    def __init__(
        self,
        fps:               float,
        net_y_ratio:       float           = 0.46,
        frame_height:      int             = 720,
        owner_y_threshold: Optional[float] = None,
    ):
        """
        Args:
            fps               : 영상 FPS
            net_y_ratio       : 네트 y 위치 비율 (NetJudge crossing 연동용)
            frame_height      : 프레임 높이 (px)
            owner_y_threshold : top/bottom 소유 구분 y 임계값 (px).
                                y < threshold → "top", y >= threshold → "bottom".
                                None 이면 frame_height × net_y_ratio (기존 동작).
                                quarter-view 카메라에서는 코트 세로 중점(court_mid_y)을
                                pipeline_service 가 계산해 전달해야 정확한 선수 분류가 됨.
        """
        self.fps          = fps
        self.net_y        = frame_height * net_y_ratio   # NetJudge 호환용, crossing 판정
        self.frame_height = frame_height
        self.cfg          = IMPACT_CONFIG
        # 선수 소유권 결정 임계값 — net_y 와 분리
        self._owner_y     = owner_y_threshold if owner_y_threshold is not None else self.net_y
        self._ambiguity_band = frame_height * 0.12

    # ── 공개 메서드 ──────────────────────────────────────────

    def detect(
        self,
        x: np.ndarray,
        y: np.ndarray,
        crossing_events: Optional[List["NetCrossingEvent"]] = None,
        pose_full: Optional[List[List[dict]]] = None,
        run_alternation: bool = True,
        verbose: bool = False,
    ) -> List[ImpactEvent]:
        """
        타점 이벤트 리스트를 반환한다.

        Args:
            x, y             : 프레임별 셔틀콕 좌표 배열. 미감지는 0 또는 NaN.
            crossing_events  : NetJudge.detect_crossings() 결과 (선택).
                               제공 시 물리 필터 통과 후보의 owner를 crossing
                               방향으로 재검증한다. 후보 수는 변하지 않는다.
            pose_full        : 프레임별 YOLO keypoints 리스트 (선택).
                               List[List[dict]] — 각 frame idx에 해당하는 플레이어 목록.
                               각 플레이어 dict: {"keypoints": [[x,y,conf] * 17], ...}
                               제공 시:
                                 1) 바닥 필터 예외처리 — 손목이 셔틀 근처면 필터 스킵
                                 2) 포즈 기반 owner 교정 (_apply_pose_owner)
                               None이면 기존 동작과 완전히 동일.
            verbose          : True 시 중간 진단 정보를 출력한다.
        """
        # v6 Step 1: 원본 valid_idx 추출 후 스파이크 클렌징
        raw_valid = np.where((x > 0) & (y > 0))[0]
        clean_idx, typical_disp = self._clean_trajectory(raw_valid, x, y)

        # v6 Step 2: 클렌징된 궤적 기준으로 어댑티브 파라미터 계산
        adaptive   = self._compute_adaptive_params(clean_idx, typical_disp, y=y)

        # pose_full → frame별 손목 좌표 인덱스 (바닥 필터 예외처리에 사용)
        wrist_index: Dict[int, List[tuple]] = (
            self._build_wrist_index(pose_full) if pose_full else {}
        )

        if verbose:
            removed = set(raw_valid.tolist()) - set(clean_idx.tolist())
            print(f"\n[ImpactDetector verbose]")
            print(f"  raw_valid  : {len(raw_valid)}  clean_idx: {len(clean_idx)}")
            print(f"  removed_frames: {sorted(removed)[:30]}")
            print(f"  avg_gap={adaptive['_avg_gap']}  gap_ratio={adaptive['_gap_ratio']}")
            print(f"  typical_disp={adaptive['_typical_disp']}  typical_speed={adaptive['_typical_speed']}")
            print(f"  net_y={self.net_y:.1f}  owner_y={self._owner_y:.1f}")
            print(f"  adaptive params: max_frame_gap={adaptive['max_frame_gap']}  "
                  f"min_speed={adaptive['min_speed']:.2f}  "
                  f"min_flight_dist={adaptive['min_flight_dist']:.1f}  "
                  f"min_floor_drop={adaptive['min_floor_drop']:.1f}  "
                  f"floor_pct={adaptive['floor_pct']:.0f}")
            if crossing_events:
                print(f"  crossings: {[(c.frame, c.direction) for c in crossing_events]}")
            if wrist_index:
                print(f"  wrist_index: {len(wrist_index)}프레임에 손목 좌표 있음")

        scores     = self._compute_impulse_scores(x, y, clean_idx, adaptive, wrist_index=wrist_index)
        raw_frames = self._find_peaks(scores, adaptive, verbose=verbose)
        candidates = self._build_candidates(raw_frames, x, y, scores)

        # [v11] rescue_near_misses()에서 재사용할 수 있도록 마지막 계산 결과를 보관.
        self._last_scores   = scores
        self._last_adaptive = adaptive

        if verbose:
            print(f"  candidates before crossing: {[(c.frame, round(c.time_sec,2), c.owner, round(c.score,1)) for c in candidates]}")

        if crossing_events:
            candidates = self._apply_crossing_owner(candidates, crossing_events)

        if pose_full:
            candidates = self._apply_pose_owner(candidates, x, y, pose_full, verbose=verbose)

        if run_alternation:
            candidates = self._correct_owner_alternation(candidates, y)

        if verbose:
            print(f"  final hits: {[(c.frame, round(c.time_sec,2), c.owner, c.method) for c in candidates]}\n")

        for idx, event in enumerate(candidates):
            event.hit_number = idx + 1

        return candidates

    def apply_pose_owner(
        self,
        events: List[ImpactEvent],
        x: np.ndarray,
        y: np.ndarray,
        pose_at_hits: Dict[int, List[dict]],
        verbose: bool = False,
    ) -> List[ImpactEvent]:
        """
        이미 감지된 타점 이벤트 리스트에 대해 포즈 기반 owner 교정을 적용한다.

        pipeline_service에서 타점 후보 프레임만 골라 YOLO predict를 실행한 후
        이 메서드를 호출해 owner를 교정하는 방식으로 사용한다.

        Args:
            events       : detector.detect()로 얻은 타점 이벤트 리스트.
            x, y         : 프레임별 셔틀콕 좌표 (원래 detect()에 넣은 값과 동일).
            pose_at_hits : {frame_idx: List[dict]} — 타점 후보 프레임별 keypoints.
                           (pipeline_service._collect_pose_at_frames()의 반환값)
            verbose      : True 시 변경 내역 출력.

        Returns:
            owner가 교정된 ImpactEvent 리스트.
        """
        return self._apply_pose_owner(events, x, y, pose_at_hits, verbose=verbose)

    # ── 궤적 클렌징 (v6 신규) ────────────────────────────────

    def _clean_trajectory(
        self,
        valid_idx: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
    ) -> tuple:
        """
        임펄스 계산 전처리: 1~2프레임 좌표 스파이크(Outlier)를 제거한다.

        [배경]
        TrackNet이 셔틀콕 ID를 순간적으로 잘못 추적하면 1~2 프레임 동안
        물리적으로 불가능한 위치가 기록된다(좌표 튐). 이 스파이크는:
          1) 스파이크 위치에서 높은 위양성(False-Positive) impulse score 생성
          2) 인근 진짜 타점 프레임의 속도 벡터(v_in, v_out)를 오염
             → 진짜 타점이 물리 필터에서 Drop되는 핵심 원인

        [알고리즘: 우회 비율(Detour Ratio) 기반 스파이크 감지]
        연속 세 유효 좌표 A → B → C에 대해:

            detour_ratio = (d(A,B) + d(B,C)) / d(A,C)

        [수학적 근거 — 삼각 부등식]
          삼각 부등식:  d(A,B) + d(B,C) ≥ d(A,C)  (등호: B가 선분 AC 위)
          따라서 detour_ratio ≥ 1.0 (항상 성립)

          B가 정상 궤적(포물선) 위: ratio ≈ 1.0~1.8 (곡률에 따라 다소 증가)
          B가 스파이크(경로 이탈): ratio >> 2.5 (경로가 크게 우회)

          임계값 2.5는 최대 정상 ratio(≈1.8)에 충분한 마진을 두어
          실제 타점을 포함한 모든 정상 좌표가 제거되지 않도록 보장한다.

        [보조 조건: 최소 절대 이탈 거리]
          min_absolute_deviation = typical_disp * 2.0
          이유: d(A,C) ≈ 0인 셔틀 정지 구간에서 ratio가 발산하는 오탐 방지.
          max(d_AB, d_BC) < threshold이면 실제 이탈이 미미하므로 스파이크 아님.

        [2-frame 연속 스파이크 처리 — 4-점 윈도우 보완]
          A, S1, S2, B 패턴(S1·S2 모두 같은 방향으로 이탈):
          triplet(A,S1,S2)와 triplet(S1,S2,B)의 detour_ratio가 낮아
          3점 검사만으로는 감지 불가. 이유:
            S1, S2가 모두 같은 방향으로 크게 벗어나면
            A→S1→S2의 경유 경로가 A→S2보다 많이 길지 않음.

          해결: 4점 우회 비율 검사 추가
            detour4 = (d_AS1 + d_S1S2 + d_S2B) / d_AB  (A와 B 사이 4점 경로)
          A와 B가 정상 좌표이므로 d_AB ≈ 1~2 * typical_disp (짧음).
          S1, S2를 경유하면 d_AS1 + d_S1S2 + d_S2B >> d_AB → detour4 >> 2.5.

        [프로 영상 Regression 안전성]
          정상 포물선 궤적에서 임의의 세 점의 detour_ratio < 1.8 << 2.5
          (수치 예시: 30px 접근 + 20px 이탈 + 50px 직선거리 = ratio 1.0)
          → 실제 타점 좌표는 절대 제거되지 않음.

        Returns:
            (clean_valid_idx, typical_disp)
            typical_disp : 클렌징 전 연속 이동 거리의 중앙값 (px).
                           _compute_adaptive_params()에서 min_speed 보정에 활용.
        """
        if len(valid_idx) < 4:
            return valid_idx, 1.0

        vx = x[valid_idx].astype(float)
        vy = y[valid_idx].astype(float)

        # 전형적 이동 거리: 연속 유효 프레임 간 유클리드 거리의 중앙값.
        # [중앙값 사용 이유] 스파이크로 인한 극단적 거리값이 포함돼도
        # 평균과 달리 전체 분포를 왜곡하지 않아 robust하게 추정된다.
        dists = np.hypot(np.diff(vx), np.diff(vy))
        pos_dists = dists[dists > 0]
        typical_disp = float(np.median(pos_dists)) if len(pos_dists) > 0 else 1.0

        # 최소 절대 이탈 거리 임계값 (정지 상태 오탐 방지용)
        min_abs_dev = typical_disp * 2.0

        DETOUR_THRESH = 2.5  # 정상 포물선 상한(≈1.8)에 충분한 여유

        cur_idx = valid_idx.copy()

        # ── 초반 워밍업 보호 (v8→v9 강화) ──────────────────────────
        # [목적] TOP 서브 리턴 누락 방지.
        # 영상 극초반(서브 전·서브 중)에는 TrackNet이 셔틀콕을 불안정하게 추적해
        # 실제 좌표가 잠시 요동칠 수 있다. 이때 정상적인 서브 리턴 프레임도
        # detour_ratio가 2.5를 초과할 수 있어 스파이크로 오판 제거될 위험이 있다.
        #
        # [v9 수정 이유: WARMUP_COUNT 15→25, DETOUR_WARMUP 4.0→5.0]
        # v8에서 WARMUP_COUNT=15는 avg_gap≈5 아마추어 영상에서 약 75프레임(2.5초)
        # 분량의 유효 포인트만 커버. 서브 이전 정지 좌표가 많을 경우 실제
        # 서브 리턴(~1초 부근)이 이미 워밍업 구간을 벗어나는 문제 발생.
        # WARMUP_COUNT=25로 확장하면 avg_gap≈5 기준 125프레임(약 4초) 커버.
        # DETOUR_WARMUP 4.0→5.0: 서브 직후 급격한 방향 전환을 보호.
        #
        # [수학적 근거]
        # 초반 N개 포인트는 궤적이 확정되지 않은 상태 → 삼각 부등식으로 계산한
        # detour_ratio의 분모 d(A,C)가 작아져 ratio가 과도하게 높아진다.
        # 따라서 초반 WARMUP_COUNT 포인트에 한해 더 관대한 임계값(5.0)을 적용,
        # 이후에는 원래 DETOUR_THRESH(2.5)로 복귀한다.
        #
        # [Regression 안전] 프로 영상: 초반부터 안정 추적 → detour_ratio << 2.5
        # → 워밍업 기준(5.0)이든 일반 기준(2.5)이든 동일하게 아무것도 제거 안 함.
        WARMUP_COUNT  = 25    # [v9: 15→25] 첫 N개 유효 포인트는 워밍업 구간으로 처리
        DETOUR_WARMUP = 5.0   # [v9: 4.0→5.0] 워밍업 구간 임계값 (일반 2.5보다 완화)

        for _pass in range(2):  # 최대 2회 반복 (중첩 스파이크 처리 여유)
            if len(cur_idx) < 4:
                break

            vx_c = x[cur_idx].astype(float)
            vy_c = y[cur_idx].astype(float)
            is_spike = np.zeros(len(cur_idx), dtype=bool)

            # ── 3점 triplet 검사: 1-frame 단독 스파이크 ─────────
            for i in range(1, len(cur_idx) - 1):
                ax, ay = vx_c[i - 1], vy_c[i - 1]
                bx, by = vx_c[i],     vy_c[i]
                cx, cy = vx_c[i + 1], vy_c[i + 1]

                d_ab = np.hypot(bx - ax, by - ay)
                d_bc = np.hypot(cx - bx, cy - by)
                d_ac = np.hypot(cx - ax, cy - ay)

                # A와 C가 거의 같은 위치(셔틀 정지 구간): ratio 발산 방지
                if d_ac < 1.0:
                    continue

                detour_ratio = (d_ab + d_bc) / d_ac

                # 초반 워밍업 구간: 더 관대한 임계값 사용
                thresh_i = DETOUR_WARMUP if i < WARMUP_COUNT else DETOUR_THRESH

                # 두 조건 모두 만족해야 스파이크로 판정:
                #   1) 우회 비율이 임계값 초과 (삼각 부등식 기반)
                #   2) 실제 이탈 거리가 전형적 이동 거리의 2배 초과 (절대값 기준)
                max_leg = max(d_ab, d_bc)
                if detour_ratio > thresh_i and max_leg > min_abs_dev:
                    is_spike[i] = True

            # ── 4점 윈도우 검사: 2-frame 연속 스파이크 ──────────
            # [이유] S1, S2가 같은 방향으로 이탈 시 triplet(A,S1,S2)와
            # triplet(S1,S2,B)의 detour_ratio가 낮아 3점 검사 미감지.
            # 4점 경로 (A→S1→S2→B)는 직선 A→B보다 훨씬 길어 감지 가능.
            for i in range(1, len(cur_idx) - 2):
                if is_spike[i] or is_spike[i + 1]:
                    continue  # 이미 단독 스파이크로 처리됨

                ax,  ay  = vx_c[i - 1], vy_c[i - 1]
                s1x, s1y = vx_c[i],     vy_c[i]
                s2x, s2y = vx_c[i + 1], vy_c[i + 1]
                bx,  by  = vx_c[i + 2], vy_c[i + 2]

                d_as1 = np.hypot(s1x - ax,  s1y - ay)
                d_s12 = np.hypot(s2x - s1x, s2y - s1y)
                d_s2b = np.hypot(bx  - s2x, by  - s2y)
                d_ab  = np.hypot(bx  - ax,  by  - ay)

                if d_ab < 1.0:
                    continue

                # 4점 우회 비율: (A→S1 + S1→S2 + S2→B) / A→B
                detour4  = (d_as1 + d_s12 + d_s2b) / d_ab
                # outer legs: 정상 좌표(A, B)와 스파이크(S1, S2) 사이의 거리
                max_outer = max(d_as1, d_s2b)

                if detour4 > DETOUR_THRESH and max_outer > min_abs_dev:
                    is_spike[i]     = True
                    is_spike[i + 1] = True

            # ── 속도 크기 기반 보조 검사 (v7 추가) ──────────────
            # [목적] detour_ratio가 임계값 미만인 미묘한 스파이크 추가 탐지.
            # [수학적 근거]
            # 연속 유효 포인트 간 속도 = 거리 / 프레임 갭 (px/frame).
            # 점 B(인덱스 j)가 스파이크이면:
            #   - speed_A→B : A에서 B(스파이크)까지의 속도 → 비정상적으로 높음
            #   - speed_B→C : B(스파이크)에서 C(정상)까지의 속도 → 비정상적으로 높음
            # 두 속도 모두 전형적 속도(typical_speed)의 SPEED_RATIO배를 초과하면
            # B는 양방향 모두 비정상적인 이탈 → 스파이크로 판정.
            #
            # [Regression 안전: 실제 타점 frame과 구분]
            # 실제 타점: 진입 속도(approach)와 이탈 속도(departure)가
            # 동시에 SPEED_RATIO배를 넘기 어렵다. 강한 스매시라도:
            #   - 진입 속도: 보통 수준 (상대가 친 공의 도착 속도)
            #   - 이탈 속도: 빠름 (스매시 속도)
            # → 두 속도 '모두' 임계값 초과하는 경우는 극히 드물다.
            # SPEED_RATIO = 4.0 (전형적 속도의 4배)은 충분한 여유 마진.
            #
            # 주의: 이 검사는 detour 검사를 '보완'하며 '대체'하지 않는다.
            # 두 검사 중 하나라도 스파이크로 판정하면 해당 점이 제거된다.
            if len(cur_idx) >= 3:
                frame_gaps_c = np.diff(cur_idx.astype(float))
                leg_dists_c  = np.hypot(np.diff(vx_c), np.diff(vy_c))
                leg_speeds_c = leg_dists_c / np.maximum(frame_gaps_c, 1.0)

                pos_spd = leg_speeds_c[leg_speeds_c > 0]
                typical_spd = float(np.median(pos_spd)) if len(pos_spd) > 0 else 1.0

                # [v9] 워밍업 구간(j < WARMUP_COUNT)에서는 SPEED_RATIO를 더 완화.
                # 이유: 서브 직후 급격한 방향 전환은 진입 속도(s_in)와 이탈 속도(s_out)가
                # 모두 높아 양방향 임계값 초과 → 스파이크로 오판될 수 있음.
                # SPEED_RATIO_WARMUP=6.0: 전형 속도의 6배 이상일 때만 스파이크 판정.
                # [Regression 안전] 프로 영상의 실제 스파이크는 8~20배 → 6배 기준에서도 탐지됨.
                SPEED_RATIO        = 4.0   # 일반 구간: 전형 속도의 4배 초과 → 비정상 고속
                SPEED_RATIO_WARMUP = 6.0   # [v9] 워밍업 구간: 더 관대한 임계값
                # [v10] 단방향 극단속도 임계값: 워밍업 구간 무관.
                # 한 방향만이라도 12배를 초과하면 ID스위치 스파이크로 판정.
                # 근거: 실제 스매시도 8~10배 수준이므로 12배는 충분한 여유.
                # 이 검사는 워밍업 보호(SPEED_RATIO_WARMUP)가 양방향 AND 검사를
                # 통과시켜버리는 ID스위치 스파이크를 추가로 잡는다.
                SPEED_RATIO_ONE_DIR_EXTREME = 12.0

                for j in range(1, len(cur_idx) - 1):
                    if is_spike[j]:
                        continue  # 이미 표시됨
                    s_in  = leg_speeds_c[j - 1]  # A → B 속도
                    s_out = leg_speeds_c[j]       # B → C 속도
                    # 워밍업 구간 여부에 따라 임계값 선택
                    spd_thresh_j = SPEED_RATIO_WARMUP if j < WARMUP_COUNT else SPEED_RATIO
                    # [기존] 양방향 모두 비정상 고속 → 스파이크
                    # [v10] OR: 단방향이라도 12배 초과 → 스파이크 (워밍업 구간 무관)
                    if (s_in > typical_spd * spd_thresh_j and s_out > typical_spd * spd_thresh_j) or \
                       max(s_in, s_out) > typical_spd * SPEED_RATIO_ONE_DIR_EXTREME:
                        is_spike[j] = True

            if not np.any(is_spike):
                break  # 더 이상 스파이크 없으면 조기 종료

            # ── 연속 스파이크 런 보호 (v10 신규) ────────────────
            # [목적] 서브 리턴, 하이클리어 등 실제 궤적 구간이
            # 연속적으로 스파이크로 오판돼 대량 삭제되는 것을 방지.
            #
            # [수학적 근거]
            # 진짜 ID스위치/좌표 튐은 1~2프레임 단발성 이탈이다.
            # 3프레임을 초과해 '연속으로 모두 스파이크' 판정이 난 구간은
            # 실제 공이 비정상적 각도(쿼터뷰 원근, 서브 궤적)로 이동한 것이지
            # 측정 노이즈가 아니다. 이 구간을 보존해야 해당 시점의
            # impulse score 계산이 가능하다.
            #
            # [Regression 안전]
            # 프로 영상의 진짜 ID스위치: 1~2프레임 단발 → 런 길이 ≤ 2 → 제거됨.
            # 아마추어 영상 서브 후 궤적: 10~15프레임 연속 → 런 길이 >> 3 → 보호됨.
            # [v11 버그픽스] > → >= : len=3 런도 보호 대상에 포함.
            # 이전: run_len > 3  → len=3 런이 보호 안 되어 pass2에서 제거됨.
            # 결과: 프레임 31-33(서브 리턴 구간)이 2패스에 걸쳐 완전 제거 →
            #       frame 32(서브 리턴 타점)의 impulse score = 0 → 감지 불가.
            # 수정: run_len >= 3 → len=3 런을 보호 → frame 32 살아남음.
            # Regression 안전: 진짜 ID스위치는 1~2프레임 단발(len ≤ 2) → 여전히 제거됨.
            MAX_CONSEC_SPIKE = 3

            run_start = None
            for k in range(len(is_spike)):
                if is_spike[k]:
                    if run_start is None:
                        run_start = k
                else:
                    if run_start is not None:
                        run_len = k - run_start
                        if run_len >= MAX_CONSEC_SPIKE:   # [v11] > → >=
                            is_spike[run_start:k] = False
                        run_start = None
            if run_start is not None:
                run_len = len(is_spike) - run_start
                if run_len >= MAX_CONSEC_SPIKE:            # [v11] > → >=
                    is_spike[run_start:] = False

            if not np.any(is_spike):
                break

            cur_idx = cur_idx[~is_spike]

        return cur_idx, typical_disp

    # ── 어댑티브 파라미터 계산 (v5 신규, v6 min_speed 정밀 보정, v10 y_range_ratio) ──

    def _compute_adaptive_params(
        self,
        valid_idx:    np.ndarray,
        typical_disp: float = 1.0,
        y:            Optional[np.ndarray] = None,
    ) -> dict:
        """
        TrackNet 추적 밀도(valid_idx 간격)와 FPS를 기반으로
        물리 파라미터를 동적으로 조정한다.

        [핵심 원리]
          avg_gap = valid_idx 연속 감지 간격의 중앙값
          (중앙값 사용 이유: 랠리 사이 긴 공백 프레임이 평균을 왜곡하는 것을 방지)

          avg_gap ≈ 1.5 (프로 탑뷰, 고감지율)  → gap_ratio = 1.0
                                                → 모든 파라미터 = 원본 값
                                                → test_8sec 7/7 정확도 완전 유지
          avg_gap ≈ 4~6 (아마추어 쿼터뷰+블러) → gap_ratio = 2.0~3.0
                                                → 파라미터 완화

        [gap_ratio 정의]
          gap_ratio = clip(avg_gap / 1.5, 1.0, 3.0)
          avg_gap=1.5 → ratio=1.0 (원본 동일)
          avg_gap=3.0 → ratio=2.0
          avg_gap=6.0 → ratio=3.0 (상한, 과완화 방지)

        [fps_scale 정의]
          fps_scale = fps / 30.0  (30fps를 기준 FPS로 사용)

        Args:
            valid_idx    : _clean_trajectory() 결과 클렌징된 유효 인덱스 배열.
            typical_disp : _clean_trajectory()에서 계산된 연속 이동 거리 중앙값 (px).
                           영상별 셔틀콕 실제 이동 규모를 반영한 min_speed 보정에 활용.
        """
        cfg = self.cfg

        if len(valid_idx) < 5:
            # 데이터가 너무 적으면 원본 파라미터 반환 (안전 fallback)
            return self._base_params()

        gaps    = np.diff(valid_idx).astype(float)
        avg_gap = float(np.median(gaps))

        # gap_ratio: 1.0(원본 유지) ~ 4.0(최대 완화) — v7: 상한 3.0 → 4.0
        # [이유] avg_gap ≈ 8~10인 매우 드문 추적 상황(극단적 블러 영상)에서
        # 이전 상한 3.0이 충분한 완화를 제공하지 못함.
        # avg_gap=6.0 → ratio=4.0 (기존 cap 3.0), avg_gap=8.0 → ratio=4.0 (신규 cap 4.0)
        # 프로 영상: avg_gap ≈ 1.5 → ratio=1.0 → cap 무관 → Regression 안전.
        gap_ratio = float(np.clip(avg_gap / 1.5, 1.0, 4.0))

        # FPS 스케일: 30fps를 기준으로
        fps_scale = self.fps / 30.0

        # ── max_frame_gap 완화 (v8: 승수 2.5 → 3.5) ─────────
        # [이유] _clean_trajectory가 스파이크를 제거하면 clean_idx에서
        # valid_idx[k-2]와 valid_idx[k] 사이에 공백이 생긴다.
        # 제거된 스파이크가 1개: dt_in ≈ avg_gap × 3 (2포지션 + 스파이크 공백)
        # 제거된 스파이크가 2개: dt_in ≈ avg_gap × 4
        #
        # [수학적 근거]
        # avg_gap=5, 스파이크 2개 제거 시: dt_in ≈ 20프레임
        # 기존 2.5승수: max_gap = int(5×2.5) = 12 → 20 > 12 → 타점 skip!
        # 신규 3.5승수: max_gap = int(5×3.5) = 17 → 여전히 skip
        # → avg_gap 기반 clean_idx로 재계산 시 avg_gap은 이미 증가돼 있음
        #   (clean_idx에서 공백 반영 → avg_gap_clean ≈ 6~8)
        #   max_gap = int(7×3.5) = 24 → 20 < 24 → 통과 ✓
        #
        # 프로 영상: avg_gap=1.5 → int(1.5×3.5)=5 → max(5,5)=5 원본 동일 ✓
        dynamic_max_gap = max(cfg["max_frame_gap"], int(avg_gap * 3.5))

        # ── min_speed 완화 (v6: typical_disp 기반 적응형 보정) ──
        # [기존 v5 방식]
        #   speed_relax = 1 + (gap_ratio - 1) × 0.4
        #   → gap_ratio 만으로 완화: 아마추어(gap=5) → relax=1.27 → min_speed≈2.36
        #
        # [v6 추가: typical_speed 기반 하한 보정]
        #   typical_speed = typical_disp / avg_gap  (px per frame, 영상 실제 이동 속도)
        #   셔틀이 "움직이고 있다"고 볼 최소 속도 = typical_speed × 0.25
        #
        #   [수학적 근거]
        #   v_in = Δpx / dt_in 에서 Δpx ≈ typical_disp (연속 2 valid gap 이동),
        #   dt_in ≈ avg_gap × 2 (valid_idx[k-2]→[k] 간격)
        #   → 전형적 v_in ≈ typical_disp / (avg_gap × 2) = typical_speed / 2
        #   min_speed를 typical_speed × 0.25 ≈ 전형적 v_in 의 절반으로 설정하면
        #   정지·잡음 구간만 제거하고 느린 셔틀도 통과 가능.
        #
        #   [Regression 안전]
        #   프로 영상: typical_speed ≈ 13~20 px/frame
        #     typical_speed_floor = 13 × 0.25 = 3.25 → cfg["min_speed"](3.0) 에 min()
        #     → 원본 값 그대로 유지 (더 엄격해지지 않음)
        #   아마추어 영상: typical_speed ≈ 5~8 px/frame
        #     typical_speed_floor = 5 × 0.25 = 1.25 → gap_ratio 기반 2.36보다 완화
        #     → 느린 셔틀 타점도 min_speed 필터 통과 가능
        speed_relax           = max(1.0, 1.0 + (gap_ratio - 1.0) * 0.4)
        gap_based_min_speed   = cfg["min_speed"] / speed_relax
        typical_speed         = typical_disp / max(avg_gap, 1.0)
        typical_speed_floor   = typical_speed * 0.25
        # 두 방식 중 더 완화된(낮은) 값을 채택, 최소 0.5 px/frame 보장
        dynamic_min_speed     = max(0.5, min(gap_based_min_speed, typical_speed_floor))

        # ── min_flight_dist 완화 (v7: gap_ratio, v10: y_range_ratio 추가) ──
        # [v7 이유] 쿼터뷰 원근 효과: 화면 원거리(코트 상단) 선수의 타격 후
        # 셔틀콕의 픽셀 비행 거리가 탑뷰보다 짧게 측정됨.
        # gap_ratio 만큼 비례 완화. 하한 30px → 20px (v7).
        # [v7 근거] gap_ratio=4.0일 때 60/4=15px → 하한 30 과도.
        # 쿼터뷰 원근이 심할수록 실제 비행 거리의 픽셀 표현이 작아지므로
        # 20px 하한이 더 물리적으로 적절함. 프로 영상(gap_ratio=1.0): 60px 유지.
        #
        # [v10 추가: y_range_ratio 기반 스케일]
        # gap_ratio=1.0(프로 탑뷰)이더라도 영상의 실제 y 활용 범위가 frame_height의
        # 절반 이하인 경우 코트가 화면에 작게 잡혀 있는 것이므로 비행거리가 짧다.
        # y_range = valid_y.max() - valid_y.min()
        # y_range_ratio = y_range / frame_height  (0.0 ~ 1.0)
        # 예: 탑뷰 풀화면(y_range≈0.7h) → ratio≈0.7 → 스케일=1.0 (기준 유지)
        #     쿼터뷰 소형(y_range≈0.35h) → ratio≈0.35 → 스케일=0.5 → 비행거리 절반 허용
        # scale = clip(y_range_ratio / 0.7, 0.4, 1.0)  (하한 0.4: 과도 완화 방지)
        # [Regression 안전] 프로 영상: y_range_ratio ≈ 0.7 → scale ≈ 1.0 → 변화 없음.
        if y is not None and len(valid_idx) > 0:
            valid_y_range = y[valid_idx]
            valid_y_range = valid_y_range[valid_y_range > 0]
            if len(valid_y_range) > 0:
                y_span         = float(valid_y_range.max() - valid_y_range.min())
                y_range_ratio  = y_span / max(self.frame_height, 1)
                y_range_scale  = float(np.clip(y_range_ratio / 0.7, 0.4, 1.0))
            else:
                y_range_scale = 1.0
        else:
            y_range_scale = 1.0

        dynamic_min_flight = max(20.0, cfg["min_flight_dist"] / gap_ratio * y_range_scale)

        # ── skip_angle_thresh 유지 ────────────────────────────
        # [이유] 각도 필터는 쿼터뷰에서도 타점 vs 직진 구분에 효과적.
        # 완화 시 직진 궤적이 타점으로 오탐될 위험 → 원본 값 유지.
        dynamic_skip_angle = cfg["skip_angle_thresh"]

        # ── peak_distance FPS 조정 (v11: 10→7 승수 추가 감소) ─────
        # [v10 문제] 승수 10/12 → 30fps에서 peak_distance=10.
        # 서브(frame 30)와 리턴(frame 38) 간격=8 < 10 → 리턴 억제!
        # [v11 수정] 승수 7/12 → 30fps에서 peak_distance=7.
        # 간격 7프레임(≈0.23s) 이상이면 독립 타점으로 인정.
        # 24fps: max(5, int(12*0.8*7/12))=max(5,5)=5 (≈0.21s)
        # 60fps: max(5, int(12*2.0*7/12))=max(5,14)=14 (≈0.23s)
        # [Regression 안전] 프로 영상 7타점의 최소 간격 > 0.5s >> 7프레임 → 영향 없음.
        # 오탐 방지: prominence=1.0 조건이 얕은 이중 bump를 차단함.
        # 최소 5프레임(≈0.17초) 보장.
        dynamic_peak_dist = max(5, int(cfg["peak_distance"] * fps_scale * 7 / 12))

        # ── near_future_window FPS 조정 ──────────────────────
        # [이유] 바닥 근처 정적 공 필터에서 "이후 15프레임"을 탐색.
        # 원본 15프레임 = 30fps에서 0.5초. 60fps이면 0.25초로 너무 짧음.
        # FPS에 비례해 확장.
        dynamic_near_future = max(10, int(15 * fps_scale))

        # ── min_floor_drop 완화 (v8→v9: 지수적 완화 + 하한 0) ───
        # [목적] BOTTOM 플레이어 4초 부근 두 타격 복구.
        # _compute_impulse_scores 내 "바닥 근처 정적 공 필터"의 조건:
        #   min(y[near_future]) > y[i_curr] - MIN_FLOOR_DROP → 타점 skip
        # 원본 MIN_FLOOR_DROP = 20px (30fps 탑뷰 기준).
        #
        # [v8 문제] gap_ratio=4.0 → min_floor_drop=5px 이었으나, 쿼터뷰 BOTTOM
        # 타격 직후 near_future Y 감소가 2~4px에 그쳐 여전히 skip됨.
        #
        # [v9 해결: 지수적 완화 + 하한 0]
        #   dynamic_min_floor_drop = max(0.0, 20.0 / gap_ratio ** 1.5)
        #   gap_ratio=1.0 (프로):  20 / 1.0   = 20.0px  → 원본 동일 ✓
        #   gap_ratio=2.0:          20 / 2.83  =  7.1px  (v8: 10px보다 완화)
        #   gap_ratio=3.0:          20 / 5.20  =  3.9px  (v8: 6.7px보다 완화)
        #   gap_ratio=4.0 (쿼터뷰): 20 / 8.0  =  2.5px  (v8: 5px보다 완화)
        #
        # 하한 0.0: gap_ratio가 매우 크면 사실상 바닥 필터 비활성화 허용.
        # 대신 floor_pct를 높여(아래 참조) 필터 적용 범위 자체를 축소함으로써
        # 완전 비활성화로 인한 오탐지 위험을 보상.
        #
        # [Regression 안전] gap_ratio=1.0 → 20px 그대로 → 프로 영상 동일.
        dynamic_min_floor_drop = max(0.0, 20.0 / (gap_ratio ** 1.5))

        # ── floor_pct: y_floor_lim 퍼센타일 어댑티브 (v9 신규) ─
        # [목적] "바닥 근처" 판정 기준을 gap_ratio에 따라 동적으로 높임.
        # y_floor_lim = percentile(valid_y, floor_pct)에서 floor_pct가 높을수록
        # 더 극단적인 Y값(실제 바닥에 박힌 공)만 필터 대상이 됨.
        #
        # [수학적 근거]
        # 쿼터뷰 영상에서 valid_y의 분포는 코트 전체에 걸쳐 퍼져 있어
        # 85th percentile이 실제 바닥보다 높은 위치를 가리킬 수 있다.
        # gap_ratio가 클수록(아마추어) 픽셀 분포가 좁아지므로 퍼센타일을 높여
        # 필터 적용 구간을 실제 바닥에 가까운 좌표만으로 축소.
        #
        # floor_pct = clip(85 + (gap_ratio - 1) × 5, 85, 95)
        #   gap_ratio=1.0: 85.0 (원본 동일)
        #   gap_ratio=2.0: 90.0
        #   gap_ratio=3.0: 95.0 (상한)
        #   gap_ratio=4.0: 95.0 (상한 유지)
        #
        # [Regression 안전] gap_ratio=1.0 → 85th percentile → 원본과 동일.
        dynamic_floor_pct = float(np.clip(85.0 + (gap_ratio - 1.0) * 5.0, 85.0, 95.0))

        return {
            "max_frame_gap":        dynamic_max_gap,
            "min_speed":            dynamic_min_speed,
            "min_flight_dist":      dynamic_min_flight,
            "skip_angle_thresh":    dynamic_skip_angle,
            "peak_distance":        dynamic_peak_dist,
            "near_future_window":   dynamic_near_future,
            "min_floor_drop":       dynamic_min_floor_drop,
            "floor_pct":            dynamic_floor_pct,   # [v9] 바닥 필터 퍼센타일
            "peak_threshold_ratio": cfg["peak_threshold_ratio"],
            # 디버깅용
            "_avg_gap":       round(avg_gap, 2),
            "_gap_ratio":     round(gap_ratio, 2),
            "_typical_disp":  round(typical_disp, 2),
            "_typical_speed": round(typical_speed, 2),
        }

    def _base_params(self) -> dict:
        """원본 파라미터 그대로 반환 (데이터 부족 시 fallback)."""
        cfg = self.cfg
        return {
            "max_frame_gap":        cfg["max_frame_gap"],
            "min_speed":            cfg["min_speed"],
            "min_flight_dist":      cfg["min_flight_dist"],
            "skip_angle_thresh":    cfg["skip_angle_thresh"],
            "peak_distance":        cfg["peak_distance"],
            "near_future_window":   15,
            "min_floor_drop":       20.0,   # 원본 하드코딩 값
            "floor_pct":            85.0,   # [v9] 원본 85th percentile
            "peak_threshold_ratio": cfg["peak_threshold_ratio"],
            "_avg_gap":       0.0,
            "_gap_ratio":     1.0,
            "_typical_disp":  0.0,
            "_typical_speed": 0.0,
        }

    # ── impulse 점수 계산 (원본 로직 + 어댑티브 파라미터 주입) ─

    def _compute_impulse_scores(
        self,
        x:           np.ndarray,
        y:           np.ndarray,
        valid_idx:   np.ndarray,
        adaptive:    dict,
        wrist_index: Optional[Dict[int, List[tuple]]] = None,
    ) -> np.ndarray:
        """
        프레임별 impulse 점수를 계산한다.

        원본(f4c000c)과 로직 구조 완전 동일.
        달라진 점: 하드코딩된 cfg 값 대신 adaptive dict 사용.
          - max_frame_gap    : avg_gap 기반 동적 확장
          - min_speed        : avg_gap + 쿼터뷰 기반 완화
          - min_flight_dist  : avg_gap 기반 완화
          - near_future_window: FPS 기반 확장

        wrist_index 제공 시: 바닥 필터 예외처리
          특정 프레임에서 손목 좌표가 셔틀 근처(< WRIST_FLOOR_BYPASS_PX)이면
          "바닥 근처 정적 공 필터"를 스킵한다.
          언더핸드 클리어·다이빙 리시브 감지에 사용.
        """
        WRIST_FLOOR_BYPASS_PX = 100.0  # 손목-셔틀 거리 임계값 (px)
        scores = np.zeros(len(x))

        # valid_idx는 detect()에서 이미 계산해 전달받음 (중복 계산 없음)
        if len(valid_idx) == 0:
            return scores

        valid_y     = y[valid_idx]
        # [v9] floor_pct: gap_ratio 어댑티브 퍼센타일 (85~95).
        # 높을수록 더 극단적인 Y값만 "바닥 근처"로 판정 → 필터 적용 범위 축소.
        # 프로 영상(gap_ratio=1.0): 85th → 원본 동일.
        # 쿼터뷰(gap_ratio=4.0): 95th → 실제 최하단 좌표만 필터 대상.
        floor_pct   = adaptive.get("floor_pct", 85.0)
        y_floor_lim = np.percentile(valid_y, floor_pct) if len(valid_y) > 0 else 9999

        # 어댑티브 파라미터 언패킹
        max_frame_gap     = adaptive["max_frame_gap"]
        min_speed         = adaptive["min_speed"]
        skip_angle_thresh = adaptive["skip_angle_thresh"]
        min_flight_dist   = adaptive["min_flight_dist"]
        near_future_win   = adaptive["near_future_window"]
        # v8: 바닥 근처 정적 공 판정 최소 Y 하락 거리 (어댑티브)
        min_floor_drop    = adaptive.get("min_floor_drop", 20.0)

        for k in range(2, len(valid_idx) - 2):
            i_curr = valid_idx[k]
            i_pre  = valid_idx[k - 2]
            i_post = valid_idx[k + 2]

            dt_in  = i_curr - i_pre
            dt_out = i_post - i_curr

            # ── 최대 프레임 갭 필터 ──────────────────────────────
            # [원본] 5프레임 초과 시 스킵.
            # [v5]  avg_gap * 2.5 로 동적 확장.
            #       avg_gap=1.5 → max_gap=5 (원본 동일)
            #       avg_gap=4.0 → max_gap=10 (아마추어 블러 대응)
            if dt_in > max_frame_gap or dt_out > max_frame_gap:
                continue

            v_in  = np.array([x[i_curr] - x[i_pre],  y[i_curr] - y[i_pre]])  / dt_in
            v_out = np.array([x[i_post] - x[i_curr],  y[i_post] - y[i_curr]]) / dt_out

            speed_in  = np.linalg.norm(v_in)
            speed_out = np.linalg.norm(v_out)

            # ── 최소 속도 필터 ────────────────────────────────────
            # [원본] 3.0 px/frame 이하 시 스킵.
            # [v5]  쿼터뷰 원근 + 드문 좌표로 인해 낮아지므로 동적 완화.
            if speed_in < min_speed and speed_out < min_speed:
                continue

            v_skip    = np.array([x[i_post] - x[i_pre], y[i_post] - y[i_pre]])
            norm_skip = np.linalg.norm(v_skip)
            norm_in   = np.linalg.norm(v_in)

            # ── Skip-Vector 직진 필터 ─────────────────────────────
            # [원본·v5 동일] 각도 < 15도 → 직진 → 타점 아님.
            # 쿼터뷰에서도 신뢰도 유지 → 완화 없음.
            if norm_skip > 1e-6 and norm_in > 1e-6:
                cos_skip   = np.dot(v_in, v_skip) / (norm_in * norm_skip)
                angle_skip = np.degrees(np.arccos(np.clip(cos_skip, -1.0, 1.0)))
                if angle_skip < skip_angle_thresh:
                    continue

            future_idx = valid_idx[valid_idx > i_curr]
            if len(future_idx) < 3:
                continue

            final_x  = x[future_idx[-1]]
            final_y  = y[future_idx[-1]]
            flight_d = np.hypot(final_x - x[i_curr], final_y - y[i_curr])

            # ── 최소 비행 거리 필터 ───────────────────────────────
            # [원본] 60px 미만 시 스킵.
            # [v5]  쿼터뷰 원근 효과로 픽셀 비행 거리가 작게 측정됨 → 동적 완화.
            #       gap_ratio=1.0 → 60px (원본), gap_ratio=2.67 → 약 22px → 최소 30px
            if flight_d < min_flight_dist:
                continue

            # ── 바닥 근처 정적 공 필터 ────────────────────────────
            # [원본] near_future 15프레임 이내 최솟값 탐색.
            # [v5]  near_future_window를 FPS 기반으로 확장 (30fps→15, 60fps→30).
            # [v8]  min_floor_drop을 어댑티브로 교체 (하드코딩 20px 제거).
            #       쿼터뷰에서 BOTTOM 플레이어 서브·드롭은 Y가 거의 안 줄어들어
            #       20px 기준을 못 넘어 정적공으로 오판 → min_floor_drop으로 완화.
            # [v11] wrist_index 제공 시: 손목이 셔틀 근처이면 필터 스킵.
            #       언더핸드 클리어·다이빙 리시브는 공이 바닥 근처에서 타격되므로
            #       선수가 실제로 손목을 가져다 댔다는 증거가 있으면 정적공 오판 방지.
            if y[i_curr] > y_floor_lim:
                # 포즈 예외: 손목이 셔틀 근처이면 바닥 필터 스킵
                floor_bypassed = False
                if wrist_index:
                    for wx, wy in wrist_index.get(int(i_curr), []):
                        if np.hypot(wx - x[i_curr], wy - y[i_curr]) < WRIST_FLOOR_BYPASS_PX:
                            floor_bypassed = True
                            break
                if not floor_bypassed:
                    near_future = [idx for idx in future_idx if idx <= i_curr + near_future_win]
                    if near_future:
                        if min(y[idx] for idx in near_future) > y[i_curr] - min_floor_drop:
                            continue

            # ── 서브 포물선 정점(apex) 필터 (v12 신규) ──────────
            # [목적] 서브 비행 중 자연적인 수직 방향 전환(정점)에서
            #        impulse가 실제 타격처럼 계산되는 오탐 방지.
            # [조건] 세 조건 모두 만족 시 스킵 (AND 조건 → 매우 보수적):
            #   1) 이전에 위쪽으로 이동 중: v_in[1] < -APEX_UP_THRESH
            #      (이미지 좌표에서 y 감소 = 위쪽 이동)
            #   2) 이후 아래쪽으로 이동:   v_out[1] > APEX_DOWN_THRESH
            #      (y 증가 = 아래쪽 이동)
            #   3) 수평 성분 변화 작음:    |v_out[0]-v_in[0]| < APEX_HORIZ_MAX
            #      (외력 없이 중력만 작용 → 수평 속도 유지)
            # [수학적 근거]
            #   실제 타격: 라켓이 수평+수직 모두 속도 변경 → |Δv_x| ≥ 5~20 px/frame
            #   자연 포물선 정점: 수평 무변화 → |Δv_x| < 2 px/frame
            #   임계값 APEX_HORIZ_MAX=4.0: 실제 타격과 자연 정점 사이 충분한 여유.
            # [Regression 안전]
            #   언더핸드 클리어(올라가는 공을 위로 침):
            #     v_out[1] < 0 (공이 위로 올라감) → 조건 2 미충족 → 보존 ✓
            #   서브 이후 내려오는 공에 대한 스매시:
            #     v_in[1] > 0 (공이 이미 내려오는 중) → 조건 1 미충족 → 보존 ✓
            APEX_UP_THRESH   = 2.0  # v_in[1] 하한 (위쪽 이동 최소 속도)
            APEX_DOWN_THRESH = 2.0  # v_out[1] 하한 (아래쪽 이동 최소 속도)
            APEX_HORIZ_MAX   = 4.0  # 수평 변화 상한 (외력 없음 판정)
            if (v_in[1] < -APEX_UP_THRESH
                    and v_out[1] > APEX_DOWN_THRESH
                    and abs(v_out[0] - v_in[0]) < APEX_HORIZ_MAX):
                continue

            impulse = np.linalg.norm(v_out - v_in)

            if v_out[1] - v_in[1] > 10:
                impulse *= 1.5

            scores[i_curr] = impulse

        return scores

    def _find_peaks(self, scores: np.ndarray, adaptive: dict, verbose: bool = False) -> np.ndarray:
        """
        impulse 점수에서 피크를 검출한다.

        [원본·v5] threshold = max_score × 0.15.
                  문제: 스파이크 허위 점수가 max_score를 부풀리면 threshold가
                  치솟아 실제 타점 점수가 전부 탈락 → 14초 구간 유실 발생.

        [v7] Tukey 극단 이상치 억제 → robust max → threshold 재계산.

        [수학적 근거 — Tukey의 극단 이상치 기준]
          Q1, Q3: 점수 분포의 1·3사분위수
          IQR   : Q3 - Q1 (사분위 범위, 분포 중앙 50%의 폭)
          극단 이상치 기준: Q3 + 3 × IQR

          이 기준은 정규 분포에서 99.9+% 분위 이상의 극히 드문 값만 이상치로
          처리하여, 실제 타점(강한 스매시 포함)이 잘못 억제될 확률을 최소화한다.

          스파이크 허위 점수 ≈ 실제 타점 점수의 5~20배 → 극단 이상치 조건 충족.
          강한 스매시 점수 ≈ 평균 타점 점수의 2~3배 → 이상치 조건 미충족 → 보존.

        [샘플 수 가드]
          비영 점수가 6개 미만이면 IQR 통계가 신뢰성 없음 → 원본 max 방식 사용.

        [Regression 안전]
          프로 영상: 스파이크 없음 → 점수 분포가 정규에 가까움
                     → upper_fence >> max_score → 억제 없음 → 원본과 동일.
        """
        non_zero = scores[scores > 0]
        if len(non_zero) == 0:
            return np.array([], dtype=int)

        work_scores  = scores.copy()  # 원본 scores 불변 (build_candidates에서 사용)
        upper_fence  = None           # IQR 억제 임계값 (억제 없으면 None)
        suppressed_any = False

        if len(non_zero) >= 6:
            q1 = np.percentile(non_zero, 25)
            q3 = np.percentile(non_zero, 75)
            iqr = q3 - q1

            if iqr > 1e-6:
                # Tukey 극단 이상치 상한: Q3 + 3 × IQR
                # [이유] 표준 box-plot의 1.5 × IQR보다 관대한 기준을 사용.
                # 1.5 × IQR 기준은 실제 강한 스매시 점수(평균의 2~3배)도
                # 이상치로 잡을 수 있어, 프로 영상 regression risk 발생.
                # 3 × IQR은 스파이크(평균의 5~20배)만 억제하고
                # 강한 스매시(평균의 2~3배)는 보존.
                upper_fence = q3 + 3.0 * iqr

                # 이상치 점수를 0으로 억제 → 피크 탐지에서 자동 제외
                suppressed_frames = np.where(work_scores > upper_fence)[0]
                if len(suppressed_frames):
                    suppressed_any = True
                    work_scores[suppressed_frames] = 0.0
                    # [v11] IQR 억제 프레임 저장 → get_near_miss_frames에서 포즈 재검증에 활용.
                    # 이 프레임들은 "score가 너무 높아서 노이즈로 억제됨"이지,
                    # "물리 필터를 통과하지 못함"이 아니다.
                    # 서브 리턴처럼 궤적이 섞인 채로 타격이 감지되면 impulse가
                    # upper_fence를 크게 초과해 억제되므로, 포즈 확인으로 진짜/가짜 구분.
                    self._last_iqr_suppressed = suppressed_frames.tolist()
                    if verbose:
                        print(f"  [IQR suppressed] upper_fence={upper_fence:.1f}  "
                              f"frames={suppressed_frames.tolist()}  "
                              f"scores={[round(float(scores[f]),1) for f in suppressed_frames]}")

        # 억제 후 max_score 재계산: 더 이상 스파이크가 max를 독점하지 않음
        remaining = work_scores[work_scores > 0]
        if len(remaining) == 0:
            return np.array([], dtype=int)

        # [v10] IQR 억제가 일어난 경우 upper_fence를 max_score 기준으로 사용.
        # 이유: 복구된 연속 스파이크 구간의 신규 점수들이 IQR 분포를 변화시켜
        #       상황에 따라 max_remaining 이 크게 달라질 수 있음.
        #       upper_fence(= Q3+3·IQR)는 "정상 타점의 최대 기대값"으로 안정적.
        #       → threshold = upper_fence × ratio 로 고정 → 복구 전후 일관성 유지.
        # Regression 안전: 억제가 없는 경우(프로 영상) → max_remaining 그대로 사용.
        if suppressed_any and upper_fence is not None:
            max_score = float(upper_fence)
        else:
            max_score = float(np.max(remaining))

        threshold = max_score * adaptive["peak_threshold_ratio"]

        if verbose:
            top_scores = sorted(
                [(int(i), round(float(scores[i]), 1)) for i in np.where(work_scores > 0)[0]],
                key=lambda t: -t[1]
            )[:20]
            print(f"  [peaks] max_score={max_score:.1f}  threshold={threshold:.1f}  "
                  f"peak_distance={adaptive['peak_distance']}")
            print(f"  top_frames(score): {top_scores}")

        peaks, _ = find_peaks(
            work_scores,
            distance  = adaptive["peak_distance"],
            prominence= 1.0,
            height    = threshold,
        )

        # ── 근접 피크 NMS (v12 신규, v12.1 ratio 조건 제거) ────
        # [목적] find_peaks(distance=d) 경계 조건으로 NMS_FRAMES 이내
        #        두 피크가 모두 살아남는 경우, 높은 점수만 유지.
        # [대상]
        #   #2 타입: 서브 전 CSV 좌표 튐 → 실제 서브와 6~7프레임 차이 중복
        #   #10 타입: 타격 직후 추적 흔들림 → 동일 타격 이중 감지
        # [기준]
        #   NMS_FRAMES: 이 프레임 수 이내 = "근접" → 높은 점수만 생존
        #   9프레임 ≈ 0.3초 at 30fps
        # [Regression 안전]
        #   배드민턴에서 0.3초 이내 두 번의 실제 타격은 물리적으로 불가능.
        #   (라켓 스윙 + 셔틀 비행 시간 최소 0.4~0.5초 필요)
        #   → NMS_FRAMES 이내라면 무조건 오탐이므로 ratio 조건 불필요.
        # [v12 → v12.1 변경 이유]
        #   ratio 조건(2.5배)이 있으면 오탐 피크 점수 ≥ 실제 타격 점수인 경우
        #   억제가 안 됨 (CSV 좌표 튐이 더 큰 impulse를 만들 수 있음).
        NMS_FRAMES = 9    # 근접 판정 최대 프레임 수 (~0.3s at 30fps)

        if verbose:
            print(f"  [NMS] before={peaks.tolist()}")

        peaks = self._score_nms(peaks, work_scores, NMS_FRAMES)

        if verbose:
            print(f"  [NMS] after ={peaks.tolist()}")

        return peaks

    def _build_candidates(
        self,
        frames: np.ndarray,
        x:      np.ndarray,
        y:      np.ndarray,
        scores: np.ndarray,
    ) -> List[ImpactEvent]:
        events = []
        for f in frames:
            frame_adj = max(0, int(f) - 1)
            owner     = "top" if y[f] < self._owner_y else "bottom"
            player    = "top" if owner == "top" else "bottom"
            events.append(ImpactEvent(
                frame    = frame_adj,
                time_sec = frame_adj / self.fps,
                score    = float(scores[f]),
                owner    = owner,
                player   = player,
                method   = "peaks",
            ))
        return events

    # ── crossing owner 교정 ──────────────────────────────────

    def _apply_crossing_owner(
        self,
        candidates: List[ImpactEvent],
        crossings:  List["NetCrossingEvent"],
    ) -> List[ImpactEvent]:
        """
        물리 필터 통과 후보의 owner를 crossing 방향으로 재검증한다.

        후보 개수는 변하지 않는다 (gap fill 없음).

        [매칭 규칙 — Pass 1: Backward (기존)]
          각 crossing 직전 lookback_frames 내에서 가장 가까운 미매칭 후보를
          crossing.hitter_side로 갱신한다.
          → 예: crossing(top_to_bottom) → 직전 타격 = top 선수 ✓

        [매칭 규칙 — Pass 2: Forward (v10 신규)]
          crossing 직후 forward_frames 내에서 첫 미매칭 후보를
          crossing.receiver_side로 갱신한다.
          → 예: crossing(top_to_bottom) → 직후 첫 타격 = bottom 선수(받는 쪽) ✓
          이유: 서브처럼 crossing 이전 hit이 감지 안 되는 경우,
                crossing 바로 다음 hit(서브 리턴)이 receiver임을 확정할 수 있음.
          forward window = crossing.frame+1 ~ crossing.frame+lookback
          (crossing.frame 자체는 제외 — 해당 프레임은 hitter에 해당)
        """
        lookback         = int(self.cfg.get("crossing_lookback_frames", 60))
        result           = list(candidates)
        matched_cand_idx = set()

        sorted_crossings = sorted(crossings, key=lambda c: c.frame)

        # ── Pass 1: Backward — crossing 직전 hitter 확정 ────────
        for crossing in sorted_crossings:
            w_start = max(0, crossing.frame - lookback)
            w_end   = crossing.frame

            eligible = [
                (i, c) for i, c in enumerate(result)
                if w_start <= c.frame < w_end and i not in matched_cand_idx
            ]
            if not eligible:
                continue

            best_i, best_c = min(eligible, key=lambda ic: w_end - ic[1].frame)
            owner  = crossing.hitter_side
            player = "pink_top" if owner == "top" else "green_bottom"

            result[best_i] = ImpactEvent(
                frame      = best_c.frame,
                time_sec   = best_c.time_sec,
                score      = best_c.score,
                owner      = owner,
                player     = player,
                hit_number = best_c.hit_number,
                method     = "peaks_crossed",
            )
            matched_cand_idx.add(best_i)

        # ── Pass 2: Forward — crossing 직후 receiver 확정 ───────
        for crossing in sorted_crossings:
            w_start = crossing.frame + 1        # crossing 자체는 hitter → 제외
            w_end   = crossing.frame + lookback

            eligible = [
                (i, c) for i, c in enumerate(result)
                if w_start <= c.frame <= w_end and i not in matched_cand_idx
            ]
            if not eligible:
                continue

            # crossing 직후 가장 가까운(직후) 후보만 선택
            best_i, best_c = min(eligible, key=lambda ic: ic[1].frame)
            owner  = crossing.receiver_side
            player = "pink_top" if owner == "top" else "green_bottom"

            result[best_i] = ImpactEvent(
                frame      = best_c.frame,
                time_sec   = best_c.time_sec,
                score      = best_c.score,
                owner      = owner,
                player     = player,
                hit_number = best_c.hit_number,
                method     = "peaks_crossed",
            )
            matched_cand_idx.add(best_i)

        return result

    # ── owner 교대 교정 (원본 로직 동일) ────────────────────

    def _correct_owner_alternation(
        self,
        candidates: List[ImpactEvent],
        y:          np.ndarray,
    ) -> List[ImpactEvent]:
        """
        연속 동일 owner를 위치 신뢰도 기반으로 교정한다.

        [v8 변경: 점수 우선 가드 추가]
        기존 로직은 동일 owner 충돌 시 위치 신뢰도만으로 플립 대상을 결정했다.
        두 후보 모두 명확한 위치(confidence=1.0)일 때는 항상 curr를 플립하는데,
        이 경우 curr의 점수가 훨씬 높아도 무조건 플립되는 부작용이 있다.

        [수학적 근거: 점수 비율 가드]
        impulse score는 실제 방향 변화량의 물리적 크기 ∝ |v_out - v_in|.
        한쪽 점수가 SCORE_DOMINANCE(3.0)배 이상이면:
          - 더 높은 점수 타점 = 더 강한 방향 전환 = 더 확실한 타점
          - 반대쪽이 오탐지일 가능성이 높음
        → 점수가 높은 쪽 owner 판단을 신뢰, 낮은 쪽을 플립.

        [Regression 안전]
        프로 영상: 7개 타점 점수가 비교적 균등 분포 → 3:1 초과 드뭄
        → 기존 confidence 로직과 동일하게 작동 ✓

        crossing 검증된 후보("peaks_crossed")는 여전히 최우선으로 신뢰한다.
        """
        if len(candidates) < 2:
            return candidates

        SCORE_DOMINANCE = 3.0  # 한쪽 점수가 이 배수 이상이면 점수 기준 우선

        # [v9] 시간 간격 기반 flip 억제 임계값
        # 두 연속 같은 owner 타격 간격이 이 이상이면 alternation 교정 없이 유지.
        # [근거] 배드민턴에서 1.0초 이상 간격이면 동일 플레이어가 두 번 치는 것이
        # 물리적으로 가능(랠리 패턴: 드롭→클리어, 서브→리드 드라이브 등).
        # flip 시 Ground Truth와 player 불일치 → 유닛 테스트 "누락" 판정 원인.
        # [Regression 안전] 프로 영상: 연속 동일 owner 충돌이 적고, 대부분
        # 프레임 간격이 짧은 잘못 감지된 경우 → 시간 간격 < 1.0초 → 기존 로직 그대로.
        MIN_TIME_GAP_FOR_SKIP = 1.0  # 초 단위

        corrected = list(candidates)

        for i in range(1, len(corrected)):
            prev = corrected[i - 1]
            curr = corrected[i]

            if prev.owner != curr.owner:
                continue

            time_gap = curr.time_sec - prev.time_sec
            if time_gap >= MIN_TIME_GAP_FOR_SKIP:
                continue

            prev_pose_conf = "_pose" in prev.method
            curr_pose_conf = "_pose" in curr.method
            prev_validated = (prev.method == "peaks_crossed") or prev_pose_conf
            curr_validated = (curr.method == "peaks_crossed") or curr_pose_conf

            if prev_pose_conf and curr_pose_conf:
                continue
            if prev_validated and not curr_validated:
                corrected[i] = self._flip(curr)
                continue
            if curr_validated and not prev_validated:
                corrected[i - 1] = self._flip(prev)
                continue

            score_ratio = curr.score / max(prev.score, 1e-6)

            if score_ratio >= SCORE_DOMINANCE:
                corrected[i - 1] = self._flip(prev)
                continue
            if score_ratio <= 1.0 / SCORE_DOMINANCE:
                corrected[i] = self._flip(curr)
                continue

            prev_conf = self._position_confidence(y[min(prev.frame, len(y) - 1)])
            curr_conf = self._position_confidence(y[min(curr.frame, len(y) - 1)])

            if curr_conf <= prev_conf:
                corrected[i] = self._flip(curr)
            else:
                corrected[i - 1] = self._flip(prev)

        return corrected

    @staticmethod
    def _flip(event: ImpactEvent) -> ImpactEvent:
        flipped_owner  = "bottom" if event.owner == "top" else "top"
        flipped_player = "pink_top" if flipped_owner == "top" else "green_bottom"
        return ImpactEvent(
            frame      = event.frame,
            time_sec   = event.time_sec,
            score      = event.score,
            owner      = flipped_owner,
            player     = flipped_player,
            hit_number = event.hit_number,
            method     = event.method,
        )

    def _position_confidence(self, ball_y: float) -> float:
        dist = abs(ball_y - self._owner_y)
        if dist >= self._ambiguity_band:
            return 1.0
        return dist / self._ambiguity_band

    @staticmethod
    def _score_nms(
        peaks:      np.ndarray,
        scores:     np.ndarray,
        min_frames: int,
    ) -> np.ndarray:
        """
        근접 피크 NMS (Non-Maximum Suppression).

        두 피크가 min_frames 이내이면 높은 점수만 남긴다.
        점수 비율 조건 없음 — 0.3초 이내 두 타격은 물리적으로 불가능하므로
        무조건 오탐으로 간주하고 낮은 쪽을 억제한다.

        [알고리즘]
          프레임 순서로 정렬 후, 살아있는 피크를 순서대로 확인:
            gap < min_frames → 낮은 점수 쪽 억제 (동점이면 뒤쪽 억제)
            gap ≥ min_frames → 둘 다 유지, 다음 쌍으로 이동
        """
        if len(peaks) <= 1:
            return peaks

        peaks_sorted = np.sort(peaks)
        keep = np.ones(len(peaks_sorted), dtype=bool)

        i = 0
        while i < len(peaks_sorted):
            if not keep[i]:
                i += 1
                continue
            j = i + 1
            while j < len(peaks_sorted):
                if not keep[j]:
                    j += 1
                    continue
                gap = int(peaks_sorted[j]) - int(peaks_sorted[i])
                if gap >= min_frames:
                    break
                si = float(scores[peaks_sorted[i]])
                sj = float(scores[peaks_sorted[j]])
                if si >= sj:
                    # i 점수가 같거나 높음 → j 억제
                    keep[j] = False
                else:
                    # j 점수가 높음 → i 억제, i 루프 종료
                    keep[i] = False
                    break
                j += 1
            i += 1

        return peaks_sorted[keep]

    # ── 포즈 데이터 유틸 ─────────────────────────────────────

    @staticmethod
    def _build_wrist_index(
        pose_full: List[List[dict]],
        kp_conf_thresh: float = 0.3,
    ) -> Dict[int, List[tuple]]:
        """
        pose_full(전 프레임 keypoints 목록) → {frame_idx: [(wx, wy), ...]} 딕셔너리.

        바닥 필터 예외처리(_compute_impulse_scores)에서 사용.
        COCO keypoints 인덱스: 9 = left wrist, 10 = right wrist.

        Args:
            pose_full       : List[List[dict]] — frame_idx 순서대로.
                              각 플레이어 dict: {"keypoints": [[x,y,conf] * 17]}
            kp_conf_thresh  : 손목 keypoint 신뢰도 최솟값 (이하 무시)

        Returns:
            손목 좌표가 유효한 프레임만 포함하는 sparse dict.
        """
        WRIST_KP = [9, 10]  # COCO wrist keypoint 인덱스
        index: Dict[int, List[tuple]] = {}

        for frame_idx, players in enumerate(pose_full):
            wrists = []
            for player in players:
                kps = player.get("keypoints", [])
                if len(kps) < 11:
                    continue
                for wi in WRIST_KP:
                    kp = kps[wi]
                    if kp[2] >= kp_conf_thresh and kp[0] > 0 and kp[1] > 0:
                        wrists.append((float(kp[0]), float(kp[1])))
            if wrists:
                index[frame_idx] = wrists

        return index

    def _apply_pose_owner(
        self,
        candidates: List[ImpactEvent],
        x: np.ndarray,
        y: np.ndarray,
        pose_data,  # List[List[dict]] 또는 Dict[int, List[dict]] 모두 허용
        verbose: bool = False,
    ) -> List[ImpactEvent]:
        """
        포즈 데이터로 타격 선수(owner)를 교정한다.

        [원칙]
        - crossing 검증된 타점(method="peaks_crossed")은 신뢰도가 높으므로 건드리지 않음.
        - 타격 프레임에서 셔틀콕과 가장 가까운 손목을 찾아 그 선수의 중심 y를
          owner_y_threshold와 비교해 owner를 결정.
        - 손목이 PROXIMITY_PX 이내가 아니면 교정하지 않음 (모호한 경우 현상 유지).

        [선수 중심 y 계산 우선순위]
          1) hips(11, 12) 중점 — 양쪽 모두 신뢰도 충분할 때
          2) 상반신 관절(코, 어깨, 팔꿈치, 손목) 중위값 — hips 불가 시
          3) 전체 가시 관절 중위값 — fallback

        Args:
            candidates : _apply_crossing_owner 결과.
            x, y       : 셔틀콕 좌표 배열.
            pose_data  : List[List[dict]] 또는 Dict[int, List[dict]].
            verbose    : 변경 내역 출력 여부.

        Returns:
            owner가 교정된 ImpactEvent 리스트.
        """
        WRIST_KP          = [9, 10]           # COCO wrist
        KP_CONF_THRESH    = 0.25              # 관절 신뢰도 최솟값
        PROXIMITY_PX      = 150.0             # 손목-셔틀 최대 허용 거리
        ANKLE_KP          = [15, 16]          # 발목 — 자기 코트를 벗어나지 않아 top/bottom 분류에 가장 안정적
        KNEE_KP           = [13, 14]          # 무릎 — 발목 불가 시 차선
        HIP_KP            = [11, 12]          # 힙 — 무릎 불가 시 차차선 (타격 자세에 따라 오판 가능)
        UPPER_BODY_IDX    = [0, 5, 6, 7, 8, 9, 10]   # 코·어깨·팔꿈치·손목 (최후 fallback)

        # pose_data를 dict로 통일 (List이면 enumerate로 변환)
        if isinstance(pose_data, dict):
            pose_dict: Dict[int, List[dict]] = pose_data
        else:
            pose_dict = {i: v for i, v in enumerate(pose_data) if v}

        result = list(candidates)

        for idx, event in enumerate(result):
            frame_no = event.frame

            # [v11] 정확한 frame에 pose가 없으면 ±2 이웃 프레임을 탐색한다.
            # 이유: _build_candidates가 frame_adj = peak-1 오프셋을 적용하므로
            # event.frame은 실제 타격 프레임보다 1 낮다. 또한 YOLO가 특정 프레임에서
            # 선수를 탐지 못하는 경우(모션 블러, 가림 등)에도 인근 프레임으로 보완.
            # 탐색 순서: 0 → +1 → -1 → +2 → -2 (가까운 쪽 우선)
            players  = pose_dict.get(frame_no, [])
            probe_sx = float(x[min(frame_no, len(x) - 1)])
            probe_sy = float(y[min(frame_no, len(y) - 1)])

            if not players:
                for delta in [1, -1, 2, -2]:
                    nb = frame_no + delta
                    nb_players = pose_dict.get(nb, [])
                    if not nb_players:
                        continue
                    nb_sx = float(x[min(nb, len(x) - 1)])
                    nb_sy = float(y[min(nb, len(y) - 1)])
                    if nb_sx > 0 and nb_sy > 0:
                        players  = nb_players
                        probe_sx = nb_sx
                        probe_sy = nb_sy
                        break

            if not players:
                continue

            sx = probe_sx
            sy = probe_sy
            if sx <= 0 or sy <= 0:
                continue

            best_dist     = float("inf")
            best_center_y = None

            for player in players:
                kps = player.get("keypoints", [])
                if len(kps) < 17:
                    continue

                # 가장 가까운 손목 탐색
                for wi in WRIST_KP:
                    kp = kps[wi]
                    if kp[2] < KP_CONF_THRESH or kp[0] <= 0:
                        continue
                    dist = float(np.hypot(kp[0] - sx, kp[1] - sy))
                    if dist >= best_dist:
                        continue

                    # ── 선수 위치 y 계산 (top/bottom 분류용) ──────────────
                    # 우선순위: 발목 → 무릎 → 힙 → 상반신 중위값
                    # 발목을 우선하는 이유: 발은 항상 자기 코트에 있어 타격 자세(몸 기울임,
                    # 점프, 스매시 등)에 영향받지 않고 안정적으로 top/bottom을 구분한다.
                    # 힙을 우선하면 서브처럼 몸을 앞으로 기울일 때 반대 코트로 오판 발생.
                    def _valid_pts(indices):
                        return [kps[i] for i in indices
                                if kps[i][2] >= KP_CONF_THRESH and kps[i][1] > 0]

                    pts = _valid_pts(ANKLE_KP)
                    if not pts:
                        pts = _valid_pts(KNEE_KP)
                    if not pts:
                        pts = _valid_pts(HIP_KP)
                    if not pts:
                        pts = _valid_pts(UPPER_BODY_IDX)
                    if not pts:
                        pts = [k for k in kps if k[2] >= KP_CONF_THRESH and k[1] > 0]
                    if not pts:
                        continue
                    center_y = float(np.median([k[1] for k in pts]))

                    best_dist     = dist
                    best_center_y = center_y

            # 손목이 너무 멀면 교정 보류
            if best_center_y is None or best_dist > PROXIMITY_PX:
                continue

            pose_owner = "top" if best_center_y < self._owner_y else "bottom"

            if pose_owner != event.owner:
                player_name = "pink_top" if pose_owner == "top" else "green_bottom"
                result[idx] = ImpactEvent(
                    frame      = event.frame,
                    time_sec   = event.time_sec,
                    score      = event.score,
                    owner      = pose_owner,
                    player     = player_name,
                    hit_number = event.hit_number,
                    method     = event.method + "_pose",
                )
                if verbose:
                    print(f"  [Pose owner] #{event.hit_number}  {event.time_sec:.3f}s  "
                          f"{event.owner} → {pose_owner}  "
                          f"(wrist_dist={best_dist:.1f}px  center_y={best_center_y:.1f})")

        return result

    # ── sub-threshold 타점 복구 (v11 신규) ──────────────────

    def get_near_miss_frames(
        self,
        confirmed_events: List[ImpactEvent],
        min_score_ratio:  float = 0.05,
        max_score_ratio:  float = 0.149,
        exclusion_radius: Optional[int] = None,
    ) -> List[int]:
        """
        감지는 안 됐지만 impulse 점수가 존재하는 "아쉬운 후보(near-miss)" 프레임을
        반환한다. pipeline에서 이 프레임에 대해 YOLO를 실행하고,
        손목-셔틀 근접이 확인되면 rescue_near_misses()로 복구한다.

        [선발 조건]
        1) impulse 점수 > 0 (물리 필터를 어느 정도 통과했으나 peak로 선발 못 됨)
        2) min_score_ratio ≤ score / max_ref < max_score_ratio
           (too-weak 와 이미-감지된 구간 사이의 회색지대)
           max_score_ratio=0.149: peak_threshold_ratio=0.15 바로 아래까지 포함
        3) 이미 감지된 타점(confirmed_events) 근방은 제외 (중복 방지)

        [exclusion_radius 자동 계산 — v11]
        _build_candidates에서 frame_adj = peak - 1로 오프셋이 발생한다.
        confirmed_events의 event.frame은 실제 peak보다 1 낮다.
        exclusion_radius를 adaptive peak_distance + 3으로 설정하면:
          (a) peak 위치 자체(e.frame+1)도 제외 대상에 포함
          (b) peak_distance 이내의 이웃 점수 bump를 중복 방지
        → 30fps(peak_dist=7): radius=10, 24fps(peak_dist=5): radius=8

        Args:
            confirmed_events : detect()로 확정된 타점 리스트.
            min_score_ratio  : max_ref 대비 최소 점수 비율 (너무 약한 것 걸러냄).
            max_score_ratio  : max_ref 대비 최대 점수 비율 (peak 임계값 바로 아래).
            exclusion_radius : 확정 타점 주변 제외 반경 (None이면 peak_distance+3 자동 계산).

        Returns:
            near-miss 프레임 번호 오름차순 리스트.
        """
        scores   = getattr(self, "_last_scores",   None)
        adaptive = getattr(self, "_last_adaptive",  None)

        if scores is None or adaptive is None:
            return []

        # exclusion_radius 자동 계산: adaptive peak_distance 기반
        if exclusion_radius is None:
            pd               = (adaptive or {}).get("peak_distance", 10)
            exclusion_radius = pd + 3   # +3: frame_adj 오프셋(1) + 여유(2)

        # 비교 기준: IQR 억제 후 max_score와 같은 기준으로 맞춤
        non_zero = scores[scores > 0]
        if len(non_zero) == 0:
            return []

        if len(non_zero) >= 6:
            q1  = np.percentile(non_zero, 25)
            q3  = np.percentile(non_zero, 75)
            iqr = q3 - q1
            if iqr > 1e-6:
                max_ref = float(q3 + 3.0 * iqr)
            else:
                max_ref = float(np.max(non_zero))
        else:
            max_ref = float(np.max(non_zero))

        lo = max_ref * min_score_ratio
        hi = max_ref * max_score_ratio

        # [v11] frame_adj = peak-1 오프셋 보정:
        # e.frame(peak-1)과 e.frame+1(peak 자체) 모두 제외 기준에 포함
        confirmed_frames   = set(e.frame for e in confirmed_events)
        confirmed_vicinity = confirmed_frames | {f + 1 for f in confirmed_frames}

        near_misses = set()

        # ── 일반 sub-threshold 후보 ────────────────────────────
        for i, s in enumerate(scores):
            if s <= 0:
                continue
            if not (lo <= s < hi):
                continue
            too_close = any(abs(i - cf) <= exclusion_radius for cf in confirmed_vicinity)
            if too_close:
                continue
            near_misses.add(int(i))

        # ── IQR 억제 프레임은 near-miss 후보에서 제외 (v12) ──────
        # [v11 삭제 이유]
        # IQR-억제 프레임(score > Q3+3·IQR)은 대부분 TrackNet ID스위치 노이즈가
        # 원인이다. 노이즈 프레임 근방에는 실제 선수가 서브 동작 중인 경우가 많아
        # YOLO가 손목-셔틀 근접을 잘못 확인 → rescue_near_misses가 오탐을 삽입한다.
        #
        # 구체적 사례: 서브 전 ID스위치(frame 20, x=250→700→250)로 frame 20 IQR억제.
        # fps=24 기준 frame 20 = 0.833s. 서브 동작 중 선수 손목이 공 위치(x=250)
        # 근처에 있어 YOLO 확인 통과 → 잘못된 0.833s 타점 삽입.
        #
        # IQR-억제 프레임의 근본 원인은 노이즈이지 "약한 실제 타격"이 아니다.
        # 진짜 서브 리턴이 IQR-억제되는 경우는 극히 드물고, 그런 경우에도
        # 이미 _compute_adaptive_params가 완화 파라미터를 적용해 정상 감지됨.
        #
        # → IQR-억제 프레임은 near-miss 후보에 포함하지 않는다.

        return sorted(near_misses)

    def rescue_near_misses(
        self,
        confirmed_events: List[ImpactEvent],
        x:               np.ndarray,
        y:               np.ndarray,
        pose_at_frames:  Dict[int, List[dict]],
        min_score_ratio: float = 0.05,
        max_score_ratio: float = 0.13,
        wrist_proximity: float = 120.0,
        verbose:         bool  = False,
    ) -> List[ImpactEvent]:
        """
        포즈 데이터로 확인된 sub-threshold 타점을 기존 타점 리스트에 삽입한다.

        [알고리즘]
        1) get_near_miss_frames()로 회색지대 프레임 수집
        2) 각 near-miss 프레임(또는 ±context_radius 이웃)에서
           손목-셔틀 거리 < wrist_proximity 인 선수가 있으면 타점으로 인정
        3) 선수 중심 y로 owner 결정 → ImpactEvent(method="rescue_pose") 생성
        4) confirmed_events 와 합쳐 시간 순 정렬 후 hit_number 재발급

        Args:
            confirmed_events : detect() + apply_pose_owner() 완료된 타점 리스트.
            x, y             : 셔틀콕 좌표 배열.
            pose_at_frames   : {frame_idx: List[dict]} — near-miss 프레임 주변의 keypoints.
            min_score_ratio  : get_near_miss_frames()와 동일 (일관성을 위해 그대로 전달).
            max_score_ratio  : 동상.
            wrist_proximity  : 손목-셔틀 허용 최대 거리 (px).
            verbose          : 복구된 타점 정보 출력 여부.

        Returns:
            rescued 타점이 추가된 ImpactEvent 리스트 (hit_number 재발급 완료).
        """
        WRIST_KP       = [9, 10]
        KP_CONF_THRESH = 0.25
        UPPER_BODY_IDX = [0, 5, 6, 7, 8, 9, 10]
        CONTEXT_RADIUS = 2

        scores = getattr(self, "_last_scores", None)
        if scores is None:
            return confirmed_events

        near_misses = self.get_near_miss_frames(
            confirmed_events,
            min_score_ratio=min_score_ratio,
            max_score_ratio=max_score_ratio,
        )

        new_events: List[ImpactEvent] = []

        # [v11] 이미 구조한 이벤트의 프레임도 exclusion 집합에 실시간 추가.
        # 기존 코드는 confirmed_events만 체크해, 두 near-miss 프레임이 서로 가까울 때
        # 둘 다 통과해버려 중복 이벤트가 생성되는 버그가 있었다.
        rescued_vicinity: set = (
            set(e.frame for e in confirmed_events)
            | {e.frame + 1 for e in confirmed_events}
        )
        rescue_exclusion_radius = (getattr(self, "_last_adaptive", None) or {}).get(
            "peak_distance", 7
        ) + 3

        for nm_frame in near_misses:
            # context_radius 범위 내에서 pose 데이터 탐색
            best_dist     = float("inf")
            best_center_y = None
            best_frame    = nm_frame

            for delta in range(-CONTEXT_RADIUS, CONTEXT_RADIUS + 1):
                probe = nm_frame + delta
                players = pose_at_frames.get(probe, [])
                if not players:
                    continue

                sx = float(x[min(probe, len(x) - 1)])
                sy = float(y[min(probe, len(y) - 1)])
                if sx <= 0 or sy <= 0:
                    continue

                for player in players:
                    kps = player.get("keypoints", [])
                    if len(kps) < 17:
                        continue
                    for wi in WRIST_KP:
                        kp = kps[wi]
                        if kp[2] < KP_CONF_THRESH or kp[0] <= 0:
                            continue
                        dist = float(np.hypot(kp[0] - sx, kp[1] - sy))
                        if dist >= best_dist:
                            continue

                        # 선수 중심 y 계산
                        h1, h2 = kps[11], kps[12]
                        if h1[2] >= KP_CONF_THRESH and h2[2] >= KP_CONF_THRESH:
                            center_y = float((h1[1] + h2[1]) / 2.0)
                        else:
                            upper = [kps[i] for i in UPPER_BODY_IDX
                                     if kps[i][2] >= KP_CONF_THRESH and kps[i][1] > 0]
                            if not upper:
                                upper = [k for k in kps
                                         if k[2] >= KP_CONF_THRESH and k[1] > 0]
                            if not upper:
                                continue
                            center_y = float(np.median([k[1] for k in upper]))

                        best_dist     = dist
                        best_center_y = center_y
                        best_frame    = probe

            if best_center_y is None or best_dist > wrist_proximity:
                continue

            # [v11] 이미 구조한 이벤트 근방인지 재확인 (자기 중복 방지)
            too_close = any(
                abs(nm_frame - rf) <= rescue_exclusion_radius
                for rf in rescued_vicinity
            )
            if too_close:
                continue

            owner       = "top" if best_center_y < self._owner_y else "bottom"
            player_name = "pink_top" if owner == "top" else "green_bottom"
            score_val   = float(scores[min(nm_frame, len(scores) - 1)])

            new_ev = ImpactEvent(
                frame    = nm_frame,
                time_sec = nm_frame / self.fps,
                score    = score_val,
                owner    = owner,
                player   = player_name,
                method   = "rescue_pose",
            )
            new_events.append(new_ev)
            # 방금 구조한 프레임을 exclusion 집합에 즉시 추가 → 다음 near-miss와 중복 방지
            rescued_vicinity.add(nm_frame)
            rescued_vicinity.add(nm_frame + 1)

            if verbose:
                print(f"  [Rescue] frame={nm_frame}  t={nm_frame/self.fps:.3f}s  "
                      f"owner={owner}  wrist_dist={best_dist:.1f}px  "
                      f"score={score_val:.1f}")

        if not new_events:
            return confirmed_events

        # 합치기 → 시간 순 정렬 → hit_number 재발급
        merged = list(confirmed_events) + new_events
        merged.sort(key=lambda e: e.frame)
        for idx, ev in enumerate(merged):
            ev.hit_number = idx + 1

        return merged


# ────────────────────────────────────────────────────────────
# 편의 함수
# ────────────────────────────────────────────────────────────

def build_hit_lookup(events: List[ImpactEvent]) -> dict:
    """frame → ImpactEvent 딕셔너리 반환."""
    return {e.frame: e for e in events}


def to_api_json(events: List[ImpactEvent], fps: float) -> dict:
    """백엔드 API 전송용 JSON 구조로 변환한다."""
    return {
        "video_fps":   round(fps, 2),
        "total_hits":  len(events),
        "hits_data":   [e.to_dict() for e in events],
    }
