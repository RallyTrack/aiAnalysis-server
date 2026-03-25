"""
RallyTrack - 스켈레톤 코트 뷰 렌더러 (v3)

[v3 변경 사항]
  1. 네트 높이 계산 방식 수정
     ─ 이전: NET_H = 250 (고정 픽셀값, 너무 높아서 셔틀이 네트 아래로 넘어가는 문제)
     ─ 수정: BWF 실제 비율로 계산
             실제 네트 높이 = 1.55m (포스트 기준)
             코트 코너(src_pts)의 실제 픽셀 높이(상단~하단 거리)를
             코트 길이(13.40m)로 나눈 비율 × 1.55m를 픽셀 높이로 환산
             → 카메라 앵글이 달라도 자동으로 적절한 네트 높이 반영

  2. 코트 라인 BWF 비율로 수정
     ─ 이전: 하드코딩된 r값 [0.0, 0.1, 0.38, 0.62, 0.9, 1.0] (부정확)
     ─ 수정: COURT_RATIOS 사용 (더블 롱서비스, 숏서비스, 네트, 센터 포함)

  3. 스켈레톤 위치 판별 로직 수정
     ─ self._net_y: shifted_pts 기반 네트 Y픽셀 정확히 계산
"""

from __future__ import annotations

import cv2
import numpy as np
from collections import deque
from typing import List, Tuple

from .config import MINIMAP_CONFIG, POSE_CONFIG, SKELETON_EDGES, BADMINTON_COURT, COURT_RATIOS
from .court  import frame_to_minimap, is_inside_court


SKELETON_VIEW_HEIGHT = 1000


class SkeletonCourtRenderer:
    """
    원근감이 살아있는 코트 위에 선수 스켈레톤과 셔틀콕 궤적을 그립니다.

    좌표 기준:
      - src_pts [TL, TR, BR, BL]: 영상 내 코트 4 코너 픽셀 좌표
      - shifted_pts: src_pts를 스켈레톤 뷰 캔버스(SKELETON_VIEW_HEIGHT×frame_w)에 맞게 이동한 좌표
      - 코트 위쪽(TL/TR)이 캔버스 상단, 아래쪽(BL/BR)이 하단에 오도록 배치
    """

    def __init__(
        self,
        frame_w:      int,
        frame_h:      int,
        homographies: dict,
        trail_length: int = MINIMAP_CONFIG["trail_length"],
    ):
        self._frame_w  = frame_w
        self._frame_h  = frame_h
        self._hg       = homographies
        self._canvas_h = SKELETON_VIEW_HEIGHT

        self._shifted_pts  = self._compute_shifted_pts()
        self._net_y        = self._compute_net_y()
        self._net_h_pixels = self._compute_net_height_pixels()

        self._trail: deque[Tuple[int, int]] = deque(maxlen=trail_length)

    # ── 초기화 헬퍼 ─────────────────────────────────────────

    def _compute_shifted_pts(self) -> np.ndarray:
        """
        영상 코너 좌표를 스켈레톤 뷰 캔버스 좌표로 이동합니다.
        캔버스 높이(SKELETON_VIEW_HEIGHT=1000px)에서 코트가 적절히 보이도록 수직 이동.
        """
        src     = self._hg["src_pts"].copy()   # [TL, TR, BR, BL]
        shifted = src.copy()
        # 영상 코트의 y 중심을 캔버스 y 중심(500px)에 맞춤
        src_cy  = (src[:, 1].min() + src[:, 1].max()) / 2
        dy      = (self._canvas_h / 2) - src_cy
        shifted[:, 1] = shifted[:, 1] + dy
        return shifted.astype(np.float32)

    def _compute_net_y(self) -> float:
        """
        shifted_pts 기준으로 네트 Y픽셀 좌표를 COURT_RATIOS["net"]으로 계산합니다.

        코트 상단(TL/TR)의 y = shifted_pts[0][1] 또는 [1][1] (평균)
        코트 하단(BL/BR)의 y = shifted_pts[3][1] 또는 [2][1] (평균)
        네트 위치: top_y + (bot_y - top_y) * COURT_RATIOS["net"]
        """
        top_y = (self._shifted_pts[0][1] + self._shifted_pts[1][1]) / 2
        bot_y = (self._shifted_pts[2][1] + self._shifted_pts[3][1]) / 2
        return top_y + (bot_y - top_y) * COURT_RATIOS["net"]   # = top_y + (bot_y - top_y) * 0.5

    def _compute_net_height_pixels(self) -> float:
        """
        BWF 네트 높이(1.55m)를 shifted_pts의 픽셀 스케일로 환산합니다.

        계산 방법:
          1. shifted_pts에서 코트 길이를 픽셀로 계산
             (TL↔BL 거리와 TR↔BR 거리의 평균)
          2. 픽셀/미터 스케일 계산
          3. net_height_m(1.55m) × 픽셀/미터 = 네트 높이 픽셀
        """
        pts = self._shifted_pts
        # 좌측 변 길이 (TL→BL)
        left_len  = np.hypot(pts[3][0] - pts[0][0], pts[3][1] - pts[0][1])
        # 우측 변 길이 (TR→BR)
        right_len = np.hypot(pts[2][0] - pts[1][0], pts[2][1] - pts[1][1])
        court_px  = (left_len + right_len) / 2   # 코트 길이(13.40m)에 대응하는 픽셀

        # 픽셀 per 미터
        px_per_m = court_px / BADMINTON_COURT["length"]   # px / 13.40m

        # 네트 높이 픽셀 (포스트 기준 1.55m)
        net_h = px_per_m * BADMINTON_COURT["net_height_post"]
        return float(net_h)

    # ── 코트 그리기 ─────────────────────────────────────────

    def _draw_court(self, canvas: np.ndarray) -> None:
        """
        원근감이 반영된 코트 라인을 그립니다.

        COURT_RATIOS를 사용하여 정확한 BWF 규격 비율로 선을 그립니다.
        r=0.0: 코트 상단(위쪽 엔드라인), r=1.0: 코트 하단(아래쪽 엔드라인)
        """
        LINE_COLOR = (60,  60, 200)   # 파란빛 흰색
        NET_COLOR  = (0,  200, 255)   # 밝은 노란색

        pts = self._shifted_pts
        p_tl, p_tr, p_br, p_bl = pts[0], pts[1], pts[2], pts[3]

        def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
            return a + (b - a) * t

        def pt_int(p: np.ndarray) -> Tuple[int, int]:
            return (int(p[0]), int(p[1]))

        r = COURT_RATIOS

        # ── 가로선 (코트 길이 방향 비율, 위→아래) ──────────
        # p_tl↔p_bl: 왼쪽 변 (r=0 → 위 엔드라인, r=1 → 아래 엔드라인)
        # p_tr↔p_br: 오른쪽 변
        h_ratios = [
            r["top_boundary"],    # 0.0   위 엔드라인 (경계이므로 사각형으로 처리)
            r["long_svc_top"],    # ≈0.057 위 더블 롱서비스
            r["short_svc_top"],   # ≈0.352 위 숏서비스
            r["net"],             # 0.5   네트 (별도 처리)
            r["short_svc_bot"],   # ≈0.648 아래 숏서비스
            r["long_svc_bot"],    # ≈0.943 아래 더블 롱서비스
            r["bottom_boundary"], # 1.0   아래 엔드라인
        ]
        for hr in h_ratios:
            lp = lerp(p_tl, p_bl, hr)
            rp = lerp(p_tr, p_br, hr)
            cv2.line(canvas, pt_int(lp), pt_int(rp), LINE_COLOR, 2)

        # ── 세로선 (코트 폭 방향 비율, 왼→오른) ─────────────
        # p_tl↔p_tr: 상단 변, p_bl↔p_br: 하단 변
        v_ratios = [
            r["left_boundary"],   # 0.0   왼쪽 더블 경계
            r["singles_left"],    # ≈0.075 왼쪽 단식 사이드라인
            r["center"],          # 0.5   센터라인
            r["singles_right"],   # ≈0.925 오른쪽 단식 사이드라인
            r["right_boundary"],  # 1.0   오른쪽 더블 경계
        ]
        for vr in v_ratios:
            tp = lerp(p_tl, p_tr, vr)
            bp = lerp(p_bl, p_br, vr)
            cv2.line(canvas, pt_int(tp), pt_int(bp), LINE_COLOR, 2)

        # ── 네트 및 포스트 ────────────────────────────────────
        # 네트 바닥 위치 (코트 정중앙)
        m_l = lerp(p_tl, p_bl, r["net"])   # 왼쪽 네트 발
        m_r = lerp(p_tr, p_br, r["net"])   # 오른쪽 네트 발

        # 네트 꼭대기: 바닥에서 위(y 감소 방향)으로 net_h_pixels만큼
        nt_l = (int(m_l[0]), int(m_l[1] - self._net_h_pixels))
        nt_r = (int(m_r[0]), int(m_r[1] - self._net_h_pixels))

        # 네트 상단 수평선
        cv2.line(canvas, nt_l, nt_r, NET_COLOR, 5)
        # 네트 포스트 (수직)
        cv2.line(canvas, pt_int(m_l), nt_l, NET_COLOR, 3)
        cv2.line(canvas, pt_int(m_r), nt_r, NET_COLOR, 3)
        # 네트 그물 느낌 (중간 수평선 몇 개)
        for t in [0.33, 0.67]:
            nl_mid = (int(m_l[0]), int(m_l[1] - self._net_h_pixels * t))
            nr_mid = (int(m_r[0]), int(m_r[1] - self._net_h_pixels * t))
            cv2.line(canvas, nl_mid, nr_mid, NET_COLOR, 1)

    # ── 퍼블릭 API ──────────────────────────────────────────

    def render_frame(
        self,
        frame_idx:      int,
        shuttle_x:      float,
        shuttle_y:      float,
        keypoints_list: list,
    ) -> np.ndarray:
        canvas = np.zeros((self._canvas_h, self._frame_w, 3), dtype=np.uint8)
        # 코트 배경 (어두운 녹색)
        canvas[:] = (20, 55, 20)
        self._draw_court(canvas)

        self._update_shuttle(shuttle_x, shuttle_y)
        self._draw_shuttle_trail(canvas)

        parsed_list = _parse_keypoints_list(keypoints_list)
        self._draw_skeletons(canvas, parsed_list)

        cv2.putText(
            canvas, f"F:{frame_idx}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 180), 2, cv2.LINE_AA,
        )
        return canvas

    # ── 내부 렌더링 ─────────────────────────────────────────

    def _update_shuttle(self, sx: float, sy: float) -> None:
        if sx <= 0 or sy <= 0:
            return
        pt  = np.array([[[sx, sy]]], dtype=np.float32)
        shifted_matrix = cv2.getPerspectiveTransform(
            self._hg["src_pts"],
            self._shifted_pts,
        )
        res = cv2.perspectiveTransform(pt, shifted_matrix)[0][0]
        self._trail.append((int(res[0]), int(res[1])))

    def _draw_shuttle_trail(self, canvas: np.ndarray) -> None:
        n = len(self._trail)
        if n == 0:
            return
        for i, (px, py) in enumerate(self._trail):
            alpha = (i + 1) / n
            radius = max(3, int(4 + 6 * alpha))
            blue   = int(200 * alpha)
            cv2.circle(canvas, (px, py), radius, (blue, 200, 255), -1)

    def _draw_skeletons(self, canvas: np.ndarray, keypoints_list: list) -> None:
        cfg = POSE_CONFIG
        shifted_matrix = cv2.getPerspectiveTransform(
            self._hg["src_pts"],
            self._shifted_pts,
        )

        for person_kpts in keypoints_list:
            if len(person_kpts) < 17:
                continue

            l_kp = person_kpts[15]
            r_kp = person_kpts[16]
            f_x, f_y = _get_foot_center(l_kp, r_kp, cfg["keypoint_conf"])
            if f_x <= 0 or f_y <= 0:
                continue

            pt  = np.array([[[f_x, f_y]]], dtype=np.float32)
            dst = cv2.perspectiveTransform(pt, shifted_matrix)[0][0]
            tx, ty = float(dst[0]), float(dst[1])

            # 코트 폴리곤 안쪽인지 확인
            poly = self._shifted_pts.astype(np.int32)
            if cv2.pointPolygonTest(poly, (tx, ty), False) < 0:
                continue

            # 네트 기준 색상 결정 (net_y 기준 위쪽=top=핑크, 아래쪽=bottom=라임)
            color = (
                MINIMAP_CONFIG["top_color"]
                if ty < self._net_y
                else MINIMAP_CONFIG["bottom_color"]
            )

            # 뼈대 그리기
            for p1_i, p2_i in SKELETON_EDGES:
                kp1 = person_kpts[p1_i - 1]
                kp2 = person_kpts[p2_i - 1]
                if len(kp1) < 3 or len(kp2) < 3:
                    continue
                if float(kp1[2]) < cfg["keypoint_conf"] or float(kp2[2]) < cfg["keypoint_conf"]:
                    continue
                x1 = int(tx + (float(kp1[0]) - f_x) * cfg["scale"])
                y1 = int(ty + (float(kp1[1]) - f_y) * cfg["scale"])
                x2 = int(tx + (float(kp2[0]) - f_x) * cfg["scale"])
                y2 = int(ty + (float(kp2[1]) - f_y) * cfg["scale"])
                cv2.line(canvas, (x1, y1), (x2, y2), color, cfg["bone_thickness"])
                cv2.circle(canvas, (x1, y1), 3, (255, 255, 255), -1)

            # 머리
            nose = person_kpts[0]
            if len(nose) >= 3 and float(nose[2]) >= cfg["keypoint_conf"]:
                hx = int(tx + (float(nose[0]) - f_x) * cfg["scale"])
                hy = int(ty + (float(nose[1]) - f_y) * cfg["scale"])
                r  = cfg["head_radius"]
                cv2.circle(canvas, (hx, hy), r, color,           -1)
                cv2.circle(canvas, (hx, hy), r, (255, 255, 255),  2)


# ────────────────────────────────────────────────────────────
# 모듈 수준 헬퍼
# ────────────────────────────────────────────────────────────

def _parse_keypoints_list(keypoints_list: list) -> list:
    result = []
    for person in keypoints_list:
        if isinstance(person, dict):
            kpts = person.get("keypoints", [])
        else:
            kpts = person.tolist() if hasattr(person, "tolist") else list(person)
        result.append(kpts)
    return result


def _get_foot_center(l_kp, r_kp, kconf: float) -> Tuple[float, float]:
    try:
        lx, ly, lc = float(l_kp[0]), float(l_kp[1]), float(l_kp[2])
        rx, ry, rc = float(r_kp[0]), float(r_kp[1]), float(r_kp[2])
    except (IndexError, TypeError):
        return 0.0, 0.0

    if lc >= kconf and rc >= kconf:
        return (lx + rx) / 2, (ly + ry) / 2
    elif lc >= kconf:
        return lx, ly
    elif rc >= kconf:
        return rx, ry
    return 0.0, 0.0
