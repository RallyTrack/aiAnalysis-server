"""
RallyTrack - 스켈레톤 코트 뷰 렌더러 (3번째 출력 영상)

Separation.ipynb / ShuttleMap.ipynb의 out_skeleton 에 해당합니다.

원본 영상의 원근감(perspective)을 유지하면서 코트 + 스켈레톤 +
셔틀콕 궤적을 하나의 1000px 높이 캔버스에 합성합니다.

미니맵(360×600 Top-Down 뷰)과는 별개의 영상입니다:
  - 미니맵:       2D 수직 시점 (Top-Down)
  - 스켈레톤 뷰:  원근감 있는 코트 시점 (Perspective Court View)
"""

from __future__ import annotations

import cv2
import numpy as np
from collections import deque
from typing import List, Tuple

from .config import MINIMAP_CONFIG, POSE_CONFIG, SKELETON_EDGES
from .court  import frame_to_minimap, is_inside_court


# 스켈레톤 코트 뷰 캔버스 고정 높이
SKELETON_VIEW_HEIGHT = 1000


class SkeletonCourtRenderer:
    """
    ShuttleMap.ipynb의 draw_pro_court + 스켈레톤 합성 로직을 구현합니다.

    원근감이 살아있는 코트 위에 선수 스켈레톤과 셔틀콕 궤적을 그립니다.
    shifted_dst_pts(원본 코트 코너를 Y축 평행이동시킨 좌표)를 기준으로
    선수 위치가 코트 내부인지 판단합니다.

    [원본과의 차이]
    원본(ShuttleMap.ipynb)은 frame_w × 1000 크기 캔버스를 사용했습니다.
    여기서는 frame_w를 생성 시점에 주입받아 동일하게 처리합니다.
    """

    def __init__(
        self,
        frame_w:       int,
        frame_h:       int,
        homographies:  dict,
        trail_length:  int  = MINIMAP_CONFIG["trail_length"],
    ):
        self._frame_w  = frame_w
        self._frame_h  = frame_h
        self._hg       = homographies
        self._canvas_h = SKELETON_VIEW_HEIGHT

        # 원근감 코트용 shifted_dst_pts 계산
        # (원본의 "src_pts를 Y축으로 올린 뒤 400px 내린" 방식 재현)
        self._shifted_pts = self._compute_shifted_pts()
        self._net_y       = self._compute_net_y()

        # 셔틀콕 궤적
        self._trail: deque[Tuple[int, int]] = deque(maxlen=trail_length)

        # 선수 경로
        self._top_path:    deque[Tuple[int, int]] = deque(maxlen=200)
        self._bottom_path: deque[Tuple[int, int]] = deque(maxlen=200)

    # ── 초기화 헬퍼 ─────────────────────────────────────────

    def _compute_shifted_pts(self) -> np.ndarray:
        """
        ShuttleMap.ipynb의 shifted_dst_pts 재현:
          dst_pts_base = src_pts - [0, frame_h * 0.4]
          shifted      = dst_pts_base + [0, 400]
        이 방식은 원근감을 유지하면서 코트를 캔버스 중앙에 배치합니다.
        """
        src = self._hg["src_pts"].copy()
        shifted = src.copy()
        shifted[:, 1] = shifted[:, 1] - self._frame_h * 0.4 + 400
        return shifted.astype(np.float32)

    def _compute_net_y(self) -> float:
        """네트 Y 좌표: 상단 두 코너의 Y 평균."""
        return (self._shifted_pts[0][1] + self._shifted_pts[3][1]) / 2

    # ── 코트 그리기 ─────────────────────────────────────────

    def _draw_court(self, canvas: np.ndarray) -> None:
        """
        ShuttleMap.ipynb의 draw_pro_court 재현.
        원근감 코트 라인 + 네트를 그립니다.
        """
        RED    = (0,   0,   255)
        YELLOW = (0, 200, 255)

        pts = self._shifted_pts
        p1, p2, p3, p4 = pts[0], pts[1], pts[2], pts[3]

        def lerp(a, b, t):
            return a + (b - a) * t

        # 가로 라인 (코트 측면 경계 포함, 서비스/백바운더리 라인)
        for r in [0.0, 0.1, 0.38, 0.62, 0.9, 1.0]:
            lp = lerp(p1, p4, r)
            rp = lerp(p2, p3, r)
            cv2.line(canvas, tuple(lp.astype(int)), tuple(rp.astype(int)), RED, 2)

        # 세로 라인
        for r in [0.0, 0.05, 0.5, 0.95, 1.0]:
            tp = lerp(p1, p2, r)
            bp = lerp(p4, p3, r)
            cv2.line(canvas, tuple(tp.astype(int)), tuple(bp.astype(int)), RED, 2)

        # 네트 (중앙 가로선 기준 위쪽으로 250px)
        NET_H = 250
        m_l = lerp(p1, p4, 0.5)
        m_r = lerp(p2, p3, 0.5)
        nt_l = (int(m_l[0]), int(m_l[1] - NET_H))
        nt_r = (int(m_r[0]), int(m_r[1] - NET_H))
        cv2.line(canvas, nt_l, nt_r, YELLOW, 6)
        cv2.line(canvas, tuple(m_l.astype(int)), nt_l, YELLOW, 3)
        cv2.line(canvas, tuple(m_r.astype(int)), nt_r, YELLOW, 3)

    # ── 퍼블릭 API ──────────────────────────────────────────

    def render_frame(
        self,
        frame_idx:      int,
        shuttle_x:      float,
        shuttle_y:      float,
        keypoints_list: list,
    ) -> np.ndarray:
        """
        스켈레톤 코트 뷰 한 프레임을 렌더링합니다.

        Returns:
            (SKELETON_VIEW_HEIGHT, frame_w, 3) uint8 이미지
        """
        canvas = np.zeros((self._canvas_h, self._frame_w, 3), dtype=np.uint8)
        self._draw_court(canvas)

        # 셔틀콕 궤적 업데이트 및 그리기
        self._update_shuttle(shuttle_x, shuttle_y)
        self._draw_shuttle_trail(canvas)

        # 선수 스켈레톤 투영
        self._draw_skeletons(canvas, keypoints_list)

        # 프레임 번호
        cv2.putText(
            canvas, f"F:{frame_idx}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 180), 2, cv2.LINE_AA,
        )

        return canvas

    # ── 내부 렌더링 ─────────────────────────────────────────

    def _update_shuttle(self, sx: float, sy: float) -> None:
        """셔틀콕 좌표를 스켈레톤 캔버스 좌표로 변환 후 저장합니다."""
        if sx <= 0 or sy <= 0:
            return
        # 원본 영상 좌표 → shifted 코트 좌표로 호모그래피 변환
        pt  = np.array([[[sx, sy]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(pt, self._hg["to_minimap"])[0][0]

        # to_minimap은 미니맵 좌표계이므로, 스켈레톤 뷰에서는
        # 원본 좌표를 shifted_pts 기반 변환 행렬로 다시 계산
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
            alpha  = (i + 1) / n
            r      = max(3, int(4 + 6 * alpha))
            blue   = int(200 * alpha)
            cv2.circle(canvas, (px, py), r, (blue, 200, 255), -1)

    def _draw_skeletons(self, canvas: np.ndarray, keypoints_list: list) -> None:
        """
        선수 스켈레톤을 shifted 코트 좌표로 투영하여 그립니다.

        ShuttleMap.ipynb의 로직:
          1. 양 발목 중심 f_x, f_y 계산
          2. getPerspectiveTransform으로 shifted 좌표 tx, ty 변환
          3. 코트 폴리곤(shifted_pts) 내부인지 확인
          4. 각 관절을 (tx + (joint_x - f_x) * scale, ty + (joint_y - f_y) * scale) 위치에 그림
        """
        cfg            = POSE_CONFIG
        shifted_matrix = cv2.getPerspectiveTransform(
            self._hg["src_pts"],
            self._shifted_pts,
        )

        for person in keypoints_list:
            # 발목 중심
            l_kp = person[15] if len(person) > 15 else [0, 0, 0]
            r_kp = person[16] if len(person) > 16 else [0, 0, 0]
            f_x, f_y = self._get_foot_center(l_kp, r_kp, cfg["keypoint_conf"])
            if f_x <= 0 or f_y <= 0:
                continue

            # shifted 코트 좌표로 변환
            pt    = np.array([[[f_x, f_y]]], dtype=np.float32)
            dst   = cv2.perspectiveTransform(pt, shifted_matrix)[0][0]
            tx, ty = float(dst[0]), float(dst[1])

            # shifted_pts 폴리곤 내부 여부 확인
            poly = self._shifted_pts.astype(np.int32)
            if cv2.pointPolygonTest(poly, (tx, ty), False) < 0:
                continue

            # 네트 기준 색상
            color = (
                MINIMAP_CONFIG["top_color"]
                if ty < self._net_y
                else MINIMAP_CONFIG["bottom_color"]
            )

            # 뼈대 그리기
            for p1_i, p2_i in SKELETON_EDGES:
                kp1 = person[p1_i - 1] if len(person) >= p1_i else [0, 0, 0]
                kp2 = person[p2_i - 1] if len(person) >= p2_i else [0, 0, 0]
                if kp1[2] < cfg["keypoint_conf"] or kp2[2] < cfg["keypoint_conf"]:
                    continue
                x1 = int(tx + (kp1[0] - f_x) * cfg["scale"])
                y1 = int(ty + (kp1[1] - f_y) * cfg["scale"])
                x2 = int(tx + (kp2[0] - f_x) * cfg["scale"])
                y2 = int(ty + (kp2[1] - f_y) * cfg["scale"])
                cv2.line(canvas, (x1, y1), (x2, y2), color, cfg["bone_thickness"])
                cv2.circle(canvas, (x1, y1), 3, (255, 255, 255), -1)

            # 머리 (Nose = 인덱스 0)
            nose = person[0] if len(person) > 0 else [0, 0, 0]
            if nose[2] >= cfg["keypoint_conf"]:
                hx = int(tx + (nose[0] - f_x) * cfg["scale"])
                hy = int(ty + (nose[1] - f_y) * cfg["scale"])
                r  = cfg["head_radius"]
                cv2.circle(canvas, (hx, hy), r, color,        -1)
                cv2.circle(canvas, (hx, hy), r, (255,255,255), 2)

    @staticmethod
    def _get_foot_center(
        l_kp: list,
        r_kp: list,
        kconf: float,
    ) -> Tuple[float, float]:
        l_x, l_y, l_c = l_kp[0], l_kp[1], l_kp[2]
        r_x, r_y, r_c = r_kp[0], r_kp[1], r_kp[2]
        if l_c >= kconf and r_c >= kconf:
            return (l_x + r_x) / 2, (l_y + r_y) / 2
        elif l_c >= kconf:
            return float(l_x), float(l_y)
        elif r_c >= kconf:
            return float(r_x), float(r_y)
        return 0.0, 0.0
