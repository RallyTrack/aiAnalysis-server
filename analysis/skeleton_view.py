"""
RallyTrack - 스켈레톤 코트 뷰 렌더러 (v2)

[이번 수정]
  extract_keypoints가 dict 형식({"track_id":..., "keypoints":...})을
  반환하도록 변경된 이후 skeleton_view.py가 raw ndarray를 기대해서
  선수가 사라지는 버그 수정.

  → _parse_keypoints() 함수로 dict / raw ndarray / raw list 모두 처리
"""

from __future__ import annotations

import cv2
import numpy as np
from collections import deque
from typing import List, Tuple

from .config import MINIMAP_CONFIG, POSE_CONFIG, SKELETON_EDGES
from .court  import frame_to_minimap, is_inside_court


SKELETON_VIEW_HEIGHT = 1000


class SkeletonCourtRenderer:
    """
    원근감이 살아있는 코트 위에 선수 스켈레톤과 셔틀콕 궤적을 그립니다.
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

        self._shifted_pts = self._compute_shifted_pts()
        self._net_y       = self._compute_net_y()

        self._trail: deque[Tuple[int, int]] = deque(maxlen=trail_length)
        self._top_path:    deque[Tuple[int, int]] = deque(maxlen=200)
        self._bottom_path: deque[Tuple[int, int]] = deque(maxlen=200)

    # ── 초기화 헬퍼 ─────────────────────────────────────────

    def _compute_shifted_pts(self) -> np.ndarray:
        src     = self._hg["src_pts"].copy()
        shifted = src.copy()
        shifted[:, 1] = shifted[:, 1] - self._frame_h * 0.4 + 400
        return shifted.astype(np.float32)

    def _compute_net_y(self) -> float:
        return (self._shifted_pts[0][1] + self._shifted_pts[3][1]) / 2

    # ── 코트 그리기 ─────────────────────────────────────────

    def _draw_court(self, canvas: np.ndarray) -> None:
        RED    = (0,   0,   255)
        YELLOW = (0, 200, 255)

        pts = self._shifted_pts
        p1, p2, p3, p4 = pts[0], pts[1], pts[2], pts[3]

        def lerp(a, b, t):
            return a + (b - a) * t

        for r in [0.0, 0.1, 0.38, 0.62, 0.9, 1.0]:
            lp = lerp(p1, p4, r)
            rp = lerp(p2, p3, r)
            cv2.line(canvas, tuple(lp.astype(int)), tuple(rp.astype(int)), RED, 2)

        for r in [0.0, 0.05, 0.5, 0.95, 1.0]:
            tp = lerp(p1, p2, r)
            bp = lerp(p4, p3, r)
            cv2.line(canvas, tuple(tp.astype(int)), tuple(bp.astype(int)), RED, 2)

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
        canvas = np.zeros((self._canvas_h, self._frame_w, 3), dtype=np.uint8)
        self._draw_court(canvas)

        self._update_shuttle(shuttle_x, shuttle_y)
        self._draw_shuttle_trail(canvas)

        # ★ dict / raw 두 형식 모두 처리
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
            r     = max(3, int(4 + 6 * alpha))
            blue  = int(200 * alpha)
            cv2.circle(canvas, (px, py), r, (blue, 200, 255), -1)

    def _draw_skeletons(self, canvas: np.ndarray, keypoints_list: list) -> None:
        """
        keypoints_list: 각 원소가 (N, 3) 형태의 list 또는 ndarray
        """
        cfg            = POSE_CONFIG
        shifted_matrix = cv2.getPerspectiveTransform(
            self._hg["src_pts"],
            self._shifted_pts,
        )

        for person_kpts in keypoints_list:
            # person_kpts: [[x, y, conf], ...] 17개
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

            poly = self._shifted_pts.astype(np.int32)
            if cv2.pointPolygonTest(poly, (tx, ty), False) < 0:
                continue

            color = (
                MINIMAP_CONFIG["top_color"]
                if ty < self._net_y
                else MINIMAP_CONFIG["bottom_color"]
            )

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

            nose = person_kpts[0]
            if len(nose) >= 3 and float(nose[2]) >= cfg["keypoint_conf"]:
                hx = int(tx + (float(nose[0]) - f_x) * cfg["scale"])
                hy = int(ty + (float(nose[1]) - f_y) * cfg["scale"])
                r  = cfg["head_radius"]
                cv2.circle(canvas, (hx, hy), r, color,         -1)
                cv2.circle(canvas, (hx, hy), r, (255, 255, 255), 2)


# ────────────────────────────────────────────────────────────
# 모듈 수준 헬퍼 함수
# ────────────────────────────────────────────────────────────

def _parse_keypoints_list(keypoints_list: list) -> list:
    """
    dict 형식과 raw 형식을 모두 raw [[x,y,c],...] 형태로 통일합니다.

    입력 형식 A (dict):
        [{"track_id": 1, "keypoints": [[x,y,c],...]}, ...]
    입력 형식 B (raw ndarray/list):
        [[[x,y,c], ...], ...]

    반환: [[[x,y,c], ...], ...]  (각 원소가 키포인트 list)
    """
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