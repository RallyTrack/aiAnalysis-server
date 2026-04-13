"""
RallyTrack - 스켈레톤 코트 뷰 렌더러 (v4)

[이번 수정]
  1. PlayerStabilizer: Y축 기반 Top/Bottom 강제 할당 + 최대 3프레임 Ghosting
     → 네트 근처 교차 시 ID 스위칭 방지, 짧은 소멸 시 깜빡임 방지
  2. BWF 표준 코트 렌더링: perspective projection으로 정확한 라인 위치
     → court.py의 BWF_X, BWF_Y 상수 기반으로 더블 라인·숏 서비스 라인·센터 라인 정확히 투영
  3. 네트 상단 좌표 기반 정확한 투영:
     → 사용자가 입력한 net_coords(Net-L, Net-R)는 네트의 최상단(Top edge) 좌표
     → 셔틀콕과 동일한 shifted_matrix로 변환해 시각적 일관성 보장
  4. 브랜드 색상 통일: Top = Royal Blue (#3B82F6), Bottom = Amber Gold (#F59E0B)
"""

from __future__ import annotations

import cv2
import numpy as np
from collections import deque
from typing import List, Optional, Tuple

from .config import MINIMAP_CONFIG, POSE_CONFIG, SKELETON_EDGES
from .court  import frame_to_minimap, is_inside_court, BWF_X, BWF_Y


SKELETON_VIEW_HEIGHT = 1000

# Top / Bottom 색상 — config.py MINIMAP_CONFIG에서 통일 관리
COLOR_TOP    = tuple(MINIMAP_CONFIG["top_color"])    # Royal Blue  #3B82F6
COLOR_BOTTOM = tuple(MINIMAP_CONFIG["bottom_color"]) # Amber Gold  #F59E0B

# 이 프레임 수 이하 동안 미감지 시 ghost(이전 위치) 유지
GHOST_MAX_FRAMES = 3


# ────────────────────────────────────────────────────────────
# 선수 안정화 추적기
# ────────────────────────────────────────────────────────────

class PlayerStabilizer:
    """
    프레임 간 Top/Bottom 플레이어 역할을 Y좌표로 강제 고정합니다.

    - 네트 Y 기준으로 위쪽 = Top, 아래쪽 = Bottom 할당
    - 두 선수가 같은 반쪽에 몰리면 Y가 작은 사람을 Top으로 강제 분리
    - 선수가 GHOST_MAX_FRAMES 프레임 이하 동안 사라지면 마지막 위치로 유지
    """

    def __init__(self, net_y_func):
        self._net_y_func  = net_y_func   # callable: () → float

        self._top:         Optional[dict] = None   # {"keypoints","tx","ty","fx","fy"}
        self._bottom:      Optional[dict] = None
        self._top_gone:    int            = 0
        self._bottom_gone: int            = 0

    def update(self, persons: list) -> Tuple[Optional[dict], Optional[dict]]:
        """
        persons: [{"keypoints", "tx", "ty", "fx", "fy"}, ...] — 이미 캔버스 좌표 포함
        returns: (top_data, bottom_data) — ghost 포함 가능
        """
        net_y = self._net_y_func()

        top_cands    = [p for p in persons if p["ty"] < net_y]
        bottom_cands = [p for p in persons if p["ty"] >= net_y]

        # 두 명 이상인데 모두 같은 쪽에 몰린 경우 → Y로 강제 분리
        if len(persons) >= 2:
            if not top_cands:
                by_y = sorted(persons, key=lambda p: p["ty"])
                top_cands    = [by_y[0]]
                bottom_cands = by_y[1:]
            elif not bottom_cands:
                by_y = sorted(persons, key=lambda p: p["ty"])
                top_cands    = by_y[:-1]
                bottom_cands = [by_y[-1]]

        # 각 역할에서 대표 한 명 선택 (top: 화면 가장 위, bottom: 화면 가장 아래)
        if top_cands:
            self._top      = min(top_cands, key=lambda p: p["ty"])
            self._top_gone = 0
        else:
            self._top_gone += 1

        if bottom_cands:
            self._bottom      = max(bottom_cands, key=lambda p: p["ty"])
            self._bottom_gone = 0
        else:
            self._bottom_gone += 1

        # Ghost 반환: 허용 프레임 초과 시 None
        top_out    = self._top    if self._top_gone    <= GHOST_MAX_FRAMES else None
        bottom_out = self._bottom if self._bottom_gone <= GHOST_MAX_FRAMES else None
        return top_out, bottom_out


# ────────────────────────────────────────────────────────────
# 메인 렌더러
# ────────────────────────────────────────────────────────────

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
        net_coords:   Optional[List[List[float]]] = None,
    ):
        self._frame_w  = frame_w
        self._frame_h  = frame_h
        self._hg       = homographies
        self._canvas_h = SKELETON_VIEW_HEIGHT

        self._shifted_pts   = self._compute_shifted_pts()
        self._bwf_H         = self._build_bwf_transform()   # BWF(m) → 캔버스(px)
        self._net_y         = self._compute_net_y()
        self._net_coords    = net_coords  # 사용자 지정 네트 상단 2점 [[L],[R]], 없으면 None

        self._shifted_matrix = cv2.getPerspectiveTransform(
            self._hg["src_pts"],
            self._shifted_pts,
        )

        self._trail:       deque[Tuple[int, int]] = deque(maxlen=trail_length)
        self._top_path:    deque[Tuple[int, int]] = deque(maxlen=200)
        self._bottom_path: deque[Tuple[int, int]] = deque(maxlen=200)

        self._stabilizer = PlayerStabilizer(lambda: self._net_y)

    # ── 초기화 헬퍼 ──────────────────────────────────────────

    def _compute_shifted_pts(self) -> np.ndarray:
        src     = self._hg["src_pts"].copy()
        shifted = src.copy()
        shifted[:, 1] = shifted[:, 1] - self._frame_h * 0.4 + 400
        return shifted.astype(np.float32)

    def _build_bwf_transform(self) -> np.ndarray:
        """
        BWF 실제 좌표(m) → 스켈레톤 캔버스 픽셀 좌표 변환 행렬.

        src_pts / shifted_pts 의 4 코너 순서 [TL, TR, BR, BL] 는
        BWF 단식 코트의 각 모서리에 대응합니다:
          TL(idx=0) → BWF (singles_left=0.46,  end_far=13.40)
          TR(idx=1) → BWF (singles_right=5.64, end_far=13.40)
          BR(idx=2) → BWF (singles_right=5.64, end_near=0.00)
          BL(idx=3) → BWF (singles_left=0.46,  end_near=0.00)
        """
        bwf_src = np.array([
            [BWF_X["singles_left"],  BWF_Y["end_far"]],    # TL
            [BWF_X["singles_right"], BWF_Y["end_far"]],    # TR
            [BWF_X["singles_right"], BWF_Y["end_near"]],   # BR
            [BWF_X["singles_left"],  BWF_Y["end_near"]],   # BL
        ], dtype=np.float32)
        return cv2.getPerspectiveTransform(bwf_src, self._shifted_pts)

    def _compute_net_y(self) -> float:
        """BWF 네트 위치(Y=6.70m) 를 캔버스 픽셀 Y로 변환."""
        return float(self._bwf_pt(BWF_X["center"], BWF_Y["net"])[1])

    def _bwf_pt(self, x_m: float, y_m: float) -> Tuple[int, int]:
        """BWF 좌표(m) → 캔버스 픽셀 좌표."""
        pt  = np.array([[[x_m, y_m]]], dtype=np.float32)
        res = cv2.perspectiveTransform(pt, self._bwf_H)[0][0]
        return int(res[0]), int(res[1])

    # ── 코트 그리기 ──────────────────────────────────────────

    def _draw_court(self, canvas: np.ndarray) -> None:
        """
        BWF 표준 규격(13.4m × 6.1m)에 따라 perspective projection으로
        코트를 그립니다.

        바닥 채색 (Two-tone Green):
          - 더블 코트 폴리곤: Dark Green (#064E3B) — 복식 앨리(단식 아웃)
          - 단식 유효 폴리곤: Bright Green (#4ADE80) — 단식 인 영역
        """
        LINE      = (255, 255, 255)                        # 코트 라인 — 흰색
        NET_C     = tuple(MINIMAP_CONFIG["net_color"])     # 미니맵과 동일한 네트 색상
        C_SINGLES = tuple(MINIMAP_CONFIG["court_singles_color"])   # Bright Green
        C_DOUBLES = tuple(MINIMAP_CONFIG["court_doubles_color"])   # Dark  Green
        lw        = 2

        # ── 바닥 채색: 더블 영역 → Dark Green (앨리 포함) ────
        d_corners = np.array([
            self._bwf_pt(BWF_X["doubles_left"],  BWF_Y["end_far"]),
            self._bwf_pt(BWF_X["doubles_right"], BWF_Y["end_far"]),
            self._bwf_pt(BWF_X["doubles_right"], BWF_Y["end_near"]),
            self._bwf_pt(BWF_X["doubles_left"],  BWF_Y["end_near"]),
        ], dtype=np.int32)
        cv2.fillPoly(canvas, [d_corners], C_DOUBLES)

        # ── 바닥 채색: 단식 유효 영역 → Bright Green ─────────
        s_corners = np.array([
            self._bwf_pt(BWF_X["singles_left"],  BWF_Y["end_far"]),
            self._bwf_pt(BWF_X["singles_right"], BWF_Y["end_far"]),
            self._bwf_pt(BWF_X["singles_right"], BWF_Y["end_near"]),
            self._bwf_pt(BWF_X["singles_left"],  BWF_Y["end_near"]),
        ], dtype=np.int32)
        cv2.fillPoly(canvas, [s_corners], C_SINGLES)

        # ── 더블 코트 외곽선 ─────────────────────────────────
        cv2.polylines(canvas, [d_corners], isClosed=True, color=LINE, thickness=lw)

        # ── 단식 사이드 라인 ─────────────────────────────────
        for x in [BWF_X["singles_left"], BWF_X["singles_right"]]:
            cv2.line(canvas,
                     self._bwf_pt(x, BWF_Y["end_far"]),
                     self._bwf_pt(x, BWF_Y["end_near"]),
                     LINE, lw)

        # ── 더블 롱 서비스 라인 (엔드 라인 안쪽 0.76m) ───────
        for y in [BWF_Y["back_service_near"], BWF_Y["back_service_far"]]:
            cv2.line(canvas,
                     self._bwf_pt(BWF_X["doubles_left"],  y),
                     self._bwf_pt(BWF_X["doubles_right"], y),
                     LINE, lw)

        # ── 숏 서비스 라인 (BWF 규격: 복식 사이드라인 전체 폭) ─
        for y in [BWF_Y["short_service_near"], BWF_Y["short_service_far"]]:
            cv2.line(canvas,
                     self._bwf_pt(BWF_X["doubles_left"],  y),
                     self._bwf_pt(BWF_X["doubles_right"], y),
                     LINE, lw)

        # ── 센터 라인 (BWF 규격: 숏 서비스 라인 → 엔드라인, 네트 구간 제외) ─
        cv2.line(canvas,
                 self._bwf_pt(BWF_X["center"], BWF_Y["end_far"]),
                 self._bwf_pt(BWF_X["center"], BWF_Y["short_service_far"]),
                 LINE, lw)
        cv2.line(canvas,
                 self._bwf_pt(BWF_X["center"], BWF_Y["short_service_near"]),
                 self._bwf_pt(BWF_X["center"], BWF_Y["end_near"]),
                 LINE, lw)

        # ── 네트 (수직 폴 + 수평 라인) ───────────────────────
        # 네트 바닥: BWF Y=6.70m 위치 (코트 중앙선)
        m_l = self._bwf_pt(BWF_X["doubles_left"],  BWF_Y["net"])
        m_r = self._bwf_pt(BWF_X["doubles_right"], BWF_Y["net"])

        if self._net_coords is not None:
            # 사용자가 지정한 Net-L / Net-R 좌표 = 네트의 최상단(Top edge)
            # 셔틀콕과 동일한 shifted_matrix로 변환 → 시각적 일관성 보장
            nl = np.array([[[float(self._net_coords[0][0]), float(self._net_coords[0][1])]]], dtype=np.float32)
            nr = np.array([[[float(self._net_coords[1][0]), float(self._net_coords[1][1])]]], dtype=np.float32)
            nt_l = tuple(cv2.perspectiveTransform(nl, self._shifted_matrix)[0][0].astype(int).tolist())
            nt_r = tuple(cv2.perspectiveTransform(nr, self._shifted_matrix)[0][0].astype(int).tolist())
        else:
            # net_coords 미제공 시 고정 오프셋으로 시각적 네트 표현
            NET_H = 250
            nt_l = (m_l[0], m_l[1] - NET_H)
            nt_r = (m_r[0], m_r[1] - NET_H)

        cv2.line(canvas, nt_l, nt_r, NET_C, 6)
        cv2.line(canvas, m_l,  nt_l, NET_C, 3)
        cv2.line(canvas, m_r,  nt_r, NET_C, 3)

    # ── 낙하 지점 X 마크 ───────────────────────────────────────

    def draw_drop_mark(
        self,
        canvas:   np.ndarray,
        frame_sx: float,
        frame_sy: float,
        is_in:    bool,
    ) -> None:
        """
        셔틀콕 낙하 지점에 X 마크를 스켈레톤 캔버스 위에 그린다.

        Args:
            canvas:   렌더링 대상 캔버스.
            frame_sx: 원본 영상 기준 셔틀콕 x 좌표.
            frame_sy: 원본 영상 기준 셔틀콕 y 좌표.
            is_in:    True이면 IN(초록), False이면 OUT(빨강).
        """
        pt  = np.array([[[frame_sx, frame_sy]]], dtype=np.float32)
        res = cv2.perspectiveTransform(pt, self._shifted_matrix)[0][0]
        cx, cy = int(res[0]), int(res[1])

        color = (0, 255, 128) if is_in else (0, 0, 255)
        size  = 22
        thick = 4
        cv2.line(canvas, (cx - size, cy - size), (cx + size, cy + size),
                 color, thick, cv2.LINE_AA)
        cv2.line(canvas, (cx - size, cy + size), (cx + size, cy - size),
                 color, thick, cv2.LINE_AA)

    # ── 공개 렌더 메서드 ──────────────────────────────────────

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

        parsed_list = _parse_keypoints_list(keypoints_list)
        self._draw_skeletons(canvas, parsed_list)

        cv2.putText(
            canvas, f"F:{frame_idx}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 180), 2, cv2.LINE_AA,
        )

        return canvas

    # ── 셔틀콕 궤적 ──────────────────────────────────────────

    def _update_shuttle(self, sx: float, sy: float) -> None:
        if sx <= 0 or sy <= 0:
            return
        pt  = np.array([[[sx, sy]]], dtype=np.float32)
        res = cv2.perspectiveTransform(pt, self._shifted_matrix)[0][0]
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

    # ── 스켈레톤 그리기 ───────────────────────────────────────

    def _draw_skeletons(self, canvas: np.ndarray, keypoints_list: list) -> None:
        cfg  = POSE_CONFIG
        poly = self._shifted_pts.astype(np.int32)

        # 1) 유효 인물 수집 (코트 안에 발이 있는 사람만)
        persons = []
        for person_kpts in keypoints_list:
            if len(person_kpts) < 17:
                continue
            l_kp = person_kpts[15]
            r_kp = person_kpts[16]
            f_x, f_y = _get_foot_center(l_kp, r_kp, cfg["keypoint_conf"])
            if f_x <= 0 or f_y <= 0:
                continue
            pt  = np.array([[[f_x, f_y]]], dtype=np.float32)
            dst = cv2.perspectiveTransform(pt, self._shifted_matrix)[0][0]
            tx, ty = float(dst[0]), float(dst[1])
            if cv2.pointPolygonTest(poly, (tx, ty), False) < 0:
                continue
            persons.append({
                "keypoints": person_kpts,
                "tx": tx,
                "ty": ty,
                "fx": f_x,
                "fy": f_y,
            })

        # 2) 안정화 분류 (ghost 포함)
        top_data, bottom_data = self._stabilizer.update(persons)

        # 3) 렌더링
        for data, color in [(top_data, COLOR_TOP), (bottom_data, COLOR_BOTTOM)]:
            if data is None:
                continue
            self._render_person(canvas, cfg, data, color)

    def _render_person(
        self,
        canvas: np.ndarray,
        cfg:    dict,
        data:   dict,
        color:  Tuple[int, int, int],
    ) -> None:
        kpts      = data["keypoints"]
        tx, ty    = data["tx"], data["ty"]
        fx, fy    = data["fx"], data["fy"]

        for p1_i, p2_i in SKELETON_EDGES:
            kp1 = kpts[p1_i - 1]
            kp2 = kpts[p2_i - 1]
            if len(kp1) < 3 or len(kp2) < 3:
                continue
            if float(kp1[2]) < cfg["keypoint_conf"] or float(kp2[2]) < cfg["keypoint_conf"]:
                continue
            x1 = int(tx + (float(kp1[0]) - fx) * cfg["scale"])
            y1 = int(ty + (float(kp1[1]) - fy) * cfg["scale"])
            x2 = int(tx + (float(kp2[0]) - fx) * cfg["scale"])
            y2 = int(ty + (float(kp2[1]) - fy) * cfg["scale"])
            cv2.line(canvas, (x1, y1), (x2, y2), color, cfg["bone_thickness"])
            cv2.circle(canvas, (x1, y1), 3, (255, 255, 255), -1)

        nose = kpts[0]
        if len(nose) >= 3 and float(nose[2]) >= cfg["keypoint_conf"]:
            hx = int(tx + (float(nose[0]) - fx) * cfg["scale"])
            hy = int(ty + (float(nose[1]) - fy) * cfg["scale"])
            r  = cfg["head_radius"]
            cv2.circle(canvas, (hx, hy), r, color,           -1)
            cv2.circle(canvas, (hx, hy), r, (255, 255, 255),  2)


# ────────────────────────────────────────────────────────────
# 모듈 수준 헬퍼 함수
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
