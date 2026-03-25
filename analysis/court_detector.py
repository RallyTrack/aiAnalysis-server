"""
RallyTrack - 배드민턴 코트 코너 자동 검출 모듈

[설계 원칙]
  수동 하드코딩(config.COURT_CORNERS) 없이 영상 프레임에서
  배드민턴 코트 4개 코너를 자동으로 검출합니다.

[검출 파이프라인]
  Stage 1 ─ 초기 프레임 수집
      영상 시작~30초 구간에서 N개 샘플 프레임을 추출합니다.

  Stage 2 ─ 그린(코트 바닥) 마스크 생성
      HSV 색공간에서 코트 바닥(녹색 계열) 영역을 검출합니다.
      그린 비율이 낮은 프레임은 코트 미노출로 판단해 제외합니다.

  Stage 3 ─ 엣지 & 허프 직선 검출
      Canny 엣지 → 확률적 허프 직선 변환으로 코트 경계선 후보를 추출합니다.

  Stage 4 ─ 코트 사각형 재구성
      직선 후보에서 4개의 코트 경계선(상·하·좌·우)을 식별하고
      교점 4개를 코너로 확정합니다.

  Stage 5 ─ 프레임 간 투표(Voting) 안정화
      여러 프레임에서 얻은 코너 후보를 평균하여 노이즈를 제거합니다.

[폴백(Fallback) 전략]
  자동 검출 실패 시, 영상 해상도 기반 경험적 추정값을 반환합니다.
  (광각/협각 카메라 두 프로파일 제공)

[웹 서비스 연동]
  detect_corners()의 반환값 dict를 compute_homographies()에 바로 전달하거나,
  별도 API 엔드포인트에서 JSON으로 응답할 수 있습니다.
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List

from .config import COURT_DETECTOR_CONFIG


# ─────────────────────────────────────────────────────────────
# 데이터 클래스
# ─────────────────────────────────────────────────────────────

@dataclass
class CourtCorners:
    """
    검출된 코트 4개 코너 좌표.

    모든 좌표는 영상 픽셀 좌표 (절대값)입니다.
    정규화 비율(0~1)은 .to_ratio()로 변환합니다.

    순서 약속: [좌상, 우상, 우하, 좌하]
    """
    top_left:     Tuple[float, float]
    top_right:    Tuple[float, float]
    bottom_right: Tuple[float, float]
    bottom_left:  Tuple[float, float]
    confidence:   float = 0.0      # 0.0(추정) ~ 1.0(확실)
    method:       str   = "auto"   # "hough" | "fallback_wide" | "fallback_narrow"

    def to_ratio(self, frame_w: int, frame_h: int) -> dict:
        """픽셀 좌표 → [0,1] 비율 딕셔너리로 변환 (court.py 호환)."""
        return {
            "top_left":     (self.top_left[0]     / frame_w, self.top_left[1]     / frame_h),
            "top_right":    (self.top_right[0]    / frame_w, self.top_right[1]    / frame_h),
            "bottom_right": (self.bottom_right[0] / frame_w, self.bottom_right[1] / frame_h),
            "bottom_left":  (self.bottom_left[0]  / frame_w, self.bottom_left[1]  / frame_h),
        }

    def to_pixel_array(self) -> np.ndarray:
        """(4, 2) float32 numpy 배열로 반환."""
        return np.array([
            self.top_left,
            self.top_right,
            self.bottom_right,
            self.bottom_left,
        ], dtype=np.float32)

    def to_dict(self) -> dict:
        """JSON 직렬화용 딕셔너리."""
        return {
            "top_left":     list(self.top_left),
            "top_right":    list(self.top_right),
            "bottom_right": list(self.bottom_right),
            "bottom_left":  list(self.bottom_left),
            "confidence":   round(self.confidence, 3),
            "method":       self.method,
        }


# ─────────────────────────────────────────────────────────────
# 메인 검출기 클래스
# ─────────────────────────────────────────────────────────────

class CourtCornerDetector:
    """
    배드민턴 코트 코너 자동 검출기.

    사용법:
        detector = CourtCornerDetector()
        corners  = detector.detect(video_path)
        # corners.to_ratio(w, h) → court.compute_homographies()에 전달
    """

    def __init__(self):
        cfg = COURT_DETECTOR_CONFIG
        self._sample_frames    = cfg["sample_frames"]
        self._max_sample_sec   = cfg["max_sample_sec"]
        self._green_ratio_min  = cfg["green_ratio_min"]
        self._hough_threshold  = cfg["hough_threshold"]
        self._hough_min_len    = cfg["hough_min_line_length"]
        self._hough_max_gap    = cfg["hough_max_line_gap"]
        self._vote_min         = cfg["vote_min_frames"]
        self._canny_low        = cfg["canny_low"]
        self._canny_high       = cfg["canny_high"]

    # ── 퍼블릭 API ─────────────────────────────────────────

    def detect(self, video_path: str) -> CourtCorners:
        """
        영상 파일에서 코트 코너를 자동 검출합니다.

        Args:
            video_path: 분석할 영상 파일 경로

        Returns:
            CourtCorners 인스턴스
              - confidence >= 0.6: 자동 검출 성공
              - confidence <  0.6: 폴백 추정값
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"영상을 열 수 없습니다: {video_path}")

        frame_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 샘플링 구간: 0 ~ min(max_sample_sec, 총 길이) 초
        max_f   = min(n_frames, int(self._max_sample_sec * fps))
        step    = max(1, max_f // self._sample_frames)
        targets = list(range(0, max_f, step))[: self._sample_frames]

        candidates: List[np.ndarray] = []   # shape (4, 2) 모음

        for t in targets:
            cap.set(cv2.CAP_PROP_POS_FRAMES, t)
            ret, frame = cap.read()
            if not ret:
                continue

            corners = self._detect_single_frame(frame, frame_w, frame_h)
            if corners is not None:
                candidates.append(corners)

        cap.release()

        return self._aggregate(candidates, frame_w, frame_h)

    def detect_from_frames(
        self,
        frames: List[np.ndarray],
        frame_w: int,
        frame_h: int,
    ) -> CourtCorners:
        """
        이미 로드된 프레임 리스트에서 코트 코너를 검출합니다.
        (웹 API 등 스트리밍 환경에서 사용)
        """
        candidates = []
        for frame in frames:
            corners = self._detect_single_frame(frame, frame_w, frame_h)
            if corners is not None:
                candidates.append(corners)
        return self._aggregate(candidates, frame_w, frame_h)

    # ── 단일 프레임 검출 ────────────────────────────────────

    def _detect_single_frame(
        self,
        frame:   np.ndarray,
        frame_w: int,
        frame_h: int,
    ) -> Optional[np.ndarray]:
        """
        단일 프레임에서 코트 코너 4개를 검출합니다.

        Returns:
            (4, 2) float32 배열 [TL, TR, BR, BL] 또는 None (검출 실패)
        """
        # ── Step 1: 전처리 ───────────────────────────────────
        resized = _resize_for_detection(frame)
        scale_x = frame_w  / resized.shape[1]
        scale_y = frame_h  / resized.shape[0]

        # ── Step 2: 코트 바닥 마스크 ─────────────────────────
        court_mask = _build_court_mask(resized)
        green_ratio = np.sum(court_mask > 0) / (resized.shape[0] * resized.shape[1])

        if green_ratio < self._green_ratio_min:
            # 코트가 충분히 보이지 않는 프레임 → 스킵
            return None

        # ── Step 3: 마스크 영역 내 엣지 검출 ─────────────────
        gray  = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
        edges = cv2.Canny(blurred, self._canny_low, self._canny_high)
        # 마스크 경계(dilated)에서만 엣지 탐색
        mask_edge = cv2.dilate(court_mask, np.ones((5, 5), np.uint8), iterations=3)
        edges = cv2.bitwise_and(edges, mask_edge)

        # ── Step 4: 허프 직선 검출 ────────────────────────────
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self._hough_threshold,
            minLineLength=self._hough_min_len,
            maxLineGap=self._hough_max_gap,
        )
        if lines is None or len(lines) < 4:
            return None

        lines = lines.reshape(-1, 4)   # (N, 4)

        # ── Step 5: 직선 분류 (수평/수직 계열) ───────────────
        h_lines, v_lines = _classify_lines(lines)
        if len(h_lines) < 2 or len(v_lines) < 2:
            return None

        # ── Step 6: 코트 경계 4선 선택 ───────────────────────
        h_sorted = sorted(h_lines, key=lambda l: (l[1] + l[3]) / 2)
        v_sorted = sorted(v_lines, key=lambda l: (l[0] + l[2]) / 2)

        top_line    = h_sorted[0]
        bottom_line = h_sorted[-1]
        left_line   = v_sorted[0]
        right_line  = v_sorted[-1]

        # ── Step 7: 4개 교점 계산 ─────────────────────────────
        tl = _line_intersect(top_line,    left_line)
        tr = _line_intersect(top_line,    right_line)
        br = _line_intersect(bottom_line, right_line)
        bl = _line_intersect(bottom_line, left_line)

        if any(p is None for p in [tl, tr, br, bl]):
            return None

        # ── Step 8: 유효성 검사 ───────────────────────────────
        corners = np.array([tl, tr, br, bl], dtype=np.float32)
        if not _is_valid_quadrilateral(corners, resized.shape[1], resized.shape[0]):
            return None

        # 원본 해상도로 스케일 복원
        corners[:, 0] *= scale_x
        corners[:, 1] *= scale_y

        return corners

    # ── 프레임 간 집계 ──────────────────────────────────────

    def _aggregate(
        self,
        candidates: List[np.ndarray],
        frame_w:    int,
        frame_h:    int,
    ) -> CourtCorners:
        """
        복수 프레임 후보를 집계해 최종 코너를 결정합니다.
        후보가 부족하면 폴백으로 전환합니다.
        """
        if len(candidates) >= self._vote_min:
            stacked    = np.stack(candidates, axis=0)   # (N, 4, 2)
            mean_pts   = np.mean(stacked, axis=0)        # (4, 2)
            confidence = min(1.0, len(candidates) / self._sample_frames)
            print(f"[CourtDetector] 자동 검출 성공: {len(candidates)}/{self._sample_frames} 프레임, "
                  f"신뢰도={confidence:.2f}")
            return CourtCorners(
                top_left     = (float(mean_pts[0, 0]), float(mean_pts[0, 1])),
                top_right    = (float(mean_pts[1, 0]), float(mean_pts[1, 1])),
                bottom_right = (float(mean_pts[2, 0]), float(mean_pts[2, 1])),
                bottom_left  = (float(mean_pts[3, 0]), float(mean_pts[3, 1])),
                confidence   = confidence,
                method       = "hough",
            )

        # ── 폴백: 경험적 추정 ─────────────────────────────────
        print(f"[CourtDetector] 자동 검출 부족 ({len(candidates)}개). "
              "폴백 추정값 사용.")
        return _fallback_corners(frame_w, frame_h)


# ─────────────────────────────────────────────────────────────
# 모듈 수준 헬퍼 함수
# ─────────────────────────────────────────────────────────────

def _resize_for_detection(frame: np.ndarray, max_w: int = 640) -> np.ndarray:
    """검출 속도 향상을 위해 프레임을 축소합니다."""
    h, w = frame.shape[:2]
    if w <= max_w:
        return frame
    scale = max_w / w
    return cv2.resize(frame, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)


def _build_court_mask(frame: np.ndarray) -> np.ndarray:
    """
    HSV 색공간 기반 코트 바닥 마스크를 생성합니다.

    배드민턴 코트 바닥 색상:
      - 녹색 계열 (일반 실내 코트, 나무 바닥 위 녹색 도장)
      - 파란색 계열 (BWF 경기장 블루 코트)
      - 나무 바닥색 계열 (목재 코트)
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 녹색 (H: 35~85)
    green_lo = np.array([35,  30,  40], dtype=np.uint8)
    green_hi = np.array([85, 255, 255], dtype=np.uint8)
    mask_green = cv2.inRange(hsv, green_lo, green_hi)

    # 청록/파란 계열 (H: 85~140) — BWF 블루 코트
    blue_lo = np.array([85,  30,  50], dtype=np.uint8)
    blue_hi = np.array([140, 255, 255], dtype=np.uint8)
    mask_blue = cv2.inRange(hsv, blue_lo, blue_hi)

    # 나무/갈색 계열 (H: 10~30, S 낮음) — 목재 코트
    wood_lo = np.array([10,  20,  80], dtype=np.uint8)
    wood_hi = np.array([30, 150, 220], dtype=np.uint8)
    mask_wood = cv2.inRange(hsv, wood_lo, wood_hi)

    combined = cv2.bitwise_or(mask_green, mask_blue)
    combined = cv2.bitwise_or(combined,   mask_wood)

    # 노이즈 제거
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  kernel)

    return combined


def _classify_lines(
    lines: np.ndarray,
    angle_thresh: float = 25.0,
) -> Tuple[List, List]:
    """
    직선을 수평(H) / 수직(V) 계열로 분류합니다.

    angle_thresh: 수평/수직 기준에서 허용하는 최대 각도 편차(도)
    """
    h_lines, v_lines = [], []
    for x1, y1, x2, y2 in lines:
        dx   = x2 - x1
        dy   = y2 - y1
        if dx == 0 and dy == 0:
            continue
        angle = abs(np.degrees(np.arctan2(dy, dx)))   # 0 ~ 180

        if angle < angle_thresh or angle > (180 - angle_thresh):
            h_lines.append((x1, y1, x2, y2))
        elif abs(angle - 90) < angle_thresh:
            v_lines.append((x1, y1, x2, y2))

    return h_lines, v_lines


def _line_intersect(
    l1: Tuple,
    l2: Tuple,
) -> Optional[Tuple[float, float]]:
    """
    두 직선(선분의 연장선)의 교점을 반환합니다.
    평행선이면 None 반환.
    """
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    ix = x1 + t * (x2 - x1)
    iy = y1 + t * (y2 - y1)
    return float(ix), float(iy)


def _is_valid_quadrilateral(
    corners: np.ndarray,
    img_w:   int,
    img_h:   int,
    min_area_ratio: float = 0.05,
    max_area_ratio: float = 0.95,
) -> bool:
    """
    검출된 사각형이 유효한 코트 영역인지 검사합니다.

    검사 항목:
      1. 모든 꼭짓점이 이미지 내(±20% 여백 허용)에 있는지
      2. 사각형 면적이 적정 범위(이미지의 5%~95%)인지
      3. [좌상 < 우상], [좌하 < 우하] X 좌표 순서
      4. [상단 < 하단] Y 좌표 순서
    """
    margin_x = img_w * 0.20
    margin_y = img_h * 0.20

    for pt in corners:
        if not (-margin_x <= pt[0] <= img_w + margin_x):
            return False
        if not (-margin_y <= pt[1] <= img_h + margin_y):
            return False

    area = cv2.contourArea(corners.reshape(-1, 1, 2).astype(np.int32))
    img_area = img_w * img_h
    if not (img_area * min_area_ratio <= area <= img_area * max_area_ratio):
        return False

    tl, tr, br, bl = corners
    if tl[0] >= tr[0] or bl[0] >= br[0]:
        return False
    if tl[1] >= bl[1] or tr[1] >= br[1]:
        return False

    return True


def _fallback_corners(frame_w: int, frame_h: int) -> CourtCorners:
    """
    자동 검출 실패 시 해상도 기반 경험적 추정값을 반환합니다.

    카메라 화각 기준:
      - 16:9 또는 4:3 영상 → 표준 중앙 촬영 프로파일
      - 세로 영상 (9:16)  → 세로 모바일 촬영 프로파일
    """
    aspect = frame_w / frame_h if frame_h > 0 else 1.78

    if aspect >= 1.0:
        # 표준 가로 촬영 (배드민턴 TV 중계 앵글 기준)
        tl = (frame_w * 0.20, frame_h * 0.35)
        tr = (frame_w * 0.80, frame_h * 0.35)
        br = (frame_w * 0.90, frame_h * 0.97)
        bl = (frame_w * 0.10, frame_h * 0.97)
        method = "fallback_landscape"
    else:
        # 세로 촬영 (모바일 9:16)
        tl = (frame_w * 0.05, frame_h * 0.25)
        tr = (frame_w * 0.95, frame_h * 0.25)
        br = (frame_w * 0.95, frame_h * 0.90)
        bl = (frame_w * 0.05, frame_h * 0.90)
        method = "fallback_portrait"

    print(f"[CourtDetector] 폴백 사용: {method} "
          f"({frame_w}×{frame_h}, aspect={aspect:.2f})")

    return CourtCorners(
        top_left     = tl,
        top_right    = tr,
        bottom_right = br,
        bottom_left  = bl,
        confidence   = 0.0,
        method       = method,
    )


# ─────────────────────────────────────────────────────────────
# 편의 함수 (외부에서 단순하게 호출 가능)
# ─────────────────────────────────────────────────────────────

def detect_court_corners(video_path: str) -> CourtCorners:
    """
    영상 파일에서 코트 코너를 자동 검출하는 편의 함수.

    Args:
        video_path: 분석할 영상 파일 경로

    Returns:
        CourtCorners 인스턴스
    """
    detector = CourtCornerDetector()
    return detector.detect(video_path)


def visualize_corners(
    frame:   np.ndarray,
    corners: CourtCorners,
    color:   Tuple = (0, 255, 0),
) -> np.ndarray:
    """
    검출된 코트 코너를 프레임에 시각화합니다.
    디버깅 및 검수용.

    Returns:
        코너가 그려진 프레임 복사본
    """
    vis   = frame.copy()
    pts   = corners.to_pixel_array().astype(np.int32)
    names = ["TL", "TR", "BR", "BL"]

    # 사각형 외곽선
    for i in range(4):
        cv2.line(vis, tuple(pts[i]), tuple(pts[(i + 1) % 4]), color, 2)

    # 코너 포인트 & 레이블
    for pt, name in zip(pts, names):
        cv2.circle(vis, tuple(pt), 8, (0, 0, 255), -1)
        cv2.putText(
            vis, f"{name}({pt[0]},{pt[1]})",
            (int(pt[0]) + 5, int(pt[1]) - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA,
        )

    # 신뢰도 표시
    cv2.putText(
        vis,
        f"Court Detection | conf={corners.confidence:.2f} | {corners.method}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA,
    )
    return vis
