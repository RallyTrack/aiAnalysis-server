"""
코트 감지 서비스
- YOLO 기반 코트 라인 감지 또는 수동 코트 좌표 설정
- 코트 영역 내 좌표 판정 (in/out)
"""
import numpy as np


# 배드민턴 코트 기본 좌표 (정규화 0~1 기준)
# 실제 비율: 코트 13.4m x 6.1m (복식), 13.4m x 5.18m (단식)
# 영상 내 코트 위치는 YOLO 감지 또는 수동 설정으로 보정
DEFAULT_COURT = {
    "singles": {
        "top_left": (0.2, 0.1),
        "top_right": (0.8, 0.1),
        "bottom_left": (0.2, 0.9),
        "bottom_right": (0.8, 0.9),
    },
    "doubles": {
        "top_left": (0.15, 0.1),
        "top_right": (0.85, 0.1),
        "bottom_left": (0.15, 0.9),
        "bottom_right": (0.85, 0.9),
    },
    # 서비스 라인 (네트에서 1.98m 떨어진 지점)
    # 전체 길이 13.4m 중 1.98m ≒ 14.8%
    "short_service_offset": 0.148,
    # 롱 서비스 라인 (복식: 뒤에서 0.76m)
    "long_service_offset": 0.057,
}


class CourtDetector:
    """코트 영역 감지 및 in/out 판정"""

    def __init__(self, court_corners=None, mode="singles"):
        """
        Args:
            court_corners: 코트 4꼭짓점 좌표 dict (YOLO 감지 결과 또는 수동 설정)
                          {"top_left": (x,y), "top_right": (x,y),
                           "bottom_left": (x,y), "bottom_right": (x,y)}
            mode: "singles" 또는 "doubles"
        """
        if court_corners:
            self.corners = court_corners
        else:
            self.corners = DEFAULT_COURT[mode]

        self.mode = mode
        self._build_court_polygon()

    def _build_court_polygon(self):
        """코트 4꼭짓점을 다각형 좌표 배열로 변환"""
        self.polygon = np.array([
            self.corners["top_left"],
            self.corners["top_right"],
            self.corners["bottom_right"],
            self.corners["bottom_left"],
        ], dtype=np.float64)

    def is_in_court(self, x: float, y: float) -> bool:
        """
        좌표가 코트 영역 안에 있는지 판정한다.
        Ray Casting 알고리즘 사용.

        Args:
            x, y: 정규화 좌표 (0~1)

        Returns:
            True = IN, False = OUT
        """
        return self._point_in_polygon(x, y, self.polygon)

    def judge_landing(self, landing_x: float, landing_y: float) -> dict:
        """
        셔틀콕 낙하지점의 in/out을 판정한다.

        Args:
            landing_x, landing_y: 셔틀콕 낙하 좌표 (정규화 0~1)

        Returns:
            {"result": "IN" or "OUT",
             "landing": {"x": float, "y": float},
             "margin": float}  # 코트 경계까지의 거리 (양수=IN, 음수=OUT)
        """
        is_in = self.is_in_court(landing_x, landing_y)
        margin = self._distance_to_boundary(landing_x, landing_y)

        return {
            "result": "IN" if is_in else "OUT",
            "landing": {"x": round(landing_x, 4), "y": round(landing_y, 4)},
            "margin": round(margin if is_in else -margin, 4),
        }

    def _point_in_polygon(self, x: float, y: float, polygon: np.ndarray) -> bool:
        """Ray Casting 알고리즘으로 점이 다각형 내부인지 판정"""
        n = len(polygon)
        inside = False

        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]

            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i

        return inside

    def _distance_to_boundary(self, x: float, y: float) -> float:
        """점에서 코트 경계(가장 가까운 변)까지의 거리"""
        min_dist = float("inf")
        n = len(self.polygon)

        for i in range(n):
            x1, y1 = self.polygon[i]
            x2, y2 = self.polygon[(i + 1) % n]
            dist = self._point_to_segment_dist(x, y, x1, y1, x2, y2)
            min_dist = min(min_dist, dist)

        return min_dist

    @staticmethod
    def _point_to_segment_dist(px, py, x1, y1, x2, y2) -> float:
        """점에서 선분까지의 최단 거리"""
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            return np.sqrt((px - x1) ** 2 + (py - y1) ** 2)

        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy

        return np.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)


def detect_court_yolo(video_path: str) -> dict | None:
    """
    YOLO로 영상에서 코트 라인을 감지하여 4꼭짓점을 반환한다.

    TODO: YOLO 코트 감지 모델 학습 후 구현
    현재는 None을 반환하여 기본 코트 좌표를 사용하게 함.

    Args:
        video_path: 영상 파일 경로

    Returns:
        코트 4꼭짓점 dict 또는 None (감지 실패)
    """
    # TODO: YOLO 코트 라인 감지 구현
    # from ultralytics import YOLO
    # model = YOLO("court_detection.pt")
    # ...
    print("[코트 감지] YOLO 미구현, 기본 코트 좌표 사용")
    return None
