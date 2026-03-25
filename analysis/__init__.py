"""
RallyTrack 분석 패키지 (v3)

모듈 구성:
  config         - 전역 설정 (경로, 파라미터, BWF 코트 규격)
  court          - 코트 기하학 / 호모그래피 / 미니맵 코트 그리기 (BWF 정규 규격)
  court_detector - 코트 코너 자동 검출 (허프 직선 + 그린 마스크)
  impact         - 물리 기반 타점 감지
  minimap        - 미니맵 통합 렌더러
  skeleton_view  - 원근감 코트 스켈레톤 렌더러
  web_exporter   - 웹 연동용 JSON 데이터 익스포터
"""

from .config        import PATHS, MINIMAP_CONFIG, IMPACT_CONFIG, POSE_CONFIG, BADMINTON_COURT, COURT_RATIOS
from .court         import compute_homographies, draw_minimap_court, create_minimap_canvas
from .court_detector import (
    CourtCornerDetector,
    CourtCorners,
    detect_court_corners,
    visualize_corners,
)
from .impact        import ImpactDetector, ImpactEvent, to_api_json
from .minimap       import MinimapRenderer
from .skeleton_view import SkeletonCourtRenderer
from .web_exporter  import WebDataCollector, save_hits_json, save_court_corners_json

__all__ = [
    # config
    "PATHS", "MINIMAP_CONFIG", "IMPACT_CONFIG", "POSE_CONFIG",
    "BADMINTON_COURT", "COURT_RATIOS",
    # court
    "compute_homographies", "draw_minimap_court", "create_minimap_canvas",
    # court_detector
    "CourtCornerDetector", "CourtCorners",
    "detect_court_corners", "visualize_corners",
    # impact
    "ImpactDetector", "ImpactEvent", "to_api_json",
    # renderers
    "MinimapRenderer", "SkeletonCourtRenderer",
    # web exporter
    "WebDataCollector", "save_hits_json", "save_court_corners_json",
]
