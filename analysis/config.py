"""
RallyTrack - 전역 설정 파일
모든 분석 파라미터와 경로를 중앙에서 관리합니다.

[v2 변경 사항]
  - COURT_CORNERS 하드코딩 제거
    → 코트 코너는 court_detector.py가 자동으로 검출합니다.
  - COURT_DETECTOR_CONFIG 추가
    → 자동 검출 알고리즘 파라미터를 중앙 관리합니다.
"""

import os

# ─────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PATHS = {
    "input_video_dir": os.path.join(BASE_DIR, "inputVideo"),
    "prediction_dir":  os.path.join(BASE_DIR, "prediction"),
    "result_dir":      os.path.join(BASE_DIR, "result"),
    "tracknet_dir":    os.path.join(BASE_DIR, "TrackNetV3"),
    "tracknet_ckpt":   os.path.join(BASE_DIR, "TrackNetV3", "ckpts", "TrackNet_best.pt"),
    "inpaintnet_ckpt": os.path.join(BASE_DIR, "TrackNetV3", "ckpts", "InpaintNet_best.pt"),
    "yolo_model":      os.path.join(BASE_DIR, "yolov8n-pose.pt"),
}

# ─────────────────────────────────────────────
# TrackNet 실행 설정
# ─────────────────────────────────────────────
TRACKNET_CONFIG = {
    # VRAM 제한 환경에서는 1 고정 (RTX 3060 이상이면 4~8로 올려도 됨)
    "batch_size": 1,
    # Windows 멀티프로세싱 오류 방지
    "num_workers": 0,
}

# ─────────────────────────────────────────────
# 미니맵 캔버스 크기
# ─────────────────────────────────────────────
MINIMAP_CONFIG = {
    "width":  320,
    "height": 670,
    # 코트 내부 패딩 (px) — 캔버스 여백
    "padding": 25,
    # 코트 라인 색상 (BGR)
    "line_color":  (255, 255, 255),
    "net_color":   (0,   80,  255),
    # 선수 색상 (BGR) — 상단(서버 쪽), 하단
    "top_color":   (255, 105, 180),   # 핑크
    "bottom_color":(50,  205,  50),   # 라임
    # 셔틀콕 궤적 색상
    "shuttle_color": (0, 200, 255),
    # 타점 마커 반지름
    "hit_radius": 12,
    # 궤적 히스토리 길이 (프레임 수)
    "trail_length": 30,
}

# ─────────────────────────────────────────────
# 배드민턴 코트 실제 치수 (국제 BWF 규격)
#
# 전체 길이 : 13.40m (단식/복식 공통)
# 전체 폭   : 6.10m  (복식) / 5.18m (단식)
# 네트 높이  : 1.55m (사이드) / 1.524m (중앙)
#
# 라인 간격 (길이 방향):
#   후방 경계선 ~ 롱 서비스 라인 (복식): 0.76m
#   후방 경계선 ~ 숏 서비스 라인      : 13.40/2 - 1.98 = 4.72m (전반부)
#   네트 ~ 숏 서비스 라인 (전방)      : 1.98m
#
# 라인 간격 (폭 방향):
#   사이드 경계선(복식) ~ 사이드라인(단식): 0.46m (각 쪽)
#   중앙 센터 라인으로 좌우 대칭
#
# 미니맵에서의 비율 매핑:
#   코트 전체 길이 = 미니맵 usable_height
#   코트 전체 폭   = 미니맵 usable_width
# ─────────────────────────────────────────────
BADMINTON_COURT = {
    # 실제 치수 (m)
    "length":         13.40,   # 코트 전체 길이 (복식/단식 공통)
    "width_doubles":   6.10,   # 복식 전체 폭
    "width_singles":   5.18,   # 단식 전체 폭
    "net_to_service":  1.98,   # 네트 ~ 숏 서비스 라인
    "back_service":    0.76,   # 후방 경계 ~ 롱 서비스 라인 (복식)
    "singles_side":    0.46,   # 복식→단식 사이드 라인 간격 (각 쪽)
    "center_service":  3.96,   # 네트 ~ 숏 서비스 라인 구간 좌우 센터 라인
    #                            (숏 서비스 라인에서만 필요)
}

# 길이 비율 (미니맵 usable_height 기준 0.0 ~ 1.0)
# Y=0: 위쪽 경계선, Y=1: 아래쪽 경계선
COURT_RATIOS = {
    # 길이 방향 (Y축)
    "back_service_top":    BADMINTON_COURT["back_service"]    / BADMINTON_COURT["length"],
    "short_service_top":   BADMINTON_COURT["net_to_service"]  / BADMINTON_COURT["length"],
    "net":                 0.5,
    "short_service_bot":   1.0 - BADMINTON_COURT["net_to_service"]  / BADMINTON_COURT["length"],
    "back_service_bot":    1.0 - BADMINTON_COURT["back_service"]    / BADMINTON_COURT["length"],

    # 폭 방향 (X축) — 복식 폭 기준 0.0~1.0
    "singles_left":        BADMINTON_COURT["singles_side"] / BADMINTON_COURT["width_doubles"],
    "singles_right":       1.0 - BADMINTON_COURT["singles_side"] / BADMINTON_COURT["width_doubles"],
    "center":              0.5,
}

# ─────────────────────────────────────────────
# 코트 코너 자동 검출 설정
#
# [하드코딩 제거]
# 이전 버전의 COURT_CORNERS(수동 좌표) 설정은 삭제되었습니다.
# 코트 코너는 court_detector.CourtCornerDetector가 영상에서 자동 검출합니다.
#
# 자동 검출 흐름:
#   1. detect_court_corners(video_path) 호출
#   2. CourtCorners 반환 → .to_ratio(w, h) → court.compute_homographies()
# ─────────────────────────────────────────────
COURT_DETECTOR_CONFIG = {
    # 분석에 사용할 샘플 프레임 수
    "sample_frames":       20,
    # 샘플링 최대 구간 (초) — 영상 앞부분에서 검출
    "max_sample_sec":      30,
    # 코트 감지 유효 판정을 위한 최소 그린 비율
    "green_ratio_min":     0.10,
    # 허프 직선 변환 최소 투표 수 (낮출수록 더 많은 직선 검출)
    "hough_threshold":     50,
    # 허프 직선 최소 길이 (픽셀, 640px 기준 리사이즈 후)
    "hough_min_line_length": 80,
    # 허프 직선 최대 연결 갭
    "hough_max_line_gap":  20,
    # 최종 채택을 위한 최소 유효 프레임 수
    "vote_min_frames":     3,
    # Canny 엣지 임계값
    "canny_low":           50,
    "canny_high":          150,
}

# ─────────────────────────────────────────────
# 타점 감지 물리 파라미터
# ─────────────────────────────────────────────
IMPACT_CONFIG = {
    # 최소 속도 (픽셀/프레임) — 이하는 노이즈로 무시
    "min_speed":         3.0,
    # 최대 프레임 갭 (좌표가 없는 구간 허용 범위)
    "max_frame_gap":     5,
    # Skip-Vector 직진 각도 임계값 (도) — 이하면 진짜 타점 아님
    "skip_angle_thresh": 15.0,
    # 타점 후 최소 비행 거리 (px) — 이하면 노이즈로 제거
    "min_flight_dist":   60,
    # 피크 감지 최소 거리 (프레임 수)
    "peak_distance":     12,
    # 임계값 = max_score * ratio
    "peak_threshold_ratio": 0.15,
}

# ─────────────────────────────────────────────
# YOLOv8-pose 설정
# ─────────────────────────────────────────────
POSE_CONFIG = {
    "confidence":    0.3,
    # 관절 가시성(confidence) 임계값
    "keypoint_conf": 0.3,
    # 스켈레톤 확장 비율 (미니맵에서 신체를 약간 크게 그림)
    "scale":         1.2,
    # 머리 원 반지름 (px)
    "head_radius":   12,
    # 뼈대 굵기 (px)
    "bone_thickness": 4,
}

# YOLOv8-pose 관절 연결 엣지 (1-indexed → 0-indexed 변환됨)
# COCO 17-keypoints 기준
SKELETON_EDGES = [
    (16, 14), (14, 12), (17, 15), (15, 13),
    (12, 13), (6, 12),  (7, 13),  (6, 7),
    (6, 8),   (7, 9),   (8, 10),  (9, 11),
]

# ─────────────────────────────────────────────
# 웹 연동 데이터 익스포트 설정
# ─────────────────────────────────────────────
WEB_EXPORT_CONFIG = {
    # 프레임별 데이터 JSON 출력 여부
    "export_frame_data":    True,
    # 선수 위치 정규화 좌표(0~1) 포함 여부
    "include_player_positions": True,
    # 셔틀콕 궤적 정규화 좌표 포함 여부
    "include_shuttle_trail": True,
    # 셔틀콕 궤적 최대 저장 길이 (프레임)
    "shuttle_trail_length":  30,
    # 출력 파일명 접미사
    "frame_data_suffix":    "_frame_data.json",
    "summary_suffix":       "_summary.json",
}
