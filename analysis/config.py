"""
RallyTrack - 전역 설정 파일
모든 분석 파라미터와 경로를 중앙에서 관리합니다.
"""

import os

# ─────────────────────────────────────────────
# 경로 설정
# BASE_DIR: aiAnalysis-server 루트
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PATHS = {
    "input_video_dir": os.path.join(BASE_DIR, "temp_videos"),
    "prediction_dir":  os.path.join(BASE_DIR, "prediction"),
    "result_dir":      os.path.join(BASE_DIR, "result"),
    "tracknet_dir":    os.path.join(BASE_DIR, "tracknetv3"),
    "tracknet_ckpt":   os.path.join(BASE_DIR, "tracknetv3", "ckpts", "TrackNet_best.pt"),
    "inpaintnet_ckpt": os.path.join(BASE_DIR, "tracknetv3", "ckpts", "InpaintNet_best.pt"),
    "yolo_model":      os.path.join(BASE_DIR, "weights", "yolov8n-pose.pt"),
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
    "width":  360,
    "height": 600,
    # 코트 내부 패딩 (px)
    "padding": 20,
    # 코트 라인 색상 (BGR)
    "line_color":  (255, 255, 255),
    "net_color":   (0,   80,  255),
    # 코트 바닥 채색 (Two-tone Green) — 단식 경기 인/아웃 직관적 구분
    "court_singles_color": ( 48, 115,  50),   # #327330 BGR — 단식 유효 영역
    "court_doubles_color": ( 26,  64,  27),   # #1B401A BGR — 복식 앨리(단식 아웃)
    # 선수 색상 (BGR) — 상단(서버 쪽), 하단
    "top_color":   (246, 130,  59),   # Royal Blue  #3B82F6 (사이트 메인 테마색)
    "bottom_color":( 11, 158, 245),   # Amber Gold  #F59E0B (보색·셔틀콕 상징)
    # 셔틀콕 궤적 색상
    "shuttle_color": (0, 200, 255),
    # 타점 마커 반지름
    "hit_radius": 12,
    # 궤적 히스토리 길이 (프레임 수)
    "trail_length": 30,
}

# ─────────────────────────────────────────────
# 호모그래피 — 코트 코너 기준점 (원본 영상 비율)
#
# [좌상, 우상, 우하, 좌하] 순서로 지정.
# 카메라 앵글에 따라 튜닝이 필요합니다.
# ─────────────────────────────────────────────
COURT_CORNERS = {
    # 영상 해상도 비율 (0~1)
    "top_left":     (0.28, 0.46),
    "top_right":    (0.72, 0.46),
    "bottom_right": (0.85, 0.97),
    "bottom_left":  (0.15, 0.97),
}

# ─────────────────────────────────────────────
# 배드민턴 코트 실제 치수 비율 (국제 규격 기준)
#   - 전체 길이: 13.4m, 전체 폭: 6.1m
#   - 서비스 라인: 네트에서 1.98m
#   - 백 바운더리 라인: 0.76m (롱 서비스 라인)
#   - 사이드 라인: 각 0.46m (복식 기준)
# ─────────────────────────────────────────────
COURT_LINES = {
    "service_ratio":   1.98 / 13.4,  # 네트-서비스 라인 거리 비율
    "back_ratio":      0.76 / 13.4,  # 백 바운더리 라인 비율
    "side_ratio":      0.46 / 6.1,   # 사이드 라인 비율
}

# ─────────────────────────────────────────────
# 타점 감지 물리 파라미터
#
# [v5 어댑티브 로직 주의]
#   아래 값들은 30fps 탑뷰 기준의 원본 값(base values)이다.
#   실제 적용 시 ImpactDetector._compute_adaptive_params()가
#   TrackNet 추적 밀도(avg_tracking_gap)와 fps에 따라 동적으로 조정한다.
#
#   avg_gap ≈ 1.5 (프로 탑뷰) → 아래 값 그대로 사용 → test_8sec 7/7 보장
#   avg_gap ≈ 4~6 (아마추어 쿼터뷰+블러) → 완화 파라미터 적용
# ─────────────────────────────────────────────
IMPACT_CONFIG = {
    # ── impulse 계산 물리 파라미터 (30fps 탑뷰 기준) ─────────
    # 최소 속도 (픽셀/프레임) — 이하는 노이즈로 무시
    # [어댑티브] 쿼터뷰 원근 + 드문 좌표 시 동적 완화
    "min_speed":         3.0,
    # 최대 프레임 갭 (valid_idx[k-2]~valid_idx[k] 허용 범위)
    # [어댑티브] avg_gap * 2.5 로 동적 확장 (최솟값 유지)
    "max_frame_gap":     5,
    # Skip-Vector 직진 각도 임계값 (도) — 이하면 진짜 타점 아님
    # [어댑티브] 완화 없음 (오탐 방지 우선)
    "skip_angle_thresh": 15.0,
    # 타점 후 최소 비행 거리 (px) — 이하면 노이즈로 제거
    # [어댑티브] 쿼터뷰 원근 효과로 짧게 측정되므로 gap_ratio 기반 완화
    "min_flight_dist":   60,

    # ── peaks 감지 파라미터 ──────────────────────────────────
    # 피크 감지 최소 거리 (프레임 수) — 30fps 기준
    # [어댑티브] fps_scale 기반 확장 (60fps → 24프레임)
    "peak_distance":     12,
    # 임계값 = max_score * ratio (원본 검증 값 0.15 유지)
    # [어댑티브] 완화 없음 (상대 비율이므로 영상 간 스케일 자동 조정)
    "peak_threshold_ratio": 0.15,

    # ── crossing owner 교정 파라미터 (하이브리드 모드) ────────
    # crossing 이전 타점 후보 탐색 최대 구간 (프레임)
    "crossing_lookback_frames": 60,
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
