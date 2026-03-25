"""
RallyTrack - 전역 설정 파일
모든 분석 파라미터와 경로를 중앙에서 관리합니다.

[v3 변경 사항]
  - BWF 코트 좌표계 완전 재정의
  - 미니맵 캔버스 비율을 실제 BWF 코트 13.40m × 6.10m 비율로 맞춤
  - 센터라인, 복식 롱서비스라인 등 모든 라인 치수 명시
  - court_lines_spec: 각 라인의 실제 좌표 JSON 스펙 추가
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
    "batch_size": 1,
    "num_workers": 0,
}

# ─────────────────────────────────────────────────────────────────────────────
# BWF 배드민턴 코트 실제 치수 (단위: 미터)
#
# 좌표계 정의:
#   원점 (0, 0) = 코트 왼쪽 아래 모서리 (더블 코트 기준 바깥쪽 모서리)
#   x축: 왼쪽 → 오른쪽,  0 ~ 6.10m  (복식 폭)
#   y축: 아래 → 위,       0 ~ 13.40m (코트 길이)
#   라인 두께: 중심선 기준 ±0.02m (총 0.04m = 40mm)
#
# 주요 y좌표 (코트 길이 방향):
#   y = 0.00  : 아래쪽 엔드라인 (외곽 복식 경계)
#   y = 0.76  : 아래 더블 롱서비스라인 (엔드라인에서 0.76m 위)
#   y = 1.98  : 아래 숏서비스라인 (네트에서 1.98m)  → y = 6.70 - 1.98 = 4.72
#   y = 6.70  : 네트 위치 (코트 정중앙)
#   y = 11.42 : 위 숏서비스라인 (= 13.40 - 1.98)
#   y = 12.64 : 위 더블 롱서비스라인 (= 13.40 - 0.76)
#   y = 13.40 : 위쪽 엔드라인
#
# 주요 x좌표 (코트 폭 방향):
#   x = 0.00  : 왼쪽 더블 사이드라인
#   x = 0.46  : 왼쪽 단식 사이드라인
#   x = 3.05  : 센터라인 (6.10 / 2)
#   x = 5.64  : 오른쪽 단식 사이드라인 (= 6.10 - 0.46)
#   x = 6.10  : 오른쪽 더블 사이드라인
# ─────────────────────────────────────────────────────────────────────────────
BADMINTON_COURT = {
    # ── 코트 전체 크기 ──────────────────────────────────────
    "length":           13.40,   # 코트 전체 길이 (y축 범위)
    "width_doubles":     6.10,   # 복식 전체 폭 (x축 범위)
    "width_singles":     5.18,   # 단식 폭 (= 6.10 - 0.46*2)

    # ── 길이 방향 (y축) 주요 치수 ───────────────────────────
    "net_y":             6.70,   # 네트 위치 y좌표 (= length / 2)
    "net_to_short_svc":  1.98,   # 네트 ~ 숏서비스라인 거리
    "back_to_long_svc":  0.76,   # 엔드라인 ~ 더블 롱서비스라인 거리

    # ── 폭 방향 (x축) 주요 치수 ─────────────────────────────
    "singles_side":      0.46,   # 복식→단식 사이드라인 간격 (각 쪽)
    "center_x":          3.05,   # 센터라인 x좌표 (= 6.10 / 2)

    # ── 네트 높이 ───────────────────────────────────────────
    "net_height_post":   1.55,   # 네트 포스트 높이 (사이드라인 위)
    "net_height_center": 1.524,  # 네트 중앙 높이

    # ── 라인 두께 ───────────────────────────────────────────
    "line_width":        0.04,   # 40mm
}

# ─────────────────────────────────────────────────────────────────────────────
# 각 라인의 실제 좌표 스펙 (JSON 형태)
#
# 좌표계: 원점 = 왼쪽 아래 모서리, x→오른쪽, y→위쪽
# (x1,y1) = 시작점, (x2,y2) = 끝점
# 라인 두께 = 0.04m (중심에서 ±0.02m)
#
# BWF 규격 근거:
#   - 코트 외곽(4모서리): (0,0), (6.10,0), (6.10,13.40), (0,13.40)
#   - 네트: y = 6.70 (= 13.40/2), x: 0 ~ 6.10
#   - 단식 사이드라인: x = 0.46, x = 5.64, y: 0 ~ 13.40
#   - 더블 롱서비스라인(아래): y = 0.76,  x: 0 ~ 6.10
#   - 더블 롱서비스라인(위):   y = 12.64, x: 0 ~ 6.10
#   - 숏서비스라인(아래):      y = 4.72,  x: 0 ~ 6.10  (= 6.70 - 1.98)
#   - 숏서비스라인(위):        y = 8.68,  x: 0 ~ 6.10  (= 6.70 + 1.98)
#   - 센터라인(아래 코트):     x = 3.05,  y: 0 ~ 4.72
#   - 센터라인(위 코트):       x = 3.05,  y: 8.68 ~ 13.40
# ─────────────────────────────────────────────────────────────────────────────

_L  = BADMINTON_COURT["length"]            # 13.40
_W  = BADMINTON_COURT["width_doubles"]     # 6.10
_NY = BADMINTON_COURT["net_y"]             # 6.70
_SS = BADMINTON_COURT["net_to_short_svc"]  # 1.98
_LS = BADMINTON_COURT["back_to_long_svc"]  # 0.76
_SD = BADMINTON_COURT["singles_side"]      # 0.46
_CX = BADMINTON_COURT["center_x"]          # 3.05

# 유도 좌표
_Y_SHORT_BOT = _NY - _SS      # 4.72  (아래 숏서비스라인)
_Y_SHORT_TOP = _NY + _SS      # 8.68  (위 숏서비스라인)
_Y_LONG_BOT  = _LS            # 0.76  (아래 더블 롱서비스라인)
_Y_LONG_TOP  = _L - _LS       # 12.64 (위 더블 롱서비스라인)
_X_SGL_L     = _SD            # 0.46  (왼쪽 단식 사이드라인)
_X_SGL_R     = _W - _SD       # 5.64  (오른쪽 단식 사이드라인)

COURT_LINES_SPEC = [
    # ── 외곽 경계선 (복식) ──────────────────────────────────
    {"name": "bottom_boundary",      "x1": 0.0,  "y1": 0.0,   "x2": _W,   "y2": 0.0,   "type": "boundary"},
    {"name": "top_boundary",         "x1": 0.0,  "y1": _L,    "x2": _W,   "y2": _L,    "type": "boundary"},
    {"name": "left_boundary",        "x1": 0.0,  "y1": 0.0,   "x2": 0.0,  "y2": _L,    "type": "boundary"},
    {"name": "right_boundary",       "x1": _W,   "y1": 0.0,   "x2": _W,   "y2": _L,    "type": "boundary"},

    # ── 단식 사이드라인 ─────────────────────────────────────
    {"name": "left_singles_sideline",  "x1": _X_SGL_L, "y1": 0.0, "x2": _X_SGL_L, "y2": _L,  "type": "sideline"},
    {"name": "right_singles_sideline", "x1": _X_SGL_R, "y1": 0.0, "x2": _X_SGL_R, "y2": _L,  "type": "sideline"},

    # ── 네트 라인 ───────────────────────────────────────────
    {"name": "net",                  "x1": 0.0,  "y1": _NY,   "x2": _W,   "y2": _NY,   "type": "net"},

    # ── 숏서비스라인 (앞 서비스라인) ─────────────────────────
    {"name": "short_service_line_bot", "x1": 0.0, "y1": _Y_SHORT_BOT, "x2": _W, "y2": _Y_SHORT_BOT, "type": "service"},
    {"name": "short_service_line_top", "x1": 0.0, "y1": _Y_SHORT_TOP, "x2": _W, "y2": _Y_SHORT_TOP, "type": "service"},

    # ── 더블 롱서비스라인 ────────────────────────────────────
    {"name": "long_service_line_bot_doubles", "x1": 0.0, "y1": _Y_LONG_BOT, "x2": _W, "y2": _Y_LONG_BOT, "type": "service"},
    {"name": "long_service_line_top_doubles", "x1": 0.0, "y1": _Y_LONG_TOP, "x2": _W, "y2": _Y_LONG_TOP, "type": "service"},

    # ── 센터 서비스 라인 ─────────────────────────────────────
    # 아래 코트: 엔드라인 ~ 아래 숏서비스라인
    {"name": "center_service_line_bot", "x1": _CX, "y1": 0.0,          "x2": _CX, "y2": _Y_SHORT_BOT, "type": "center"},
    # 위 코트: 위 숏서비스라인 ~ 위 엔드라인
    {"name": "center_service_line_top", "x1": _CX, "y1": _Y_SHORT_TOP, "x2": _CX, "y2": _L,           "type": "center"},
]

# ─────────────────────────────────────────────────────────────────────────────
# 서비스 코트 구역 정의 (인/아웃 판별용)
#
# 원점 기준 직사각형: (x_min, y_min, x_max, y_max)
# 단식: 폭 = 5.18m (x: 0.46 ~ 5.64), 더블: 폭 = 6.10m (x: 0 ~ 6.10)
# ─────────────────────────────────────────────────────────────────────────────
SERVICE_COURTS = {
    # ── 단식 서비스 코트 ─────────────────────────────────────
    # 오른쪽 서비스코트 (아래): 네트 기준 오른쪽 대각선 방향 서브
    "singles_right_service_bot": {
        "x_min": _CX, "y_min": 0.0,          "x_max": _X_SGL_R, "y_max": _Y_SHORT_BOT,
        "desc": "단식 아래쪽 오른쪽 서비스코트 (엔드라인 ~ 숏서비스라인, 센터~오른쪽 단식)"
    },
    # 왼쪽 서비스코트 (아래)
    "singles_left_service_bot": {
        "x_min": _X_SGL_L, "y_min": 0.0,     "x_max": _CX,      "y_max": _Y_SHORT_BOT,
        "desc": "단식 아래쪽 왼쪽 서비스코트"
    },
    # 오른쪽 서비스코트 (위)
    "singles_right_service_top": {
        "x_min": _CX, "y_min": _Y_SHORT_TOP, "x_max": _X_SGL_R, "y_max": _L,
        "desc": "단식 위쪽 오른쪽 서비스코트"
    },
    # 왼쪽 서비스코트 (위)
    "singles_left_service_top": {
        "x_min": _X_SGL_L, "y_min": _Y_SHORT_TOP, "x_max": _CX, "y_max": _L,
        "desc": "단식 위쪽 왼쪽 서비스코트"
    },

    # ── 복식 서비스 코트 ─────────────────────────────────────
    # 더블 서브: 엔드라인 ~ 더블 롱서비스라인(0.76m 안쪽), 더블 폭 전체
    "doubles_right_service_bot": {
        "x_min": _CX,  "y_min": _Y_LONG_BOT, "x_max": _W,   "y_max": _Y_SHORT_BOT,
        "desc": "복식 아래쪽 오른쪽 서비스코트 (더블롱서비스라인 ~ 숏서비스라인)"
    },
    "doubles_left_service_bot": {
        "x_min": 0.0,  "y_min": _Y_LONG_BOT, "x_max": _CX,  "y_max": _Y_SHORT_BOT,
        "desc": "복식 아래쪽 왼쪽 서비스코트"
    },
    "doubles_right_service_top": {
        "x_min": _CX,  "y_min": _Y_SHORT_TOP, "x_max": _W,  "y_max": _Y_LONG_TOP,
        "desc": "복식 위쪽 오른쪽 서비스코트"
    },
    "doubles_left_service_top": {
        "x_min": 0.0,  "y_min": _Y_SHORT_TOP, "x_max": _CX, "y_max": _Y_LONG_TOP,
        "desc": "복식 위쪽 왼쪽 서비스코트"
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# 미니맵 렌더링 비율 (0.0 ~ 1.0)
#
# 미니맵 좌표계:
#   y=0.0 → 미니맵 상단 (코트 위쪽 엔드라인, y=13.40m)
#   y=1.0 → 미니맵 하단 (코트 아래쪽 엔드라인, y=0m)
#   x=0.0 → 미니맵 왼쪽 (코트 왼쪽, x=0m)
#   x=1.0 → 미니맵 오른쪽 (코트 오른쪽, x=6.10m)
#
# 중요: 미니맵은 Y축이 반전됨 (화면 좌표계: 위 = 작은 y값)
#   실제 y=0.00 (아래 엔드라인) → 미니맵 y_ratio = 1.0 (하단)
#   실제 y=13.40 (위 엔드라인) → 미니맵 y_ratio = 0.0 (상단)
# ─────────────────────────────────────────────────────────────────────────────
def _y_ratio(y_m: float) -> float:
    """실제 y좌표(m) → 미니맵 비율 (y축 반전)"""
    return 1.0 - y_m / _L

def _x_ratio(x_m: float) -> float:
    """실제 x좌표(m) → 미니맵 비율"""
    return x_m / _W

COURT_RATIOS = {
    # ── Y축 비율 (미니맵 상단=0, 하단=1) ────────────────────
    # 위쪽 코트 (미니맵 상단 영역)
    "top_boundary":      _y_ratio(_L),           # 0.0   ← 위 엔드라인
    "long_svc_top":      _y_ratio(_Y_LONG_TOP),  # ≈0.057 ← 위 더블 롱서비스
    "short_svc_top":     _y_ratio(_Y_SHORT_TOP), # ≈0.352 ← 위 숏서비스
    "net":               _y_ratio(_NY),           # 0.5   ← 네트
    "short_svc_bot":     _y_ratio(_Y_SHORT_BOT), # ≈0.648 ← 아래 숏서비스
    "long_svc_bot":      _y_ratio(_Y_LONG_BOT),  # ≈0.943 ← 아래 더블 롱서비스
    "bottom_boundary":   _y_ratio(0.0),           # 1.0   ← 아래 엔드라인

    # ── X축 비율 ─────────────────────────────────────────────
    "left_boundary":     _x_ratio(0.0),           # 0.0
    "singles_left":      _x_ratio(_X_SGL_L),      # ≈0.075
    "center":            _x_ratio(_CX),            # 0.5
    "singles_right":     _x_ratio(_X_SGL_R),      # ≈0.925
    "right_boundary":    _x_ratio(_W),             # 1.0
}

# ─────────────────────────────────────────────────────────────────────────────
# 미니맵 캔버스 설정
#
# 비율: BWF 코트 13.40m(길이) × 6.10m(폭) = 2.197:1
# 캔버스: width=304px 기준 → height = 304 * 13.40/6.10 ≈ 668px
# → 패딩 25px 포함하여 width=304+50=354, height=668+50=718로 맞춤
# 실용 크기: width=310, height=680 (살짝 여유 포함)
# ─────────────────────────────────────────────────────────────────────────────
MINIMAP_CONFIG = {
    "width":   310,
    "height":  680,
    "padding":  25,
    # 라인 색상 (BGR)
    "line_color":    (255, 255, 255),
    "net_color":     (0,   80,  255),
    # 선수 색상
    "top_color":     (255, 105, 180),   # 핑크 (미니맵 상단 선수)
    "bottom_color":  (50,  205,  50),   # 라임 (미니맵 하단 선수)
    # 셔틀콕
    "shuttle_color": (0, 200, 255),
    "hit_radius":    12,
    "trail_length":  30,
}

# ─────────────────────────────────────────────────────────────────────────────
# 코트 코너 자동 검출 설정
# ─────────────────────────────────────────────────────────────────────────────
COURT_DETECTOR_CONFIG = {
    "sample_frames":         20,
    "max_sample_sec":        30,
    "green_ratio_min":       0.10,
    "hough_threshold":       50,
    "hough_min_line_length": 80,
    "hough_max_line_gap":    20,
    "vote_min_frames":       3,
    "canny_low":             50,
    "canny_high":            150,
}

# ─────────────────────────────────────────────────────────────────────────────
# 타점 감지 파라미터
# ─────────────────────────────────────────────────────────────────────────────
IMPACT_CONFIG = {
    "min_speed":             3.0,
    "max_frame_gap":         5,
    "skip_angle_thresh":     15.0,
    "min_flight_dist":       60,
    "peak_distance":         12,
    "peak_threshold_ratio":  0.15,
}

# ─────────────────────────────────────────────────────────────────────────────
# YOLOv8-pose 설정
# ─────────────────────────────────────────────────────────────────────────────
POSE_CONFIG = {
    "confidence":     0.3,
    "keypoint_conf":  0.3,
    "scale":          1.2,
    "head_radius":    12,
    "bone_thickness": 4,
}

SKELETON_EDGES = [
    (16, 14), (14, 12), (17, 15), (15, 13),
    (12, 13), (6, 12),  (7, 13),  (6, 7),
    (6, 8),   (7, 9),   (8, 10),  (9, 11),
]

# ─────────────────────────────────────────────────────────────────────────────
# 웹 연동 데이터 익스포트 설정
# ─────────────────────────────────────────────────────────────────────────────
WEB_EXPORT_CONFIG = {
    "export_frame_data":         True,
    "include_player_positions":  True,
    "include_shuttle_trail":     True,
    "shuttle_trail_length":      30,
    "frame_data_suffix":         "_frame_data.json",
    "summary_suffix":            "_summary.json",
}
