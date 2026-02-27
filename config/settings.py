"""
RallyTrack AI 분석서버 설정
"""
import os

# 백엔드 서버 주소
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")

# 모델 가중치 경로
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights")
MLP_MODEL_PATH = os.path.join(WEIGHTS_DIR, "RT_90plus_model.pth")
STGCN_MODEL_PATH = os.path.join(WEIGHTS_DIR, "RT_stgcn_model.pth")

# 영상 임시 저장 경로
TEMP_VIDEO_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp_videos")
os.makedirs(TEMP_VIDEO_DIR, exist_ok=True)

# MediaPipe 설정
POSE_MIN_DETECTION_CONFIDENCE = 0.5
POSE_MIN_TRACKING_CONFIDENCE = 0.5

# 임팩트 감지 설정
IMPACT_SENSITIVITY = 0.4
IMPACT_COOL_DOWN = 0.5

# 스트로크 라벨
STROKE_LABELS = ["Smash", "Clear", "Drop", "Drive"]
