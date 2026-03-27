"""
RallyTrack AI 분석서버 설정
"""
import os

# 백엔드 서버 주소
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")

# 영상 임시 저장 경로
TEMP_VIDEO_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp_videos")
os.makedirs(TEMP_VIDEO_DIR, exist_ok=True)

# 결과물 저장 경로
RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "result")
os.makedirs(RESULT_DIR, exist_ok=True)

# TrackNet 예측 CSV 저장 경로
PREDICTION_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prediction")
os.makedirs(PREDICTION_DIR, exist_ok=True)
