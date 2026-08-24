"""
RallyTrack AI 분석서버 설정
"""
import os

# 백엔드 서버 주소
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")

# 백엔드 콜백 인증용 공유 시크릿 (backend의 ANALYSIS_CALLBACK_SECRET와 동일 값)
CALLBACK_SECRET = os.getenv("ANALYSIS_CALLBACK_SECRET", "")

# Slack Incoming Webhook — rally-track-analysis-log 채널 (비우면 알림 비활성)
SLACK_ANALYSIS_WEBHOOK = os.getenv("SLACK_ANALYSIS_WEBHOOK", "")

# 영상 임시 저장 경로
TEMP_VIDEO_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp_videos")
os.makedirs(TEMP_VIDEO_DIR, exist_ok=True)

# 결과물 저장 경로
RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "result")
os.makedirs(RESULT_DIR, exist_ok=True)

# TrackNet 예측 CSV 저장 경로
PREDICTION_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prediction")
os.makedirs(PREDICTION_DIR, exist_ok=True)
