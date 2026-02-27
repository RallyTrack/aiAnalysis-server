"""
분석 라우터
- POST /analyze: 백엔드에서 호출하는 영상 분석 엔드포인트
"""
import json
import httpx
from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel

from config.settings import BACKEND_URL
from services.video_service import download_video, cleanup_video
from services.pose_service import detect_impacts, extract_player_positions
from services.stroke_service import classify_strokes_stgcn
from services.heatmap_service import generate_heatmap
from services.score_service import calculate_score, generate_timeline_events
from services.feedback_service import generate_feedback, generate_ability_metrics

router = APIRouter()


class AnalyzeRequest(BaseModel):
    videoId: int
    s3Url: str


@router.post("/analyze")
def analyze_video(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    """
    백엔드에서 영상 업로드 후 호출하는 엔드포인트.
    분석 작업을 백그라운드로 실행하고 즉시 응답을 반환한다.
    """
    print(f"[분석 요청 수신] videoId={request.videoId}, s3Url={request.s3Url}")
    background_tasks.add_task(run_analysis, request.videoId, request.s3Url)
    return {"message": "분석이 시작되었습니다.", "videoId": request.videoId}


def run_analysis(video_id: int, s3_url: str):
    """
    실제 분석 파이프라인.

    1. S3에서 영상 다운로드
    2. MediaPipe로 관절 추출 + 임팩트 감지
    3. ST-GCN으로 스트로크 분류
    4. 히트맵 데이터 생성
    5. 점수 계산 + 타임라인 생성
    6. AI 피드백 생성
    7. 백엔드 콜백
    """
    local_path = None

    try:
        print(f"[분석 시작] videoId={video_id}")

        # 1. S3에서 영상 다운로드
        local_path = download_video(s3_url)

        # 2. MediaPipe로 관절 추출 + 임팩트 감지
        sequences, timestamps = detect_impacts(local_path)

        # 3. ST-GCN으로 스트로크 분류
        if len(sequences) > 0:
            stroke_results = classify_strokes_stgcn(sequences)
        else:
            stroke_results = []
            print(f"[경고] videoId={video_id}: 임팩트 감지 실패, 빈 결과 반환")

        # 4. 선수 위치 추출 + 히트맵 생성
        positions = extract_player_positions(local_path)
        heatmap_data = generate_heatmap(positions)

        # 5. 점수 계산 + 타임라인 생성
        score = calculate_score(stroke_results, timestamps)
        timeline_events = generate_timeline_events(stroke_results, timestamps)

        # 6. AI 피드백 + 능력치 생성
        feedback = generate_feedback(stroke_results, score)
        ability_metrics = generate_ability_metrics(stroke_results, score)

        # 7. 스트로크 분포 집계
        stroke_types = {}
        for result in stroke_results:
            label = result["label"].lower()
            stroke_types[label] = stroke_types.get(label, 0) + 1

        # 콜백 데이터 구성
        callback_data = {
            "videoId": video_id,
            "myScore": score["myScore"],
            "opponentScore": score["opponentScore"],
            "matchOutcome": score["matchOutcome"],
            "totalStrokeCount": score["totalStrokeCount"],
            "matchTime": score["matchTime"],
            "heatmapData": json.dumps(heatmap_data),
            "strokeTypes": json.dumps(stroke_types),
            "abilityMetrics": json.dumps(ability_metrics),
            "aiFeedback": feedback,
            "timelineEvents": timeline_events,
        }

        # 백엔드 콜백 호출
        response = httpx.post(
            f"{BACKEND_URL}/api/v1/analysis/complete",
            json=callback_data,
            timeout=30.0,
        )
        print(f"[콜백 완료] videoId={video_id}, status={response.status_code}")

    except Exception as e:
        print(f"[분석 실패] videoId={video_id}, error={e}")

        # 실패 시에도 콜백을 보내 상태를 FAILED로 변경할 수 있도록 함
        # TODO: 백엔드에 실패 콜백 엔드포인트 추가 시 활성화
        # try:
        #     httpx.post(f"{BACKEND_URL}/api/v1/analysis/failed",
        #                json={"videoId": video_id, "error": str(e)}, timeout=10.0)
        # except:
        #     pass

    finally:
        # 임시 영상 파일 삭제
        if local_path:
            cleanup_video(local_path)
