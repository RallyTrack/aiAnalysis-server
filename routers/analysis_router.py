"""
분석 라우터 (리팩토링 - RallyTrackPipeline 기반)

- POST /analyze : 백엔드에서 호출하는 영상 분석 엔드포인트
- 동료(feat/intergration 브랜치)의 파이프라인을 FastAPI에 통합
- 콜백 데이터: hits_data JSON 형식으로 전송
"""
import json
import httpx
from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel

from config.settings import BACKEND_URL
from services.video_service import download_video, cleanup_video
from services.pipeline_service import RallyTrackPipeline

router = APIRouter()

# 파이프라인 싱글턴 (YOLO 모델을 서버 시작 시 한 번만 로드)
_pipeline: RallyTrackPipeline | None = None


def _get_pipeline() -> RallyTrackPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RallyTrackPipeline()
    return _pipeline


class AnalyzeRequest(BaseModel):
    videoId: int
    s3Url: str
    skeletonUploadUrl: str = ""
    skeletonVideoUrl: str = ""


@router.post("/analyze")
def analyze_video(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    """
    백엔드에서 영상 업로드 후 호출하는 엔드포인트.
    분석 작업을 백그라운드로 실행하고 즉시 응답을 반환한다.
    """
    print(f"[분석 요청 수신] videoId={request.videoId}, s3Url={request.s3Url}")
    background_tasks.add_task(
        run_analysis,
        request.videoId,
        request.s3Url,
        request.skeletonUploadUrl,
        request.skeletonVideoUrl,
    )
    return {"message": "분석이 시작되었습니다.", "videoId": request.videoId}


def run_analysis(
    video_id: int,
    s3_url: str,
    skeleton_upload_url: str = "",
    skeleton_video_url: str = "",
):
    """
    실제 분석 파이프라인.

    1. S3에서 영상 다운로드
    2. RallyTrackPipeline 실행
       → TrackNetV3 셔틀콕 트래킹
       → YOLOv8-pose 관절 추출
       → 물리 기반 타점 감지 (ImpactDetector)
       → 영상 3종 렌더링 (main / minimap / skeleton)
    3. 스켈레톤 영상 S3 업로드 (선택)
    4. 백엔드 콜백 — hits_data JSON 형식
    """
    local_path = None

    try:
        print(f"[분석 시작] videoId={video_id}")

        # 1. S3에서 영상 다운로드
        local_path = download_video(s3_url)

        # 2. RallyTrackPipeline 실행
        pipeline = _get_pipeline()
        api_data = pipeline.run(local_path)

        # result_paths는 콜백에 포함하지 않음 (내부용)
        result_paths = api_data.pop("result_paths", {})

        # 3. 스켈레톤 영상 S3 업로드 (skeleton_upload_url 제공 시)
        skeleton_video_path = result_paths.get("skeleton_video", "")
        if skeleton_upload_url and skeleton_video_path:
            try:
                import os
                if os.path.exists(skeleton_video_path):
                    with open(skeleton_video_path, "rb") as f:
                        httpx.put(
                            skeleton_upload_url,
                            content=f.read(),
                            headers={"Content-Type": "video/mp4"},
                            timeout=120.0,
                        )
                    print("[스켈레톤 업로드] 완료")
                else:
                    print(f"[스켈레톤 업로드] 파일 없음: {skeleton_video_path}")
            except Exception as e:
                print(f"[스켈레톤 업로드 실패] {e}")

        # 4. 백엔드 콜백 데이터 구성
        callback_data = {
            "videoId":          video_id,
            "videoFps":         api_data["video_fps"],
            "totalHits":        api_data["total_hits"],
            "hitsData":         json.dumps(api_data["hits_data"], ensure_ascii=False),
        }

        # skeletonVideoUrl이 있으면 포함
        if skeleton_video_url:
            callback_data["skeletonVideoUrl"] = skeleton_video_url

        # 5. 백엔드 콜백 호출
        response = httpx.post(
            f"{BACKEND_URL}/api/v1/analysis/complete",
            json=callback_data,
            timeout=30.0,
        )
        print(f"[콜백 완료] videoId={video_id}, status={response.status_code}")

    except Exception as e:
        print(f"[분석 실패] videoId={video_id}, error={e}")
        import traceback
        traceback.print_exc()

        # 실패 시 백엔드에 에러 알림
        try:
            httpx.post(
                f"{BACKEND_URL}/api/v1/analysis/fail",
                json={"videoId": video_id, "error": str(e)},
                timeout=10.0,
            )
        except Exception:
            pass

    finally:
        if local_path:
            cleanup_video(local_path)
