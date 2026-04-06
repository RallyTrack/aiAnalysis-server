"""
RallyTrack AI 분석 라우터

- POST /analyze : 백엔드에서 호출하는 영상 분석 엔드포인트
- 분석 완료 후 minimap / skeleton 영상을 S3에 업로드하고 백엔드에 콜백
"""
import os
import httpx
from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from typing import Optional

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


class CourtCornerPoint(BaseModel):
    x: int
    y: int


class CourtCorners(BaseModel):
    topLeft: CourtCornerPoint
    topRight: CourtCornerPoint
    bottomLeft: CourtCornerPoint
    bottomRight: CourtCornerPoint


class AnalyzeRequest(BaseModel):
    videoId: int
    s3Url: str
    # skeleton
    skeletonUploadUrl: str = ""
    skeletonVideoUrl: str = ""
    # minimap (NEW)
    minimapUploadUrl: str = ""
    minimapVideoUrl: str = ""
    # courtCorncers
    courtCorners: Optional[CourtCorners] = None


@router.post("/analyze")
def analyze_video(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    """
    백엔드에서 영상 업로드 후 호출하는 엔드포인트.
    분석 작업을 백그라운드로 실행하고 즉시 응답을 반환한다.
    """
    print(f"[분석 요청 수신] videoId={request.videoId}")
    background_tasks.add_task(
        run_analysis,
        request.videoId,
        request.s3Url,
        request.skeletonUploadUrl,
        request.skeletonVideoUrl,
        request.minimapUploadUrl,
        request.minimapVideoUrl,
        request.courtCorners,
    )
    return {"message": "분석이 시작되었습니다.", "videoId": request.videoId}


def _upload_to_s3(local_path: str, upload_url: str, label: str) -> bool:
    """
    로컬 파일을 S3 presigned PUT URL로 업로드한다.

    Returns:
        True = 성공, False = 실패
    """
    if not upload_url:
        print(f"[{label} 업로드] upload_url 없음 → 생략")
        return False
    if not os.path.exists(local_path):
        print(f"[{label} 업로드] 파일 없음: {local_path}")
        return False

    try:
        with open(local_path, "rb") as f:
            response = httpx.put(
                upload_url,
                content=f.read(),
                headers={"Content-Type": "video/mp4"},
                timeout=180.0,
            )
        if response.status_code in (200, 204):
            print(f"[{label} 업로드] 완료")
            return True
        else:
            print(f"[{label} 업로드] 실패 — HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"[{label} 업로드] 실패 — {e}")
        return False


def run_analysis(
    video_id: int,
    s3_url: str,
    skeleton_upload_url: str = "",
    skeleton_video_url: str = "",
    minimap_upload_url: str = "",
    minimap_video_url: str = "",
    court_corners=None,
):
    """
    실제 분석 파이프라인.

    1. S3에서 영상 다운로드
    2. RallyTrackPipeline 실행
       → TrackNetV3 셔틀콕 트래킹
       → YOLOv8-pose 관절 추출
       → 물리 기반 타점 감지 (ImpactDetector)
       → 영상 3종 렌더링 (main / minimap / skeleton)
    3. skeleton + minimap 영상 S3 업로드
    4. 백엔드 콜백
    """
    local_path = None

    try:
        print(f"[분석 시작] videoId={video_id}")

        # 1. S3에서 영상 다운로드
        local_path = download_video(s3_url)

        # 2. RallyTrackPipeline 실행
        pipeline = _get_pipeline()
        api_data = pipeline.run(local_path, court_corners)
        result_paths = api_data.pop("result_paths", {})

        # 3. S3 업로드 (skeleton + minimap)
        _upload_to_s3(result_paths.get("skeleton_video", ""), skeleton_upload_url, "skeleton")
        _upload_to_s3(result_paths.get("minimap_video", ""),  minimap_upload_url,  "minimap")

        # 4. 백엔드 콜백 데이터 구성
        callback_data = {
            "videoId":   video_id,
            "videoFps":  api_data["video_fps"],
            "totalHits": api_data["total_hits"],
            "hitsData":  api_data["hits_data"],  # list 그대로 전송 (JSON 직렬화는 httpx가 처리)
        }
        if skeleton_video_url:
            callback_data["skeletonVideoUrl"] = skeleton_video_url
        if minimap_video_url:
            callback_data["minimapVideoUrl"] = minimap_video_url

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
