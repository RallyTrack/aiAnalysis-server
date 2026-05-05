"""
RallyTrack AI 분석 라우터

- POST /analyze : 백엔드에서 호출하는 영상 분석 엔드포인트
- 분석 완료 후 minimap / skeleton 영상을 S3에 업로드하고 백엔드에 콜백

[수정 사항]
  - AnalyzeRequest에 singles_corners (코트 4점) 및 net_coords (네트 2점) 필드 추가
  - run_analysis()에 좌표 데이터 전달
"""
import os
from typing import Optional

import httpx
from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel

from config.settings import BACKEND_URL
from services.video_service import cleanup_video, download_video
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
    netTopLeft: Optional[CourtCornerPoint] = None
    netTopRight: Optional[CourtCornerPoint] = None


class AnalyzeRequest(BaseModel):
    videoId: int
    s3Url: str
    skeletonUploadUrl: str = ""
    skeletonVideoUrl: str = ""
    minimapUploadUrl: str = ""
    minimapVideoUrl: str = ""
    courtCorners: Optional[CourtCorners] = None


@router.post("/analyze")
def analyze_video(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    """
    백엔드에서 영상 업로드 후 호출하는 엔드포인트.
    분석 작업을 백그라운드로 실행하고 즉시 응답을 반환한다.
    """
    print(f"[분석 요청 수신] videoId={request.videoId}")
    print(f"  courtCorners 제공 여부: {request.courtCorners is not None}")
    if request.courtCorners is not None:
        print(f"  netTopLeft 제공 여부:   {request.courtCorners.netTopLeft is not None}")

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
    court_corners: Optional[CourtCorners] = None,
):
    """
    실제 분석 파이프라인.

    1. S3에서 영상 다운로드
    2. courtCorners → user_corners / net_coords 변환
    3. RallyTrackPipeline 실행
    4. skeleton + minimap 영상 S3 업로드
    5. 백엔드 콜백
    """
    local_path = None

    try:
        print(f"[분석 시작] videoId={video_id}")

        # 1. S3에서 영상 다운로드
        local_path = download_video(s3_url)

        # 2. courtCorners → pipeline 내부 형식으로 변환
        # user_corners 순서: [TL, TR, BR, BL] (compute_homographies 기대 순서)
        user_corners = None
        net_coords   = None
        if court_corners is not None:
            user_corners = [
                [court_corners.topLeft.x,     court_corners.topLeft.y],
                [court_corners.topRight.x,    court_corners.topRight.y],
                [court_corners.bottomRight.x, court_corners.bottomRight.y],
                [court_corners.bottomLeft.x,  court_corners.bottomLeft.y],
            ]
            if court_corners.netTopLeft is not None and court_corners.netTopRight is not None:
                net_coords = [
                    [court_corners.netTopLeft.x,  court_corners.netTopLeft.y],
                    [court_corners.netTopRight.x, court_corners.netTopRight.y],
                ]

        # 3. RallyTrackPipeline 실행
        pipeline = _get_pipeline()
        api_data = pipeline.run(
            local_path,
            user_corners=user_corners,
            net_coords=net_coords,
        )
        result_paths = api_data.pop("result_paths", {})

        # 3. S3 업로드 (skeleton + minimap)
        _upload_to_s3(result_paths.get("skeleton_video", ""), skeleton_upload_url, "skeleton")
        _upload_to_s3(result_paths.get("minimap_video", ""),  minimap_upload_url,  "minimap")

        # 4. 백엔드 콜백 데이터 구성
        callback_data = {
            "videoId":        video_id,
            "videoFps":       api_data["video_fps"],
            "totalHits":      api_data["total_hits"],
            "hitsData":       api_data["hits_data"],
            "netFaultEvents": api_data.get("net_fault_events", []),
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