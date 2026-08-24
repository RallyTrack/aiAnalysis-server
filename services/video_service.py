"""
영상 다운로드 서비스
- S3 URL에서 영상을 로컬 임시 디렉토리로 다운로드
"""
import os
import uuid
import httpx
from config.settings import TEMP_VIDEO_DIR


def download_video(s3_url: str) -> str:
    """
    S3 URL에서 영상을 다운로드하여 로컬 임시 경로를 반환한다.

    Args:
        s3_url: AWS S3 영상 URL

    Returns:
        다운로드된 로컬 파일 경로
    """
    file_name = f"{uuid.uuid4()}.mp4"
    local_path = os.path.join(TEMP_VIDEO_DIR, file_name)

    print(f"[영상 다운로드] {s3_url} → {local_path}")

    # 스트리밍 다운로드: 영상 전체를 RAM에 올리지 않고 1MB 단위로 디스크에 기록
    with httpx.stream("GET", s3_url, timeout=httpx.Timeout(10.0, read=600.0)) as response:
        response.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in response.iter_bytes(1024 * 1024):
                f.write(chunk)

    size_mb = os.path.getsize(local_path) / 1024 / 1024
    print(f"[영상 다운로드 완료] 크기: {size_mb:.1f}MB")
    return local_path


def cleanup_video(local_path: str):
    """임시 영상 파일 삭제"""
    if os.path.exists(local_path):
        os.remove(local_path)
        print(f"[임시 파일 삭제] {local_path}")
