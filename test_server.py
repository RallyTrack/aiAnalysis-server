"""
RallyTrack - 서버 통합 테스트

실행 중인 서버에 HTTP 요청을 보내서 엔드포인트를 검증합니다.

사전 조건:
    uvicorn main:app --reload --port 8000  # 서버 먼저 실행

실행:
    python test_server.py                          # health + mock 요청
    python test_server.py --s3-url <url>           # 실제 S3 URL로 분석 트리거
    python test_server.py --host http://0.0.0.0:8000  # 호스트 변경
"""

import argparse
import json
import time
import httpx


def test_health(base_url: str) -> bool:
    """GET /health"""
    print("\n[1단계] Health check")
    try:
        r = httpx.get(f"{base_url}/health", timeout=5.0)
        if r.status_code == 200 and r.json().get("status") == "ok":
            print(f"  ✓ {r.status_code} {r.json()}")
            return True
        print(f"  ✗ 예상치 못한 응답: {r.status_code} {r.text}")
        return False
    except Exception as e:
        print(f"  ✗ 연결 실패: {e}")
        print(f"    서버가 실행 중인지 확인: uvicorn main:app --reload --port 8000")
        return False


def test_root(base_url: str) -> bool:
    """GET /"""
    print("\n[2단계] Root endpoint")
    try:
        r = httpx.get(f"{base_url}/", timeout=5.0)
        print(f"  ✓ {r.status_code} {r.json()}")
        return r.status_code == 200
    except Exception as e:
        print(f"  ✗ 실패: {e}")
        return False


def test_analyze_mock(base_url: str) -> bool:
    """
    POST /analyze — 잘못된 S3 URL로 요청
    백그라운드 작업이므로 즉시 202 응답이 오는지만 확인
    (실제 분석 실패는 서버 로그에서 확인)
    """
    print("\n[3단계] POST /analyze (mock 요청)")
    payload = {
        "videoId": 9999,
        "s3Url": "https://invalid.example.com/test.mp4",
        "skeletonUploadUrl": "",
        "skeletonVideoUrl": "",
        "minimapUploadUrl": "",
        "minimapVideoUrl": "",
    }
    try:
        r = httpx.post(f"{base_url}/analyze", json=payload, timeout=10.0)
        body = r.json()
        if r.status_code == 200 and body.get("videoId") == 9999:
            print(f"  ✓ {r.status_code} — 백그라운드 작업 접수됨")
            print(f"    응답: {body}")
            print(f"    ℹ 실제 분석 결과는 서버 로그에서 확인하세요")
            return True
        print(f"  ✗ 예상치 못한 응답: {r.status_code} {r.text}")
        return False
    except Exception as e:
        print(f"  ✗ 실패: {e}")
        return False


def test_analyze_real(base_url: str, s3_url: str) -> bool:
    """
    POST /analyze — 실제 S3 URL로 분석 트리거
    요청 후 콜백 수신은 백엔드 서버에서 확인
    """
    print(f"\n[4단계] POST /analyze (실제 요청)")
    print(f"  s3_url: {s3_url}")

    payload = {
        "videoId": 1,
        "s3Url": s3_url,
        "skeletonUploadUrl": "",
        "skeletonVideoUrl": "",
        "minimapUploadUrl": "",
        "minimapVideoUrl": "",
    }
    try:
        r = httpx.post(f"{base_url}/analyze", json=payload, timeout=10.0)
        body = r.json()
        print(f"  ✓ {r.status_code} — 분석 시작됨")
        print(f"    응답: {body}")
        print(f"  ℹ 분석 진행 상황은 서버 로그를 확인하세요")
        return r.status_code == 200
    except Exception as e:
        print(f"  ✗ 실패: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="RallyTrack 서버 통합 테스트")
    parser.add_argument("--host", default="http://localhost:8000", help="서버 주소")
    parser.add_argument("--s3-url", default=None, help="실제 분석에 사용할 S3 영상 URL")
    args = parser.parse_args()

    base = args.host.rstrip("/")

    print("=" * 55)
    print("  RallyTrack 서버 통합 테스트")
    print(f"  대상: {base}")
    print("=" * 55)

    results = {}
    results["health"]  = test_health(base)

    if not results["health"]:
        print("\n서버에 연결할 수 없어 테스트를 중단합니다.")
        print("아래 명령으로 서버를 먼저 실행하세요:")
        print("  uvicorn main:app --reload --port 8000")
        return

    results["root"]    = test_root(base)
    results["mock"]    = test_analyze_mock(base)

    if args.s3_url:
        results["real"] = test_analyze_real(base, args.s3_url)

    print("\n" + "=" * 55)
    print("  테스트 결과 요약")
    print("=" * 55)
    for name, ok in results.items():
        icon = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {icon}  {name}")
    print("=" * 55)


if __name__ == "__main__":
    main()
