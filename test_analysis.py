"""
RallyTrack - analysis/ 패키지 독립 실행 테스트

서버 없이 analysis/ 모듈만 직접 테스트합니다.
영상 파일 경로만 지정하면 전체 파이프라인을 로컬에서 돌려볼 수 있어요.

실행:
    python test_analysis.py --video /path/to/video.mp4
    python test_analysis.py --video /path/to/video.mp4 --skip-tracknet
"""

import argparse
import json
import sys
import traceback
from pathlib import Path


def test_imports():
    """1단계: 모든 모듈 import 가능한지 확인"""
    print("\n[1단계] import 테스트")
    try:
        from analysis.config import PATHS, MINIMAP_CONFIG, IMPACT_CONFIG, POSE_CONFIG
        print(f"  ✓ analysis.config")
        print(f"    tracknet_dir : {PATHS['tracknet_dir']}")
        print(f"    yolo_model   : {PATHS['yolo_model']}")
        print(f"    result_dir   : {PATHS['result_dir']}")

        from analysis.court import compute_homographies, create_minimap_canvas
        print(f"  ✓ analysis.court")

        from analysis.impact import ImpactDetector, to_api_json
        print(f"  ✓ analysis.impact")

        from analysis.minimap import MinimapRenderer
        print(f"  ✓ analysis.minimap")

        from analysis.skeleton_view import SkeletonCourtRenderer
        print(f"  ✓ analysis.skeleton_view")

        from services.pipeline_service import RallyTrackPipeline
        print(f"  ✓ services.pipeline_service")

        return True
    except ImportError as e:
        print(f"  ✗ import 실패: {e}")
        traceback.print_exc()
        return False


def test_paths():
    """2단계: 필수 파일/디렉토리 존재 확인"""
    import os
    from analysis.config import PATHS

    print("\n[2단계] 경로 확인")
    results = []

    checks = [
        ("tracknet_dir",    PATHS["tracknet_dir"],    "dir"),
        ("tracknet_ckpt",   PATHS["tracknet_ckpt"],   "file"),
        ("inpaintnet_ckpt", PATHS["inpaintnet_ckpt"], "file"),
        ("prediction_dir",  PATHS["prediction_dir"],  "dir_or_create"),
        ("result_dir",      PATHS["result_dir"],      "dir_or_create"),
    ]

    for name, path, kind in checks:
        if kind == "file":
            ok = os.path.isfile(path)
        elif kind == "dir":
            ok = os.path.isdir(path)
        else:  # dir_or_create
            os.makedirs(path, exist_ok=True)
            ok = os.path.isdir(path)

        icon = "✓" if ok else "✗"
        print(f"  {icon} {name}: {path}")
        results.append(ok)

    # yolov8n-pose.pt — 없으면 첫 실행 시 자동 다운로드됨
    yolo_path = PATHS["yolo_model"]
    if os.path.isfile(yolo_path):
        print(f"  ✓ yolo_model: {yolo_path}")
    else:
        print(f"  ℹ yolo_model: 없음 → 파이프라인 실행 시 자동 다운로드")

    return all(results)


def test_unit_impact():
    """3단계: ImpactDetector 유닛 테스트 (더미 궤적)"""
    import numpy as np
    from analysis.impact import ImpactDetector, to_api_json

    print("\n[3단계] ImpactDetector 유닛 테스트")
    try:
        # 간단한 왕복 궤적 생성 (타점 2개 포함)
        n = 300
        x = np.zeros(n)
        y = np.zeros(n)
        fps = 30.0

        # 0~100프레임: 위로 이동
        for i in range(100):
            x[i] = 320 + i * 2
            y[i] = 500 - i * 2

        # 타점 1 (100프레임 부근): 방향 전환
        for i in range(100, 200):
            x[i] = 520 + (i - 100) * 2
            y[i] = 300 + (i - 100) * 2

        # 타점 2 (200프레임 부근): 다시 전환
        for i in range(200, 300):
            x[i] = 720 - (i - 200) * 2
            y[i] = 500 - (i - 200)

        detector = ImpactDetector(fps=fps, frame_height=720)
        events = detector.detect(x, y)
        api_data = to_api_json(events, fps)

        print(f"  ✓ 감지된 타점 수: {api_data['total_hits']}")
        for hit in api_data["hits_data"]:
            print(f"    #{hit['hit_number']:02d}  {hit['time_sec']:.2f}s  {hit['player']}")

        return True
    except Exception as e:
        print(f"  ✗ 실패: {e}")
        traceback.print_exc()
        return False


def test_pipeline(video_path: str, skip_tracknet: bool = False):
    """4단계: 전체 파이프라인 실행 테스트 (실제 영상 필요)"""
    from services.pipeline_service import RallyTrackPipeline

    print(f"\n[4단계] 전체 파이프라인 테스트")
    print(f"  영상: {video_path}")
    print(f"  skip_tracknet: {skip_tracknet}")

    try:
        pipeline = RallyTrackPipeline(skip_tracknet=skip_tracknet)
        api_data = pipeline.run(video_path)

        result_paths = api_data.pop("result_paths", {})

        print(f"\n  ✓ 파이프라인 완료")
        print(f"    video_fps  : {api_data['video_fps']}")
        print(f"    total_hits : {api_data['total_hits']}")
        print(f"    hits_data  :")
        for hit in api_data["hits_data"]:
            print(f"      #{hit['hit_number']:02d}  {hit['time_sec']:.3f}s  {hit['player']}")
        print(f"\n  출력 파일:")
        for k, v in result_paths.items():
            import os
            exists = "✓" if os.path.exists(v) else "✗"
            print(f"    {exists} {k}: {v}")

        return True
    except Exception as e:
        print(f"  ✗ 실패: {e}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="RallyTrack analysis/ 독립 테스트")
    parser.add_argument("--video", "-v", default=None, help="테스트할 영상 경로 (없으면 1~3단계만 실행)")
    parser.add_argument("--skip-tracknet", action="store_true", help="TrackNet 생략 (CSV 이미 있을 때)")
    args = parser.parse_args()

    print("=" * 55)
    print("  RallyTrack analysis/ 패키지 테스트")
    print("=" * 55)

    results = {}
    results["import"]  = test_imports()
    results["paths"]   = test_paths()
    results["impact"]  = test_unit_impact()

    if args.video:
        if not Path(args.video).exists():
            print(f"\n[4단계] 영상 파일 없음: {args.video}")
            results["pipeline"] = False
        else:
            results["pipeline"] = test_pipeline(args.video, args.skip_tracknet)
    else:
        print("\n[4단계] --video 옵션 없음 → 파이프라인 테스트 생략")

    # 결과 요약
    print("\n" + "=" * 55)
    print("  테스트 결과 요약")
    print("=" * 55)
    all_pass = True
    for name, ok in results.items():
        icon = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {icon}  {name}")
        if not ok:
            all_pass = False
    print("=" * 55)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
