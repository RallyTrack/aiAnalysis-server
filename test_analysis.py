"""
RallyTrack - analysis/ 패키지 독립 실행 테스트

서버 없이 analysis/ 모듈만 직접 테스트합니다.
영상 파일 경로만 지정하면 전체 파이프라인을 로컬에서 돌려볼 수 있어요.

실행:
    python test_analysis.py --video /path/to/video.mp4
    python test_analysis.py --video /path/to/video.mp4 --skip-tracknet

[좌표 입력 안내]
  - 파이프라인 실행 직전에 첫 프레임 창이 열립니다.
  - 코트의 네 모서리(좌상 → 우상 → 우하 → 좌하)를 순서대로 클릭하세요.
  - 4점을 모두 찍으면 자동으로 네트 입력 단계로 넘어갑니다.
  - 네트 상단 좌/우 2점을 찍고 Enter를 누르거나, 바로 Enter를 눌러 생략하세요.
  - 4점을 다 찍지 않고 창을 닫으면 분석이 중단됩니다.
"""

import argparse
import io
import json
import sys
import traceback
from pathlib import Path

# Windows cp949 터미널에서 유니코드 출력 오류 방지
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import cv2
import numpy as np


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


def collect_court_points(video_path: str):
    """
    첫 프레임을 화면에 띄우고 마우스 클릭으로 코트 좌표를 수집합니다.

    [Phase 1 - Court Picker]
      TL → TR → BR → BL 순서로 4점 클릭 → 보라색 폴리곤 표시
    [Phase 2 - Net Picker]  (선택)
      Net-L → Net-R 2점 클릭 또는 Enter로 생략
    창 강제 종료 / ESC → SystemExit

    Returns:
        user_corners : [[x,y] x 4]
        net_coords   : [[x,y] x 2] | None
    """
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    cap.release()

    if not ret:
        print("  [ERROR] Cannot read the first frame.")
        sys.exit(1)

    # ── 상수 (모두 영문 — 한글 깨짐 방지) ──────────────────────
    WINDOW       = "Court Picker"
    CORNER_ORDER = ["TL", "TR", "BR", "BL"]          # 클릭 순서 안내
    NET_ORDER    = ["Net-L", "Net-R"]
    C_CORNER     = (0, 255, 0)    # 초록  — 코트 점
    C_NET        = (0, 140, 255)  # 주황  — 네트 점
    C_POLY       = (200, 0, 255)  # 보라  — 코트 폴리곤
    C_TEXT       = (255, 255, 255)

    collected = []          # [[x, y], ...]  최대 6점
    phase     = "corner"    # "corner" | "net"

    # ── 화면 갱신 함수 ────────────────────────────────────────
    def _redraw() -> np.ndarray:
        img = first_frame.copy()

        # 코트 폴리곤 (4점 완료 시)
        if len(collected) >= 4:
            pts = np.array(collected[:4], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts], isClosed=True, color=C_POLY, thickness=2)

        # 수집된 점 표시
        for i, pt in enumerate(collected):
            cx, cy = int(pt[0]), int(pt[1])
            if i < 4:
                color, label = C_CORNER, CORNER_ORDER[i]
            else:
                color, label = C_NET, NET_ORDER[i - 4]
            cv2.circle(img, (cx, cy), 9, color, -1)
            cv2.putText(img, f"{i + 1}:{label}", (cx + 12, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

        # 상단 안내 문구 (영문 고정)
        n = len(collected)
        if phase == "corner":
            next_pt = CORNER_ORDER[n] if n < 4 else ""
            guide   = f"Click court corner  [{n}/4]: {next_pt}"
        else:
            net_n   = n - 4
            if net_n < 2:
                guide = f"Click net top point [{net_n}/2]: {NET_ORDER[net_n]}  |  Enter = skip"
            else:
                guide = "Net done. Press Enter to start analysis."

        # 반투명 배경바
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (img.shape[1], 55), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
        cv2.putText(img, guide, (14, 36),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85, C_TEXT, 2, cv2.LINE_AA)
        return img

    # ── 마우스 콜백 ────────────────────────────────────────────
    def mouse_callback(event, x, y, _flags, _param):
        nonlocal phase
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if phase == "corner" and len(collected) < 4:
            collected.append([float(x), float(y)])
            print(f"  [{len(collected)}/4] {CORNER_ORDER[len(collected)-1]}: ({x}, {y})")
            if len(collected) == 4:
                phase = "net"
                print("  Court corners done. Click Net-L / Net-R, or press Enter to skip.")
        elif phase == "net" and len(collected) < 6:
            collected.append([float(x), float(y)])
            print(f"  [Net {len(collected)-4}/2] {NET_ORDER[len(collected)-5]}: ({x}, {y})")
            if len(collected) == 6:
                print("  Net points done. Press Enter to start analysis.")

    # ── 창 초기화 ──────────────────────────────────────────────
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, mouse_callback)

    print("\n  [Court Picker] Click TL -> TR -> BR -> BL on the first frame.")
    print("  ESC or close window to abort.\n")

    # ── 이벤트 루프 ────────────────────────────────────────────
    while True:
        cv2.imshow(WINDOW, _redraw())
        key = cv2.waitKey(30) & 0xFF

        # 창이 강제로 닫혔는지 확인
        try:
            visible = cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE)
        except cv2.error:
            visible = 0
        if visible < 1:
            break

        if key == 27:           # ESC
            collected.clear()
            break
        if key in (13, 32):     # Enter / Space
            if phase == "net":  # 코트 4점이 완료된 상태에서만 진행
                break

    cv2.destroyAllWindows()

    # ── 가드 ───────────────────────────────────────────────────
    if len(collected) < 4:
        print("\n  코트 좌표가 설정되지 않아 분석을 중단합니다.")
        sys.exit(1)

    user_corners = collected[:4]
    net_coords   = collected[4:6] if len(collected) >= 6 else None

    print(f"\n  Court corners collected:")
    for label, pt in zip(CORNER_ORDER, user_corners):
        print(f"    {label}: ({pt[0]:.0f}, {pt[1]:.0f})")

    if net_coords:
        print(f"  Net points collected:")
        for label, pt in zip(NET_ORDER, net_coords):
            print(f"    {label}: ({pt[0]:.0f}, {pt[1]:.0f})")
    else:
        print("  Net points: skipped")

    return user_corners, net_coords


def test_pipeline(video_path: str, skip_tracknet: bool = False, verbose: bool = False):
    """4단계: 전체 파이프라인 실행 테스트 (실제 영상 필요)"""
    import os
    from services.pipeline_service import RallyTrackPipeline

    print(f"\n[4단계] 전체 파이프라인 테스트")
    print(f"  영상: {video_path}")
    print(f"  skip_tracknet: {skip_tracknet}")
    print(f"  verbose: {verbose}")

    # ── 코트 좌표 수집 (필수) ──────────────────────────────────
    print("\n  파이프라인 시작 전 코트 좌표를 입력합니다.")
    user_corners, net_coords = collect_court_points(video_path)

    try:
        pipeline = RallyTrackPipeline(skip_tracknet=skip_tracknet)
        api_data = pipeline.run(
            video_path,
            user_corners=user_corners,
            net_coords=net_coords,
            verbose=verbose,
        )

        result_paths = api_data.pop("result_paths", {})

        print(f"\n  ✓ 파이프라인 완료")
        print(f"    coordinate_mode : {api_data.get('coordinate_mode', '-')}")
        print(f"    video_fps       : {api_data['video_fps']}")
        print(f"    total_hits      : {api_data['total_hits']}")
        print(f"    hits_data       :")
        for hit in api_data["hits_data"]:
            print(f"      #{hit['hit_number']:02d}  {hit['time_sec']:.3f}s  {hit['player']}")
        print(f"\n  출력 파일:")
        for k, v in result_paths.items():
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
    parser.add_argument("--verbose", action="store_true", help="ImpactDetector 중간 진단 출력")
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
            results["pipeline"] = test_pipeline(args.video, args.skip_tracknet, args.verbose)
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
