"""
RallyTrack - CSV만으로 미니맵 빠른 테스트 스크립트 (v3)

[v3 변경 사항]
  - 코트 코너 자동 검출 통합
  - --corners 수동 지정 옵션 추가
  - --web-export 옵션 추가 (프레임 데이터 JSON 출력)
  - --show-corners 옵션 추가 (코트 코너 시각화 이미지 저장)

TrackNet 없이 이미 생성된 CSV + 영상으로
미니맵 + 웹 연동 데이터를 빠르게 테스트할 때 사용합니다.

실행 예시:
    # 기본 (미니맵 영상만)
    python test_minimap.py --video inputVideo/test_8sec.mp4 \\
                           --csv   prediction/test_8sec_ball.csv

    # 웹 데이터 JSON 포함
    python test_minimap.py --video inputVideo/test_8sec.mp4 \\
                           --csv   prediction/test_8sec_ball.csv \\
                           --web-export

    # 코트 코너 수동 지정 (자동 검출 대신)
    python test_minimap.py --video inputVideo/test_8sec.mp4 \\
                           --csv   prediction/test_8sec_ball.csv \\
                           --corners 0.20,0.35,0.80,0.35,0.90,0.97,0.10,0.97
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from analysis.config  import PATHS, MINIMAP_CONFIG, POSE_CONFIG
from analysis.court   import compute_homographies
from analysis.court_detector import CourtCornerDetector, CourtCorners, visualize_corners
from analysis.impact  import ImpactDetector
from analysis.minimap import MinimapRenderer
from analysis.web_exporter import WebDataCollector, save_hits_json, save_court_corners_json


def extract_keypoints(pose_result) -> list:
    if pose_result.keypoints is None:
        return []
    try:
        kpts = pose_result.keypoints.data.cpu().numpy()
    except Exception:
        return []
    boxes = pose_result.boxes
    ids   = []
    if boxes is not None and boxes.id is not None:
        try:
            ids = boxes.id.cpu().numpy().astype(int).tolist()
        except Exception:
            ids = []
    result = []
    for i, k in enumerate(kpts):
        tid = int(ids[i]) if i < len(ids) else -(i + 1)
        result.append({"track_id": tid, "keypoints": k.tolist()})
    return result


def resolve_corners(
    video_path:     str,
    manual_corners: str | None,
    frame_w:        int,
    frame_h:        int,
) -> CourtCorners:
    """코트 코너를 수동 지정 또는 자동 검출로 결정합니다."""
    if manual_corners:
        vals = [float(v) for v in manual_corners.split(",")]
        if len(vals) != 8:
            raise ValueError("--corners 인자는 8개 값이 필요합니다")
        print("[코너] 수동 지정 사용")
        return CourtCorners(
            top_left     = (vals[0] * frame_w, vals[1] * frame_h),
            top_right    = (vals[2] * frame_w, vals[3] * frame_h),
            bottom_right = (vals[4] * frame_w, vals[5] * frame_h),
            bottom_left  = (vals[6] * frame_w, vals[7] * frame_h),
            confidence   = 1.0,
            method       = "manual",
        )

    print("[코너] 자동 검출 시작...")
    detector = CourtCornerDetector()
    corners  = detector.detect(video_path)
    print(f"[코너] 방법={corners.method}, 신뢰도={corners.confidence:.2f}")
    return corners


def run(
    video_path:     str,
    csv_path:       str,
    output_dir:     str     = "result",
    web_export:     bool    = False,
    manual_corners: str | None = None,
    show_corners:   bool    = False,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    video_name = Path(video_path).stem

    # CSV 로드
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    x_arr = df["x"].fillna(0).values.astype(float)
    y_arr = df["y"].fillna(0).values.astype(float)

    # 영상 정보
    cap     = cv2.VideoCapture(video_path)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps     = cap.get(cv2.CAP_PROP_FPS)
    n       = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"영상: {frame_w}×{frame_h} @ {fps:.1f}fps ({n}프레임)")

    # 코트 코너 결정
    corners = resolve_corners(video_path, manual_corners, frame_w, frame_h)

    # 코트 코너 시각화 저장
    if show_corners:
        cap_vis = cv2.VideoCapture(video_path)
        ret, first_frame = cap_vis.read()
        cap_vis.release()
        if ret:
            vis      = visualize_corners(first_frame, corners)
            vis_path = os.path.join(output_dir, f"{video_name}_court_viz.jpg")
            cv2.imwrite(vis_path, vis)
            print(f"[코너] 시각화 저장 → {vis_path}")

    # 코트 코너 JSON 저장
    corners_json_path = os.path.join(output_dir, f"{video_name}_court_corners.json")
    save_court_corners_json(corners.to_dict(), corners_json_path)

    # 호모그래피
    hg = compute_homographies(frame_w, frame_h, corners)

    # 타점 감지
    detector   = ImpactDetector(fps=fps, frame_height=frame_h)
    hit_events = detector.detect(x_arr, y_arr)
    print(f"타점 감지: {len(hit_events)}개")

    # 미니맵 렌더러
    renderer = MinimapRenderer(hg, hit_events)
    mw       = MINIMAP_CONFIG["width"]
    mh       = MINIMAP_CONFIG["height"]

    # 웹 데이터 수집기
    web_collector: WebDataCollector | None = None
    if web_export:
        web_collector = WebDataCollector(
            homographies  = hg,
            hit_events    = hit_events,
            fps           = fps,
            net_y_minimap = hg["net_y_minimap"],
        )

    # YOLO 모델
    pose_model = YOLO(PATHS["yolo_model"])

    tmp_map = os.path.join(output_dir, f"_tmp_{video_name}_minimap.mp4")
    fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
    out_mm  = cv2.VideoWriter(tmp_map, fourcc, fps, (mw, mh))

    cap       = cv2.VideoCapture(video_path)
    frame_idx = 0
    t0        = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        sx = float(x_arr[frame_idx]) if frame_idx < len(x_arr) else 0.0
        sy = float(y_arr[frame_idx]) if frame_idx < len(y_arr) else 0.0

        pose_result = pose_model.track(
            frame, persist=True, verbose=False,
            conf=POSE_CONFIG["confidence"],
        )[0]
        kpts = extract_keypoints(pose_result)

        canvas = renderer.render_frame(frame_idx, sx, sy, kpts)
        out_mm.write(canvas)

        if web_collector is not None:
            web_collector.record(frame_idx, sx, sy, kpts)

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  {frame_idx}/{n} ({time.time()-t0:.1f}s)")

    cap.release()
    out_mm.release()

    # ffmpeg 재인코딩
    final  = os.path.join(output_dir, f"{video_name}_minimap.mp4")
    import subprocess
    encoded = False
    for cmd in ["static_ffmpeg", "ffmpeg"]:
        try:
            subprocess.run(
                [cmd, "-y", "-i", tmp_map, "-vcodec", "libx264",
                 "-crf", "23", "-pix_fmt", "yuv420p", final],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            os.remove(tmp_map)
            encoded = True
            break
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    if not encoded:
        os.rename(tmp_map, final)

    # 타점 JSON 저장
    hits_json_path = os.path.join(output_dir, f"{video_name}_hits.json")
    save_hits_json(hit_events, fps, hits_json_path)

    # 웹 데이터 저장
    if web_collector is not None:
        print("\n[웹 데이터] 저장 중...")
        web_files = web_collector.save(output_dir, video_name)

    print(f"\n완료: {final}")
    print(f"타점 수: {len(hit_events)}개")
    for e in hit_events:
        print(f"  #{e.hit_number:02d} {e.time_sec:.2f}s ({e.player})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RallyTrack 미니맵 빠른 테스트 (v3)")
    parser.add_argument("--video",        required=True)
    parser.add_argument("--csv",          required=True)
    parser.add_argument("--out",          default="result")
    parser.add_argument("--web-export",   action="store_true",
                        help="웹 연동용 JSON 데이터 추가 출력")
    parser.add_argument("--show-corners", action="store_true",
                        help="코트 코너 시각화 이미지 저장")
    parser.add_argument(
        "--corners",
        default=None,
        metavar="x1,y1,x2,y2,x3,y3,x4,y4",
        help="코트 코너 수동 지정 (정규화 비율, 자동 검출 대신 사용)",
    )
    args = parser.parse_args()

    run(
        video_path     = args.video,
        csv_path       = args.csv,
        output_dir     = args.out,
        web_export     = args.web_export,
        manual_corners = args.corners,
        show_corners   = args.show_corners,
    )
