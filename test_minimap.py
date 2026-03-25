"""
RallyTrack - CSV만으로 미니맵 빠른 테스트 스크립트

TrackNet 없이 이미 생성된 CSV + 영상으로
미니맵만 빠르게 테스트할 때 사용합니다.

실행:
    python test_minimap.py --video inputVideo/test_8sec.mp4 \
                           --csv   prediction/test_8sec_ball.csv
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
from analysis.impact  import ImpactDetector, to_api_json
from analysis.minimap import MinimapRenderer


def extract_keypoints(pose_result) -> list:
    """
    YOLO pose track 결과에서 키포인트를 dict 형식으로 추출합니다.
 
    반환 형식:
        [{"track_id": int, "keypoints": [[x, y, conf], ...]}, ...]
 
    track_id가 없는 경우(detect 모드) 인덱스 기반 음수 ID를 부여합니다.
    """
    if pose_result.keypoints is None:
        return []
 
    try:
        kpts  = pose_result.keypoints.data.cpu().numpy()   # (N, 17, 3)
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
        result.append({
            "track_id":  tid,
            "keypoints": k.tolist(),   # [[x, y, conf], ...] 17개
        })
 
    return result


def run(video_path: str, csv_path: str, output_dir: str = "result") -> None:
    os.makedirs(output_dir, exist_ok=True)

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
    print(f"영상: {frame_w}×{frame_h} @ {fps:.1f}fps ({n}프레임)")

    # 호모그래피
    hg = compute_homographies(frame_w, frame_h)

    # 타점 감지
    detector   = ImpactDetector(fps=fps, frame_height=frame_h)
    hit_events = detector.detect(x_arr, y_arr)
    print(f"타점 감지: {len(hit_events)}개")

    # 미니맵 렌더러
    renderer = MinimapRenderer(hg, hit_events)
    mw       = MINIMAP_CONFIG["width"]
    mh       = MINIMAP_CONFIG["height"]

    # YOLO 모델
    pose_model = YOLO(PATHS["yolo_model"])

    video_name  = Path(video_path).stem
    tmp_map     = os.path.join(output_dir, f"_tmp_{video_name}_minimap.mp4")
    fourcc      = cv2.VideoWriter_fourcc(*"mp4v")
    out_minimap = cv2.VideoWriter(tmp_map, fourcc, fps, (mw, mh))

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
            conf=POSE_CONFIG["confidence"]
        )[0]
        kpts = extract_keypoints(pose_result)

        canvas = renderer.render_frame(frame_idx, sx, sy, kpts)
        out_minimap.write(canvas)

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  {frame_idx}/{n} ({time.time()-t0:.1f}s)")

    cap.release()
    out_minimap.release()

    # ffmpeg 재인코딩 시도
    final = os.path.join(output_dir, f"{video_name}_minimap.mp4")
    encoded = False
    for cmd in ["static_ffmpeg", "ffmpeg"]:
        import subprocess
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

    print(f"\n완료: {final}")
    print(f"타점 수: {len(hit_events)}개")
    for e in hit_events:
        print(f"  #{e.hit_number:02d} {e.time_sec:.2f}s ({e.player})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="미니맵 빠른 테스트")
    parser.add_argument("--video", required=True)
    parser.add_argument("--csv",   required=True)
    parser.add_argument("--out",   default="result")
    args = parser.parse_args()
    run(args.video, args.csv, args.out)