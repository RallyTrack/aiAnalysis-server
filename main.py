"""
RallyTrack - 메인 분석 파이프라인 (v3)

[v3 변경 사항]
  1. 코트 코너 자동 검출 통합
     - 이전: config.COURT_CORNERS 하드코딩
     - 변경: CourtCornerDetector가 영상에서 자동으로 코너 검출
             검출 실패 시 해상도 기반 폴백 자동 전환

  2. 웹 연동 데이터 출력 추가 (--web-export 플래그)
     - result/{name}_frame_data.json  ← 프레임별 선수/셔틀 좌표 (프론트 애니메이션용)
     - result/{name}_summary.json     ← 경기 요약 (대시보드용)
     - result/{name}_court_corners.json ← 검출된 코트 코너 좌표

  3. 미니맵 코트 라인 BWF 표준 규격 적용
     (court.py draw_minimap_court() 업데이트 반영)

출력물:
  result/{name}_1_main.mp4           ← 원본 + YOLO 스켈레톤 오버레이
  result/{name}_2_minimap.mp4        ← 2D Top-Down 미니맵 (320×670)
  result/{name}_3_skeleton.mp4       ← 원근감 코트 스켈레톤 뷰
  result/{name}_hits.json            ← 타점 API 데이터
  [--web-export 시 추가]
  result/{name}_frame_data.json      ← 프레임별 정규화 좌표 스트림
  result/{name}_summary.json         ← 경기 요약
  result/{name}_court_corners.json   ← 코트 코너 검출 결과

실행:
  # 기본 (영상 3개 + 타점 JSON)
  python main.py --video inputVideo/test_8sec.mp4

  # TrackNet 생략 (CSV가 이미 있을 때)
  python main.py --video inputVideo/test_8sec.mp4 --skip-tracknet

  # 웹 연동 데이터까지 출력
  python main.py --video inputVideo/test_8sec.mp4 --web-export

  # 코트 코너 수동 지정 (자동 검출 대신)
  python main.py --video inputVideo/test_8sec.mp4 \\
      --corners 0.20,0.35,0.80,0.35,0.90,0.97,0.10,0.97
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from analysis.config import PATHS, TRACKNET_CONFIG, MINIMAP_CONFIG, POSE_CONFIG
from analysis.court  import compute_homographies
from analysis.court_detector import CourtCornerDetector, CourtCorners, visualize_corners
from analysis.impact import ImpactDetector, to_api_json
from analysis.minimap import MinimapRenderer
from analysis.skeleton_view import SkeletonCourtRenderer
from analysis.web_exporter import (
    WebDataCollector,
    save_hits_json,
    save_court_corners_json,
)


# ────────────────────────────────────────────────────────────
# TrackNet 실행
# ────────────────────────────────────────────────────────────

def run_tracknet(video_path: str) -> str:
    """TrackNetV3 predict.py 실행 → CSV 반환. 이미 있으면 재사용."""
    video_name = Path(video_path).stem
    csv_path   = os.path.join(PATHS["prediction_dir"], f"{video_name}_ball.csv")

    if os.path.exists(csv_path):
        print(f"[TrackNet] 기존 CSV 재사용: {csv_path}")
        return csv_path

    os.makedirs(PATHS["prediction_dir"], exist_ok=True)

    predict_script = os.path.join(PATHS["tracknet_dir"], "predict.py")
    cmd = [
        sys.executable, predict_script,
        "--video_file",      video_path,
        "--tracknet_file",   PATHS["tracknet_ckpt"],
        "--inpaintnet_file", PATHS["inpaintnet_ckpt"],
        "--batch_size",      str(TRACKNET_CONFIG["batch_size"]),
        "--save_dir",        PATHS["prediction_dir"],
    ]
    print("[TrackNet] 분석 시작...")
    result = subprocess.run(cmd, cwd=PATHS["tracknet_dir"])
    if result.returncode != 0:
        raise RuntimeError("TrackNet 실행 실패")
    print(f"[TrackNet] 완료 → {csv_path}")
    return csv_path


def load_trajectory_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    x = df["x"].fillna(0).values.astype(float)
    y = df["y"].fillna(0).values.astype(float)
    return x, y


# ────────────────────────────────────────────────────────────
# 코트 코너 검출
# ────────────────────────────────────────────────────────────

def detect_corners(
    video_path:      str,
    manual_corners:  Optional[str] = None,
    frame_w:         int = 0,
    frame_h:         int = 0,
) -> CourtCorners:
    """
    코트 코너를 결정합니다.

    우선순위:
      1. --corners CLI 인자 (수동 지정)
      2. CourtCornerDetector 자동 검출
      3. 해상도 기반 폴백

    Args:
        video_path:     분석 영상 경로
        manual_corners: "x1,y1,x2,y2,x3,y3,x4,y4" 형태 정규화 비율 문자열
                        순서: [좌상, 우상, 우하, 좌하] X,Y 쌍
        frame_w, frame_h: 영상 해상도

    Returns:
        CourtCorners 인스턴스
    """
    if manual_corners:
        # CLI 수동 지정
        vals = [float(v) for v in manual_corners.split(",")]
        if len(vals) != 8:
            raise ValueError("--corners 인자는 정확히 8개 값이 필요합니다 (x1,y1,...,x4,y4)")
        print(f"[Step 3] 코트 코너 수동 지정 사용")
        return CourtCorners(
            top_left     = (vals[0] * frame_w, vals[1] * frame_h),
            top_right    = (vals[2] * frame_w, vals[3] * frame_h),
            bottom_right = (vals[4] * frame_w, vals[5] * frame_h),
            bottom_left  = (vals[6] * frame_w, vals[7] * frame_h),
            confidence   = 1.0,
            method       = "manual",
        )

    # 자동 검출
    print("[Step 3] 코트 코너 자동 검출 시작...")
    detector = CourtCornerDetector()
    corners  = detector.detect(video_path)
    print(f"         → 방법={corners.method}, 신뢰도={corners.confidence:.2f}")
    return corners


# ────────────────────────────────────────────────────────────
# 키포인트 추출
# ────────────────────────────────────────────────────────────

def extract_keypoints(pose_result) -> list:
    if pose_result.keypoints is None:
        return []
    try:
        kpts = pose_result.keypoints.data.cpu().numpy()   # (N, 17, 3)
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
            "keypoints": k.tolist(),
        })
    return result


# ────────────────────────────────────────────────────────────
# H.264 재인코딩
# ────────────────────────────────────────────────────────────

def reencode_h264(src: str, dst: str) -> None:
    for cmd in ["static_ffmpeg", "ffmpeg"]:
        try:
            subprocess.run(
                [cmd, "-y", "-i", src,
                 "-vcodec", "libx264", "-crf", "23",
                 "-pix_fmt", "yuv420p", dst],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if os.path.exists(src) and src != dst:
                os.remove(src)
            print(f"  저장: {dst}")
            return
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    os.rename(src, dst)
    print(f"  저장(재인코딩 생략): {dst}")


# ────────────────────────────────────────────────────────────
# 메인 파이프라인
# ────────────────────────────────────────────────────────────

class RallyTrackPipeline:
    """
    배드민턴 경기 영상 분석 전체 파이프라인.

    웹 연동 시 사용법:
        pipeline = RallyTrackPipeline()
        result   = pipeline.run("inputVideo/match.mp4", web_export=True)
        # result = { "video_fps": ..., "total_hits": ..., "hits_data": [...],
        #            "web_files": {"frame_data": ..., "summary": ...} }
    """

    def __init__(
        self,
        skip_tracknet:  bool = False,
        manual_corners: Optional[str] = None,
    ):
        self.skip_tracknet  = skip_tracknet
        self.manual_corners = manual_corners
        self._pose_model: Optional[YOLO] = None

    @property
    def pose_model(self) -> YOLO:
        if self._pose_model is None:
            print("[YOLO] 모델 로딩 중...")
            self._pose_model = YOLO(PATHS["yolo_model"])
        return self._pose_model

    def run(self, video_path: str, web_export: bool = False) -> dict:
        video_path = os.path.abspath(video_path)
        video_name = Path(video_path).stem
        os.makedirs(PATHS["result_dir"],     exist_ok=True)
        os.makedirs(PATHS["prediction_dir"], exist_ok=True)

        # ── Step 1: TrackNet ──────────────────────────────────
        if self.skip_tracknet:
            csv_path = os.path.join(PATHS["prediction_dir"], f"{video_name}_ball.csv")
            print(f"[Step 1] TrackNet 생략 → {csv_path}")
        else:
            print("[Step 1] TrackNetV3 실행")
            csv_path = run_tracknet(video_path)

        x_arr, y_arr = load_trajectory_csv(csv_path)

        # ── Step 2: 영상 메타데이터 ──────────────────────────
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"영상을 열 수 없습니다: {video_path}")

        frame_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps      = cap.get(cv2.CAP_PROP_FPS)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        print(f"[Step 2] {frame_w}×{frame_h} @ {fps:.1f}fps, {n_frames}프레임")

        # ── Step 3: 코트 코너 자동 검출 ─────────────────────
        corners = detect_corners(
            video_path,
            manual_corners = self.manual_corners,
            frame_w        = frame_w,
            frame_h        = frame_h,
        )

        # ── Step 4: 호모그래피 계산 ──────────────────────────
        print("[Step 4] 호모그래피 계산")
        hg = compute_homographies(frame_w, frame_h, corners)

        # 코트 코너 시각화 이미지 저장 (디버그용)
        cap_debug = cv2.VideoCapture(video_path)
        ret, debug_frame = cap_debug.read()
        cap_debug.release()
        if ret:
            vis = visualize_corners(debug_frame, corners)
            vis_path = os.path.join(PATHS["result_dir"], f"{video_name}_court_viz.jpg")
            cv2.imwrite(vis_path, vis)
            print(f"         코트 시각화 → {vis_path}")

        # ── Step 5: 타점 감지 ────────────────────────────────
        print("[Step 5] 타점 감지")
        detector   = ImpactDetector(fps=fps, frame_height=frame_h)
        hit_events = detector.detect(x_arr, y_arr)
        print(f"         → {len(hit_events)}개")

        # ── Step 6: 렌더러 초기화 ────────────────────────────
        minimap_renderer  = MinimapRenderer(hg, hit_events)
        skeleton_renderer = SkeletonCourtRenderer(frame_w, frame_h, hg)

        # 웹 데이터 수집기 (web_export 모드)
        web_collector: Optional[WebDataCollector] = None
        if web_export:
            web_collector = WebDataCollector(
                homographies  = hg,
                hit_events    = hit_events,
                fps           = fps,
                net_y_minimap = hg["net_y_minimap"],
            )

        mw   = MINIMAP_CONFIG["width"]
        mh   = MINIMAP_CONFIG["height"]
        sk_h = 1000

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        rd     = PATHS["result_dir"]
        tmp_1  = os.path.join(rd, f"_tmp_{video_name}_1.mp4")
        tmp_2  = os.path.join(rd, f"_tmp_{video_name}_2.mp4")
        tmp_3  = os.path.join(rd, f"_tmp_{video_name}_3.mp4")

        cap = cv2.VideoCapture(video_path)
        out_1 = cv2.VideoWriter(tmp_1, fourcc, fps, (frame_w, frame_h))
        out_2 = cv2.VideoWriter(tmp_2, fourcc, fps, (mw, mh))
        out_3 = cv2.VideoWriter(tmp_3, fourcc, fps, (frame_w, sk_h))

        hit_frames = {e.frame for e in hit_events}

        # ── Step 7: 프레임 루프 ──────────────────────────────
        print("[Step 6] 렌더링 시작")
        frame_idx = 0
        t0        = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            sx = float(x_arr[frame_idx]) if frame_idx < len(x_arr) else 0.0
            sy = float(y_arr[frame_idx]) if frame_idx < len(y_arr) else 0.0

            # YOLOv8-pose 추론
            pose_result    = self.pose_model.track(
                frame, persist=True, verbose=False,
                conf=POSE_CONFIG["confidence"],
            )[0]
            keypoints_list = extract_keypoints(pose_result)

            # ── 영상 1: 원본 + 스켈레톤 오버레이 ────────────
            annotated = pose_result.plot()
            annotated = cv2.resize(annotated, (frame_w, frame_h))

            if sx > 0 and sy > 0:
                cv2.circle(annotated, (int(sx), int(sy)), 7, (0, 215, 255), -1)

            if any(abs(frame_idx - hf) <= 2 for hf in hit_frames):
                cv2.putText(
                    annotated, "IMPACT",
                    (frame_w // 2 - 90, 130),
                    cv2.FONT_HERSHEY_DUPLEX, 2.2,
                    (0, 215, 255), 5, cv2.LINE_AA,
                )
                annotated[:, :, 2] = np.clip(
                    annotated[:, :, 2].astype(np.int16) + 60, 0, 255
                ).astype(np.uint8)

            # 코너 검출 신뢰도 표시
            cv2.putText(
                annotated,
                f"F:{frame_idx}  Court:{corners.method}({corners.confidence:.2f})",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (200, 200, 200), 2, cv2.LINE_AA,
            )
            out_1.write(annotated)

            # ── 영상 2: 2D Top-Down 미니맵 ───────────────────
            minimap_canvas = minimap_renderer.render_frame(
                frame_idx, sx, sy, keypoints_list
            )
            out_2.write(minimap_canvas)

            # ── 영상 3: 원근감 코트 스켈레톤 뷰 ─────────────
            skeleton_canvas = skeleton_renderer.render_frame(
                frame_idx, sx, sy, keypoints_list
            )
            out_3.write(skeleton_canvas)

            # ── 웹 데이터 수집 ────────────────────────────────
            if web_collector is not None:
                web_collector.record(frame_idx, sx, sy, keypoints_list)

            frame_idx += 1
            if frame_idx % 50 == 0:
                print(f"         {frame_idx}/{n_frames} ({time.time()-t0:.1f}s)")

        cap.release()
        out_1.release()
        out_2.release()
        out_3.release()
        print(f"[Step 6] 완료 ({frame_idx}프레임, {time.time()-t0:.1f}s)")

        # ── Step 8: H.264 재인코딩 ───────────────────────────
        print("[Step 7] H.264 인코딩")
        f1 = os.path.join(rd, f"{video_name}_1_main.mp4")
        f2 = os.path.join(rd, f"{video_name}_2_minimap.mp4")
        f3 = os.path.join(rd, f"{video_name}_3_skeleton.mp4")
        reencode_h264(tmp_1, f1)
        reencode_h264(tmp_2, f2)
        reencode_h264(tmp_3, f3)

        # ── Step 9: 타점 JSON ────────────────────────────────
        json_path = os.path.join(rd, f"{video_name}_hits.json")
        save_hits_json(hit_events, fps, json_path)

        # 코트 코너 JSON
        corners_json_path = os.path.join(rd, f"{video_name}_court_corners.json")
        save_court_corners_json(hg["corners_info"], corners_json_path)

        # ── Step 10: 웹 연동 데이터 저장 (옵션) ──────────────
        web_files = {}
        if web_collector is not None:
            print("[Step 8] 웹 연동 데이터 저장")
            web_files = web_collector.save(rd, video_name)

        # 기존 to_api_json 형식도 함께 반환
        from analysis.impact import to_api_json
        api_data  = to_api_json(hit_events, fps)

        self._print_summary(
            hit_events, corners, video_name,
            f1, f2, f3, json_path, corners_json_path,
            web_files,
        )

        return {
            **api_data,
            "corners":   hg["corners_info"],
            "web_files": web_files,
        }

    @staticmethod
    def _print_summary(
        hit_events, corners, name,
        f1, f2, f3, json_p, corners_p,
        web_files,
    ):
        print("\n" + "═" * 60)
        print(f"  RallyTrack 완료 — {name}")
        print("═" * 60)
        print(f"  타점 수             : {len(hit_events)}개")
        print(f"  코트 검출 방법      : {corners.method}  (신뢰도 {corners.confidence:.2f})")
        print(f"  영상 1 (원본)       : {f1}")
        print(f"  영상 2 (미니맵)     : {f2}")
        print(f"  영상 3 (스켈레톤)   : {f3}")
        print(f"  타점 JSON           : {json_p}")
        print(f"  코트 코너 JSON      : {corners_p}")
        if web_files:
            print(f"  프레임 데이터 JSON  : {web_files.get('frame_data', '-')}")
            print(f"  요약 JSON           : {web_files.get('summary', '-')}")
        print("─" * 60)
        for e in hit_events:
            side = "상단(핑크)" if e.owner == "top" else "하단(라임)"
            print(f"    #{e.hit_number:02d}  {e.time_sec:6.2f}s  {side}")
        print("═" * 60 + "\n")


# ────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RallyTrack - 배드민턴 경기 분석")
    parser.add_argument("--video",    "-v", required=True,
                        help="분석할 영상 경로")
    parser.add_argument("--skip-tracknet", action="store_true",
                        help="TrackNet 생략 (CSV가 이미 있을 때)")
    parser.add_argument("--web-export", action="store_true",
                        help="웹 연동용 JSON 데이터 추가 출력")
    parser.add_argument(
        "--corners",
        default=None,
        metavar="x1,y1,x2,y2,x3,y3,x4,y4",
        help=(
            "코트 코너 수동 지정 (정규화 비율 0~1, 자동 검출 대신 사용).\n"
            "순서: 좌상X,좌상Y, 우상X,우상Y, 우하X,우하Y, 좌하X,좌하Y\n"
            "예: --corners 0.20,0.35,0.80,0.35,0.90,0.97,0.10,0.97"
        ),
    )
    args = parser.parse_args()

    pipeline = RallyTrackPipeline(
        skip_tracknet  = args.skip_tracknet,
        manual_corners = args.corners,
    )
    pipeline.run(args.video, web_export=args.web_export)


if __name__ == "__main__":
    main()
