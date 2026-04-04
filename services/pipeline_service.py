"""
RallyTrack - 메인 분석 파이프라인 서비스

출력물:
  result/{name}_1_main.mp4       ← 원본 + YOLO 스켈레톤 오버레이
  result/{name}_2_minimap.mp4    ← 2D Top-Down 미니맵 (360×600)
  result/{name}_3_skeleton.mp4   ← 원근감 코트 스켈레톤 뷰
  result/{name}_hits.json        ← 타점 + 네트판정 데이터 (백엔드 콜백용)

[수정 사항]
  - run()에 user_corners / net_coords 파라미터 추가
    · user_corners: 사용자가 지정한 단식 코트 4점 (픽셀, [좌상, 우상, 우하, 좌하])
    · net_coords:   사용자가 지정한 네트 상단 2점 (픽셀, [좌, 우])
  - compute_homographies()에 user_corners 주입
  - NetJudge를 통한 네트 판정 결과를 api_data에 포함
  - 네트 판정 이벤트를 렌더링 시 오버레이로 표시
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from analysis.config        import PATHS, TRACKNET_CONFIG, MINIMAP_CONFIG, POSE_CONFIG
from analysis.court         import compute_homographies
from analysis.impact        import ImpactDetector, to_api_json
from analysis.minimap       import MinimapRenderer
from analysis.net_judge     import NetFaultEvent, NetJudge
from analysis.skeleton_view import SkeletonCourtRenderer


# ────────────────────────────────────────────────────────────
# TrackNet subprocess 실행
# ────────────────────────────────────────────────────────────

def run_tracknet(video_path: str) -> str:
    """TrackNetV3 predict.py 실행 → CSV 경로 반환. 이미 있으면 재사용."""
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
    """CSV에서 x, y 궤적 배열 로드."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    x = df["x"].fillna(0).values.astype(float)
    y = df["y"].fillna(0).values.astype(float)
    return x, y


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
# 네트 렌더링 헬퍼
# ────────────────────────────────────────────────────────────

def _draw_net_overlay(
    frame: np.ndarray,
    net_coords: List[List[float]],
) -> None:
    """원본 영상 프레임에 네트 상단 라인을 파란색으로 오버레이."""
    pt1 = (int(net_coords[0][0]), int(net_coords[0][1]))
    pt2 = (int(net_coords[1][0]), int(net_coords[1][1]))
    cv2.line(frame, pt1, pt2, (255, 100, 0), 2)   # BGR 파랑


def _draw_fault_overlay(frame: np.ndarray, label: str = "NET FAULT") -> None:
    """네트 걸림 발생 프레임에 경고 텍스트 오버레이."""
    h, w = frame.shape[:2]
    cv2.putText(
        frame, label,
        (w // 2 - 130, 180),
        cv2.FONT_HERSHEY_DUPLEX, 2.0,
        (0, 0, 255), 5, cv2.LINE_AA,
    )


# ────────────────────────────────────────────────────────────
# 메인 파이프라인 클래스
# ────────────────────────────────────────────────────────────

class RallyTrackPipeline:
    """
    배드민턴 경기 영상 분석 전체 파이프라인.

    FastAPI 라우터에서 사용::

        pipeline = RallyTrackPipeline()
        api_data = pipeline.run(
            local_video_path,
            user_corners=[[x,y], ...],   # 사용자 지정 코트 코너 (선택)
            net_coords=[[x,y], [x,y]],   # 사용자 지정 네트 좌표 (선택)
        )

    반환값::

        {
            "video_fps":        float,
            "total_hits":       int,
            "hits_data":        [...],
            "net_fault_events": [...],   # net_coords 제공 시에만 포함
            "coordinate_mode":  str,     # "user_singles" | "auto_ratio"
            "result_paths":     {...},
        }
    """

    def __init__(self, skip_tracknet: bool = False):
        self.skip_tracknet = skip_tracknet
        self._pose_model: Optional[YOLO] = None

    @property
    def pose_model(self) -> YOLO:
        if self._pose_model is None:
            print("[YOLO] 모델 로딩 중...")
            self._pose_model = YOLO(PATHS["yolo_model"])
        return self._pose_model

    def run(
        self,
        video_path:   str,
        user_corners: Optional[List[List[float]]] = None,
        net_coords:   Optional[List[List[float]]] = None,
    ) -> dict:
        """
        영상을 분석하고 타점 + 네트 판정 API 데이터를 반환합니다.

        Args:
            video_path   : 로컬 영상 파일 경로
            user_corners : 단식 코트 4꼭짓점 픽셀 좌표 [[좌상],[우상],[우하],[좌하]]
                           None이면 config.py COURT_CORNERS 비율 기반 자동 계산
            net_coords   : 네트 상단 2점 [[좌],[우]]
                           None이면 네트 판정 생략

        Returns:
            api_data dict (result_paths 포함)
        """
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
        print(f"[Step 2] {frame_w}×{frame_h} @ {fps:.1f}fps, {n_frames}프레임")

        # ── Step 3: 호모그래피 ──────────────────────────────
        # user_corners가 있으면 단식 규격 비율 사용, 없으면 자동 비율
        print(f"[Step 3] 호모그래피 계산 (mode={'user_singles' if user_corners else 'auto_ratio'})")
        user_corners_np = (
            np.array(user_corners, dtype=np.float32) if user_corners is not None else None
        )
        hg = compute_homographies(frame_w, frame_h, user_corners=user_corners_np)
        print(f"         coordinate_mode: {hg['coordinate_mode']}")

        # ── Step 4: 타점 감지 ────────────────────────────────
        print("[Step 4] 타점 감지")
        detector   = ImpactDetector(fps=fps, frame_height=frame_h)
        hit_events = detector.detect(x_arr, y_arr)
        print(f"         → {len(hit_events)}개")

        # ── Step 4-b: 네트 판정 (net_coords 제공 시) ────────
        net_fault_events: List[NetFaultEvent] = []
        if net_coords is not None:
            print("[Step 4b] 네트 판정")
            judge = NetJudge(net_coords=net_coords, fps=fps)
            net_fault_events = judge.detect_faults(x_arr, y_arr)
            print(f"          → NET_FAULT {len(net_fault_events)}건")
        else:
            print("[Step 4b] 네트 판정 생략 (net_coords 없음)")

        # 렌더링에서 빠른 조회를 위한 세트
        hit_frames   = {e.frame for e in hit_events}
        fault_frames = {e.frame for e in net_fault_events}

        # ── Step 5: 렌더러 초기화 ────────────────────────────
        minimap_renderer  = MinimapRenderer(hg, hit_events)
        skeleton_renderer = SkeletonCourtRenderer(frame_w, frame_h, hg)

        mw   = MINIMAP_CONFIG["width"]
        mh   = MINIMAP_CONFIG["height"]
        sk_h = 1000

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        rd     = PATHS["result_dir"]
        tmp_1  = os.path.join(rd, f"_tmp_{video_name}_1.mp4")
        tmp_2  = os.path.join(rd, f"_tmp_{video_name}_2.mp4")
        tmp_3  = os.path.join(rd, f"_tmp_{video_name}_3.mp4")

        out_1 = cv2.VideoWriter(tmp_1, fourcc, fps, (frame_w, frame_h))
        out_2 = cv2.VideoWriter(tmp_2, fourcc, fps, (mw, mh))
        out_3 = cv2.VideoWriter(tmp_3, fourcc, fps, (frame_w, sk_h))

        # ── Step 6: 프레임 루프 ──────────────────────────────
        print("[Step 5] 렌더링 시작")
        frame_idx   = 0
        t0          = time.time()

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

            # 네트 라인 오버레이 (net_coords 제공 시 항상 표시)
            if net_coords is not None:
                _draw_net_overlay(annotated, net_coords)

            # 타점 표시
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

            # 네트 걸림 표시
            if any(abs(frame_idx - ff) <= 3 for ff in fault_frames):
                _draw_fault_overlay(annotated)

            cv2.putText(
                annotated, f"F:{frame_idx}",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
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

            frame_idx += 1
            if frame_idx % 50 == 0:
                print(f"         {frame_idx}/{n_frames} ({time.time()-t0:.1f}s)")

        cap.release()
        out_1.release()
        out_2.release()
        out_3.release()
        print(f"[Step 5] 완료 ({frame_idx}프레임, {time.time()-t0:.1f}s)")

        # ── Step 7: H.264 재인코딩 ───────────────────────────
        print("[Step 6] H.264 인코딩")
        f1 = os.path.join(rd, f"{video_name}_1_main.mp4")
        f2 = os.path.join(rd, f"{video_name}_2_minimap.mp4")
        f3 = os.path.join(rd, f"{video_name}_3_skeleton.mp4")
        reencode_h264(tmp_1, f1)
        reencode_h264(tmp_2, f2)
        reencode_h264(tmp_3, f3)

        # ── Step 8: JSON 생성 ─────────────────────────────────
        api_data  = to_api_json(hit_events, fps)
        api_data["net_fault_events"]  = [e.to_dict() for e in net_fault_events]
        api_data["coordinate_mode"]   = hg["coordinate_mode"]

        json_path = os.path.join(rd, f"{video_name}_hits.json")
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(api_data, fh, ensure_ascii=False, indent=4)

        self._print_summary(hit_events, net_fault_events, video_name, f1, f2, f3, json_path)

        api_data["result_paths"] = {
            "main_video":     f1,
            "minimap_video":  f2,
            "skeleton_video": f3,
            "hits_json":      json_path,
        }
        return api_data

    # ────────────────────────────────────────────────────────
    @staticmethod
    def _print_summary(
        hit_events:        list,
        net_fault_events:  list,
        name:              str,
        f1: str, f2: str, f3: str,
        json_p: str,
    ):
        print("\n" + "═" * 55)
        print(f"  RallyTrack 완료 — {name}")
        print("═" * 55)
        print(f"  타점 수             : {len(hit_events)}개")
        print(f"  네트 걸림           : {len(net_fault_events)}건")
        print(f"  영상 1 (원본)       : {f1}")
        print(f"  영상 2 (미니맵)     : {f2}")
        print(f"  영상 3 (스켈레톤)   : {f3}")
        print(f"  타점 JSON           : {json_p}")
        print("─" * 55)
        for e in hit_events:
            side = "상단(핑크)" if e.owner == "top" else "하단(라임)"
            print(f"    #{e.hit_number:02d}  {e.time_sec:6.2f}s  {side}")
        if net_fault_events:
            print("─" * 55)
            print("  [NET_FAULT]")
            for fe in net_fault_events:
                print(
                    f"    {fe.time_sec:6.2f}s  "
                    f"clearance={fe.clearance_px:+.1f}px  "
                    f"shuttle_y={fe.shuttle_y:.1f}  net_y={fe.net_top_y_at_x:.1f}"
                )
        print("═" * 55 + "\n")