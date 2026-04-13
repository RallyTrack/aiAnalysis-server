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
from analysis.court         import (
    compute_homographies, classify_drop_location, frame_to_minimap,
)
from analysis.impact        import ImpactDetector, to_api_json
from analysis.minimap       import MinimapRenderer
from analysis.net_judge     import NetCrossingEvent, NetFaultEvent, NetJudge
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
    # predict.py 89번 줄: video_name = video_file.split('/')[-1][:-4]
    # Windows 절대 경로(\)를 그대로 넘기면 split('/')가 전체 경로를 video_name으로 잘못 추출.
    # → forward slash로 변환해서 넘겨야 prediction/ 폴더에 올바르게 저장됨.
    video_path_fwd = video_path.replace("\\", "/")
    cmd = [
        sys.executable, predict_script,
        "--video_file",      video_path_fwd,
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
    cv2.line(frame, pt1, pt2, (246, 130, 59), 2)   # BGR Royal Blue (#3B82F6)


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
        verbose:      bool = False,
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

        # ── Step 4-b: 네트 판정 (net_coords 제공 시) ────────
        # 타점 감지보다 먼저 실행 → crossing_events를 ImpactDetector에 전달
        net_fault_events: List[NetFaultEvent]    = []
        crossing_events:  List[NetCrossingEvent] = []
        net_y_ratio      = 0.46   # 기본값 (net_coords 없을 때)
        owner_y_threshold: Optional[float] = None  # None → net_y 그대로 사용

        if net_coords is not None:
            print("[Step 4b] 네트 판정")
            judge           = NetJudge(net_coords=net_coords, fps=fps)
            crossing_events = judge.detect_crossings(x_arr, y_arr)
            net_fault_events = [
                NetFaultEvent(
                    frame           = ev.frame,
                    time_sec        = ev.time_sec,
                    crossing_x      = (net_coords[0][0] + net_coords[1][0]) / 2.0,
                    shuttle_y       = float(y_arr[min(ev.frame, len(y_arr) - 1)]),
                    net_top_y_at_x  = judge.net_y_at(
                        (net_coords[0][0] + net_coords[1][0]) / 2.0
                    ),
                    clearance_px    = ev.clearance_px,
                )
                for ev in crossing_events if ev.is_fault
            ]
            # net_y_ratio: 네트 중심 y를 frame_height 비율로 변환 → ImpactDetector에 주입
            net_cx    = (net_coords[0][0] + net_coords[1][0]) / 2.0
            net_y_abs = judge.net_y_at(net_cx)
            net_y_ratio = net_y_abs / max(frame_h, 1)
            print(f"          → crossing {len(crossing_events)}건 / NET_FAULT {len(net_fault_events)}건")
            print(f"          → net_y_ratio={net_y_ratio:.3f}")
        else:
            print("[Step 4b] 네트 판정 생략 (net_coords 없음)")

        # ── owner_y_threshold 계산 ────────────────────────────
        # [목적] top/bottom 선수 구분 기준 y값.
        # net_y(네트 상단 픽셀)는 코트 상단에 가까운 쿼터뷰에서
        # 코트 내 거의 모든 y > net_y 가 되어 전원 "bottom" 오분류 발생.
        #
        # [해결] user_corners가 있으면 코트의 실제 세로 중점 y를 계산:
        #   court_top_y    = (TL.y + TR.y) / 2   ← 코트 상단 라인 중점
        #   court_bottom_y = (BR.y + BL.y) / 2   ← 코트 하단 라인 중점
        #   owner_y_threshold = (court_top_y + court_bottom_y) / 2
        #
        # 이 값은 코트 실제 중앙선 y와 일치하므로,
        # 상반부(top half)에 있는 공 → "top" 선수, 하반부 → "bottom" 선수.
        #
        # user_corners 없는 경우: 기존대로 net_y(frame_height × net_y_ratio) 사용.
        if user_corners is not None:
            corners = np.array(user_corners, dtype=np.float32)  # [TL,TR,BR,BL]
            court_top_y    = float((corners[0, 1] + corners[1, 1]) / 2.0)
            court_bottom_y = float((corners[2, 1] + corners[3, 1]) / 2.0)
            owner_y_threshold = (court_top_y + court_bottom_y) / 2.0
            print(f"[Step 4c] owner_y_threshold={owner_y_threshold:.1f}  "
                  f"(court_top={court_top_y:.1f}, court_bottom={court_bottom_y:.1f})")

        # ── Step 4: 타점 감지 ────────────────────────────────
        print("[Step 4] 타점 감지")
        detector   = ImpactDetector(
            fps               = fps,
            net_y_ratio       = net_y_ratio,
            frame_height      = frame_h,
            owner_y_threshold = owner_y_threshold,
        )
        hit_events = detector.detect(
            x_arr, y_arr,
            crossing_events=crossing_events if crossing_events else None,
            run_alternation=False,   # [v11] 포즈 교정 후 Step 4.5 말미에 실행
            verbose=verbose,
        )
        print(f"         → {len(hit_events)}개")

        # ── Step 4.5: 포즈 기반 owner 교정 + sub-threshold 타점 복구 ──
        # (A) 확정된 타점 프레임에 YOLO predict 실행 → owner 교정
        #     crossing 검증된 타점은 건드리지 않음.
        # (B) near-miss 프레임(impulse 점수가 있으나 임계값 미달)에도 YOLO 실행
        #     → 손목-셔틀 근접 확인 시 rescue_near_misses()로 강제 복구
        #     목적: 전역 임계값이 너무 높아 누락된 서브 리턴·약한 타격 복원
        if hit_events:
            print("[Step 4.5] 포즈 기반 선수 분류 교정")

            # (A) 확정 타점 owner 교정
            pose_at_hits = self._collect_pose_at_frames(
                video_path, [e.frame for e in hit_events]
            )
            if pose_at_hits:
                hit_events = detector.apply_pose_owner(
                    hit_events, x_arr, y_arr, pose_at_hits, verbose=verbose
                )
                print(f"           → owner 교정 완료 ({len(pose_at_hits)}프레임)")

            # (B) near-miss 복구 (IQR 억제된 서브 리턴 포함)
            near_miss_frames = detector.get_near_miss_frames(hit_events)
            if near_miss_frames:
                pose_at_near_miss = self._collect_pose_at_frames(
                    video_path, near_miss_frames, context_radius=2
                )
                if pose_at_near_miss:
                    before = len(hit_events)
                    hit_events = detector.rescue_near_misses(
                        hit_events, x_arr, y_arr,
                        pose_at_near_miss,
                        verbose=verbose,
                    )
                    rescued = len(hit_events) - before
                    if rescued > 0:
                        print(f"           → near-miss 복구: +{rescued}개 "
                              f"(총 {len(hit_events)}개)")

        # ── Step 4.6: 포즈 교정 완료 후 교대 교정 실행 ──────────
        # [v11] detect()에서 run_alternation=False로 호출했으므로,
        # 포즈 교정(Step 4.5A)과 rescue(4.5B)가 완전히 끝난 뒤 한 번만 실행.
        # 이 시점엔 포즈 확정 타점(method에 "_pose"/"rescue_pose" 포함)이
        # _correct_owner_alternation 내에서 flip 금지 처리된다.
        hit_events = detector._correct_owner_alternation(hit_events, y_arr)
        if verbose:
            print(f"[Step 4.6] 교대 교정 완료 → {len(hit_events)}개")

        # ── Step 4.7: In/Out 판정 (랠리 종료 낙하 지점) ────────
        rally_drops = self._find_rally_drops(hit_events, x_arr, y_arr, fps)
        drop_results = []   # 판정 결과 (JSON 출력용)
        drop_by_frame = {}  # {frame: {...}} — 프레임 루프에서 빠른 조회

        if rally_drops:
            print(f"[Step 4.7] In/Out 판정 ({len(rally_drops)}건)")
            for i, drop in enumerate(rally_drops):
                minimap_pt = frame_to_minimap(
                    (drop["sx"], drop["sy"]), hg["to_minimap"]
                )
                result = classify_drop_location(
                    hg["minimap_pts"], hg["net_y_minimap"], minimap_pt
                )

                is_in = result["location"] != "out"
                label = "IN" if is_in else "OUT"
                loc_kr = {"in_top": "상단 코트", "in_bottom": "하단 코트", "out": "아웃"}

                margin = result["margin_cm"]
                if margin >= 0:
                    dist_str = f"{result['nearest_line']}까지 약 {margin:.1f}cm"
                else:
                    dist_str = f"{result['nearest_line']} 밖 약 {abs(margin):.1f}cm"

                print(
                    f"  [In/Out 판정] 결과: {label} ({loc_kr[result['location']]}) "
                    f"| {dist_str} "
                    f"(좌표: {drop['sx']:.0f}, {drop['sy']:.0f})"
                )

                drop_entry = {
                    **drop,
                    "is_in":    is_in,
                    "result":   result,
                    "rally_idx": i + 1,
                }
                drop_results.append(drop_entry)

        if drop_results:
            # ── 각 랠리의 X마크 지속 범위 계산 ──────────────────
            # 다음 랠리 시작 프레임 목록: hit_events 사이의 3초+ 갭 지점
            RALLY_GAP_SEC = 3.0
            next_rally_starts: list[int] = []
            for k in range(1, len(hit_events)):
                gap = (hit_events[k].frame - hit_events[k - 1].frame) / max(fps, 1)
                if gap > RALLY_GAP_SEC:
                    next_rally_starts.append(hit_events[k].frame)

            for drop_entry in drop_results:
                drop_frame = drop_entry["frame"]
                # 이 드롭 이후 처음 오는 다음 랠리 시작 프레임 탐색
                persist_end = n_frames  # 마지막 랠리 → 영상 끝까지 유지
                for rs in next_rally_starts:
                    if rs > drop_frame:
                        persist_end = rs
                        break
                # drop 프레임부터 다음 랠리 직전까지 등록
                for f in range(drop_frame, persist_end):
                    drop_by_frame[f] = drop_entry
        else:
            print("[Step 4.7] In/Out 판정 — 낙하 감지 없음")

        # 렌더링에서 빠른 조회를 위한 세트
        hit_frames   = {e.frame for e in hit_events}
        fault_frames = {e.frame for e in net_fault_events}

        # ── Step 5: 렌더러 초기화 ────────────────────────────
        minimap_renderer  = MinimapRenderer(hg, hit_events)
        skeleton_renderer = SkeletonCourtRenderer(frame_w, frame_h, hg, net_coords=net_coords)

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

            # 타점 표시 (브랜드 컬러: Royal Blue #3B82F6)
            if any(abs(frame_idx - hf) <= 2 for hf in hit_frames):
                cv2.putText(
                    annotated, "IMPACT",
                    (frame_w // 2 - 90, 130),
                    cv2.FONT_HERSHEY_DUPLEX, 2.2,
                    (246, 130, 59), 5, cv2.LINE_AA,   # BGR Royal Blue
                )
                # B 채널 증가 → 파란 하이라이트 (브랜드 컬러 방향)
                annotated[:, :, 0] = np.clip(
                    annotated[:, :, 0].astype(np.int16) + 60, 0, 255
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

            # 낙하 지점 X 마크 (drop 프레임 이후 ~1.5초 지속)
            if frame_idx in drop_by_frame:
                d = drop_by_frame[frame_idx]
                skeleton_renderer.draw_drop_mark(
                    skeleton_canvas, d["sx"], d["sy"], d["is_in"]
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
        api_data["drop_judgments"]     = [
            {
                "rally_idx":       d["rally_idx"],
                "frame":           d["frame"],
                "location":        d["result"]["location"],
                "margin_cm":       d["result"]["margin_cm"],
                "nearest_line":    d["result"]["nearest_line"],
                "bwf_x":           d["result"]["bwf_x"],
                "bwf_y":           d["result"]["bwf_y"],
                "last_hit_number": d["last_hit_number"],
                "last_hit_owner":  d["last_hit_owner"],
            }
            for d in drop_results
        ]

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

    def _collect_pose_at_frames(
        self,
        video_path: str,
        frames: List[int],
        context_radius: int = 2,
    ) -> dict:
        """
        지정된 프레임들에 대해 YOLO pose predict를 실행하고
        {frame_idx: List[dict]} 형태로 반환한다.

        [왜 track이 아닌 predict를 쓰는가]
        여기서는 track ID 일관성이 필요 없다. 타점 시점의 선수 위치(손목·중심)만
        알면 되므로 predict()가 더 빠르고 render loop의 track 상태를 오염시키지 않는다.

        Args:
            video_path      : 영상 파일 경로.
            frames          : 포즈 수집 대상 frame 인덱스 리스트.
            context_radius  : 각 frame 앞뒤로 추가 수집할 프레임 수.
                              타점 frame 자체에 셔틀이 없어도 인근 프레임으로 보완.

        Returns:
            {frame_idx: List[dict]} — keypoints 있는 프레임만 포함.
        """
        if not frames:
            return {}

        # context 포함 대상 프레임 집합
        target_set = set()
        for f in frames:
            for delta in range(-context_radius, context_radius + 1):
                target_set.add(f + delta)

        cap = cv2.VideoCapture(video_path)
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target_set = {f for f in target_set if 0 <= f < n_total}

        result: dict = {}

        for frame_idx in sorted(target_set):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            # predict (no tracking) — 빠르고 track 상태 오염 없음
            pose_res = self.pose_model.predict(
                frame, verbose=False, conf=POSE_CONFIG["confidence"]
            )[0]
            kp_list = extract_keypoints(pose_res)
            if kp_list:
                result[frame_idx] = kp_list

        cap.release()
        return result

    # ────────────────────────────────────────────────────────
    @staticmethod
    def _find_rally_drops(
        hit_events: list,
        x_arr: np.ndarray,
        y_arr: np.ndarray,
        fps: float,
    ) -> list:
        """
        각 랠리의 마지막 타점 이후 셔틀콕 **착지** 프레임을 찾는다.

        랠리 분리: 연속 타점 간 gap > RALLY_GAP_SEC → 별도 랠리.
        착지 감지: ``_find_landing_frame()`` — 궤적 방향 급변(바운스) 직전 프레임.

        Returns:
            [{"frame", "sx", "sy", "last_hit_number", "last_hit_owner"}, ...]
        """
        if not hit_events:
            return []

        RALLY_GAP_SEC  = 3.0
        MAX_SEARCH_SEC = 5.0
        n = len(x_arr)

        # ── 각 랠리의 마지막 타점 인덱스 ──────────────────────
        last_hit_indices: list[int] = []
        for i in range(len(hit_events)):
            is_last = (i == len(hit_events) - 1)
            if not is_last:
                gap = (hit_events[i + 1].frame - hit_events[i].frame) / max(fps, 1)
                if gap > RALLY_GAP_SEC:
                    is_last = True
            if is_last:
                last_hit_indices.append(i)

        drops: list[dict] = []
        for idx in last_hit_indices:
            hit = hit_events[idx]
            # 타점 직후 몇 프레임은 라켓 접촉 노이즈 → 건너뛰기
            skip         = max(3, int(fps * 0.1))
            search_start = hit.frame + skip
            search_end   = min(hit.frame + int(fps * MAX_SEARCH_SEC), n)

            landing = RallyTrackPipeline._find_landing_frame(
                x_arr, y_arr, search_start, search_end,
            )
            if landing is not None:
                frame, sx, sy = landing
                drops.append({
                    "frame":           frame,
                    "sx":              sx,
                    "sy":              sy,
                    "last_hit_number": hit.hit_number,
                    "last_hit_owner":  hit.owner,
                })

        return drops

    @staticmethod
    def _find_landing_frame(
        x_arr: np.ndarray,
        y_arr: np.ndarray,
        start_frame: int,
        search_end: int,
    ):
        """
        셔틀콕 착지 프레임을 궤적 물리 분석으로 찾는다.

        [탑뷰(프로) / 쿼터뷰(아마) 공통 대응]

        핵심 설계:
          비중첩(non-overlapping) 윈도우로 바운스 전/후 속도 벡터를 계산.

          이전 버전(겹치는 smoothed_vel)의 문제:
            smoothed_vel(i-1)과 smoothed_vel(i+1)이 i 주변 구간을 공유해
            바운스 전/후 방향이 혼합 → 실제 각도가 축소 → 아마추어 영상 미감지.

          비중첩 방식:
            before_vel : pts[i - W_BEFORE ~ i-1] 구간만 사용
            after_vel  : pts[i+1 ~ i + W_AFTER] 구간만 사용
            → 바운스 전/후 방향이 깨끗하게 분리, 실제 각도 정확히 포착.
            → 네트 부근 TrackNet 1~2프레임 오류는 W=3 평균으로 희석.

          탐지 조건:
            방향 변화 > 80° AND 속도 < 이전의 70%
            · 네트 통과: 방향 변화 ≤60°, 속도 거의 유지 → 미감지
            · 실제 바운스: 방향 >80° 반전 + 속도 급감 → 감지

          fallback:
            바운스 미감지(탑뷰에서 착지가 2D에 잘 안 보이는 경우) →
            첫 번째 큰 추적 갭(>4프레임) 직전 프레임.

        Returns:
            (frame, x, y) 또는 None
        """
        BOUNCE_ANGLE = 80    # 방향 전환 임계각
        SPEED_RATIO  = 0.70  # 착지 확정: 바운스 후 속도가 이전의 70% 이하
        W_BEFORE     = 3     # 착지 직전 평균 구간 길이
        W_AFTER      = 3     # 착지 직후 평균 구간 길이
        GAP_THRESH   = 4     # 연속 미감지 프레임 수 > 이 값 → 궤적 종료 신호

        n   = min(search_end, len(x_arr))
        pts: list = []
        for f in range(start_frame, n):
            if x_arr[f] > 0 and y_arr[f] > 0:
                pts.append((f, float(x_arr[f]), float(y_arr[f])))

        if not pts:
            return None

        # ── 공용 fallback: 첫 번째 큰 추적 갭 직전 프레임 ───
        def first_before_gap():
            for k in range(1, len(pts)):
                if pts[k][0] - pts[k - 1][0] > GAP_THRESH:
                    return pts[k - 1]
            return pts[-1]

        if len(pts) < W_BEFORE + W_AFTER + 1:
            return first_before_gap()

        # ── 1차: 비중첩 윈도우 방향 급변 + 속도 감소 감지 ───
        for i in range(W_BEFORE, len(pts) - W_AFTER):
            # before: pts[i-W_BEFORE] → pts[i-1]  (착지 직전 구간)
            i0b = max(0, i - W_BEFORE)
            i1b = i - 1
            if i1b <= i0b:
                continue
            dt_b = max(pts[i1b][0] - pts[i0b][0], 1)
            vxb  = (pts[i1b][1] - pts[i0b][1]) / dt_b
            vyb  = (pts[i1b][2] - pts[i0b][2]) / dt_b
            spd_b = (vxb ** 2 + vyb ** 2) ** 0.5

            if spd_b < 1e-6:
                continue

            # after: pts[i+1] → pts[i+W_AFTER]  (착지 직후 구간)
            i0a = i + 1
            i1a = min(len(pts) - 1, i + W_AFTER)
            if i1a <= i0a:
                continue
            dt_a = max(pts[i1a][0] - pts[i0a][0], 1)
            vxa  = (pts[i1a][1] - pts[i0a][1]) / dt_a
            vya  = (pts[i1a][2] - pts[i0a][2]) / dt_a
            spd_a = (vxa ** 2 + vya ** 2) ** 0.5

            cos_a = float(np.clip(
                (vxb * vxa + vyb * vya) / (spd_b * max(spd_a, 1e-6)),
                -1.0, 1.0,
            ))
            angle = float(np.degrees(np.arccos(cos_a)))

            # 방향 급변 AND 속도 급감 → 착지 후보 감지
            if angle > BOUNCE_ANGLE and spd_a < spd_b * SPEED_RATIO:
                # ── 정제: 조건이 유지되는 마지막 프레임으로 전진 ─
                # before 윈도우에 post-bounce 프레임이 섞일수록 각도 축소 →
                # 조건 미충족 시점 직전이 실제 착지 프레임에 가장 가깝다
                last_valid = i
                for j in range(i + 1, min(i + 6, len(pts) - W_AFTER)):
                    i0b_j  = max(0, j - W_BEFORE)
                    i1b_j  = j - 1
                    if i1b_j <= i0b_j:
                        break

                    dt_bj  = max(pts[i1b_j][0] - pts[i0b_j][0], 1)
                    vxbj   = (pts[i1b_j][1] - pts[i0b_j][1]) / dt_bj
                    vybj   = (pts[i1b_j][2] - pts[i0b_j][2]) / dt_bj
                    spd_bj = (vxbj ** 2 + vybj ** 2) ** 0.5

                    i0a_j  = j + 1
                    i1a_j  = min(len(pts) - 1, j + W_AFTER)
                    if i1a_j <= i0a_j:
                        break

                    dt_aj  = max(pts[i1a_j][0] - pts[i0a_j][0], 1)
                    vxaj   = (pts[i1a_j][1] - pts[i0a_j][1]) / dt_aj
                    vyaj   = (pts[i1a_j][2] - pts[i0a_j][2]) / dt_aj
                    spd_aj = (vxaj ** 2 + vyaj ** 2) ** 0.5

                    if spd_bj < 1e-6:
                        break

                    cos_aj = float(np.clip(
                        (vxbj * vxaj + vybj * vyaj) / (spd_bj * max(spd_aj, 1e-6)),
                        -1.0, 1.0,
                    ))
                    if (float(np.degrees(np.arccos(cos_aj))) > BOUNCE_ANGLE
                            and spd_aj < spd_bj * SPEED_RATIO):
                        last_valid = j
                    else:
                        break

                return pts[last_valid]

        # ── 2차: 첫 번째 큰 추적 갭 직전 프레임 ─────────────
        return first_before_gap()

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
            side = "상단(블루)" if e.owner == "top" else "하단(골드)"
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