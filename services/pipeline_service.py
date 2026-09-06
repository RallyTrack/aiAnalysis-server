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

import bisect
import json
import math
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

from analysis.config        import PATHS, TRACKNET_CONFIG, MINIMAP_CONFIG, POSE_CONFIG, STROKE_CONFIG
from analysis.court         import (
    compute_homographies, classify_drop_location, frame_to_minimap,
)
from analysis.impact        import ImpactDetector, to_api_json
from analysis.minimap       import MinimapRenderer
from analysis.net_judge     import NetCrossingEvent, NetFaultEvent, NetJudge
from analysis.skeleton_view import SkeletonCourtRenderer
from services.analysis_mode import get_analysis_mode_profile


_stroke_classifier = None
_stroke_classifier_loaded = False


def _get_stroke_classifier():
    """ViT 5-fold ensemble — ViT-only fallback 및 feature classifier 의 teacher 로 사용."""
    global _stroke_classifier, _stroke_classifier_loaded
    if not _stroke_classifier_loaded:
        _stroke_classifier_loaded = True
        try:
            from analysis.stroke_classifier import StrokeClassifierEnsemble
            _stroke_classifier = StrokeClassifierEnsemble()
        except Exception as e:
            print(f"[StrokeClassifier] 초기화 실패 (스트로크 분류 생략): {e}")
    return _stroke_classifier


# Feature-based classifier 는 mode 별로 다른 pkl. lazy + per-mode memoize.
_feature_classifier_cache: dict = {}


def _get_feature_classifier(mode: str):
    """sklearn feature classifier lazy load.

    Args:
        mode: "pro" | "amateur". 다른 값이면 pro fallback.

    Returns:
        ``TrainedStrokeModel`` 또는 None (pkl 없거나 로드 실패 시).
    """
    key = mode if mode in ("pro", "amateur") else "pro"
    if key in _feature_classifier_cache:
        return _feature_classifier_cache[key]

    pkl_key = (f"stroke_feature_classifier_{key}")
    pkl_path = PATHS.get(pkl_key)
    if not pkl_path or not os.path.exists(pkl_path):
        print(f"[FeatureClassifier:{key}] pkl 없음: {pkl_path} → 캐시 None")
        _feature_classifier_cache[key] = None
        return None

    try:
        from analysis.stroke_feature_classifier import TrainedStrokeModel
        model = TrainedStrokeModel.load(Path(pkl_path))
        print(f"[FeatureClassifier:{key}] loaded "
              f"{model.model_name}/{model.label_scheme} "
              f"({len(model.class_names)}-class)")
        _feature_classifier_cache[key] = model
        return model
    except Exception as e:
        print(f"[FeatureClassifier:{key}] 로드 실패: {e}")
        _feature_classifier_cache[key] = None
        return None


def _build_feature_inputs_for_event(
    ev,
    video_path: str,
    df_ball,
    pose_at_hits: dict,
    hg: dict,
    net_y_pixel,
    vit_ensemble,
):
    """단일 ImpactEvent → ``FeatureInputs`` 변환.

    Args:
        ev: ImpactEvent (frame, hit_number 포함)
        video_path: 원본 영상 경로 (ViT teacher offset frame 추출용)
        df_ball: TrackNet 출력 DataFrame (컬럼 Frame/X/Y/Visibility)
        pose_at_hits: {frame_idx: [pose_dict, ...]} (Step 4.5 에서 수집)
        hg: compute_homographies 결과 dict
        net_y_pixel: 네트 픽셀 y (frame_h * net_y_ratio)
        vit_ensemble: StrokeClassifierEnsemble (offset 별 frame 추론용)
    """
    import cv2 as _cv2
    import numpy as _np
    from analysis.stroke_features import FeatureInputs, VIT_OFFSETS

    # ── 1) ViT teacher 5-offset probs (-2..+2) ──
    probs_by_off: dict = {}
    cap = _cv2.VideoCapture(str(video_path))
    try:
        for off in VIT_OFFSETS:
            tgt = ev.frame + off
            if tgt < 0:
                continue
            cap.set(_cv2.CAP_PROP_POS_FRAMES, tgt)
            ok, bgr = cap.read()
            if not ok:
                continue
            r = vit_ensemble.classify(bgr)
            probs_by_off[off] = r.probs.astype(_np.float32)
    finally:
        cap.release()

    # ── 2) shuttle 위치 (hit frame) ──
    shuttle_xy = None
    if df_ball is not None and len(df_ball) > 0:
        at = df_ball[(df_ball["Frame"] == ev.frame) & (df_ball["Visibility"] == 1)]
        if not at.empty:
            shuttle_xy = (float(at["X"].iloc[0]), float(at["Y"].iloc[0]))

    # ── 3) hitter pose — 셔틀에 가장 가까운 손목의 사람 ──
    def _find_hitter_kp(pose_list, shuttle):
        if not pose_list or shuttle is None:
            return None
        sx, sy = shuttle
        best, best_d = None, float("inf")
        for p in pose_list:
            kp = _np.asarray(p.get("keypoints", []), dtype=_np.float32)
            if kp.shape != (17, 3):
                continue
            for idx in (9, 10):    # COCO wrist
                x, y, c = kp[idx]
                if c < 0.3:
                    continue
                d = _np.hypot(x - sx, y - sy)
                if d < best_d:
                    best_d = d
                    best = kp
        return best

    kp     = _find_hitter_kp(pose_at_hits.get(ev.frame), shuttle_xy)
    kp_pre = _find_hitter_kp(pose_at_hits.get(ev.frame - 2), shuttle_xy)
    kp_post = _find_hitter_kp(pose_at_hits.get(ev.frame + 2), shuttle_xy)

    return FeatureInputs(
        video_path=Path(video_path),
        hit_frame=ev.frame,
        hit_number=ev.hit_number,
        ball_df=df_ball,
        net_y=net_y_pixel,
        impact_xy_pixel=shuttle_xy,
        homography_matrix=hg.get("M") if isinstance(hg, dict) else None,
        prev_hit_gap_sec=None,
        next_hit_gap_sec=None,
        hitter_kp_at_hit=kp,
        hitter_kp_pre=kp_pre,
        hitter_kp_post=kp_post,
        shuttle_xy_at_hit=shuttle_xy,
        vit_probs_by_offset=probs_by_off,
    )


def _mode_profile(mode: str) -> tuple[str, dict]:
    return get_analysis_mode_profile(mode)


# ────────────────────────────────────────────────────────────
# TrackNet subprocess 실행
# ────────────────────────────────────────────────────────────

def run_tracknet(video_path: str, batch_size: int | None = None) -> str:
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
    resolved_batch_size = batch_size or TRACKNET_CONFIG["batch_size"]

    cmd = [
        sys.executable, predict_script,
        "--video_file",      video_path_fwd,
        "--tracknet_file",   PATHS["tracknet_ckpt"],
        "--inpaintnet_file", PATHS["inpaintnet_ckpt"],
        "--batch_size",      str(resolved_batch_size),
        "--save_dir",        PATHS["prediction_dir"],
    ]
    print(f"[TrackNet] 분석 시작... batch_size={resolved_batch_size}")
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
# 타격 owner 단순 교대 재할당
# ────────────────────────────────────────────────────────────

def _reassign_owners_alternating(
    events: list,
    y_arr: np.ndarray,
    fps: float,
    owner_y: float,
    rally_gap_sec: float = 3.0,
) -> list:
    """
    각 랠리의 첫 타점 owner를 결정하고 이후 타점은 top ↔ bottom 교대로 재할당한다.

    첫 hitter 결정:
      Step 4.5(포즈 교정)의 결과를 그대로 사용한다.
      - peaks_crossed 포함 모든 타점에 포즈가 적용되므로 (apply_pose_owner skip 제거)
        first_ev.owner가 가장 신뢰할 수 있는 값이다.
      - 포즈 데이터가 없어 교정이 안 된 경우 → 셔틀 Y 위치 fallback

    Args:
        events       : ImpactEvent 리스트 (hit_number 이미 부여된 상태, Step 4.5 교정 포함)
        y_arr        : 프레임별 셔틀콕 Y 좌표 배열
        fps          : 영상 FPS (랠리 갭 계산용)
        owner_y      : top/bottom 구분 Y 기준 (px)
        rally_gap_sec: 이 값(초) 초과 갭이면 새 랠리 시작
        crossings    : 미사용 (API 호환성 유지)
    """
    if not events:
        return events

    evs = list(events)
    rally_start = 0

    while rally_start < len(evs):
        # ── 이 랠리의 끝 인덱스 탐색 ──────────────────────────
        rally_end = rally_start
        while rally_end + 1 < len(evs):
            gap = (evs[rally_end + 1].frame - evs[rally_end].frame) / max(fps, 1)
            if gap > rally_gap_sec:
                break
            rally_end += 1

        rally_first_frame = evs[rally_start].frame
        first_ev          = evs[rally_start]

        # Step 4.5 포즈 교정 결과를 그대로 사용
        # apply_pose_owner가 peaks_crossed 포함 모든 타점에 적용되므로
        # first_ev.owner가 가장 신뢰할 수 있는 값이다.
        cur_owner = first_ev.owner if first_ev.owner in ("top", "bottom") else None

        # Fallback: 포즈 데이터 완전 부재로 owner 미설정 시 셔틀 Y 위치 사용
        if cur_owner is None:
            fy = float(y_arr[min(rally_first_frame, len(y_arr) - 1)])
            cur_owner = "top" if fy < owner_y else "bottom"

        # ── 랠리 내 교대 할당 ──────────────────────────────────
        for k in range(rally_start, rally_end + 1):
            evs[k].owner  = cur_owner
            evs[k].player = "top" if cur_owner == "top" else "bottom"
            cur_owner = "bottom" if cur_owner == "top" else "top"

        rally_start = rally_end + 1

    return evs


# ────────────────────────────────────────────────────────────
# 선수 중심 좌표 추출 헬퍼
# ────────────────────────────────────────────────────────────

def _extract_player_center(
    pose_data: dict,
    target_frame: int,
    owner: str,
    owner_y_threshold: float,
    frame_w: int,
    frame_h: int,
    context_radius: int = 2,
) -> "tuple[float, float] | None":
    """
    타격 프레임 근방의 포즈 데이터에서 해당 선수(owner)의 신체 중심 좌표를 추출한다.

    Args:
        pose_data         : {frame_idx: List[dict]} — _collect_pose_at_frames() 반환값
        target_frame      : 타격 프레임 인덱스
        owner             : "top" | "bottom"
        owner_y_threshold : top/bottom 구분 기준 y (픽셀)
        frame_w, frame_h  : 영상 해상도 (정규화 기준)
        context_radius    : 탐색 범위 (target_frame ± context_radius)

    Returns:
        (x_norm, y_norm) 0~1 정규화 좌표, 또는 None (포즈 미감지 시)
    """
    KP_HIP      = [11, 12]
    KP_KNEE     = [13, 14]
    KP_SHOULDER = [5, 6]
    KP_CONF     = 0.25

    def _valid_pts(kps, indices):
        return [kps[i] for i in indices
                if len(kps) > i and kps[i][2] >= KP_CONF and kps[i][1] > 0]

    for delta in range(-context_radius, context_radius + 1):
        probe = target_frame + delta
        players = pose_data.get(probe, [])
        if not players:
            continue

        for player in players:
            kps = player.get("keypoints", [])
            if len(kps) < 13:
                continue

            # 힙 → 무릎 → 어깨 순서로 fallback하며 중심 좌표 계산
            pts = _valid_pts(kps, KP_HIP)
            if not pts:
                pts = _valid_pts(kps, KP_KNEE)
            if not pts:
                pts = _valid_pts(kps, KP_SHOULDER)
            if not pts:
                pts = [k for k in kps if k[2] >= KP_CONF and k[1] > 0]
            if not pts:
                continue

            center_y = float(np.median([k[1] for k in pts]))
            center_x = float(np.median([k[0] for k in pts]))

            # owner 판별: y 기준으로 top/bottom 확인
            pose_owner = "top" if center_y < owner_y_threshold else "bottom"
            if pose_owner != owner:
                continue

            # 0~1 정규화 후 반환
            x_norm = max(0.0, min(1.0, center_x / max(frame_w, 1)))
            y_norm = max(0.0, min(1.0, center_y / max(frame_h, 1)))
            return x_norm, y_norm

    return None  # 해당 owner 포즈 미감지


# ────────────────────────────────────────────────────────────
# In/Out 오판정 검증 헬퍼
# ────────────────────────────────────────────────────────────

def validate_drop(
    drop: dict,
    hit_events: list,
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    fps: float,
    hit_check_sec: float = 2.0,
    shuttle_check_sec: float = 1.0,
    shuttle_skip_sec: float = 0.1,
    max_bounce_disp_px: float = 60.0,
    max_bounce_speed_px: float = 8.0,
) -> bool:
    """
    Determine whether a detected shuttle drop is a genuine landing or a false positive.

    Two-stage validation:

    Stage 1 — Hit-event continuity check (high confidence when hit detection is working)
        If any ImpactEvent occurs within `hit_check_sec` after the drop frame,
        the rally is ongoing → drop is a false positive.

    Stage 2 — Shuttle kinematic check (covers gaps in hit detection)
        After skipping `shuttle_skip_sec` from drop frame (to avoid bounce noise),
        examine shuttle position data for `shuttle_check_sec`.
        A genuine landing shows:
            - maximum displacement from drop point < max_bounce_disp_px
            - mean inter-frame speed < max_bounce_speed_px
        Exceeding either threshold indicates the shuttle is still in play.

    Returns:
        True  → valid landing, include in rally results
        False → false positive, discard
    """
    drop_frame = drop["frame"]
    drop_x     = float(drop["sx"])
    drop_y     = float(drop["sy"])

    # ── Stage 1: hit-event continuity ────────────────────────
    hits_after = [
        ev for ev in hit_events
        if ev.frame > drop_frame
        and (ev.frame - drop_frame) / max(fps, 1.0) <= hit_check_sec
    ]
    if hits_after:
        return False

    # ── Stage 2: shuttle kinematics ──────────────────────────
    skip_frames  = max(1, int(fps * shuttle_skip_sec))
    check_frames = max(1, int(fps * shuttle_check_sec))
    start_f = drop_frame + skip_frames
    end_f   = min(drop_frame + skip_frames + check_frames, len(x_arr))

    valid_pos = [
        (float(x_arr[f]), float(y_arr[f]))
        for f in range(start_f, end_f)
        if x_arr[f] > 0 and y_arr[f] > 0
    ]

    if len(valid_pos) < 3:
        # Insufficient data after drop → assume genuine landing
        return True

    max_disp = max(
        math.sqrt((px - drop_x) ** 2 + (py - drop_y) ** 2)
        for px, py in valid_pos
    )
    speeds = [
        math.sqrt(
            (valid_pos[i][0] - valid_pos[i - 1][0]) ** 2 +
            (valid_pos[i][1] - valid_pos[i - 1][1]) ** 2
        )
        for i in range(1, len(valid_pos))
    ]
    avg_speed = sum(speeds) / len(speeds) if speeds else 0.0

    if max_disp > max_bounce_disp_px or avg_speed > max_bounce_speed_px:
        return False

    return True


def classify_rally_result(
    last_hit_owner: str,
    valid_drops: list,
    rally_idx: int,
) -> dict:
    matching = [d for d in valid_drops if d.get("rally_idx") == rally_idx]
    if matching:
        drop        = matching[0]
        is_in       = drop.get("is_in", False)
        result_dict = drop.get("result", {})
        location    = result_dict.get("location", "out")
        margin_cm   = result_dict.get("margin_cm", None)

        if margin_cm is not None:
            confidence = "HIGH" if abs(margin_cm) >= 10.0 else "LOW"
        else:
            confidence = None

        return {
            "lastHitOwner": last_hit_owner,
            "resultType":   "WINNER" if is_in else "CONFIRMED_ERROR",
            "isIn":         is_in,
            "location":     location,
            "marginCm":     round(margin_cm, 1) if margin_cm is not None else None,
            "confidence":   confidence,
        }

    return {
        "lastHitOwner": last_hit_owner,
        "resultType":   "UNKNOWN",
        "isIn":         None,
        "location":     None,
        "marginCm":     None,
        "confidence":   None,
    }


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
        mode:         str = "pro",
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
            mode         : "pro" | "amateur" 분석 프로파일

        Returns:
            api_data dict (result_paths 포함)
        """
        video_path = os.path.abspath(video_path)
        video_name = Path(video_path).stem
        mode, profile = _mode_profile(mode)
        os.makedirs(PATHS["result_dir"],     exist_ok=True)
        os.makedirs(PATHS["prediction_dir"], exist_ok=True)
        print(f"[Mode] analysis mode={mode}")

        # ── Step 1: TrackNet ──────────────────────────────────
        if self.skip_tracknet:
            csv_path = os.path.join(PATHS["prediction_dir"], f"{video_name}_ball.csv")
            print(f"[Step 1] TrackNet 생략 → {csv_path}")
        else:
            print("[Step 1] TrackNetV3 실행")
            csv_path = run_tracknet(
                video_path,
                batch_size=profile["tracknet_batch_size"],
            )

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

        if net_coords is None and user_corners is not None:
            corners_np = np.array(user_corners, dtype=np.float32)  # [TL, TR, BR, BL]
            # 코트 상단/하단 라인의 y 중점 → 네트는 코트 세로 중앙에 위치
            court_top_y = float((corners_np[0, 1] + corners_np[1, 1]) / 2.0)
            court_bottom_y = float((corners_np[2, 1] + corners_np[3, 1]) / 2.0)
            estimated_net_y = (court_top_y + court_bottom_y) / 2.0
            net_y_ratio = estimated_net_y / max(frame_h, 1)
            print(f"[Step 4b] net_y_ratio → user_corners 기반 추정값: {net_y_ratio:.3f}")
        elif net_coords is None:
            print(f"[Step 4b] net_y_ratio → 디폴트 사용: {net_y_ratio:.3f}")

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
            pose_conf_threshold = profile["pose_conf_threshold"],
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
                video_path,
                [e.frame for e in hit_events],
                pose_model_conf=profile["pose_model_conf"],
            )
            if pose_at_hits:
                hit_events = detector.apply_pose_owner(
                    hit_events, x_arr, y_arr, pose_at_hits, verbose=verbose
                )
                print(f"           → owner 교정 완료 ({len(pose_at_hits)}프레임)")

                # ── 선수 위치 주입 (이미 수집된 pose_at_hits 재사용, 추가 YOLO 없음) ──
                _owner_y = (
                    owner_y_threshold if owner_y_threshold is not None
                    else frame_h * net_y_ratio
                )
                injected = 0
                for ev in hit_events:
                    result = _extract_player_center(
                        pose_data         = pose_at_hits,
                        target_frame      = ev.frame,
                        owner             = ev.owner,
                        owner_y_threshold = _owner_y,
                        frame_w           = frame_w,
                        frame_h           = frame_h,
                    )
                    if result is not None:
                        ev.player_x, ev.player_y = result
                        injected += 1
                print(f"           → 선수 위치 주입: {injected}/{len(hit_events)}개")

            # (B) near-miss 복구 (IQR 억제된 서브 리턴 포함)
            if profile["run_near_miss_rescue"]:
                near_miss_frames = detector.get_near_miss_frames(hit_events)
                if near_miss_frames:
                    pose_at_near_miss = self._collect_pose_at_frames(
                        video_path,
                        near_miss_frames,
                        context_radius=2,
                        pose_model_conf=profile["pose_model_conf"],
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
            else:
                print("           → near-miss 복구 생략 (pro mode)")

        # ── Step 4.6: 단순 교대 owner 재할당 ────────────────────
        # 첫 타점 owner: Step 4.5 포즈 교정 결과(손목-셔틀 거리) 신뢰
        # 이후 타점은 top ↔ bottom 교대 (YOLO 포즈 감지가 일부 실패해도 보정)
        _owner_y_ref = (
            owner_y_threshold if owner_y_threshold is not None
            else frame_h * net_y_ratio
        )
        hit_events = _reassign_owners_alternating(hit_events, y_arr, fps, _owner_y_ref)
        print(f"[Step 4.6] 교대 owner 재할당 완료 → {len(hit_events)}개")

        # ── Step 4.8: 스트로크 분류 ──────────────────────────────
        # 우선순위: Feature classifier (sklearn, traj+pose+ViT teacher 융합)
        #          → 없거나 실패 시 ViT-only fallback (legacy)
        #          → 둘 다 없으면 stroke 필드 None 유지
        # mode 별로 다른 feature classifier 로드 (pro 9-class / amateur 4-class).
        print(f"[Step 4.8] 스트로크 분류 (mode={mode})")
        feature_clf = _get_feature_classifier(mode)
        vit_ensemble = _get_stroke_classifier()  # teacher 겸 fallback

        feature_classified = 0
        if feature_clf is not None and vit_ensemble is not None and hit_events:
            print(f"           feature classifier ({feature_clf.label_scheme}, "
                  f"{len(feature_clf.class_names)}-class)")
            df_ball = pd.read_csv(csv_path)
            # 컬럼 통일 — TrackNet CSV 는 'Frame X Y Visibility' (대문자).
            # load_trajectory_csv 는 소문자화하지만 여기선 원본 그대로 사용.
            df_ball.columns = [
                {"frame": "Frame", "x": "X", "y": "Y", "visibility": "Visibility"}
                .get(c.strip().lower(), c) for c in df_ball.columns
            ]
            net_y_pixel = frame_h * net_y_ratio

            for ev in hit_events:
                try:
                    from analysis.stroke_features import compute_features
                    inputs = _build_feature_inputs_for_event(
                        ev, video_path, df_ball, pose_at_hits or {},
                        hg, net_y_pixel, vit_ensemble,
                    )
                    feats = compute_features(inputs)
                    feats_df = pd.DataFrame([feats])
                    # 모델이 학습 시 본 컬럼만, 순서 유지. 누락 컬럼은 NaN.
                    for col in feature_clf.feature_columns:
                        if col not in feats_df.columns:
                            feats_df[col] = float("nan")
                    feats_df = feats_df[feature_clf.feature_columns]

                    pred = feature_clf.predict_with_meta(
                        feats_df,
                        uncertainty_margin=STROKE_CONFIG["uncertainty_margin"],
                    )[0]
                    ev.stroke_type            = pred["label_name"]
                    ev.stroke_confidence      = float(pred["confidence"])
                    ev.stroke_top2            = pred["top2_name"]
                    ev.stroke_top2_confidence = float(pred["top2_conf"])
                    ev.stroke_is_uncertain    = bool(pred["is_uncertain"])
                    ev.stroke_probs           = pred["probs"].tolist()
                    ev.stroke_source          = "feature_classifier"
                    ev.stroke_class_scheme    = feature_clf.label_scheme
                    feature_classified += 1
                except Exception as exc:
                    print(f"           #{ev.hit_number:02d} feature 분류 실패: {exc}")
            print(f"           → feature classifier: {feature_classified}/{len(hit_events)}개 완료")

            # ── Rally-aware Serve 강등 (학습 데이터의 rally context 부재 보정) ──
            gap_thresh = STROKE_CONFIG.get("serve_rally_gap_sec", 3.0)
            if gap_thresh > 0:
                events_sorted = sorted(
                    [e for e in hit_events if e.stroke_type is not None],
                    key=lambda e: e.frame,
                )
                serve_demoted = 0
                for i in range(1, len(events_sorted)):
                    e = events_sorted[i]
                    if e.stroke_type != "Serve" or e.stroke_top2 is None:
                        continue
                    gap = e.time_sec - events_sorted[i - 1].time_sec
                    if gap >= gap_thresh:
                        continue
                    # top1 ↔ top2 swap.
                    e.stroke_type, e.stroke_top2 = e.stroke_top2, e.stroke_type
                    e.stroke_confidence, e.stroke_top2_confidence = (
                        e.stroke_top2_confidence, e.stroke_confidence)
                    serve_demoted += 1
                    print(f"           #{e.hit_number:02d} Serve 강등 → {e.stroke_type} "
                          f"({e.stroke_confidence:.2f}) [gap={gap:.1f}s < {gap_thresh}s]")
                if serve_demoted:
                    print(f"           [rally-aware] Serve 강등 {serve_demoted}건 "
                          f"(gap < {gap_thresh}s)")

            # ── 최종 분류 결과 출력 (강등 후 실제 저장되는 값 기준) ──
            for ev in hit_events:
                if ev.stroke_type is not None and ev.stroke_source == "feature_classifier":
                    uncertain_flag = " [불확실]" if ev.stroke_is_uncertain else ""
                    print(f"           #{ev.hit_number:02d} → "
                          f"{ev.stroke_type} ({ev.stroke_confidence:.2f})"
                          f"{uncertain_flag}")

        # ── ViT-only fallback — feature classifier 안 돌았거나 일부 실패 ──
        unclassified = [e for e in hit_events if e.stroke_type is None]
        if vit_ensemble is not None and unclassified:
            print(f"           ViT-only fallback ({len(unclassified)}개)")
            for ev in unclassified:
                try:
                    result = vit_ensemble.classify_at(video_path, ev.frame)
                    ev.stroke_type       = result.class_name
                    ev.stroke_confidence = float(result.confidence)
                    ev.stroke_source     = "vit_only"
                    print(f"           #{ev.hit_number:02d} → "
                          f"{result.class_name} ({result.confidence:.2f})")
                except Exception as exc:
                    print(f"           #{ev.hit_number:02d} ViT 분류 실패: {exc}")

        if feature_clf is None and vit_ensemble is None:
            print("           → 스트로크 분류기 없음 (weights/stroke/ 확인)")

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

        # ── Step 4.7-post: Filter false-positive In/Out detections ──
        raw_drop_count = len(drop_results)
        drop_results = [
            d for d in drop_results
            if validate_drop(d, hit_events, x_arr, y_arr, fps)
        ]
        print(
            f"[In/Out 검증] {raw_drop_count}건 → {len(drop_results)}건 유효 "
            f"({raw_drop_count - len(drop_results)}건 오판정 제거)"
        )

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

        # ── Build rally_results for backend ability metrics ──────────────────
        rally_results: list = []
        RALLY_GAP_SEC = 3.0
        rally_idx = 1
        for i, ev in enumerate(hit_events):
            is_last = (i == len(hit_events) - 1)
            if not is_last:
                gap = (hit_events[i + 1].frame - ev.frame) / max(fps, 1.0)
                if gap > RALLY_GAP_SEC:
                    is_last = True
            if is_last:
                result = classify_rally_result(
                    last_hit_owner = ev.owner,
                    valid_drops    = drop_results,
                    rally_idx      = rally_idx,
                )
                rally_results.append({
                    "rallyIdx":      rally_idx,
                    "lastHitNumber": ev.hit_number,
                    "lastHitOwner":  ev.owner,
                    "resultType":    result["resultType"],
                    "isIn":          result["isIn"],
                    "location":      result["location"],
                    "marginCm":      result.get("marginCm"),
                    "confidence":    result.get("confidence"),
                })
                rally_idx += 1
        print(f"[rally_results] {len(rally_results)}개 랠리 분류 완료")
        for r in rally_results:
            margin_str = f"marginCm={r['marginCm']}" if r['marginCm'] is not None else "marginCm=None"
            print(f"  Rally #{r['rallyIdx']}: {r['resultType']}, {r['location']}, {margin_str}, confidence={r['confidence']}")

        # 렌더링에서 빠른 조회를 위한 세트/딕셔너리
        hit_frames    = {e.frame for e in hit_events}
        fault_frames  = {e.frame for e in net_fault_events}
        # frame → stroke_type 빠른 조회 (타점 ±2 프레임에 오버레이)
        stroke_by_frame: dict[int, str] = {}
        for ev in hit_events:
            if ev.stroke_type:
                for delta in range(-2, 3):
                    stroke_by_frame[ev.frame + delta] = ev.stroke_type

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

            # 스트로크 분류 표시
            if frame_idx in stroke_by_frame:
                cv2.putText(
                    annotated, stroke_by_frame[frame_idx],
                    (frame_w // 2 - 90, 185),
                    cv2.FONT_HERSHEY_DUPLEX, 1.6,
                    (0, 255, 180), 4, cv2.LINE_AA,
                )

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
                frame_idx, keypoints_list
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

        # ── Compute player mobility metrics (home-zone return rate) ──
        from analysis.minimap import compute_home_zone_minimap, point_in_zone

        _pad = MINIMAP_CONFIG["padding"]
        _mw  = MINIMAP_CONFIG["width"]
        _mh  = MINIMAP_CONFIG["height"]

        player_metrics: dict = {}

        def _get_path_segment(
            path: list,
            frame_keys: list,
            frame_a: int,
            frame_b: int,
        ) -> list:
            """full_path[(frame_idx, mx, my)] 에서 [frame_a, frame_b] 구간의 (mx, my) 반환."""
            i = bisect.bisect_left(frame_keys, frame_a)
            j = bisect.bisect_right(frame_keys, frame_b)
            return [(p[1], p[2]) for p in path[i:j]]

        for side in ("top", "bottom"):
            full_path  = minimap_renderer._player.get_full_path(side)
            zone_min, zone_max = compute_home_zone_minimap(
                side, _mw, _mh, _pad
            )

            side_hits = [ev for ev in hit_events if ev.owner == side]

            if len(side_hits) < 2 or len(full_path) < 5:
                print(f"[기동력] {side} 데이터 부족 — side_hits={len(side_hits)}, full_path={len(full_path)}")
                player_metrics[side] = {"home_return_rate": 0.0}
                continue

            frame_keys = [p[0] for p in full_path]

            return_count = 0
            pair_count   = 0

            for k in range(len(side_hits) - 1):
                segment = _get_path_segment(
                    full_path, frame_keys,
                    side_hits[k].frame, side_hits[k + 1].frame,
                )

                if not segment:
                    continue

                pair_count += 1
                if any(point_in_zone(pt, zone_min, zone_max) for pt in segment):
                    return_count += 1

            home_return_rate = (
                round(return_count / pair_count, 2)
                if pair_count > 0 else 0.0
            )
            player_metrics[side] = {"home_return_rate": home_return_rate}

        print(f"[기동력] top={player_metrics.get('top')}, bottom={player_metrics.get('bottom')}")

        # ── Step 7: H.264 재인코딩 ───────────────────────────
        print("[Step 6] H.264 인코딩")
        f1 = os.path.join(rd, f"{video_name}_1_main.mp4")
        f2 = os.path.join(rd, f"{video_name}_2_minimap.mp4")
        f3 = os.path.join(rd, f"{video_name}_3_skeleton.mp4")
        reencode_h264(tmp_1, f1)
        reencode_h264(tmp_2, f2)
        reencode_h264(tmp_3, f3)

        # ── Step 7.9: 각 타점에 히트맵 좌표 주입 ──────────────
        # minimap_renderer._player.get_full_path(side)에서
        # 미니맵 영상에 실제로 그려진 선수 위치(mx, my)를 가져온다.
        # 이 좌표가 미니맵과 완전히 동일한 소스.
        _mm_pts  = hg["minimap_pts"]   # [TL, TR, BR, BL]
        _net_y   = float(hg["net_y_minimap"])
        _ct_x_min = float(min(_mm_pts[0][0], _mm_pts[3][0]))
        _ct_x_max = float(max(_mm_pts[1][0], _mm_pts[2][0]))
        _ct_y_min = float(min(_mm_pts[0][1], _mm_pts[1][1]))
        _ct_y_max = float(max(_mm_pts[2][1], _mm_pts[3][1]))
        _ct_w = max(_ct_x_max - _ct_x_min, 1.0)
        _ct_h = max(_ct_y_max - _ct_y_min, 1.0)
        _net_pct = (_net_y - _ct_y_min) / _ct_h * 100

        # 선수별 full_path: {frame_idx: (mx, my)} 로 변환
        _player_pos: dict = {}
        for _side in ("top", "bottom"):
            _fp = minimap_renderer._player.get_full_path(_side)
            for (_fi, _mx, _my) in _fp:
                if _side not in _player_pos:
                    _player_pos[_side] = {}
                _player_pos[_side][_fi] = (_mx, _my)

        for ev in hit_events:
            heatmap_set = False

            # ── 우선: minimap_renderer의 실제 선수 위치 사용 ──
            side_path = _player_pos.get(ev.owner, {})
            for delta in [0, 1, -1, 2, -2, 3, -3]:
                fi = ev.frame + delta
                if fi not in side_path:
                    continue
                mx, my = side_path[fi]
                if mx <= 0 and my <= 0:
                    continue
                # 코트 내 0~100 정규화
                hm_x = (float(mx) - _ct_x_min) / _ct_w * 100
                hm_y = (float(my) - _ct_y_min) / _ct_h * 100
                hm_x = max(2.0, min(98.0, hm_x))
                # top/bottom 절반 강제
                if ev.owner == "top":
                    hm_y = max(2.0, min(_net_pct - 2, hm_y))
                else:
                    hm_y = max(_net_pct + 2, min(98.0, hm_y))
                ev.minimap_x = round(hm_x, 2)
                ev.minimap_y = round(hm_y, 2)
                heatmap_set = True
                break

            # ── 폴백: 코트 중앙 ──
            if not heatmap_set:
                ev.minimap_x = 50.0
                ev.minimap_y = (_net_pct / 2) if ev.owner == "top" else (_net_pct + (100 - _net_pct) / 2)

        # ── Step 8: JSON 생성 ─────────────────────────────────
        api_data  = to_api_json(hit_events, fps)
        api_data["net_fault_events"]  = [e.to_dict() for e in net_fault_events]
        api_data["coordinate_mode"]   = hg["coordinate_mode"]
        api_data["rallyResults"]       = rally_results
        api_data["playerMetrics"]      = player_metrics
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
        pose_model_conf: float | None = None,
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
            pose_model_conf : YOLO pose detection confidence.

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
                frame,
                verbose=False,
                conf=pose_model_conf or POSE_CONFIG["confidence"],
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
