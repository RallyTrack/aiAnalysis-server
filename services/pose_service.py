"""
자세 추정 서비스 (MediaPipe 기반)
- 영상에서 33개 관절 좌표를 프레임별로 추출
- 손목 속도 기반 임팩트(타구) 순간 감지
- 임팩트 시점 앞뒤 15프레임 시퀀스 추출 → ST-GCN 입력용
"""
import cv2
import numpy as np
import mediapipe as mp
from config.settings import (
    POSE_MIN_DETECTION_CONFIDENCE,
    POSE_MIN_TRACKING_CONFIDENCE,
    IMPACT_SENSITIVITY,
    IMPACT_COOL_DOWN,
)


def detect_impacts(video_path: str) -> tuple:
    """
    영상에서 타구 임팩트 순간을 감지한다.

    Args:
        video_path: 로컬 영상 파일 경로

    Returns:
        (sequences, timestamps)
        - sequences: (N, 30, 33, 3) 각 임팩트 시점 ±15프레임의 관절 시퀀스
        - timestamps: 각 임팩트의 발생 시각(초)
    """
    mp_pose = mp.solutions.pose
    pose_engine = mp_pose.Pose(
        min_detection_confidence=POSE_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=POSE_MIN_TRACKING_CONFIDENCE,
    )

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    velocities = []
    frames_data = []

    print(f"[자세 추정] 영상 스캔 중... (FPS: {fps:.2f})")

    prev_wrist = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = pose_engine.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:
            # 양쪽 손목 좌표 평균으로 스윙 속도 계산
            w_r = results.pose_landmarks.landmark[16]
            w_l = results.pose_landmarks.landmark[15]
            curr_pos = np.array([w_r.x + w_l.x, w_r.y + w_l.y]) / 2

            if prev_wrist is not None:
                vel = np.linalg.norm(curr_pos - prev_wrist)
                velocities.append(vel)
            else:
                velocities.append(0)

            frames_data.append(
                [[l.x, l.y, l.z] for l in results.pose_landmarks.landmark]
            )
            prev_wrist = curr_pos
        else:
            velocities.append(0)
            frames_data.append(None)

    cap.release()
    pose_engine.close()

    # 속도 급증 지점(임팩트) 감지
    if len(velocities) < 31:
        print("[자세 추정] 영상이 너무 짧아 임팩트 감지 불가")
        return np.array([]), []

    threshold = np.mean(velocities) * IMPACT_SENSITIVITY
    event_indices = []

    for i in range(15, len(velocities) - 15):
        if velocities[i] > threshold and velocities[i] == max(velocities[i - 15:i + 15]):
            if not event_indices or i - event_indices[-1] > (fps * IMPACT_COOL_DOWN):
                event_indices.append(i)

    # 임팩트 시점 ±15프레임 시퀀스 추출
    sequences = []
    timestamps = []
    for idx in event_indices:
        seq = frames_data[idx - 15:idx + 15]
        if None not in seq:
            sequences.append(np.array(seq))
            timestamps.append(idx / fps)

    print(f"[자세 추정 완료] 타구 임팩트 {len(sequences)}개 감지")
    return np.array(sequences) if sequences else np.array([]), timestamps


def extract_player_positions(video_path: str) -> list:
    """
    영상에서 프레임별 선수 위치(엉덩이 중심)를 추출한다.
    히트맵 데이터 생성에 사용.

    Returns:
        [{"x": float, "y": float, "frame": int}, ...]
    """
    mp_pose = mp.solutions.pose
    pose_engine = mp_pose.Pose(
        min_detection_confidence=POSE_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=POSE_MIN_TRACKING_CONFIDENCE,
    )

    cap = cv2.VideoCapture(video_path)
    positions = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = pose_engine.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:
            # 엉덩이 중심점 (landmark 23, 24)
            hip_l = results.pose_landmarks.landmark[23]
            hip_r = results.pose_landmarks.landmark[24]
            positions.append({
                "x": (hip_l.x + hip_r.x) / 2,
                "y": (hip_l.y + hip_r.y) / 2,
                "frame": frame_idx,
            })

        frame_idx += 1

    cap.release()
    pose_engine.close()

    print(f"[위치 추출 완료] {len(positions)}개 프레임")
    return positions
