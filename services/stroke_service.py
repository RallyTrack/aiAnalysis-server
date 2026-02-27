"""
스트로크 분류 서비스
- MLP: 좌표 기반 물리 피처 → 스트로크 분류 (보조)
- ST-GCN: 관절 시퀀스 → 스트로크 분류 (메인)
"""
import os
import torch
import numpy as np
from models.mlp_model import UltraStrokeClassifier
from models.stgcn_model import BadmintonSTGCN
from config.settings import MLP_MODEL_PATH, STGCN_MODEL_PATH, STROKE_LABELS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 캐싱 (서버 시작 시 한 번만 로드)
_mlp_model = None
_mlp_scaler = None
_stgcn_model = None


def _load_mlp_model():
    """MLP 모델과 스케일러를 로드하고 캐싱"""
    global _mlp_model, _mlp_scaler
    if _mlp_model is not None:
        return _mlp_model, _mlp_scaler

    if not os.path.exists(MLP_MODEL_PATH):
        print(f"[경고] MLP 가중치 없음: {MLP_MODEL_PATH}")
        return None, None

    checkpoint = torch.load(MLP_MODEL_PATH, map_location=device, weights_only=False)
    _mlp_model = UltraStrokeClassifier().to(device)
    _mlp_model.load_state_dict(checkpoint["model_state_dict"])
    _mlp_model.eval()
    _mlp_scaler = checkpoint["scaler"]

    print(f"[모델 로드] MLP 모델 로드 완료 ({device})")
    return _mlp_model, _mlp_scaler


def _load_stgcn_model():
    """ST-GCN 모델을 로드하고 캐싱"""
    global _stgcn_model
    if _stgcn_model is not None:
        return _stgcn_model

    if not os.path.exists(STGCN_MODEL_PATH):
        print(f"[경고] ST-GCN 가중치 없음: {STGCN_MODEL_PATH}")
        return None

    checkpoint = torch.load(STGCN_MODEL_PATH, map_location=device, weights_only=False)
    _stgcn_model = BadmintonSTGCN().to(device)
    _stgcn_model.load_state_dict(checkpoint["model_state_dict"])
    _stgcn_model.eval()

    print(f"[모델 로드] ST-GCN 모델 로드 완료 ({device})")
    return _stgcn_model


def extract_physics_features(raw_features: np.ndarray) -> np.ndarray:
    """
    좌표 데이터를 10차원 물리 피처로 변환.
    (player_x, player_y, landing_x, landing_y) → 10개 피처
    """
    px, py = raw_features[:, 0], raw_features[:, 1]
    lx, ly = raw_features[:, 2], raw_features[:, 3]

    dist = np.sqrt((lx - px) ** 2 + (ly - py) ** 2)
    dx, dy = lx - px, ly - py
    angle = np.arctan2(dy, dx)
    dist_x, dist_y = np.abs(dx), np.abs(dy)

    return np.stack([px, py, lx, ly, dist, dx, dy, angle, dist_x, dist_y], axis=1)


def classify_strokes_stgcn(sequences: np.ndarray) -> list:
    """
    ST-GCN으로 관절 시퀀스에서 스트로크를 분류한다.

    Args:
        sequences: (N, 30, 33, 3) 관절 시퀀스 배열

    Returns:
        [{"label": str, "confidence": float}, ...]
    """
    model = _load_stgcn_model()
    if model is None:
        print("[분류] ST-GCN 모델 없음, 스킵")
        return []

    with torch.no_grad():
        # (N, 30, 33, 3) → (N, 3, 30, 33) : (Batch, Channel, Time, Vertex)
        inputs = torch.FloatTensor(sequences).permute(0, 3, 1, 2).to(device)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        confidences, preds = torch.max(probs, 1)

    results = []
    for i in range(len(sequences)):
        results.append({
            "label": STROKE_LABELS[preds[i].item()],
            "confidence": round(confidences[i].item() * 100, 2),
        })

    return results


def classify_strokes_mlp(raw_features: np.ndarray) -> list:
    """
    MLP로 좌표 기반 스트로크를 분류한다 (보조 판단용).

    Args:
        raw_features: (N, 4) [player_x, player_y, landing_x, landing_y]

    Returns:
        [{"label": str, "confidence": float}, ...]
    """
    model, scaler = _load_mlp_model()
    if model is None:
        print("[분류] MLP 모델 없음, 스킵")
        return []

    physics = extract_physics_features(raw_features)
    scaled = scaler.transform(physics)

    with torch.no_grad():
        inputs = torch.FloatTensor(scaled).to(device)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        confidences, preds = torch.max(probs, 1)

    results = []
    for i in range(len(raw_features)):
        results.append({
            "label": STROKE_LABELS[preds[i].item()],
            "confidence": round(confidences[i].item() * 100, 2),
        })

    return results
