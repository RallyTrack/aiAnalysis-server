"""ViT BallType 스트로크 분류기 — wish44165 9클래스 사전학습.

입력 : 타점 프레임 (원본 BGR) — hit_frame + 1 오프셋 자동 적용
출력 : 9클래스 스트로크 분류

가중치 배치 (5-fold):
    weights/stroke/fold{1..5}_BallType_ViT-B_16_checkpoint.bin

가중치 출처:
    wish44165/A-New-Perspective-for-Shuttlecock-Hitting-Event-Detection
    → assets/weights/ViT/BallType/output/fold{N}_BallType_ViT-B_16_checkpoint.bin

가중치 없으면 초기화 단계에서 FileNotFoundError 발생 → pipeline_service.py 에서
    예외를 잡아 분류기를 None으로 처리하고 파이프라인을 계속 진행한다.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from analysis.config import PATHS

VIT_DIR = Path(__file__).resolve().parents[1] / "third_party" / "vit_pytorch"
WEIGHTS_DIR = Path(PATHS["stroke_weights_dir"])

if str(VIT_DIR) not in sys.path:
    sys.path.insert(0, str(VIT_DIR))

from models.modeling import VisionTransformer, CONFIGS  # noqa: E402

CLASS_NAMES = [
    "Serve",    # label 1
    "Defense",  # label 2
    "Lob",      # label 3
    "Smash",    # label 4
    "Drop",     # label 5
    "Drive",    # label 6
    "Net",      # label 7
    "Clear",    # label 8
    "Push",     # label 9
]

_DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


@dataclass
class StrokeResult:
    label:      int
    class_name: str
    confidence: float
    probs:      np.ndarray

    def __repr__(self) -> str:
        return f"StrokeResult(label={self.label} '{self.class_name}', conf={self.confidence:.3f})"


class StrokeClassifier:
    def __init__(
        self,
        fold: int = 1,
        model_type: str = "ViT-B_16",
        img_size: int = 480,
        num_classes: int = 9,
        device: str = _DEVICE,
    ):
        ckpt = WEIGHTS_DIR / f"fold{fold}_BallType_{model_type}_checkpoint.bin"
        if not ckpt.exists():
            raise FileNotFoundError(f"스트로크 분류 가중치 없음: {ckpt}")

        self.device = device
        config = CONFIGS[model_type]
        self.model = VisionTransformer(config, img_size=img_size, zero_head=False, num_classes=num_classes)
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.model.to(device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    @torch.no_grad()
    def classify(self, bgr_image: np.ndarray) -> StrokeResult:
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        x = self.transform(rgb).unsqueeze(0).to(self.device)
        logits, _ = self.model(x)
        logits = logits.squeeze(0)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        idx = int(np.argmax(probs))
        return StrokeResult(label=idx + 1, class_name=CLASS_NAMES[idx], confidence=float(probs[idx]), probs=probs)

    def classify_at(self, video_path: str | Path, hit_frame: int, frame_offset: int = 1) -> StrokeResult:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"영상을 열 수 없음: {video_path}")
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, hit_frame + frame_offset)
            ret, frame = cap.read()
            if not ret:
                raise IOError(f"frame {hit_frame + frame_offset} 읽기 실패")
        finally:
            cap.release()
        return self.classify(frame)


class StrokeClassifierEnsemble:
    """5-fold ensemble — 가중치 파일이 있는 fold만 로드."""

    def __init__(self, folds: list[int] | None = None, **kwargs):
        if folds is None:
            folds = [1, 2, 3, 4, 5]
        self.classifiers: list[StrokeClassifier] = []
        for f in folds:
            try:
                self.classifiers.append(StrokeClassifier(fold=f, **kwargs))
            except FileNotFoundError as e:
                print(f"  [StrokeEnsemble] fold{f} 건너뜀: {e}")
        if not self.classifiers:
            raise RuntimeError("로드된 fold가 없습니다. weights/stroke/ 에 가중치를 배치하세요.")
        print(f"  [StrokeEnsemble] {len(self.classifiers)}-fold 로드 완료 (device={self.classifiers[0].device})")

    def classify(self, bgr_image: np.ndarray) -> StrokeResult:
        probs_list = [clf.classify(bgr_image).probs for clf in self.classifiers]
        avg = np.mean(probs_list, axis=0)
        idx = int(np.argmax(avg))
        return StrokeResult(label=idx + 1, class_name=CLASS_NAMES[idx], confidence=float(avg[idx]), probs=avg)

    def classify_at(self, video_path: str | Path, hit_frame: int, frame_offset: int = 1) -> StrokeResult:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"영상을 열 수 없음: {video_path}")
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, hit_frame + frame_offset)
            ret, frame = cap.read()
            if not ret:
                raise IOError(f"frame {hit_frame + frame_offset} 읽기 실패")
        finally:
            cap.release()
        return self.classify(frame)
