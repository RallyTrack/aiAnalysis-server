"""Feature-based stroke classifier (sklearn).

설계
----
- 작은 데이터 + 라벨 노이즈 환경을 가정해 **CPU 친화 sklearn 모델** 만 묶는다.
- 입력은 ``data/features/stroke_features.parquet`` 의 numeric feature 컬럼.
- 9-class canonical 라벨을 학습. 추론 시 9-class 출력을 받은 뒤 호출자가
  ``src.labels.schema.map_9_to_6_pro / map_9_to_5_amateur`` 로 매핑한다.

구성 모델
---------
- ``logreg``: ``LogisticRegression(class_weight="balanced", max_iter=2000)`` — 가장 안정,
  feature scale 에 민감하므로 StandardScaler 와 함께 사용.
- ``rf``: ``RandomForestClassifier`` — 비선형, NaN 처리 못함 (preprocess 필요).
- ``hgb``: ``HistGradientBoostingClassifier`` — **NaN 자체 지원** + 빠름. 1순위 권장.
- ``extra``: ``ExtraTreesClassifier`` — RF 변형, ensemble 후보.

전처리
------
- HGB 외 모델은 NaN → median imputation + ``StandardScaler``.
- HGB 는 NaN 자체 처리, scaler 불필요 (tree-based).

추론 인터페이스
-----------------
``predict_with_meta(features_df)`` 가 dict 반환:
  {
    'label_idx': int (9-class 0-indexed),
    'label_name': str,
    'confidence': float,
    'probs': np.ndarray (9,),
    'top2_idx': int,
    'top2_name': str,
    'top2_conf': float,
    'is_uncertain': bool (top1 - top2 < margin),
  }
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from analysis.stroke_labels import CLASS_NAMES_9


# ──────────────────────────────────────────────────────────────────────
# Model registry
# ──────────────────────────────────────────────────────────────────────
def make_model(name: str, seed: int = 42) -> Pipeline:
    """이름에 해당하는 sklearn pipeline 생성.

    Args:
        name: ``logreg`` | ``rf`` | ``hgb`` | ``extra``.
        seed: random_state.
    """
    if name == "logreg":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                C=1.0,
                solver="lbfgs",
                random_state=seed,
                n_jobs=-1,
            )),
        ])
    if name == "rf":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            )),
        ])
    if name == "extra":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", ExtraTreesClassifier(
                n_estimators=400,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            )),
        ])
    if name == "hgb":
        # HGB 는 NaN 자체 처리, scaler/imputer 불필요.
        return Pipeline([
            ("clf", HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.05,
                max_depth=None,
                class_weight="balanced",
                random_state=seed,
            )),
        ])
    raise ValueError(f"unknown model name: {name!r}")


MODEL_CHOICES: list[str] = ["logreg", "rf", "extra", "hgb"]


# ──────────────────────────────────────────────────────────────────────
# Trained artifact wrapper — 추론 + 저장/로드
# ──────────────────────────────────────────────────────────────────────
@dataclass
class TrainedStrokeModel:
    """학습된 분류기 + 메타. pickle 로 저장/로드 가능.

    Pkl 이 자기 schema (``class_names``) 를 들고 다님 → 추론 코드는 모델이
    알려주는 클래스 이름으로 출력. Pro (9-class) / amateur (4-class) 등 다양한
    schema 를 같은 추론 파이프라인에서 다룰 수 있게 함.

    backwards compat: 옛 pkl 에 ``class_names`` 가 없으면 ``CLASS_NAMES_9`` 로
    fallback (label_scheme=='9class' 가정).
    """

    pipeline: Pipeline                   # sklearn pipeline
    feature_columns: list[str]           # 학습 시 사용한 컬럼 순서
    classes_: np.ndarray                 # 라벨 인덱스 배열 (class_names 기준 부분집합)
    model_name: str
    class_names: list[str] = None        # 인덱스→이름 매핑. None 이면 CLASS_NAMES_9.
    label_scheme: str = "9class"         # 9class | 6class_pro | 5class_amateur | 4class_amateur
    metrics: dict = None                 # macro_f1, balanced_acc 등

    def __post_init__(self):
        if self.class_names is None:
            self.class_names = list(CLASS_NAMES_9)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "pipeline": self.pipeline,
                "feature_columns": self.feature_columns,
                "classes_": self.classes_,
                "model_name": self.model_name,
                "class_names": list(self.class_names),
                "label_scheme": self.label_scheme,
                "metrics": self.metrics or {},
            }, f)

    @classmethod
    def load(cls, path: Path) -> "TrainedStrokeModel":
        with open(path, "rb") as f:
            d = pickle.load(f)
        return cls(
            pipeline=d["pipeline"],
            feature_columns=d["feature_columns"],
            classes_=d["classes_"],
            model_name=d["model_name"],
            # 옛 pkl 호환: class_names 없으면 CLASS_NAMES_9 default
            class_names=d.get("class_names") or list(CLASS_NAMES_9),
            label_scheme=d.get("label_scheme", "9class"),
            metrics=d.get("metrics", {}),
        )

    def predict_proba(self, features_df: pd.DataFrame) -> np.ndarray:
        """class_names 길이만큼 probs 반환. classes_ 부분집합은 zero-padding.

        예) pro (9-class) 학습 시 Defense/Net 데이터 0개여서 classes_=[0,2,3,4,5,7,8]
            → 길이 9 padding, idx 1/6 은 0.0.
        예) amateur (4-class) 학습 → classes_=[0,1,2,3], 패딩 없음.
        """
        X = features_df[self.feature_columns].to_numpy(dtype=np.float64)
        raw = self.pipeline.predict_proba(X)
        n_classes = len(self.class_names)
        out = np.zeros((len(features_df), n_classes), dtype=np.float64)
        for j, cls_idx in enumerate(self.classes_):
            out[:, int(cls_idx)] = raw[:, j]
        return out

    # backwards compat alias: 옛 코드가 부르던 이름.
    def predict_proba_9(self, features_df: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(features_df)

    def predict_with_meta(
        self,
        features_df: pd.DataFrame,
        uncertainty_margin: float = 0.10,
    ) -> list[dict]:
        """row 별 풍부한 예측 결과 dict 리스트.

        라벨 이름은 ``self.class_names`` 기준 (pro / amateur 자동 분기).
        """
        probs = self.predict_proba(features_df)
        names = self.class_names
        results = []
        for p in probs:
            order = np.argsort(p)[::-1]
            top1, top2 = int(order[0]), int(order[1])
            results.append({
                "label_idx": top1,
                "label_name": names[top1],
                "confidence": float(p[top1]),
                "probs": p,
                "top2_idx": top2,
                "top2_name": names[top2],
                "top2_conf": float(p[top2]),
                "is_uncertain": bool(p[top1] - p[top2] < uncertainty_margin),
            })
        return results


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    """metadata/label 컬럼 제외한 numeric feature 컬럼만 선택.

    feature 컬럼은 ``meta_``, ``traj_``, ``court_``, ``pose_``, ``vit_`` 접두로 시작.
    """
    prefixes = ("meta_", "traj_", "court_", "pose_", "vit_")
    return [c for c in df.columns if c.startswith(prefixes)]


def filter_trainable(df: pd.DataFrame, label_col: str = "label_9class") -> pd.DataFrame:
    """학습 가능한 row 만 남기기.

    제외 조건:
      - label_9class 가 ``Defense`` (모호 상태 라벨)
      - label_9class 가 ``Net`` 인데 데이터 자체가 거의 없으면 학습 안정성 위해 제외
        (호출자 판단; 본 함수는 우선 Defense 만 제외)
    """
    return df[df[label_col] != "Defense"].reset_index(drop=True)
