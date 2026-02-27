"""
MLP 기반 스트로크 분류 모델 (엔진 구축 단계)
- 10차원 물리 피처를 입력받아 4가지 스트로크(Smash, Clear, Drop, Drive)를 분류
- CoachAI 데이터셋 기반 학습, 90%+ 정확도
"""
import torch.nn as nn


class UltraStrokeClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # 입력층: 10개 물리 피처 → 1024 노드
            nn.Linear(10, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),

            # 은닉층 1
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            # 은닉층 2
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            # 출력층: 4가지 스트로크 확률
            nn.Linear(256, 4)
        )

    def forward(self, x):
        return self.net(x)
