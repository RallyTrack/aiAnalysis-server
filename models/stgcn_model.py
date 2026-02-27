"""
ST-GCN 기반 영상 스트로크 분류 모델 (분류 모델 구축 단계)
- MediaPipe에서 추출한 관절 시퀀스를 입력받아 스트로크를 분류
- GCN(공간: 관절 연결 구조) + TCN(시간: 움직임 흐름) 결합
"""
import torch.nn as nn


class BadmintonSTGCN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        # 공간 계층(GCN): 신체 부위 간 연결 구조 학습
        self.gcn = nn.Sequential(
            nn.Conv2d(3, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # 시간 계층(TCN): 시간에 따른 움직임 변화 패턴 학습
        self.tcn = nn.Sequential(
            nn.Conv2d(64, 128, (9, 1), padding=(4, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        # 최종 분류: 4개 스트로크 확률 산출
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.gcn(x)
        x = self.tcn(x)
        return self.fc(x.mean(dim=[2, 3]))  # 전역 평균 풀링
