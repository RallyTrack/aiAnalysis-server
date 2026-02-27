## 프로젝트 구조
```
aiAnalysis-server/
├── main.py                     # 엔트리포인트 (라우터 등록만)
├── config/
│   └── settings.py             # 설정값 (경로, 하이퍼파라미터)
├── routers/
│   └── analyze.py              # POST /analyze + 분석 파이프라인
├── services/
│   ├── video_service.py        # S3 영상 다운로드/삭제
│   ├── pose_service.py         # MediaPipe 관절 추출 + 임팩트 감지
│   ├── stroke_service.py       # MLP + ST-GCN 스트로크 분류
│   ├── heatmap_service.py      # 히트맵 데이터 생성
│   ├── score_service.py        # 점수 계산 + 타임라인 생성
│   └── feedback_service.py     # AI 코칭 피드백 + 능력치 생성
├── models/
│   ├── mlp_model.py            # UltraStrokeClassifier (엔진 구축 PDF)
│   └── stgcn_model.py          # BadmintonSTGCN (분류 모델 PDF)
├── weights/                    # .pth 가중치 파일 (git 제외)
├── requirements.txt
└── Dockerfile
```