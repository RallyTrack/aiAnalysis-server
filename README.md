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
-----------------------------------------------------------------------------


🏸 RallyTrack AI Analysis Engine

✨ 주요 기능
Shuttlecock Tracking: TrackNetV3 모델을 사용하여 고속으로 이동하는 셔틀콕의 좌표를 정밀하게 추적함.

Pose Estimation: YOLOv8-pose를 활용해 선수의 스켈레톤을 추출하고 코트 위 실시간 위치를 파악함.

Ironclad Physics Hits Detection: 단순 좌표 변화가 아닌 물리적 가속도와 방향 전환 분석을 통해 실제 타구 시점을 감지함. (영상 시작 1초 이내 초반 타점 포함)

Smart Mini-map & Heatmap: 분석된 데이터를 바탕으로 코트 미니맵에 타구 순서와 선수의 이동 동선을 시각화함.

🛠 기술 스택
Deep Learning: TrackNetV3, YOLOv8-pose (Ultralytics)

Computer Vision: OpenCV

Data Analysis: Pandas, NumPy, SciPy (Peak Detection)

Environment: Google Colab, FFmpeg

📊 분석 결과 (Output 예시)
분석이 완료되면 다음과 같은 시각화 데이터가 포함된 영상을 생성

메인 화면: 셔틀콕 추적 및 선수 포즈 렌더링

우측 상단: 타점 번호가 마킹된 코트 히트맵 (선수별 색상 구분)

하단 영역: 원근 변환(Homography)이 적용된 프로 코트 미니맵 및 궤적 잔상

⚙️ 실행 순서
환경 세팅: TrackNetV3 소스 복제 및 가중치(Weights) 다운로드

영상 업로드: 분석할 배드민턴 경기 영상 (.mp4) 선택

데이터 추출: 셔틀콕 좌표 데이터 및 타점 분석 실행

렌더링: 미니맵과 히트맵이 합성된 최종 결과 영상 생성 및 다운로드
