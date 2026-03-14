🏸 RallyTrack: AI Badminton Analysis Server

딥러닝 기반 배드민턴 경기 영상 분석 및 데이터 시각화 자동화 서버

RallyTrack은 배드민턴 경기 영상에서 셔틀콕의 궤적, 선수의 움직임, 타격 지점(Impact)을 자동으로 분석하여 데이터 기반의 분석 리포트를 생성하는 시스템입니다.

📂 사전 준비 및 파일 배치 (Prerequisites)

이 프로젝트는 대용량 영상 및 AI 모델 가중치 파일을 포함하지 않습니다. 실행 전 아래 지침에 따라 폴더를 생성하고 파일을 배치해야 합니다.

1. 필수 폴더 생성

프로젝트 최상위 경로(Root)에 다음 폴더들을 직접 생성해 주세요.

inputVideo/: 분석할 원본 영상(.mp4, .avi 등)을 넣는 폴더입니다.

prediction/: 분석 과정에서 생성되는 중간 데이터(CSV 좌표 등)가 저장되는 폴더입니다.

result/: 모든 분석과 합성이 완료된 최종 리포트 영상이 저장되는 폴더입니다.

2. AI 모델 가중치 파일 배치
모델 실행에 필요한 가중치 파일을 아래 경로에 맞게 배치해야 합니다.

TrackNetV3: TrackNetV3/ckpts/ 폴더 내에 학습된 가중치 파일(예: TrackNetV3.pt)을 배치합니다.

YOLOv8-pose: 실행 시 자동 다운로드되거나, 최상위 경로에 yolov8n-pose.pt 파일을 배치합니다.

🏗️ 프로젝트 구조 (Directory Structure)
Plaintext
aiAnalysis-server/
├── TrackNetV3/            # 셔틀콕 추적 엔진
│   ├── ckpts/             # <-- [필수] TrackNetV3 가중치 파일 배치
│   ├── predict.py         # 셔틀콕 좌표 추출 실행 스크립트
│   └── dataset.py         # 메모리 최적화 패치가 적용된 데이터 로더
├── inputVideo/            # <-- [필수] 분석할 원본 영상 배치
├── prediction/            # <-- [필수] 생성된 좌표 데이터(CSV) 저장
├── result/                # <-- [필수] 최종 분석 리포트 영상 저장
├── ShuttleMap.ipynb       # 호모그래피 변환 및 미니맵 시각화 로직
├── HIt2.ipynb             # 물리 엔진 기반 타점 탐지 및 임팩트 효과 합성
├── .gitignore             # 대용량 파일 업로드 방지 설정
└── README.md

🚀 주요 기능 (Key Features)

Shuttlecock Tracking (TrackNetV3): 고속 이동하는 셔틀콕의 위치를 실시간으로 탐지 및 좌표화합니다.

Pose Estimation (YOLOv8-pose): 선수들의 17개 관절 스켈레톤을 추출하여 역동적인 움직임을 시각화합니다.

Court Mini-map: 호모그래피 변환을 통해 영상 속 선수의 위치를 2D 평면 미니맵에 실시간 투영합니다.

Impact Detection: 셔틀콕의 속도와 방향 변화를 분석하여 타구 시점(Impact)을 정밀 탐지하고 시각 효과를 적용합니다.

📈 기술적 최적화 (Optimization)
메모리 효율화 (Frame Sampling): 1분 이상의 영상 분석 시 발생하는 RAM 부족 문제를 해결하기 위해, 배경 연산 시 10프레임 간격 샘플링 기법을 적용하여 메모리 점유율을 90% 이상 절감했습니다.

코덱 호환성: static-ffmpeg를 연동하여 윈도우 환경에서도 별도 설치 없이 모든 재생기에서 호환되는 H.264 영상 인코딩을 지원합니다.

💻 시작하기 (Usage)

1. 가상환경 및 라이브러리 설치
```
Bash
python -m venv .venv
source .venv/Scripts/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install static-ffmpeg
```
2. 실행 프로세스

inputVideo/에 영상을 넣습니다.

TrackNetV3/predict.py를 실행하여 셔틀콕 좌표를 추출합니다.

ShuttleMap.ipynb 또는 HIt2.ipynb를 실행하여 최종 분석 리포트 영상을 생성합니다.

⚠️ 주의사항

.gitignore 설정으로 인해 inputVideo/, prediction/, result/ 폴더 내의 파일과 가중치 파일은 Git에 업로드되지 않습니다. 협업 시 해당 파일들은 별도 공유가 필요합니다.

윈도우 환경에서는 num_workers=0 설정을 권장합니다.
