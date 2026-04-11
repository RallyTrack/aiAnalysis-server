# 🏸 RallyTrack AI Analysis Server

배드민턴 경기 영상을 분석하여 셔틀콕 궤적, 선수 동선, 타점, 네트 판정 데이터를 자동으로 추출하고 시각화하는 AI 분석 엔진입니다.

---

## 주요 기능

| 기능 | 설명 |
|---|---|
| **Shuttlecock Tracking** | TrackNetV3로 고속 셔틀콕 좌표 정밀 추적 |
| **Pose Estimation** | YOLOv8-pose 기반 선수 스켈레톤 추출 및 코트 위 위치 실시간 파악 |
| **Impact Detection** | 물리적 가속도·방향 전환 분석으로 실제 타구 시점 감지 (영상 초반 포함) |
| **Net Fault Detection** | 사용자 지정 네트 상단 좌표 기반 네트 걸림 자동 판정 |
| **Mini-map Visualization** | 타구 순서 및 선수 이동 동선을 BWF 규격 코트 위에 시각화 |
| **Skeleton Court View** | 원근 투영된 코트 위에 선수 스켈레톤과 셔틀콕 궤적을 3D 감각으로 렌더링 |

---

## 출력물

분석 완료 시 `result/` 디렉토리에 4종 파일이 생성됩니다.

```
result/
├── {name}_1_main.mp4       ← 원본 + YOLO 스켈레톤 오버레이
├── {name}_2_minimap.mp4    ← 2D Top-Down BWF 규격 미니맵 (360×600)
├── {name}_3_skeleton.mp4   ← 원근감 코트 스켈레톤 뷰 (1920×1000)
└── {name}_hits.json        ← 타점 + 네트판정 데이터 (백엔드 콜백용)
```

---

## 프로젝트 구조

```
aiAnalysis-server/
├── main.py                       # FastAPI 엔트리포인트
├── routers/
│   └── analyze.py                # POST /analyze 라우터
├── services/
│   ├── pipeline_service.py       # 메인 분석 파이프라인
│   └── video_service.py          # S3 영상 다운로드/삭제
├── analysis/
│   ├── config.py                 # 전역 설정 (색상, 경로, 파라미터)
│   ├── court.py                  # BWF 코트 기하학 / 호모그래피
│   ├── minimap.py                # 2D Top-Down 미니맵 렌더러
│   ├── skeleton_view.py          # 원근 코트 스켈레톤 뷰 렌더러
│   ├── impact.py                 # 타점 감지 (물리 기반)
│   └── net_judge.py              # 네트 판정 로직
├── tracknetv3/                   # 셔틀콕 추적 서브모듈
├── weights/                      # YOLO 가중치 (git 제외)
├── temp_videos/                  # 처리 중 임시 영상 (git 제외)
├── prediction/                   # TrackNet CSV 캐시 (git 제외)
└── result/                       # 분석 결과물 (git 제외)
```

---

## 기술 스택

- **Deep Learning**: TrackNetV3, YOLOv8-pose (Ultralytics)
- **Computer Vision**: OpenCV
- **Data Analysis**: Pandas, NumPy, SciPy
- **API Server**: FastAPI
- **Infra**: AWS S3, FFmpeg

---

## API 사용법

```http
POST /analyze
Content-Type: application/json

{
  "video_url": "s3://...",
  "user_corners": [[x,y],[x,y],[x,y],[x,y]],  // 단식 코트 4점 (좌상·우상·우하·좌하)
  "net_coords":   [[x,y],[x,y]]                // 네트 상단 좌·우 2점
}
```

`user_corners` / `net_coords` 는 영상 픽셀 좌표이며 생략 시 자동 비율로 추정합니다.

---

## 시각화 규격

### 색상 시스템

| 요소 | 색상 | HEX |
|---|---|---|
| Top 선수 (화면 상단) | Royal Blue | `#3B82F6` |
| Bottom 선수 (화면 하단) | Amber Gold | `#F59E0B` |
| 단식 유효 구역 | Forest Green | `#327330` |
| 복식 앨리 (단식 아웃) | Dark Green | `#1B401A` |
| 네트 | Blue | `MINIMAP_CONFIG["net_color"]` |

### BWF 코트 규격 (단위: m)

| 항목 | 값 |
|---|---|
| 코트 전장 | 13.40 |
| 복식 폭 | 6.10 |
| 단식 폭 | 5.18 (사이드라인 각 0.46 내측) |
| 네트 위치 | 6.70 (중앙) |
| 숏 서비스 라인 | 네트 ± 1.98 → 4.72 / 8.68 |
| 복식 롱 서비스 라인 | 엔드 ± 0.76 → 0.76 / 12.64 |
| 센터라인 | 숏 서비스 라인 → 엔드라인 (네트 구간 미포함) |

---

## 수정 이력

### feat/court-picker-pipeline

#### 타점 감지 v12 — 오탐지 제거 및 rescue 로직 개선

**근접 피크 NMS (`_score_nms`)**
- `find_peaks(distance=d)` 경계 조건으로 d 이내 두 피크가 모두 통과하던 문제 해결
- 9프레임(≈0.3s) 이내 두 피크 → 점수 높은 쪽만 유지 (ratio 조건 없음)
- 배드민턴에서 0.3초 이내 두 번 타격은 물리적으로 불가능하므로 무조건 오탐으로 처리

**서브 포물선 정점(apex) 필터**
- 서브 비행 중 자연적인 수직 방향 전환(정점)을 타격으로 오인하는 문제 해결
- 조건: 위로 이동(v_in[1] < -2) + 아래로 반전(v_out[1] > 2) + 수평 변화 없음(|Δv_x| < 4) → 스킵
- 실제 타격은 라켓이 수평+수직 모두 변경 → |Δv_x| ≥ 5+ → 보존

**IQR-억제 near-miss 후보 제외**
- IQR-억제 프레임(TrackNet ID 스위치 노이즈)을 near-miss 후보에서 완전 제거
- 기존(v11): IQR 억제 프레임을 포즈로 재검증 → ID 스위치 구간의 선수 손목이 공 위치 근처에 있으면 오탐 삽입
- 수정(v12): IQR 억제 = 노이즈 원인이므로 rescue 대상에서 제외

---

#### 네트 상단 좌표 기반 물리 로직 교정
- `net_coords`(Net-L, Net-R)를 네트 **최상단(Top edge)** 좌표로 재해석
- `SkeletonCourtRenderer`에 `net_coords` 파라미터 추가
- 셔틀콕과 **동일한 `_shifted_matrix`** 로 네트 상단을 투영 → 공과 네트의 전후/상하 관계 물리적 일치
- `pipeline_service.py`에서 렌더러 초기화 시 `net_coords` 자동 전달

#### 브랜드 색상 통일
- Top 선수: 빨강 → **Royal Blue (#3B82F6)**
- Bottom 선수: 파랑 → **Amber Gold (#F59E0B)**
- IMPACT 텍스트 / 하이라이트 효과 → Royal Blue 계열
- 네트 오버레이 색상 → Royal Blue
- `config.py` 단일 진실 소스로 모든 파일 자동 동기화

#### Two-tone Green 코트 채색
- 단식 유효 구역: **Forest Green (#327330)**
- 복식 앨리 (단식 아웃): **Dark Dark Green (#1B401A)**
- 미니맵: `cv2.rectangle` fillPoly 방식
- 스켈레톤 뷰: `cv2.fillPoly` perspective 폴리곤 방식

#### BWF 규격 라인 수정
- **센터라인**: 전장 관통 → 숏 서비스 라인↔엔드라인 2분할 (네트 구간 제외)
- **숏 서비스 라인**: 단식 폭(0.46~5.64) → **복식 전폭(0~6.10)**

#### 선수 ID 안정화
- `PlayerStabilizer`: 화면 Y좌표 기반 Top/Bottom 강제 고정
- 네트 근처 겹침 시에도 색상 스위칭 없음, 최대 3프레임 Ghost 유지
