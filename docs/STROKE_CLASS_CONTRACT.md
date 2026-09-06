# Stroke classifier contract

분석 모드와 실제 스트로크 출력 스키마는 서로 다른 정보입니다. `mode`는 TrackNet batch,
pose threshold, near-miss rescue 및 feature classifier 파일을 선택하고, 출력 라벨은 선택된
`.pkl`의 `label_scheme`, `class_names`, `classes_` 메타데이터가 결정합니다.

2026-09-06 현재 모델 아티팩트와 운영 로그에서 확인한 계약은 다음과 같습니다.

| mode | feature artifact | scheme | 실제 학습된 출력 라벨 |
|---|---|---|---|
| `pro` | `feature_classifier_pro.pkl` | `9class` | Serve, Lob, Smash, Drop, Drive, Clear |
| `amateur` | `feature_classifier_amateur.pkl` | `4class_amateur` | Serve, Smash, Clear, Drive |

프로 아티팩트는 9칸 확률 스키마를 사용하지만 `classes_=[0,2,3,4,5,7]`만 학습되어 실제로는
6개 라벨만 출력할 수 있습니다. 이는 코드에 별도로 정의된 목표 `6class_pro`
(Serve, Smash, Clear, Drive, Drop, Net)와 같지 않습니다. 현재 모델에는 Lob가 있고 Net은 없습니다.
따라서 프로를 `6class_pro`라고 표시하려면 라벨을 사후에 이름만 바꾸는 것으로는 부족하며,
canonical mapping을 적용한 데이터로 모델을 다시 학습하고 대표 영상 회귀 검증을 해야 합니다.

분석 로그의 `[StrokeContract]`는 매 영상마다 mode, 모델, scheme, 실제 학습 라벨 수와 목록,
ViT fallback 스키마를 출력합니다. AI 완료 콜백은 `analysisMode`와
`strokeClassSchemes`를 백엔드로 보내 DB 저장 모드와 비교할 수 있게 합니다.
