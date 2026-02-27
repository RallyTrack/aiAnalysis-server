"""
스코어 측정 서비스
- 스트로크 분류 결과를 기반으로 점수 및 타임라인 이벤트를 생성
- TODO: 실제 in/out 판정 로직, 스코어보드 OCR 추가
"""


def calculate_score(stroke_results: list, timestamps: list) -> dict:
    """
    스트로크 분류 결과를 기반으로 경기 점수를 계산한다.

    현재는 스트로크 수 기반 더미 점수 로직.
    추후 in/out 판정, 스코어보드 OCR로 교체 예정.

    Args:
        stroke_results: [{"label": str, "confidence": float}, ...]
        timestamps: 각 스트로크의 발생 시각(초)

    Returns:
        {"myScore": int, "opponentScore": int, "matchOutcome": str,
         "totalStrokeCount": int, "matchTime": str}
    """
    total_strokes = len(stroke_results)

    # TODO: 실제 점수 판정 로직으로 교체
    # 현재는 스트로크 수 기반 임시 계산
    my_score = 0
    opponent_score = 0

    for result in stroke_results:
        if result["confidence"] > 70:
            my_score += 1
        else:
            opponent_score += 1

    # 최대 21점 제한
    my_score = min(my_score, 21)
    opponent_score = min(opponent_score, 21)

    if my_score > opponent_score:
        outcome = "WIN"
    elif my_score < opponent_score:
        outcome = "LOSE"
    else:
        outcome = "DRAW"

    # 경기 시간 계산
    if timestamps:
        total_seconds = int(max(timestamps))
        match_time = f"{total_seconds // 60}:{total_seconds % 60:02d}"
    else:
        match_time = "0:00"

    return {
        "myScore": my_score,
        "opponentScore": opponent_score,
        "matchOutcome": outcome,
        "totalStrokeCount": total_strokes,
        "matchTime": match_time,
    }


def generate_timeline_events(stroke_results: list, timestamps: list) -> list:
    """
    스트로크 분류 결과를 타임라인 이벤트로 변환한다.

    Args:
        stroke_results: [{"label": str, "confidence": float}, ...]
        timestamps: 각 스트로크의 발생 시각(초)

    Returns:
        [{"timestamp": int, "displayTime": str, "eventType": str,
          "eventTitle": str, "eventDescription": str, "eventScore": str}, ...]
    """
    events = []
    my_score = 0
    opp_score = 0

    # 경기 시작 이벤트
    events.append({
        "timestamp": 0,
        "displayTime": "0:00",
        "eventType": "GAME_START",
        "eventTitle": "경기 시작",
        "eventDescription": "분석 시작",
        "eventScore": "0:0",
    })

    for i, (result, ts) in enumerate(zip(stroke_results, timestamps)):
        ts_int = int(ts)
        display_time = f"{ts_int // 60}:{ts_int % 60:02d}"

        # 높은 신뢰도 스트로크는 득점으로 처리 (임시 로직)
        if result["confidence"] > 70:
            my_score += 1
            event_type = result["label"].upper()
            description = f"{result['label']} (신뢰도 {result['confidence']}%)"
        else:
            opp_score += 1
            event_type = "CONCEDE"
            description = f"상대 득점 (분석 신뢰도 낮음: {result['confidence']}%)"

        events.append({
            "timestamp": ts_int,
            "displayTime": display_time,
            "eventType": event_type,
            "eventTitle": result["label"],
            "eventDescription": description,
            "eventScore": f"{min(my_score, 21)}:{min(opp_score, 21)}",
        })

    return events
