"""
스코어 측정 서비스
- 셔틀콕 낙하지점 in/out 판정 기반 점수 계산
- 배드민턴 랠리 규칙 기반 점수 추론
- 타임라인 이벤트 자동 생성
"""
from services.court_service import CourtDetector, detect_court_yolo


def judge_rallies(stroke_results: list, timestamps: list, landing_points: list,
                  court_detector: CourtDetector = None) -> list:
    """
    각 랠리의 셔틀콕 낙하지점을 in/out 판정한다.

    Args:
        stroke_results: [{"label": str, "confidence": float}, ...]
        timestamps: 각 스트로크의 발생 시각(초)
        landing_points: [{"x": float, "y": float}, ...] 셔틀콕 낙하 좌표
        court_detector: CourtDetector 인스턴스 (None이면 기본 코트 사용)

    Returns:
        [{"stroke": str, "timestamp": float, "landing": dict,
          "inout": "IN"/"OUT", "margin": float, "confidence": float}, ...]
    """
    if court_detector is None:
        court_detector = CourtDetector(mode="singles")

    rally_results = []

    for i in range(min(len(stroke_results), len(landing_points))):
        landing = landing_points[i]
        judgment = court_detector.judge_landing(landing["x"], landing["y"])

        rally_results.append({
            "stroke": stroke_results[i]["label"],
            "timestamp": timestamps[i] if i < len(timestamps) else 0,
            "landing": judgment["landing"],
            "inout": judgment["result"],
            "margin": judgment["margin"],
            "confidence": stroke_results[i]["confidence"],
        })

    return rally_results


def calculate_score(rally_results: list) -> dict:
    """
    랠리 in/out 판정 결과를 기반으로 점수를 계산한다.

    배드민턴 점수 규칙:
    - 상대 코트에 IN → 내 득점
    - 내가 친 공이 OUT → 상대 득점
    - 서브권과 무관하게 매 랠리 득점 (랠리 포인트제)
    - 21점 선취 시 세트 승리

    Args:
        rally_results: judge_rallies()의 반환값

    Returns:
        {"myScore": int, "opponentScore": int, "matchOutcome": str,
         "totalStrokeCount": int, "matchTime": str,
         "rallyDetails": list}
    """
    my_score = 0
    opponent_score = 0
    rally_details = []

    for rally in rally_results:
        # 높은 신뢰도 + IN = 내 득점 (상대 코트에 정확히 떨어짐)
        # 높은 신뢰도 + OUT = 상대 득점 (내가 친 공이 나감)
        # 낮은 신뢰도 = 판정 불확실, OUT 쪽으로 처리
        if rally["inout"] == "IN" and rally["confidence"] > 50:
            my_score += 1
            point_winner = "MY"
        else:
            opponent_score += 1
            point_winner = "OPP"

        rally_details.append({
            **rally,
            "pointWinner": point_winner,
            "scoreAfter": f"{min(my_score, 21)}:{min(opponent_score, 21)}",
        })

    # 21점 제한
    my_score = min(my_score, 21)
    opponent_score = min(opponent_score, 21)

    # 승패 판정
    if my_score > opponent_score:
        outcome = "WIN"
    elif my_score < opponent_score:
        outcome = "LOSE"
    else:
        outcome = "DRAW"

    # 경기 시간
    if rally_results:
        total_seconds = int(max(r["timestamp"] for r in rally_results))
        match_time = f"{total_seconds // 60}:{total_seconds % 60:02d}"
    else:
        match_time = "0:00"

    return {
        "myScore": my_score,
        "opponentScore": opponent_score,
        "matchOutcome": outcome,
        "totalStrokeCount": len(rally_results),
        "matchTime": match_time,
        "rallyDetails": rally_details,
    }


def calculate_score_simple(stroke_results: list, timestamps: list) -> dict:
    """
    in/out 판정 없이 스트로크 분류 결과만으로 점수를 계산한다.
    (landing_points가 없을 때 폴백용)

    Args:
        stroke_results: [{"label": str, "confidence": float}, ...]
        timestamps: 각 스트로크의 발생 시각(초)

    Returns:
        calculate_score()와 동일한 형식
    """
    my_score = 0
    opponent_score = 0

    for result in stroke_results:
        if result["confidence"] > 70:
            my_score += 1
        else:
            opponent_score += 1

    my_score = min(my_score, 21)
    opponent_score = min(opponent_score, 21)

    if my_score > opponent_score:
        outcome = "WIN"
    elif my_score < opponent_score:
        outcome = "LOSE"
    else:
        outcome = "DRAW"

    if timestamps:
        total_seconds = int(max(timestamps))
        match_time = f"{total_seconds // 60}:{total_seconds % 60:02d}"
    else:
        match_time = "0:00"

    return {
        "myScore": my_score,
        "opponentScore": opponent_score,
        "matchOutcome": outcome,
        "totalStrokeCount": len(stroke_results),
        "matchTime": match_time,
        "rallyDetails": [],
    }


def generate_timeline_events(rally_results: list) -> list:
    """
    랠리 판정 결과를 타임라인 이벤트로 변환한다.

    Args:
        rally_results: calculate_score()["rallyDetails"]

    Returns:
        [{"timestamp": int, "displayTime": str, "eventType": str,
          "eventTitle": str, "eventDescription": str, "eventScore": str}, ...]
    """
    events = []

    # 경기 시작
    events.append({
        "timestamp": 0,
        "displayTime": "0:00",
        "eventType": "GAME_START",
        "eventTitle": "경기 시작",
        "eventDescription": "분석 시작",
        "eventScore": "0:0",
    })

    for rally in rally_results:
        ts_int = int(rally["timestamp"])
        display_time = f"{ts_int // 60}:{ts_int % 60:02d}"

        if rally.get("pointWinner") == "MY":
            event_type = rally["stroke"].upper()
            title = f"{rally['stroke']} 득점"
            desc = f"{rally['stroke']} ({rally['inout']}, 신뢰도 {rally['confidence']}%)"
        else:
            event_type = "CONCEDE"
            title = "실점"
            if rally["inout"] == "OUT":
                desc = f"{rally['stroke']} OUT (마진 {abs(rally['margin']):.3f})"
            else:
                desc = f"상대 득점 (분석 신뢰도 {rally['confidence']}%)"

        events.append({
            "timestamp": ts_int,
            "displayTime": display_time,
            "eventType": event_type,
            "eventTitle": title,
            "eventDescription": desc,
            "eventScore": rally.get("scoreAfter", ""),
        })

    return events
