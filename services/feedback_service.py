"""
AI 피드백 생성 서비스
- 스트로크 분석 결과를 기반으로 코칭 피드백 문장을 생성
"""
from config.settings import STROKE_LABELS


def generate_feedback(stroke_results: list, score: dict) -> str:
    """
    분석 결과를 기반으로 AI 코칭 피드백을 생성한다.

    Args:
        stroke_results: [{"label": str, "confidence": float}, ...]
        score: calculate_score()의 반환값

    Returns:
        피드백 문자열
    """
    if not stroke_results:
        return "분석할 스트로크 데이터가 부족합니다."

    # 스트로크별 횟수 집계
    stroke_counts = {label: 0 for label in STROKE_LABELS}
    for result in stroke_results:
        stroke_counts[result["label"]] += 1

    total = len(stroke_results)

    # 가장 많이 사용한 스트로크
    most_used = max(stroke_counts, key=stroke_counts.get)
    most_count = stroke_counts[most_used]

    # 가장 적게 사용한 스트로크
    least_used = min(stroke_counts, key=stroke_counts.get)

    # 평균 신뢰도
    avg_confidence = sum(r["confidence"] for r in stroke_results) / total

    # 피드백 생성
    feedback_parts = []

    feedback_parts.append(
        f"총 {total}회의 스트로크 중 {most_used}을(를) {most_count}회로 가장 많이 사용했습니다."
    )

    if avg_confidence > 80:
        feedback_parts.append("전반적으로 스트로크가 안정적이며 정확도가 높습니다.")
    elif avg_confidence > 60:
        feedback_parts.append("스트로크 정확도가 보통 수준입니다. 폼 교정을 통해 향상 가능합니다.")
    else:
        feedback_parts.append("스트로크 정확도가 낮습니다. 기본기 훈련을 추천합니다.")

    if stroke_counts.get("Smash", 0) > total * 0.3:
        feedback_parts.append("스매시 비중이 높습니다. 다양한 샷 배합으로 상대를 교란해보세요.")

    if stroke_counts.get(least_used, 0) == 0:
        feedback_parts.append(f"{least_used} 사용이 없습니다. 다양한 기술 활용을 추천합니다.")

    if score["matchOutcome"] == "WIN":
        feedback_parts.append("승리를 축하합니다! 현재 전략을 유지하며 약점을 보완해보세요.")
    else:
        feedback_parts.append("아쉬운 결과입니다. 분석 데이터를 참고하여 부족한 부분을 연습해보세요.")

    return " ".join(feedback_parts)


def generate_ability_metrics(stroke_results: list, score: dict) -> dict:
    """
    분석 결과를 기반으로 능력치 지표를 생성한다.

    Returns:
        {"smash": int, "defense": int, "speed": int,
         "stamina": int, "accuracy": int, "strategy": int}
    """
    if not stroke_results:
        return {k: 50 for k in ["smash", "defense", "speed", "stamina", "accuracy", "strategy"]}

    stroke_counts = {label: 0 for label in STROKE_LABELS}
    total_confidence = 0

    for result in stroke_results:
        stroke_counts[result["label"]] += 1
        total_confidence += result["confidence"]

    total = len(stroke_results)
    avg_conf = total_confidence / total

    # 능력치 계산 (0~100)
    smash_ratio = stroke_counts.get("Smash", 0) / max(total, 1)
    clear_ratio = stroke_counts.get("Clear", 0) / max(total, 1)
    drop_ratio = stroke_counts.get("Drop", 0) / max(total, 1)
    drive_ratio = stroke_counts.get("Drive", 0) / max(total, 1)

    return {
        "smash": min(int(smash_ratio * 200 + avg_conf * 0.3), 100),
        "defense": min(int((clear_ratio + drive_ratio) * 150 + avg_conf * 0.2), 100),
        "speed": min(int(drive_ratio * 200 + smash_ratio * 100), 100),
        "stamina": min(int(70 + total * 0.5), 100),
        "accuracy": min(int(avg_conf), 100),
        "strategy": min(int(len([s for s in stroke_counts.values() if s > 0]) * 25), 100),
    }
