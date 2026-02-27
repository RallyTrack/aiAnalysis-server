"""
히트맵 데이터 생성 서비스
- 선수의 프레임별 위치 데이터를 기반으로 코트 히트맵 좌표 + 강도를 계산
"""
import numpy as np


def generate_heatmap(positions: list) -> list:
    """
    선수 위치 데이터를 히트맵 격자로 변환한다.

    코트를 10x10 격자로 나누고, 각 셀에 머문 빈도를 강도(intensity)로 계산.

    Args:
        positions: [{"x": float, "y": float, "frame": int}, ...]

    Returns:
        [{"x": float, "y": float, "intensity": float}, ...]
    """
    if not positions:
        return []

    grid_size = 10
    grid = np.zeros((grid_size, grid_size))

    for pos in positions:
        gx = min(int(pos["x"] * grid_size), grid_size - 1)
        gy = min(int(pos["y"] * grid_size), grid_size - 1)
        grid[gy][gx] += 1

    # 최대값으로 정규화 (0.0 ~ 1.0)
    max_val = grid.max()
    if max_val == 0:
        return []

    grid = grid / max_val

    # 강도가 0보다 큰 셀만 반환
    heatmap_data = []
    for gy in range(grid_size):
        for gx in range(grid_size):
            if grid[gy][gx] > 0:
                heatmap_data.append({
                    "x": round((gx + 0.5) / grid_size * 100, 1),
                    "y": round((gy + 0.5) / grid_size * 100, 1),
                    "intensity": round(float(grid[gy][gx]), 2),
                })

    return heatmap_data
