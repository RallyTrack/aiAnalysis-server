"""
RallyTrack - 웹 서비스 연동용 데이터 익스포터 (v1)

[역할]
  영상 분석 결과를 웹 프론트엔드 / 백엔드 API가 소비할 수 있는
  JSON 형식으로 변환하고 파일로 저장합니다.

[출력 파일 구조]
  {name}_summary.json      ← 경기 요약 (프론트 대시보드용)
  {name}_frame_data.json   ← 프레임별 상세 데이터 (프론트 코트 애니메이션용)
  {name}_hits.json         ← 타점 이벤트 목록 (기존 API 형식 유지)

[웹 연동 시나리오]
  시나리오 A — 영상 기반 렌더링 (현재):
      영상 → main.py → result/*.mp4  (영상 파일로 확인)

  시나리오 B — 웹 프론트 실시간 렌더링 (향후):
      영상 → main.py → _frame_data.json
          → 프론트(Canvas/Three.js): 프레임별 좌표를 읽어 코트/선수/셔틀콕 직접 렌더링
          → 백엔드(FastAPI 등): _summary.json / _hits.json을 REST API로 서빙

[프레임 데이터 스키마]
  각 프레임 레코드:
  {
    "frame":    int,          // 프레임 인덱스
    "time_sec": float,        // 영상 시각 (초)
    "shuttle": {
      "x": float | null,      // 미니맵 정규화 X (0~1), 감지 못 하면 null
      "y": float | null       // 미니맵 정규화 Y (0~1)
    },
    "players": [
      {
        "track_id": int,
        "side":     "top" | "bottom",
        "x": float,           // 미니맵 정규화 X (0~1)
        "y": float            // 미니맵 정규화 Y (0~1)
      }, ...
    ],
    "hit": {                  // 해당 프레임에 타점이 있을 때만 포함
      "hit_number": int,
      "player":     "pink_top" | "green_bottom"
    } | null
  }
"""

from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import MINIMAP_CONFIG, POSE_CONFIG, WEB_EXPORT_CONFIG
from .court import frame_to_normalized
from .impact import ImpactEvent, build_hit_lookup


# ─────────────────────────────────────────────────────────────
# 프레임 데이터 레코드
# ─────────────────────────────────────────────────────────────

@dataclass
class ShuttleRecord:
    x: Optional[float]   # 정규화 (0~1), 감지 안 됨 → None
    y: Optional[float]

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y}


@dataclass
class PlayerRecord:
    track_id: int
    side:     str         # "top" | "bottom"
    x:        float       # 정규화 (0~1)
    y:        float

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "side":     self.side,
            "x":        round(self.x, 4),
            "y":        round(self.y, 4),
        }


@dataclass
class FrameRecord:
    frame:    int
    time_sec: float
    shuttle:  ShuttleRecord
    players:  List[PlayerRecord] = field(default_factory=list)
    hit:      Optional[dict]     = None

    def to_dict(self) -> dict:
        d = {
            "frame":    self.frame,
            "time_sec": round(self.time_sec, 3),
            "shuttle":  self.shuttle.to_dict(),
            "players":  [p.to_dict() for p in self.players],
        }
        if self.hit is not None:
            d["hit"] = self.hit
        return d


# ─────────────────────────────────────────────────────────────
# 웹 데이터 수집기 (프레임 루프 내에서 호출)
# ─────────────────────────────────────────────────────────────

class WebDataCollector:
    """
    프레임 루프에서 호출하며 웹 전송용 데이터를 수집합니다.

    사용법:
        collector = WebDataCollector(hg, hit_events, fps)
        for frame_idx in ...:
            collector.record(frame_idx, sx, sy, keypoints_list)
        collector.save(output_dir, video_name)
    """

    def __init__(
        self,
        homographies: dict,
        hit_events:   List[ImpactEvent],
        fps:          float,
        net_y_minimap: float,
    ):
        self._hg          = homographies
        self._hit_lookup  = build_hit_lookup(hit_events)
        self._fps         = fps
        self._net_y_mm    = net_y_minimap  # 미니맵 픽셀 기준 네트 Y

        self._frames: List[FrameRecord] = []
        self._cfg = WEB_EXPORT_CONFIG

        # 미니맵 크기 (정규화 계산에 사용)
        self._mm_w  = MINIMAP_CONFIG["width"]
        self._mm_h  = MINIMAP_CONFIG["height"]
        self._pad   = MINIMAP_CONFIG["padding"]
        self._kconf = POSE_CONFIG["keypoint_conf"]

        # 셔틀콕 궤적 버퍼 (trail)
        self._trail: deque[dict] = deque(
            maxlen=self._cfg["shuttle_trail_length"]
        )

        # track_id → 누적 Y (side 결정용)
        self._player_y_acc: Dict[int, list] = {}

    # ── 퍼블릭 API ─────────────────────────────────────────

    def record(
        self,
        frame_idx:      int,
        shuttle_x:      float,
        shuttle_y:      float,
        keypoints_list: list,
    ) -> None:
        """
        한 프레임 데이터를 수집합니다.
        (main.py 프레임 루프 내에서 MinimapRenderer.render_frame()과 함께 호출)
        """
        time_sec = frame_idx / self._fps

        # ── 셔틀콕 ──────────────────────────────────────────
        if shuttle_x > 0 and shuttle_y > 0:
            nx, ny = frame_to_normalized((shuttle_x, shuttle_y),
                                          self._hg["to_normalized"])
            nx = float(np.clip(nx, 0.0, 1.0))
            ny = float(np.clip(ny, 0.0, 1.0))
            shuttle_rec = ShuttleRecord(
                x=round(nx, 4),
                y=round(ny, 4),
            )
            self._trail.append({"x": round(nx, 4), "y": round(ny, 4)})
        else:
            shuttle_rec = ShuttleRecord(x=None, y=None)

        # ── 선수 위치 ────────────────────────────────────────
        player_recs = self._extract_players(keypoints_list)

        # ── 타점 이벤트 ─────────────────────────────────────
        hit_dict = None
        if frame_idx in self._hit_lookup:
            ev = self._hit_lookup[frame_idx]
            hit_dict = {
                "hit_number": ev.hit_number,
                "player":     ev.player,
                "owner":      ev.owner,
            }

        self._frames.append(FrameRecord(
            frame    = frame_idx,
            time_sec = time_sec,
            shuttle  = shuttle_rec,
            players  = player_recs,
            hit      = hit_dict,
        ))

    def save(self, output_dir: str, video_name: str) -> dict:
        """
        수집된 데이터를 JSON 파일로 저장합니다.

        Returns:
            저장된 파일 경로 dict:
              {"frame_data": path, "summary": path}
        """
        os.makedirs(output_dir, exist_ok=True)
        cfg = self._cfg

        # ── frame_data.json ──────────────────────────────────
        frame_data_path = os.path.join(
            output_dir,
            f"{video_name}{cfg['frame_data_suffix']}",
        )
        frame_data = {
            "schema_version": "1.0",
            "video_name":     video_name,
            "fps":            round(self._fps, 2),
            "total_frames":   len(self._frames),
            "minimap": {
                "width":   self._mm_w,
                "height":  self._mm_h,
                "padding": self._pad,
            },
            "court": {
                "description": "BWF standard badminton court (doubles)",
                "ratios": {
                    "back_service_top":  round(0.76 / 13.40, 4),
                    "short_service_top": round(1.98 / 13.40, 4),
                    "net":               0.5,
                    "short_service_bot": round(1.0 - 1.98 / 13.40, 4),
                    "back_service_bot":  round(1.0 - 0.76 / 13.40, 4),
                    "singles_left":      round(0.46 / 6.10, 4),
                    "singles_right":     round(1.0 - 0.46 / 6.10, 4),
                    "center":            0.5,
                },
            },
            "frames": [fr.to_dict() for fr in self._frames],
        }

        with open(frame_data_path, "w", encoding="utf-8") as fh:
            json.dump(frame_data, fh, ensure_ascii=False, separators=(",", ":"))
        print(f"  [WebExporter] frame_data → {frame_data_path}")

        # ── summary.json ─────────────────────────────────────
        summary_path = os.path.join(
            output_dir,
            f"{video_name}{cfg['summary_suffix']}",
        )
        summary = self._build_summary(video_name)

        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
        print(f"  [WebExporter] summary    → {summary_path}")

        return {
            "frame_data": frame_data_path,
            "summary":    summary_path,
        }

    # ── 내부 메서드 ─────────────────────────────────────────

    def _extract_players(self, keypoints_list: list) -> List[PlayerRecord]:
        """
        keypoints_list(dict 또는 raw 형식)에서 선수 정규화 좌표를 추출합니다.
        """
        records = []
        parsed  = _to_dict_format(keypoints_list)

        # 누적 Y 기반 side 결정을 위해 먼저 모든 사람 처리
        candidates: List[Tuple[int, float, float]] = []   # (track_id, nx, ny)

        for person in parsed:
            track_id = person.get("track_id", -1)
            kpts     = person.get("keypoints", [])

            fx, fy = _foot_center(kpts, self._kconf)
            if fx <= 0 or fy <= 0:
                continue

            # 영상 좌표 → 정규화
            nx, ny = frame_to_normalized((fx, fy), self._hg["to_normalized"])
            nx = float(np.clip(nx, -0.1, 1.1))
            ny = float(np.clip(ny, -0.1, 1.1))

            # 누적 Y 저장 (side 결정용)
            if track_id not in self._player_y_acc:
                self._player_y_acc[track_id] = []
            self._player_y_acc[track_id].append(ny)
            if len(self._player_y_acc[track_id]) > 200:
                self._player_y_acc[track_id].pop(0)

            candidates.append((track_id, nx, ny))

        # 누적 평균 Y 기반 side 결정
        sorted_by_avgy = sorted(
            candidates,
            key=lambda c: (
                sum(self._player_y_acc.get(c[0], [c[2]])) /
                max(1, len(self._player_y_acc.get(c[0], [c[2]])))
            ),
        )

        for rank, (track_id, nx, ny) in enumerate(sorted_by_avgy):
            side = "top" if rank == 0 else "bottom"
            records.append(PlayerRecord(
                track_id = track_id,
                side     = side,
                x        = round(nx, 4),
                y        = round(ny, 4),
            ))

        return records

    def _build_summary(self, video_name: str) -> dict:
        """경기 요약 데이터를 생성합니다."""
        hit_frames = [fr for fr in self._frames if fr.hit is not None]
        hit_list   = [fr.hit for fr in hit_frames]

        # 선수별 타격 수
        top_hits = sum(1 for h in hit_list if h and h.get("owner") == "top")
        bot_hits = sum(1 for h in hit_list if h and h.get("owner") == "bottom")

        # 셔틀콕 궤적 (정규화 좌표, 감지된 것만)
        trail = [
            {"frame": fr.frame, "x": fr.shuttle.x, "y": fr.shuttle.y}
            for fr in self._frames
            if fr.shuttle.x is not None
        ]

        # 타점 목록
        hits = [
            {
                "hit_number": fr.hit["hit_number"],
                "frame":      fr.frame,
                "time_sec":   round(fr.time_sec, 3),
                "player":     fr.hit["player"],
                "owner":      fr.hit["owner"],
                # 해당 프레임에 선수 위치가 있으면 타점 위치도 기록
                "court_pos":  next(
                    (
                        {"x": p.x, "y": p.y}
                        for p in fr.players
                        if p.side == fr.hit.get("owner")
                    ),
                    None,
                ),
            }
            for fr in hit_frames
            if fr.hit
        ]

        return {
            "schema_version": "1.0",
            "video_name":     video_name,
            "fps":            round(self._fps, 2),
            "total_frames":   len(self._frames),
            "duration_sec":   round(len(self._frames) / max(self._fps, 1), 2),
            "analysis": {
                "total_hits":  len(hit_list),
                "top_hits":    top_hits,
                "bottom_hits": bot_hits,
            },
            "shuttle_trail": trail,
            "hits":          hits,
        }


# ─────────────────────────────────────────────────────────────
# 모듈 수준 헬퍼
# ─────────────────────────────────────────────────────────────

def _to_dict_format(keypoints_list: list) -> list:
    """raw 형식을 dict 형식으로 통일합니다."""
    if not keypoints_list:
        return []
    if isinstance(keypoints_list[0], dict):
        return keypoints_list
    result = []
    for i, person in enumerate(keypoints_list):
        kpts = person.tolist() if hasattr(person, "tolist") else list(person)
        result.append({"track_id": -(i + 1), "keypoints": kpts})
    return result


def _foot_center(kpts: list, kconf: float) -> Tuple[float, float]:
    """키포인트에서 양발 중심 좌표 추출 (COCO idx 15, 16)."""
    if len(kpts) < 17:
        return 0.0, 0.0
    try:
        lx, ly, lc = float(kpts[15][0]), float(kpts[15][1]), float(kpts[15][2])
        rx, ry, rc = float(kpts[16][0]), float(kpts[16][1]), float(kpts[16][2])
    except (IndexError, TypeError):
        return 0.0, 0.0

    if lc >= kconf and rc >= kconf:
        return (lx + rx) / 2, (ly + ry) / 2
    elif lc >= kconf:
        return lx, ly
    elif rc >= kconf:
        return rx, ry
    return 0.0, 0.0


# ─────────────────────────────────────────────────────────────
# 정적 분석 결과 저장 유틸리티
# ─────────────────────────────────────────────────────────────

def save_hits_json(
    hit_events:  List[ImpactEvent],
    fps:         float,
    output_path: str,
) -> None:
    """
    타점 이벤트를 hits.json으로 저장합니다.
    기존 to_api_json()과 동일한 형식에 추가 필드 포함.
    """
    data = {
        "schema_version": "1.0",
        "video_fps":   round(fps, 2),
        "total_hits":  len(hit_events),
        "hits_data": [
            {
                **ev.to_dict(),
                "owner":  ev.owner,
                "player": ev.player,
            }
            for ev in hit_events
        ],
    }
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    print(f"  [WebExporter] hits        → {output_path}")


def save_court_corners_json(
    corners_info: dict,
    output_path:  str,
) -> None:
    """
    자동 검출된 코트 코너 정보를 JSON으로 저장합니다.
    프론트엔드가 호모그래피 재계산 없이 코너 위치를 알 수 있도록 합니다.
    """
    data = {
        "schema_version": "1.0",
        "description":    "Auto-detected badminton court corners",
        "corners":        corners_info,
    }
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    print(f"  [WebExporter] court_corners → {output_path}")
