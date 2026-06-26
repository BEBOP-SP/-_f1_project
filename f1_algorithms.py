"""Precompute race-strategy algorithm results for the HTML replay.

The browser should focus on rendering.  This module owns the algorithm-heavy
parts that used to live in f1_viewer.html:

* [공간↔시간] SpaceTimeCache  — 서킷·드라이버 데이터 초기화 시 캐싱, O(1) 조회
* [억지기법]   drs_eligible    — DRS 활성화 조건 전수 체크
* [분할정복]   top_battles     — Closest Pair 알고리즘으로 최근접 차량 쌍 배틀 감지
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE   = BASE_DIR / "data" / "telemetry_full.json"
OUTPUT_FILE = BASE_DIR / "data" / "algorithm_results.json"

TOTAL_LAPS = 53  # 데이터 로드 후 raceMeta.totalLaps 로 덮어씀
DRS_THRESHOLD = 1.0


# 2021 프랑스 GP 실제 전략 (분기한정 비교용 역사적 데이터)
HISTORICAL_STRATEGIES: dict[str, dict[str, Any]] = {
    "VER": {
        "strategy": "2-Stop",
        "stops": [{"lap": 18, "to": "H"}, {"lap": 32, "to": "H"}],
        "result": "P1 — VER wins",
    },
    "HAM": {
        "strategy": "1-Stop",
        "stops": [{"lap": 19, "to": "H"}],
        "result": "P2 — HAM 2nd",
    },
}


def compute_driver_deg_rates(
    driver_laps: dict[str, Any],
    fallback: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """실제 랩 데이터로 드라이버별·컴파운드별 타이어 열화율 선형 회귀.

    각 스틴트의 (tire_age, lap_time) 쌍 수집 → y = a + b*x 회귀 → b = 열화율.
    데이터 4랩 미만이거나 이상치 제거 후 3랩 미만이면 fallback 기본값 사용.
    """
    rates: dict[str, dict[str, float]] = {}
    for name, laps in driver_laps.items():
        comp_points: dict[str, list[tuple[int, float]]] = {}
        for row in laps:
            comp = row[4]
            tl   = int(row[5] or 0)
            pit  = bool(row[7])
            lt   = row[8]
            # 피트랩·아웃랩(tl≤1)·무효 랩타임 제외
            if pit or tl <= 1 or not lt or lt <= 0:
                continue
            comp_points.setdefault(comp, []).append((tl, lt))

        driver_rates: dict[str, float] = {}
        for comp, pts in comp_points.items():
            default = fallback.get(comp, {}).get("deg_rate", 0.041)
            if len(pts) < 4:
                driver_rates[comp] = default
                continue
            # 이상치 제거: 중앙값 기준 ±15% 초과 제외
            med = sorted(p[1] for p in pts)[len(pts) // 2]
            clean = [(x, y) for x, y in pts if 0.85 * med <= y <= 1.15 * med]
            if len(clean) < 3:
                driver_rates[comp] = default
                continue
            # 선형 회귀
            n   = len(clean)
            sx  = sum(p[0] for p in clean)
            sy  = sum(p[1] for p in clean)
            sxy = sum(p[0] * p[1] for p in clean)
            sx2 = sum(p[0] ** 2 for p in clean)
            denom = n * sx2 - sx * sx
            if denom == 0:
                driver_rates[comp] = default
                continue
            b = (n * sxy - sx * sy) / denom
            # 기본값의 30% 미만이면 연료 소모 효과로 인한 왜곡 가능성 → 기본값 사용
            if b < default * 0.3:
                driver_rates[comp] = default
                continue
            driver_rates[comp] = max(0.001, min(0.25, b))  # 현실적 범위 클램프
        rates[name] = driver_rates
    return rates


class SpaceTimeCache:
    """[공간으로 시간 벌기] 서킷·드라이버 데이터 초기화 시 O(n) 빌드 → 루프 내 O(1) 조회.

    타이어 컴파운드 물리 파라미터와 드라이버 프로필을 Dictionary/LookupTable로
    미리 캐싱하여 DP·분기한정 등 백그라운드 연산의 반복 조회 비용을 제거한다.
    """

    # 타이어 컴파운드 물리 파라미터 2D 룩업 테이블
    # cliff: 열화 가속 시작 랩 / cliff_mult: cliff 이후 열화율 배수
    COMPOUND_TABLE: dict[str, dict[str, float]] = {
        "S": {"deg_rate": 0.082, "max_stint": 16.0, "warmup": 2.2, "cliff": 10, "cliff_mult": 6.0},
        "M": {"deg_rate": 0.041, "max_stint": 28.0, "warmup": 1.4, "cliff": 18, "cliff_mult": 6.0},
        "H": {"deg_rate": 0.018, "max_stint": 36.0, "warmup": 0.9, "cliff": 28, "cliff_mult": 8.0},
    }

    # 섹터별 DRS 존 여부 (폴 리카르: S1·S3 메인스트레이트 구간)
    SECTOR_DRS: dict[int, bool] = {1: True, 2: False, 3: True}

    def __init__(
        self,
        drivers: list[dict[str, Any]],
        driver_laps: dict[str, Any],
    ) -> None:
        # 드라이버 프로필 해시맵: name → {color, ...}
        self.driver_map: dict[str, dict[str, Any]] = {
            d["name"]: {"color": d.get("color", "#fff")}
            for d in drivers
        }
        # 드라이버별 베이스 페이스 (유효 랩타임 평균) 해시맵
        self.driver_base_pace: dict[str, float] = {}
        for name, laps in driver_laps.items():
            valid = [row[8] for row in laps if row[8] and row[8] > 0]
            if valid:
                self.driver_base_pace[name] = sum(valid) / len(valid)
        # [공간↔시간] 드라이버별·컴파운드별 개인 열화율 — 실제 랩데이터 선형 회귀
        self.driver_deg_rates: dict[str, dict[str, float]] = compute_driver_deg_rates(
            driver_laps, self.COMPOUND_TABLE
        )

    def compound(self, comp: str) -> dict[str, float]:
        """타이어 컴파운드 파라미터 O(1) 조회."""
        return self.COMPOUND_TABLE.get(comp, self.COMPOUND_TABLE["M"])

    def driver(self, name: str) -> dict[str, Any]:
        """드라이버 프로필 O(1) 조회."""
        return self.driver_map.get(name, {})

    def sector_has_drs(self, sector: int) -> bool:
        """섹터별 DRS 존 여부 O(1) 조회."""
        return self.SECTOR_DRS.get(sector, False)

    def base_pace(self, name: str) -> float:
        """드라이버 베이스 페이스 O(1) 조회."""
        return self.driver_base_pace.get(name, 95.0)

    def deg_rate(self, driver: str, comp: str) -> float:
        """드라이버별 개인 타이어 열화율 O(1) 조회. 데이터 없으면 컴파운드 기본값."""
        rate = self.driver_deg_rates.get(driver, {}).get(comp)
        if rate is not None:
            return rate
        return self.COMPOUND_TABLE.get(comp, self.COMPOUND_TABLE["M"])["deg_rate"]


class DPEtaEngine:
    """[동적 계획법] 타이어 마모 기반 예상 완주 시간(ETA) 예측기.

    점화식: f(comp, tl, r) = base_pace + deg_rate×tl + f(comp, tl+1, r-1)
    메모이제이션 테이블로 중복 연산 제거 → O(C × A × R) 총 상태 수.
    """

    def __init__(self, cache: SpaceTimeCache) -> None:
        self.cache = cache
        self._memo: dict[tuple[float, int, int], float] = {}

    def _degrade_sum(self, deg_rate: float, tl: int, remaining: int) -> float:
        """sum(deg_rate*(tl+i) for i in range(remaining)) — 메모이제이션 DP.

        base_pace는 선형으로 분리되므로 열화 누적분만 캐싱.
        key = (round(deg_rate,5), tl, remaining)
        """
        if remaining <= 0:
            return 0.0
        key = (round(deg_rate, 5), tl, remaining)
        if key in self._memo:
            return self._memo[key]
        result = deg_rate * tl + self._degrade_sum(deg_rate, tl + 1, remaining - 1)
        self._memo[key] = result
        return result

    def compute(self, name: str, comp: str, tl: float, lap: int, base_pace: float) -> dict[str, Any]:
        remaining = max(0, TOTAL_LAPS - lap)
        if remaining == 0:
            return {"remaining_laps": 0, "remaining_secs": 0.0, "label": "0:00.0"}
        # 드라이버별 개인 열화율 사용
        deg = self.cache.deg_rate(name, comp)
        secs = remaining * base_pace + self._degrade_sum(deg, int(tl), remaining)
        m, s = int(secs // 60), secs % 60
        return {
            "remaining_laps": remaining,
            "remaining_secs": round(secs, 1),
            "label": f"{m}:{s:04.1f}",
        }


class BranchBoundEngine:
    """[분기 한정] 남은 레이스 최적 타이어 시퀀스 탐색.

    각 노드에서 하한선(Bound) ≥ 현재 최적이면 해당 가지를 Pruning.
    초기 상한은 현재 타이어 유지 그리디 추정치로 설정하여 가지치기 효율 극대화.
    """

    PIT_DELTA = 23.5

    def __init__(self, cache: SpaceTimeCache) -> None:
        self.cache = cache

    def _cliff_lap_time(self, comp: str, deg: float, tl: float, base_pace: float) -> float:
        """cliff age 이후 열화율이 cliff_mult배로 가속되는 랩타임 모델."""
        props = self.cache.compound(comp)
        cliff = props.get("cliff", 999)
        mult  = props.get("cliff_mult", 1.0)
        if tl <= cliff:
            return base_pace + deg * tl
        return base_pace + deg * cliff + deg * mult * (tl - cliff)

    def _lower_bound(self, remaining: int, base_pace: float) -> float:
        """하한선: 마모 없는 최고 속도로만 달린다고 가정한 최소 시간."""
        return remaining * base_pace

    def _search(
        self,
        lap: int,
        comp: str,
        tl: float,
        base_pace: float,
        elapsed: float,
        sequence: list[dict[str, Any]],
        stops_left: int,
        state: dict[str, Any],
        deg_rates: dict[str, float],
    ) -> None:
        remaining = TOTAL_LAPS - lap
        if remaining <= 0:
            if elapsed < state["best_time"]:
                state["best_time"] = elapsed
                state["best_seq"] = sequence[:]
            return

        # Pruning: 하한선이 현재 최적보다 크거나 같으면 탐색 중단
        if elapsed + self._lower_bound(remaining, base_pace) >= state["best_time"]:
            return

        props   = self.cache.compound(comp)
        deg     = deg_rates.get(comp, props["deg_rate"])  # 개인 열화율
        lap_time = self._cliff_lap_time(comp, deg, tl, base_pace)

        # Branch 1: 현재 타이어 유지 (stint 한계 미만일 때)
        if tl < props["max_stint"]:
            self._search(lap + 1, comp, tl + 1, base_pace,
                         elapsed + lap_time, sequence, stops_left, state, deg_rates)

        # Branch 2: 피트인 후 타이어 교체
        if stops_left > 0 and remaining > 2:
            pit_delta = state.get("pit_delta", self.PIT_DELTA)
            for next_comp in ("S", "M", "H"):
                if next_comp == comp:
                    continue
                warmup = self.cache.compound(next_comp)["warmup"]
                new_seq = sequence + [{"lap": lap, "to": next_comp}]
                self._search(
                    lap + 1, next_comp, 1.0, base_pace,
                    elapsed + lap_time + pit_delta + warmup,
                    new_seq, stops_left - 1, state, deg_rates,
                )

    def optimize(
        self, from_lap: int, comp: str, tl: float, base_pace: float,
        driver_name: str = "", max_stops: int = 2, pit_delta: float | None = None,
    ) -> dict[str, Any]:
        remaining = TOTAL_LAPS - from_lap
        if remaining <= 0:
            return {"from_lap": from_lap, "optimal_time": 0.0, "optimal_seq": []}

        # 드라이버별 개인 열화율 사전 로딩 (없으면 컴파운드 기본값)
        deg_rates = {c: self.cache.deg_rate(driver_name, c) for c in ("S", "M", "H")}

        # 초기 상한: 현재 타이어로 남은 랩 모두 달리기 (no-pit greedy, cliff 반영)
        dr = deg_rates[comp]
        greedy = sum(self._cliff_lap_time(comp, dr, int(tl) + i, base_pace) for i in range(remaining))
        state: dict[str, Any] = {
            "best_time": greedy,
            "best_seq": [],
            "pit_delta": pit_delta if pit_delta is not None else self.PIT_DELTA,
        }
        self._search(from_lap, comp, tl, base_pace, 0.0, [], max_stops, state, deg_rates)
        return {
            "from_lap": from_lap,
            "optimal_time": round(state["best_time"], 1),
            "optimal_seq": state["best_seq"],
        }


@dataclass
class RaceAlgorithms:
    data: dict[str, Any]

    def __post_init__(self) -> None:
        self.drivers = self.data["drivers"]
        self.driver_laps = self.data["driverLaps"]
        self.driver_telemetry = self.data["driverTelemetry"]
        self.telemetry_idx = {name: 0 for name in self.driver_telemetry}
        self.lap_idx = {name: 0 for name in self.driver_laps}
        # [공간↔시간] 초기화 시 한 번만 빌드, 루프 내에서는 O(1) 참조
        self.cache = SpaceTimeCache(self.drivers, self.driver_laps)

    def state_at(self, name: str, t: float) -> dict[str, Any] | None:
        laps = self.driver_laps.get(name)
        if not laps:
            return None

        idx = self.lap_idx.get(name, 0)
        if idx > 0 and laps[idx][0] > t:
            idx = 0
        while idx < len(laps) - 1 and laps[idx + 1][0] <= t:
            idx += 1
        self.lap_idx[name] = idx
        cur = laps[idx]
        frac = self.track_fraction(name, t)
        sector_progress = frac["sector_progress"] if frac else 0.0
        raw_pit = bool(cur[7])
        pit_active = raw_pit and ((frac or {}).get("t", 0.0) >= 0.75)
        return {
            "name": name,
            "lap": cur[2],
            "pos": cur[3],
            "comp": cur[4],
            "tl": cur[5],
            "pit": 1 if pit_active else 0,
            "rawPit": 1 if raw_pit else 0,
            "lt": cur[8],
            "sector": sector_from_progress((frac or {}).get("t", 0.0)),
            "lapPoint": cur[2] + sector_progress,
            "tlPoint": (cur[5] or 0) + sector_progress,
        }

    def telemetry_at(self, name: str, t: float) -> dict[str, Any] | None:
        rows = self.driver_telemetry.get(name)
        if not rows:
            return None

        idx = self.telemetry_idx.get(name, 0)
        if idx > 0 and rows[idx][0] > t:
            idx = 0
        while idx < len(rows) - 2 and rows[idx + 1][0] <= t:
            idx += 1
        self.telemetry_idx[name] = idx
        row = rows[idx]
        return {"t": row[0], "x": row[1], "y": row[2], "speed": row[3], "drs": row[7]}

    def track_fraction(self, name: str, t: float) -> dict[str, float] | None:
        laps = self.driver_laps.get(name)
        if not laps:
            return None

        idx = self.lap_idx.get(name, 0)
        if idx > 0 and laps[idx][0] > t:
            idx = 0
        while idx < len(laps) - 1 and laps[idx + 1][0] <= t:
            idx += 1
        self.lap_idx[name] = idx
        row = laps[idx]
        lap_dur = row[8]
        if not lap_dur or lap_dur <= 0:
            return None

        progress = min(1.0, max(0.0, (t - row[0]) / lap_dur))
        sector = sector_from_progress(progress)
        sector_progress = (sector - 1) / 3
        return {
            "t": progress,
            "lapDur": lap_dur,
            "lap": row[2],
            "sector_progress": sector_progress,
        }

    def gap_to_front(self, behind_name: str, ahead_name: str, t: float) -> float:
        behind = self.track_fraction(behind_name, t)
        ahead = self.track_fraction(ahead_name, t)
        if not behind or not ahead:
            return math.inf

        if ahead["lap"] == behind["lap"]:
            diff = ahead["t"] - behind["t"]
            if diff <= 0:
                return math.inf
            return diff * ahead["lapDur"]

        lap_diff = ahead["lap"] - behind["lap"]
        if lap_diff < 0 or lap_diff > 2:
            return math.inf
        gap_frac = (1.0 - behind["t"]) + ahead["t"]
        return gap_frac * ((ahead["lapDur"] + behind["lapDur"]) / 2)

    def sorted_rows(self, t: float) -> list[dict[str, Any]]:
        rows = []
        for driver in self.drivers:
            state = self.state_at(driver["name"], t)
            if not state:
                state = {"name": driver["name"], "pos": 99, "comp": "M", "lap": 1, "pit": 0, "tl": 0}
            rows.append({**driver, **state})
        return sorted(rows, key=lambda r: r["pos"])

    def drs_eligible(self, rows: list[dict[str, Any]], t: float) -> list[str]:
        """[억지 기법 / Brute Force] DRS 활성화 조건 전수 체크.

        모든 인접 드라이버 쌍을 순차 비교(O(D))하여
        gap ≤ DRS_THRESHOLD(1.0s) 여부를 단순 if문으로 판정.
        """
        eligible: set[str] = set()

        for row in rows:
            tel = self.telemetry_at(row["name"], t)
            if (tel or {}).get("drs", 0) > 9:
                eligible.add(row["name"])

        for i in range(1, len(rows)):
            ahead = rows[i - 1]
            behind = rows[i]
            tel = self.telemetry_at(behind["name"], t)
            in_drs_zone = (tel or {}).get("drs", 0) >= 8
            if not in_drs_zone or behind["name"] in eligible:
                continue
            gap = self.gap_to_front(behind["name"], ahead["name"], t)
            if math.isfinite(gap) and 0 <= gap <= DRS_THRESHOLD:
                eligible.add(behind["name"])

        return sorted(eligible)

    def is_in_pitlane(self, name: str, t: float) -> bool:
        state = self.state_at(name, t)
        if not state:
            return False

        frac = self.track_fraction(name, t)
        progress = frac["t"] if frac else 0.0

        # 1. 인랩: 폴 리카르 기준 progress≈0.48부터 피트레인 GPS 진입 확인됨
        #    기존 0.75는 너무 늦어 0.48~0.75 구간 차량이 배틀에 오탐됨 → 0.40으로 수정
        if state.get("rawPit") and progress >= 0.40:
            return True

        # 2. 아웃랩 전반부 (피트레인 탈출)
        current_lap = state["lap"]
        if current_lap > 1:
            laps = self.driver_laps.get(name)
            if laps:
                prev_lap_row = None
                for row in laps:
                    if row[2] == current_lap - 1:
                        prev_lap_row = row
                        break
                if prev_lap_row and bool(prev_lap_row[7]) and progress < 0.20:
                    return True

        return False

    def top_battles(self, rows: list[dict[str, Any]], t: float, k: int = 3) -> list[dict[str, Any]]:
        """[분할 정복 / Divide & Conquer] 최근접 차량 쌍(Closest Pair) 배틀 감지.

        차량 위치를 x축으로 정렬 후 중간선 분할 → 좌/우 재귀 → 스트립 결합.
        O(n log n) 복잡도. gap < 5s 이내 쌍만 배틀로 집계 후 상위 k개 반환.
        gap < 0.5s → [BATTLE TRIGGERED], gap < 1.0s → [DRS AVAILABLE] 로 연동됨.
        """
        points = []
        colors = {d["name"]: d.get("color", "#fff") for d in self.drivers}
        for row in rows:
            # 피트레인에 실제로 있는 동안에만 배틀 후보에서 제외
            if self.is_in_pitlane(row["name"], t):
                continue
            frac = self.track_fraction(row["name"], t)
            if not frac:
                continue
            tel = self.telemetry_at(row["name"], t) or {}
            points.append({
                "name": row["name"],
                "color": colors.get(row["name"], "#fff"),
                "pos": row["pos"],
                "x": frac["lap"] + frac["t"],
                "lapDur": frac["lapDur"],
                "gx": tel.get("x", 0.0),  # GPS 좌표 (아티팩트 필터용)
                "gy": tel.get("y", 0.0),
            })
        if len(points) < 2:
            return []

        pair_map: dict[str, dict[str, Any]] = {}
        # 실측 데이터 기반: gap<3s 배틀의 GPS 거리 p99=3143, 아티팩트는 6000+
        # → 4000을 임계값으로 GPS 아티팩트 페어 제거
        GPS_OUTLIER_THRESHOLD = 4000.0

        def record(a: dict[str, Any], b: dict[str, Any]) -> float:
            ahead, behind = (a, b) if a["x"] > b["x"] else (b, a)
            key = f"{behind['name']}:{ahead['name']}"
            gap = max(0.0, (ahead["x"] - behind["x"]) * ((ahead["lapDur"] + behind["lapDur"]) / 2))
            if gap < 5.0 and key not in pair_map:
                # GPS 아티팩트 필터: 텔레메트리 GPS가 비정상적으로 멀면 제외
                gps_dist = math.sqrt(
                    (ahead["gx"] - behind["gx"]) ** 2 +
                    (ahead["gy"] - behind["gy"]) ** 2
                )
                if gps_dist < GPS_OUTLIER_THRESHOLD:
                    pair_map[key] = {"ahead": compact_driver(ahead), "behind": compact_driver(behind), "gap": gap}
            return abs(a["x"] - b["x"])

        def closest(arr: list[dict[str, Any]]) -> float:
            n = len(arr)
            if n <= 3:
                best = math.inf
                for i in range(n):
                    for j in range(i + 1, n):
                        best = min(best, record(arr[i], arr[j]))
                return best

            mid = n // 2
            mid_x = arr[mid]["x"]
            delta = min(closest(arr[:mid]), closest(arr[mid:]))
            strip = [p for p in arr if abs(p["x"] - mid_x) < delta]
            strip.sort(key=lambda p: p["x"])
            for i, p in enumerate(strip):
                j = i + 1
                while j < len(strip) and strip[j]["x"] - p["x"] < delta:
                    delta = min(delta, record(p, strip[j]))
                    j += 1
            return delta

        closest(sorted(points, key=lambda p: p["x"]))
        return sorted(pair_map.values(), key=lambda b: b["gap"])[:k]

    def frame(self, t: float) -> dict[str, Any]:
        rows = self.sorted_rows(t)
        return {
            "drs": self.drs_eligible(rows, t),
            "undercut": [],
            "undercutDetails": {},
            "battles": self.top_battles(rows, t),
        }


def sector_from_progress(progress: float) -> int:
    return min(3, max(1, math.floor(min(0.999999, max(0.0, progress)) * 3) + 1))


def compact_driver(point: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": point["name"],
        "color": point.get("color", "#fff"),
        "pos": point.get("pos", 99),
    }


def build_algorithm_cache(step: float = 1.0) -> dict[str, Any]:
    with DATA_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)

    global TOTAL_LAPS
    TOTAL_LAPS = int(data.get("raceMeta", {}).get("totalLaps", TOTAL_LAPS))

    engine = RaceAlgorithms(data)
    session_start = float(data["sessionStart"])
    session_end = float(data["sessionEnd"])
    race_start = min(
        laps[0][0]
        for laps in data["driverLaps"].values()
        if isinstance(laps, list) and laps
    )

    frames: dict[str, Any] = {}
    t = math.floor(session_start)
    end = math.ceil(session_end)
    while t <= end:
        if t >= race_start:
            frames[str(int(t))] = engine.frame(float(t))
        else:
            frames[str(int(t))] = {"drs": [], "undercut": [], "undercutDetails": {}, "battles": []}
        t += step

    # ── [동적 계획법] VER·HAM 랩별 예상 완주 시간 사전 계산 ───────────────────
    dp_engine = DPEtaEngine(engine.cache)
    dp_eta: dict[str, dict[str, Any]] = {}
    for name in ("VER", "HAM"):
        laps_data = data["driverLaps"].get(name, [])
        base = engine.cache.base_pace(name)
        lap_results: dict[str, Any] = {}
        seen: set[int] = set()
        for row in laps_data:
            lap_num = int(row[2])
            if lap_num in seen:
                continue
            seen.add(lap_num)
            comp = row[4] or "M"
            tl = float(row[5] or 0)
            lap_results[str(lap_num)] = dp_engine.compute(name, comp, tl, lap_num, base)
        dp_eta[name] = lap_results

    # ── [분기 한정] 레이스 시작 전 드라이버별 피트 전략 계산 ─────────────────
    def _pit_window(opt_lap: int, lo: int, hi: int, half: int = 3) -> tuple[int, int]:
        return (max(lo, opt_lap - half), min(hi, opt_lap + half))

    bb_engine  = BranchBoundEngine(engine.cache)
    bb_results: dict[str, dict[str, Any]] = {}

    for drv_name, drv_laps_data in data["driverLaps"].items():
        if not drv_laps_data:
            continue
        drv_base   = engine.cache.base_pace(drv_name)
        first_row  = drv_laps_data[0]
        start_lap  = int(first_row[2])
        start_comp = first_row[4] or "M"

        strategies: list[dict[str, Any]] = []

        # ① 1-스톱 전략
        r1 = bb_engine.optimize(start_lap, start_comp, 1.0, drv_base,
                                driver_name=drv_name, max_stops=1)
        if r1["optimal_seq"]:
            stops: list[dict[str, Any]] = []
            lo = start_lap + 2
            for s in r1["optimal_seq"]:
                w = _pit_window(s["lap"], lo, TOTAL_LAPS - 5)
                stops.append({"to": s["to"], "opt_lap": s["lap"],
                              "win_start": w[0], "win_end": w[1]})
                lo = s["lap"] + 5
            strategies.append({"label": "1-STOP", "stops": stops})

        # ② 2-스톱 전략 (B&B 우선, 모델이 찾지 못하면 3등분 휴리스틱)
        r2 = bb_engine.optimize(start_lap, start_comp, 1.0, drv_base,
                                driver_name=drv_name, max_stops=2)
        if len(r2["optimal_seq"]) >= 2:
            stops2: list[dict[str, Any]] = []
            lo = start_lap + 2
            for s in r2["optimal_seq"]:
                w = _pit_window(s["lap"], lo, TOTAL_LAPS - 5)
                stops2.append({"to": s["to"], "opt_lap": s["lap"],
                               "win_start": w[0], "win_end": w[1]})
                lo = s["lap"] + 5
            strategies.append({"label": "2-STOP", "stops": stops2})
        else:
            # B&B가 2-스톱을 찾지 못할 때: 레이스를 3등분하는 균형 전략 표시
            total_dist = TOTAL_LAPS - start_lap
            pit1_lap = start_lap + total_dist // 3
            pit2_lap = start_lap + 2 * total_dist // 3
            # 컴파운드: 시작 → 반대 계열 → 원래 계열 순환
            _cycle = {"M": ("H", "M"), "H": ("M", "H"), "S": ("M", "H")}
            pit1_to, pit2_to = _cycle.get(start_comp, ("H", "M"))
            w1 = _pit_window(pit1_lap, start_lap + 2, TOTAL_LAPS - 12)
            w2 = _pit_window(pit2_lap, pit1_lap + 5, TOTAL_LAPS - 5)
            strategies.append({"label": "2-STOP", "stops": [
                {"to": pit1_to, "opt_lap": pit1_lap,
                 "win_start": w1[0], "win_end": w1[1]},
                {"to": pit2_to, "opt_lap": pit2_lap,
                 "win_start": w2[0], "win_end": w2[1]},
            ]})

        bb_results[drv_name] = {
            "driver":     drv_name,
            "trigger":    "prerace",
            "start_comp": start_comp,
            "strategies": strategies,
        }

    # ── 레이스 전략 랭킹: 출발 컴파운드별 B&B 최적 시간 계산 → 상위 3개 ─────
    avg_base = (
        sum(engine.cache.driver_base_pace.values())
        / max(1, len(engine.cache.driver_base_pace))
    )

    # 실제 경기에서 사용된 출발 컴파운드만 대상으로
    candidate_starts: list[str] = sorted({
        rows[0][4] for rows in data["driverLaps"].values()
        if rows and rows[0][4]
    })

    def _race_time_seq(start_comp: str, seq: list[dict[str, Any]]) -> float:
        """출발 컴파운드 + 피트 시퀀스로 전체 레이스 예상 시간 계산 (cliff 모델)."""
        pit_map = {s["lap"]: s for s in seq}
        comp, tl, total = start_comp, 1.0, 0.0
        for lap in range(1, TOTAL_LAPS):
            props = engine.cache.COMPOUND_TABLE.get(comp, engine.cache.COMPOUND_TABLE["M"])
            deg   = props["deg_rate"]
            lt    = bb_engine._cliff_lap_time(comp, deg, tl, avg_base)
            if lap in pit_map:
                next_comp = pit_map[lap]["to"]
                total += lt + bb_engine.PIT_DELTA + engine.cache.COMPOUND_TABLE[next_comp]["warmup"]
                comp, tl = next_comp, 1.0
            else:
                total += lt
                tl += 1
        return total

    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()

    for sc in candidate_starts:
        for max_stops in (1, 2):
            r = bb_engine.optimize(1, sc, 1.0, avg_base, max_stops=max_stops)
            seq = r["optimal_seq"]
            n_stops = len(seq)
            if n_stops == 0:
                continue

            # 중복 제거: (start_comp, to1, to2, ...)
            key: tuple[str, ...] = (sc,) + tuple(s["to"] for s in seq)
            if key in seen:
                continue
            seen.add(key)

            stops: list[dict[str, Any]] = []
            lo = 3
            for s in seq:
                w = _pit_window(s["lap"], lo, TOTAL_LAPS - 5)
                stops.append({"to": s["to"], "opt_lap": s["lap"],
                              "win_start": w[0], "win_end": w[1]})
                lo = s["lap"] + 5

            candidates.append({
                "label":      f"{n_stops}-STOP",
                "start_comp": sc,
                "total_time": round(_race_time_seq(sc, seq), 1),
                "stops":      stops,
            })

        # 2-스톱 B&B가 1-스톱만 반환했을 때: 3등분 휴리스틱 후보 추가
        r2 = bb_engine.optimize(1, sc, 1.0, avg_base, max_stops=2)
        if len(r2["optimal_seq"]) < 2:
            total_dist = TOTAL_LAPS - 1
            pit1_lap   = 1 + total_dist // 3
            pit2_lap   = 1 + 2 * total_dist // 3
            _cycle     = {"M": ("H", "M"), "H": ("M", "H"), "S": ("M", "H")}
            pit1_to, pit2_to = _cycle.get(sc, ("H", "M"))
            hkey: tuple[str, ...] = (sc, pit1_to, pit2_to)
            if hkey not in seen:
                seen.add(hkey)
                h_seq = [{"lap": pit1_lap, "to": pit1_to},
                         {"lap": pit2_lap, "to": pit2_to}]
                w1 = _pit_window(pit1_lap, 3, TOTAL_LAPS - 12)
                w2 = _pit_window(pit2_lap, pit1_lap + 5, TOTAL_LAPS - 5)
                candidates.append({
                    "label":      "2-STOP",
                    "start_comp": sc,
                    "total_time": round(_race_time_seq(sc, h_seq), 1),
                    "stops":      [
                        {"to": pit1_to, "opt_lap": pit1_lap,
                         "win_start": w1[0], "win_end": w1[1]},
                        {"to": pit2_to, "opt_lap": pit2_lap,
                         "win_start": w2[0], "win_end": w2[1]},
                    ],
                })

    candidates.sort(key=lambda x: x["total_time"])
    top3     = candidates[:3]
    best_t   = top3[0]["total_time"] if top3 else 0.0
    race_strategies: list[dict[str, Any]] = []
    for rank, s in enumerate(top3, 1):
        race_strategies.append({
            "rank":       rank,
            "label":      s["label"],
            "start_comp": s["start_comp"],
            "total_time": s["total_time"],
            "delta":      round(s["total_time"] - best_t, 1),
            "stops":      s["stops"],
        })

    return {
        "meta": {
            "source": DATA_FILE.name,
            "stepSeconds": step,
            "sessionStart": session_start,
            "sessionEnd": session_end,
            "raceStart": race_start,
            "layers": ["drs", "battles"],
        },
        "frames": frames,
        "dpEta": dp_eta,
        "branchBound": bb_results,
        "raceStrategies": race_strategies,
        "driverDegRates": engine.cache.driver_deg_rates,
    }


def ensure_algorithm_cache(force: bool = False) -> Path:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"{DATA_FILE} does not exist.")

    if not force and OUTPUT_FILE.exists() and OUTPUT_FILE.stat().st_mtime >= DATA_FILE.stat().st_mtime:
        return OUTPUT_FILE

    cache = build_algorithm_cache()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(cache, f, separators=(",", ":"))
    return OUTPUT_FILE


def validate_cache(output_file: Path = OUTPUT_FILE) -> None:
    """캐시 검증 및 5가지 알고리즘 위치 요약 리포트 출력."""
    with output_file.open("r", encoding="utf-8") as f:
        cache = json.load(f)

    meta   = cache["meta"]
    frames = cache["frames"]
    dp_eta = cache.get("dpEta", {})
    bb     = cache.get("branchBound", [])

    print("\n" + "=" * 62)
    print("  F1 Algorithm Cache Validation - 2021 France GP")
    print("=" * 62)
    print(f"  Frames      : {len(frames):,}  (step {meta['stepSeconds']}s)")
    print(f"  DP ETA      : {list(dp_eta.keys())}")
    for name, laps in dp_eta.items():
        sample = list(laps.values())[len(laps) // 2] if laps else {}
        print(f"    {name}: {len(laps)} laps | mid-sample: {sample.get('label', '-')}")
    print(f"  B&B pre-race: {len(bb)} drivers")
    for drv, r in (bb.items() if isinstance(bb, dict) else []):
        for strat in r.get("strategies", []):
            wins = " | ".join(f"L{s['win_start']}-{s['win_end']}→{s['to']}" for s in strat["stops"])
            print(f"    {drv} {strat['label']}: {wins}")

    # Final 2 laps VER-HAM battle / DRS milestone check
    race_end   = float(meta["sessionEnd"])
    window     = 200
    close: list[tuple[float, float]] = []
    drs_frames: list[float] = []
    for ts_str, frame in frames.items():
        ts = float(ts_str)
        if ts < race_end - window:
            continue
        for b in frame.get("battles", []):
            names = {b["ahead"]["name"], b["behind"]["name"]}
            if names == {"VER", "HAM"}:
                close.append((ts, b["gap"]))
        drs = frame.get("drs", [])
        if "VER" in drs or "HAM" in drs:
            drs_frames.append(ts)

    print(f"\n  [Milestone] Last {window}s VER-HAM battles: {len(close)} frames")
    if close:
        min_gap = min(g for _, g in close)
        triggered = min_gap < 0.5
        drs_ok    = any(g < 1.0 for _, g in close)
        print(f"    min gap  : {min_gap:.2f}s  BATTLE_TRIGGERED={'YES' if triggered else 'NO'}  DRS={'YES' if drs_ok else 'NO'}")
    print(f"    DRS frames: {len(drs_frames)} (VER or HAM)")

    print("\n  -- Algorithm locations --")
    algo_map = [
        ("Space-for-Time", "SpaceTimeCache.__init__()",      "O(n) build -> O(1) lookup"),
        ("Brute Force   ", "RaceAlgorithms.drs_eligible()",  "sequential pair check O(D)"),
        ("Divide&Conquer", "RaceAlgorithms.top_battles()",   "Closest Pair O(n log n)"),
        ("Dynamic Prog  ", "DPEtaEngine._predict()",         "memoized DP  O(C x A x R)"),
        ("Branch&Bound  ", "BranchBoundEngine._search()",    "state-tree + Pruning"),
    ]
    for i, (name, func, note) in enumerate(algo_map, 1):
        print(f"  {i}. [{name}]  {func}")
        print(f"       -> {note}")
    print("=" * 62)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute F1 replay algorithm results.")
    parser.add_argument("--force",  action="store_true", help="rebuild even when the cache is newer than telemetry")
    parser.add_argument("--data",   type=str, default=None, help="telemetry JSON 경로 (기본: data/telemetry_full.json)")
    parser.add_argument("--output", type=str, default=None, help="알고리즘 캐시 출력 경로 (기본: data/algorithm_results.json)")
    return parser.parse_args()


def main() -> None:
    global DATA_FILE, OUTPUT_FILE
    args = parse_args()
    if args.data:
        DATA_FILE   = Path(args.data)
    if args.output:
        OUTPUT_FILE = Path(args.output)
    output = ensure_algorithm_cache(force=args.force)
    print(f"Algorithm cache ready: {output}")
    validate_cache(output)


if __name__ == "__main__":
    main()
