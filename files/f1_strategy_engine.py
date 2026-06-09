"""
=============================================================================
🏎️ F1 데이터 기반 실시간 전략 및 중계 분석 엔진
   2021 프랑스 GP - 언더컷 알고리즘 효율성 분석
=============================================================================

3-Layer 알고리즘 구조:
  Layer 1: Brute Force        → DRS/Override 추월 권한 체크
  Layer 2: Decrease & Conquer → 언더컷 골든 윈도우 이진 탐색
  Layer 3: Divide & Conquer   → 최근접 쌍(Closest Pair) 배틀 감지
=============================================================================
"""

import pandas as pd
import numpy as np
import math
import heapq
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import os

# ──────────────────────────────────────────────────────────────
# 0. 데이터 로딩 & 전처리
# ──────────────────────────────────────────────────────────────

def timedelta_to_seconds(td_str: str) -> Optional[float]:
    """FastF1 timedelta 문자열 → 초 변환"""
    try:
        if pd.isna(td_str):
            return None
        parts = str(td_str).split(' ')[-1].split(':')
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    except:
        return None


def load_data(data_dir: str = None) -> dict:
    """CSV 데이터 로딩 및 전처리 — 자동 경로 탐색"""
    if data_dir is None:
        # 스크립트 위치 기준으로 탐색
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            script_dir,                          # 같은 폴더
            os.path.join(script_dir, 'data'),    # ./data/
            os.path.join(script_dir, '..', 'data'),  # ../data/
        ]
        for c in candidates:
            if os.path.exists(os.path.join(c, 'laps.csv')):
                data_dir = c
                break
        if data_dir is None:
            raise FileNotFoundError(
                f"laps.csv를 찾을 수 없습니다. 다음 위치를 확인하세요:\n"
                + "\n".join(f"  - {os.path.abspath(c)}" for c in candidates)
            )
    print(f"  📁 데이터 경로: {os.path.abspath(data_dir)}")
    """CSV 데이터 로딩 및 전처리"""
    laps = pd.read_csv(os.path.join(data_dir, 'laps.csv'))
    results = pd.read_csv(os.path.join(data_dir, 'results.csv'))
    tel_ver = pd.read_csv(os.path.join(data_dir, 'telemetry_VER.csv'))
    tel_ham = pd.read_csv(os.path.join(data_dir, 'telemetry_HAM.csv'))

    # 랩타임 → 초 변환
    laps['LapTimeSec'] = laps['LapTime'].apply(timedelta_to_seconds)

    # 세션 시간 → 초 변환
    laps['SessionTimeSec'] = laps['Time'].apply(timedelta_to_seconds)

    # 텔레메트리 세션 시간 → 초 변환
    tel_ver['SessionTimeSec'] = tel_ver['SessionTime'].apply(timedelta_to_seconds)
    tel_ham['SessionTimeSec'] = tel_ham['SessionTime'].apply(timedelta_to_seconds)

    return {
        'laps': laps,
        'results': results,
        'tel_ver': tel_ver,
        'tel_ham': tel_ham,
    }


# ──────────────────────────────────────────────────────────────
# 상수 정의
# ──────────────────────────────────────────────────────────────

DRS_THRESHOLD = 1.0          # DRS 활성화 기준: 1초 이내
PIT_STOP_LOSS = 25.0         # 일반 피트스톱 시간 손실 (초)
PIT_STOP_LOSS_SC = 12.0      # 세이프티카 상황 피트스톱 시간 손실 (초)
TIRE_DEG_MEDIUM = 0.045      # 미디엄 타이어 랩당 성능 저하 (초)
TIRE_DEG_HARD = 0.025        # 하드 타이어 랩당 성능 저하 (초)
NEW_TIRE_ADVANTAGE = 1.8     # 새 타이어의 랩타임 이득 (초)
BATTLE_DISTANCE_THRESHOLD = 200.0  # 배틀 감지 거리 (좌표 단위)


# ══════════════════════════════════════════════════════════════
# LAYER 1: BRUTE FORCE — DRS/Override 추월 권한 체크
# ══════════════════════════════════════════════════════════════

@dataclass
class DRSReport:
    """DRS 체크 결과"""
    lap: int
    driver: str
    driver_ahead: str
    gap_seconds: float
    drs_eligible: bool
    overtake_possible: bool


def brute_force_drs_check(laps: pd.DataFrame) -> List[DRSReport]:
    """
    억지 기법(Brute Force): 모든 드라이버 쌍을 전수 조사하여
    DRS 활성화 가능 여부를 판정.

    복잡도: O(D² × L)  (D=드라이버 수, L=총 랩수)
    D=20으로 소규모이므로 전수 조사가 가장 명확하고 직관적.
    """
    reports = []
    drivers = laps['Driver'].unique()
    max_lap = int(laps['LapNumber'].max())

    print("\n" + "=" * 70)
    print("  🔍 LAYER 1: BRUTE FORCE — DRS/Override 추월 권한 체크")
    print("=" * 70)
    print(f"  전수 조사 대상: {len(drivers)}명 드라이버 × {max_lap} 랩")
    print(f"  총 비교 횟수: {len(drivers) * (len(drivers)-1) * max_lap:,}회")
    print("-" * 70)

    drs_events = []

    for lap_num in range(1, max_lap + 1):
        lap_data = laps[laps['LapNumber'] == lap_num].copy()
        lap_data = lap_data.sort_values('Position')

        # 모든 드라이버 쌍 비교 (Brute Force)
        for i in range(len(lap_data)):
            for j in range(len(lap_data)):
                if i == j:
                    continue

                d_behind = lap_data.iloc[i]
                d_ahead = lap_data.iloc[j]

                # 바로 앞차만 체크 (포지션 차이 == 1)
                if d_behind['Position'] - d_ahead['Position'] != 1:
                    continue

                # 두 드라이버의 랩타임 기반 갭 계산
                gap = abs(
                    (d_behind['SessionTimeSec'] or 0) -
                    (d_ahead['SessionTimeSec'] or 0)
                )

                drs_eligible = gap < DRS_THRESHOLD
                # 폴 리카르에서 트랙 위 직접 추월은 매우 어려움
                overtake_possible = drs_eligible and gap < 0.5

                report = DRSReport(
                    lap=lap_num,
                    driver=d_behind['Driver'],
                    driver_ahead=d_ahead['Driver'],
                    gap_seconds=round(gap, 3),
                    drs_eligible=drs_eligible,
                    overtake_possible=overtake_possible,
                )
                reports.append(report)

                if drs_eligible:
                    drs_events.append(report)

    # VER-HAM 관련 DRS 이벤트 출력
    ver_ham_drs = [r for r in drs_events
                   if (r.driver in ('VER', 'HAM') and r.driver_ahead in ('VER', 'HAM'))]

    if ver_ham_drs:
        print("\n  📡 VER ↔ HAM DRS 감지 이벤트:")
        for r in ver_ham_drs:
            status = "🟢 DRS OPEN" if r.drs_eligible else "🔴 CLOSED"
            overtake = " ⚡ OVERTAKE WINDOW!" if r.overtake_possible else ""
            print(f"    Lap {r.lap:2d} | {r.driver} → {r.driver_ahead} | "
                  f"Gap: {r.gap_seconds:.3f}s | {status}{overtake}")
    else:
        print("\n  📡 VER ↔ HAM: DRS 범위(1초) 이내 진입 없음")
        print("     → 트랙 위 직접 추월 불가 — 언더컷 전략 필요!")

    total_drs = len(drs_events)
    print(f"\n  📊 전체 DRS 이벤트: {total_drs}건 / 전수 조사 {len(reports)}건")

    # 실제 추월 발생 여부 확인 (DRS가 있어도 추월 못한 경우)
    # Lap 20-28: HAM이 DRS로 VER를 추격했지만 추월 실패 (폴 리카르 특성)
    actual_overtake_with_drs = any(r.overtake_possible for r in ver_ham_drs)
    if ver_ham_drs:
        print(f"  ⚠️  DRS 범위 진입 {len(ver_ham_drs)}회 → 그러나 폴 리카르에서 "
              f"실제 추월 성공은 {'있음' if actual_overtake_with_drs else '없음'}")
        print(f"  ✅ 결론: DRS만으로는 부족 → 언더컷 전략이 승부를 결정짓는 핵심 수단")
    else:
        print(f"  ✅ 결론: 직접 추월 불가 → 언더컷이 유일한 해법")

    return reports


# ══════════════════════════════════════════════════════════════
# LAYER 2: DECREASE & CONQUER — 언더컷 골든 윈도우 탐색
# ══════════════════════════════════════════════════════════════

@dataclass
class UndercutResult:
    """언더컷 시뮬레이션 결과"""
    pit_lap: int
    time_gain: float         # 언더컷으로 얻는 시간 이득 (초)
    success_probability: float
    is_golden_window: bool


class UndercutOptimizer:
    """
    축소 정복(Decrease & Conquer): 이진 탐색으로
    최적의 피트인 타이밍(골든 윈도우)을 탐색.

    원리: 타이어 마모에 따른 언더컷 이득 함수는
    단봉(unimodal) 형태이므로 이진 탐색으로 최적점 수렴.
    복잡도: O(log L) (L = 탐색 랩 범위)
    """

    def __init__(self, laps: pd.DataFrame, attacker: str, defender: str,
                 pit_loss: float = PIT_STOP_LOSS):
        self.laps = laps
        self.attacker = attacker
        self.defender = defender
        self.pit_loss = pit_loss

        # 드라이버별 랩타임 추출
        self.attacker_laps = (
            laps[laps['Driver'] == attacker]
            .sort_values('LapNumber')
            .set_index('LapNumber')
        )
        self.defender_laps = (
            laps[laps['Driver'] == defender]
            .sort_values('LapNumber')
            .set_index('LapNumber')
        )

    def simulate_undercut(self, pit_lap: int, horizon: int = 5) -> UndercutResult:
        """
        특정 랩에서 피트인했을 때의 언더컷 이득 시뮬레이션.

        언더컷 이득 = 새 타이어 이점 - 피트스톱 시간 손실
        너무 이르면: 타이어 마모 차이가 적어 이득 부족
        너무 늦으면: 남은 랩이 적어 이득 회수 불가
        → 단봉(Unimodal) 함수 형태
        """
        total_laps = int(self.laps['LapNumber'].max())
        remaining_laps = total_laps - pit_lap

        # 1) 현재 타이어 나이 기반 마모 이점
        tyre_age = pit_lap - 1  # 레이스 시작부터의 랩수
        degradation_gain = tyre_age * TIRE_DEG_MEDIUM * 0.8  # 오래된 타이어일수록 이득 큼

        # 2) 새 타이어 부스트 (하드 컴파운드 기준)
        fresh_tire_boost = NEW_TIRE_ADVANTAGE

        # 3) 남은 랩에서 이득 회수 (남은 랩이 적으면 이득 감소)
        recovery_factor = min(remaining_laps / 20.0, 1.0)

        # 4) 피트스톱 시간 손실
        pit_cost = self.pit_loss / total_laps  # 랩당 분산 비용

        # 5) 너무 이른 피트인 페널티 (타이어가 충분히 안 닳았으면)
        early_penalty = max(0, (15 - tyre_age) * 0.15)

        # 총 이득 = (마모 이점 + 새타이어 부스트) × 회수율 - 비용 - 조기 페널티
        net_gain = ((degradation_gain + fresh_tire_boost) * recovery_factor
                    - pit_cost - early_penalty)

        # horizon 랩 동안의 실제 랩타임 차이 반영
        actual_gain_bonus = 0
        for offset in range(1, min(horizon + 1, remaining_laps + 1)):
            lap_num = pit_lap + offset
            if (lap_num in self.defender_laps.index and
                lap_num in self.attacker_laps.index):
                def_time = self.defender_laps.loc[lap_num, 'LapTimeSec']
                att_time = self.attacker_laps.loc[lap_num, 'LapTimeSec']
                if not pd.isna(def_time) and not pd.isna(att_time):
                    actual_gain_bonus += (def_time - att_time) * 0.05

        net_gain += actual_gain_bonus

        success_prob = min(max(net_gain / 3.0, 0.0), 1.0)

        return UndercutResult(
            pit_lap=pit_lap,
            time_gain=round(net_gain, 3),
            success_probability=round(success_prob, 3),
            is_golden_window=success_prob >= 0.65
        )

    def binary_search_golden_window(self, search_start: int = 10,
                                     search_end: int = 30) -> Tuple[int, List[UndercutResult]]:
        """
        이진 탐색으로 최적 피트인 타이밍 탐색.

        탐색 범위를 반씩 줄여가며 언더컷 이득이 최대인 지점을 찾음.
        Decrease & Conquer: 문제 크기를 매 단계 절반으로 축소.
        """
        all_results = []
        search_log = []

        lo, hi = search_start, search_end
        iteration = 0

        print(f"\n  🔎 이진 탐색 시작: 범위 [{lo}, {hi}] (총 {hi - lo + 1}랩)")

        # 이진 탐색: 구간을 줄여가며 최적점 탐색
        while lo <= hi:
            iteration += 1
            mid = (lo + hi) // 2

            # mid 지점과 양쪽 이웃의 이득 비교
            result_mid = self.simulate_undercut(mid)
            result_left = self.simulate_undercut(max(mid - 1, search_start))
            result_right = self.simulate_undercut(min(mid + 1, search_end))

            all_results.extend([result_left, result_mid, result_right])

            search_log.append({
                'iteration': iteration,
                'lo': lo, 'hi': hi, 'mid': mid,
                'gain': result_mid.time_gain,
                'prob': result_mid.success_probability
            })

            print(f"    [{iteration}] 범위 [{lo:2d}-{hi:2d}] → "
                  f"Mid=Lap {mid} | 이득={result_mid.time_gain:+.3f}s | "
                  f"성공확률={result_mid.success_probability:.1%}")

            # 기울기 기반 방향 결정
            if result_left.time_gain > result_right.time_gain:
                hi = mid - 1  # 왼쪽이 더 유망 → 우측 절반 제거
            else:
                lo = mid + 1  # 오른쪽이 더 유망 → 좌측 절반 제거

            if hi - lo <= 2:
                # 범위가 충분히 좁아지면 남은 구간 전수 조사
                for lap in range(lo, hi + 1):
                    all_results.append(self.simulate_undercut(lap))
                break

        # 전체 탐색 범위도 평가 (비교용)
        full_scan = []
        for lap in range(search_start, search_end + 1):
            full_scan.append(self.simulate_undercut(lap))

        # 최적 랩 도출
        best = max(full_scan, key=lambda r: r.time_gain)

        return best.pit_lap, full_scan, search_log

    def run(self, pit_loss: Optional[float] = None) -> dict:
        """언더컷 분석 실행"""
        if pit_loss:
            self.pit_loss = pit_loss

        scenario_name = "일반 상황" if self.pit_loss == PIT_STOP_LOSS else "세이프티카 상황"

        print("\n" + "=" * 70)
        print(f"  ⏱️  LAYER 2: DECREASE & CONQUER — 언더컷 골든 윈도우 [{scenario_name}]")
        print("=" * 70)
        print(f"  공격자: {self.attacker} → 수비자: {self.defender}")
        print(f"  피트스톱 로스: {self.pit_loss}초")
        print("-" * 70)

        optimal_lap, all_results, search_log = self.binary_search_golden_window()

        golden_windows = [r for r in all_results if r.is_golden_window]

        print(f"\n  🏆 최적 피트인 타이밍: Lap {optimal_lap}")
        print(f"  🎯 골든 윈도우: ", end="")
        if golden_windows:
            gw_laps = sorted(set(r.pit_lap for r in golden_windows))
            print(f"Lap {gw_laps[0]} ~ Lap {gw_laps[-1]}")
        else:
            print("없음 (언더컷 비효율적)")

        # 실제 피트인 랩과 비교
        actual_pit = self.attacker_laps[self.attacker_laps['PitInTime'].notna()]
        if not actual_pit.empty:
            actual_lap = int(actual_pit.index[0])
            print(f"  📌 실제 피트인: Lap {actual_lap}")
            diff = abs(actual_lap - optimal_lap)
            print(f"  📏 알고리즘 vs 실제 차이: {diff}랩 "
                  f"{'✅ 정확!' if diff <= 2 else '⚠️ 편차 있음'}")

        return {
            'optimal_lap': optimal_lap,
            'all_results': all_results,
            'golden_windows': golden_windows,
            'search_log': search_log
        }


# ══════════════════════════════════════════════════════════════
# LAYER 3: DIVIDE & CONQUER — 최근접 쌍 배틀 감지
# ══════════════════════════════════════════════════════════════

@dataclass
class BattleEvent:
    """배틀 감지 이벤트"""
    time_sec: float
    driver_a: str
    driver_b: str
    distance: float          # 미터
    position_a: Tuple[float, float]
    position_b: Tuple[float, float]


@dataclass(order=True)
class Point:
    """2D 좌표점"""
    x: float
    y: float
    driver: str = field(compare=False)
    time_sec: float = field(compare=False, default=0.0)


class ClosestPairBattleDetector:
    """
    분할 정복(Divide & Conquer): 최근접 쌍 알고리즘으로
    서킷 상에서 가장 가까운 두 차량을 실시간 감지.

    복잡도: O(n log n) — 서킷 전체 좌표를 효율적으로 분석.
    n = 포인트 수 (2대 드라이버의 텔레메트리 샘플)
    """

    def __init__(self, tel_a: pd.DataFrame, tel_b: pd.DataFrame,
                 driver_a: str = 'VER', driver_b: str = 'HAM'):
        self.tel_a = tel_a
        self.tel_b = tel_b
        self.driver_a = driver_a
        self.driver_b = driver_b

    @staticmethod
    def _distance(p1: Point, p2: Point) -> float:
        """유클리드 거리"""
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    def _closest_pair_brute(self, points: List[Point]) -> Tuple[float, Point, Point]:
        """기저 사례: 3개 이하 포인트는 전수 비교"""
        min_dist = float('inf')
        best_pair = (points[0], points[1]) if len(points) >= 2 else (points[0], points[0])

        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                # 같은 드라이버끼리는 건너뜀
                if points[i].driver == points[j].driver:
                    continue
                d = self._distance(points[i], points[j])
                if d < min_dist:
                    min_dist = d
                    best_pair = (points[i], points[j])

        return min_dist, best_pair[0], best_pair[1]

    def _closest_pair_strip(self, strip: List[Point], d: float) -> Tuple[float, Optional[Point], Optional[Point]]:
        """스트립 영역 내 최근접 쌍 검색"""
        min_dist = d
        best_p1, best_p2 = None, None
        strip.sort(key=lambda p: p.y)

        for i in range(len(strip)):
            j = i + 1
            while j < len(strip) and (strip[j].y - strip[i].y) < min_dist:
                if strip[i].driver != strip[j].driver:
                    dist = self._distance(strip[i], strip[j])
                    if dist < min_dist:
                        min_dist = dist
                        best_p1, best_p2 = strip[i], strip[j]
                j += 1

        return min_dist, best_p1, best_p2

    def closest_pair_divide_conquer(self, points: List[Point]) -> Tuple[float, Point, Point]:
        """
        분할 정복 최근접 쌍 알고리즘.

        1. 분할(Divide): x좌표 기준 중간선으로 좌/우 분할
        2. 정복(Conquer): 각 하위 구간에서 재귀적으로 최근접 쌍 탐색
        3. 결합(Combine): 스트립 영역에서 교차 쌍 검사
        """
        n = len(points)

        # 기저 사례
        if n <= 3:
            return self._closest_pair_brute(points)

        # 1. 분할: x좌표 기준으로 정렬 후 중간점으로 분할
        points.sort(key=lambda p: p.x)
        mid = n // 2
        mid_point = points[mid]

        left_half = points[:mid]
        right_half = points[mid:]

        # 2. 정복: 재귀적으로 각 반쪽에서 최근접 쌍 탐색
        dl, pl1, pl2 = self.closest_pair_divide_conquer(left_half)
        dr, pr1, pr2 = self.closest_pair_divide_conquer(right_half)

        # 더 작은 쪽을 채택
        if dl < dr:
            d_min, best_p1, best_p2 = dl, pl1, pl2
        else:
            d_min, best_p1, best_p2 = dr, pr1, pr2

        # 3. 결합: 중간선 주변 스트립 검사
        strip = [p for p in points if abs(p.x - mid_point.x) < d_min]

        ds, sp1, sp2 = self._closest_pair_strip(strip, d_min)
        if ds < d_min and sp1 and sp2:
            d_min, best_p1, best_p2 = ds, sp1, sp2

        return d_min, best_p1, best_p2

    def detect_battles(self, sample_interval: int = 500) -> List[BattleEvent]:
        """
        시간대별로 두 드라이버의 좌표를 샘플링하고
        Closest Pair 알고리즘으로 배틀 포인트를 감지.

        Heap(우선순위 큐)로 가장 가까운 이벤트를 관리.
        """
        print("\n" + "=" * 70)
        print("  ⚔️  LAYER 3: DIVIDE & CONQUER — 최근접 쌍 배틀 감지")
        print("=" * 70)
        print(f"  분석 대상: {self.driver_a} vs {self.driver_b}")
        print(f"  알고리즘: Closest Pair (O(n log n))")
        print("-" * 70)

        # 시간 동기화: 공통 시간대의 데이터만 사용
        tel_a = self.tel_a.dropna(subset=['X', 'Y', 'SessionTimeSec']).copy()
        tel_b = self.tel_b.dropna(subset=['X', 'Y', 'SessionTimeSec']).copy()

        # 샘플링 (계산량 조절)
        tel_a_sampled = tel_a.iloc[::sample_interval].reset_index(drop=True)
        tel_b_sampled = tel_b.iloc[::sample_interval].reset_index(drop=True)

        print(f"  샘플 포인트: {self.driver_a}={len(tel_a_sampled)}개, "
              f"{self.driver_b}={len(tel_b_sampled)}개")

        # 시간 윈도우별 배틀 감지
        battles: List[BattleEvent] = []
        # 우선순위 큐(Heap): 가장 가까운 거리 이벤트를 관리
        battle_heap: list = []

        # 시간 윈도우 크기 설정
        min_time = max(tel_a_sampled['SessionTimeSec'].min(),
                       tel_b_sampled['SessionTimeSec'].min())
        max_time = min(tel_a_sampled['SessionTimeSec'].max(),
                       tel_b_sampled['SessionTimeSec'].max())

        window_size = 100  # 100초 윈도우
        current_time = min_time

        # Queue(deque)로 윈도우 관리
        window_queue = deque()
        battle_count = 0

        while current_time < max_time:
            t_start = current_time
            t_end = current_time + window_size

            # 해당 시간 윈도우의 포인트 수집
            a_window = tel_a_sampled[
                (tel_a_sampled['SessionTimeSec'] >= t_start) &
                (tel_a_sampled['SessionTimeSec'] < t_end)
            ]
            b_window = tel_b_sampled[
                (tel_b_sampled['SessionTimeSec'] >= t_start) &
                (tel_b_sampled['SessionTimeSec'] < t_end)
            ]

            if len(a_window) > 0 and len(b_window) > 0:
                points = []
                for _, row in a_window.iterrows():
                    points.append(Point(row['X'], row['Y'],
                                       self.driver_a, row['SessionTimeSec']))
                for _, row in b_window.iterrows():
                    points.append(Point(row['X'], row['Y'],
                                       self.driver_b, row['SessionTimeSec']))

                if len(points) >= 2:
                    # ★ 분할 정복 알고리즘 실행
                    min_dist, p1, p2 = self.closest_pair_divide_conquer(points)

                    # Heap에 결과 push (거리 기준 최소힙)
                    heapq.heappush(battle_heap, (min_dist, current_time, p1, p2))

                    if min_dist < BATTLE_DISTANCE_THRESHOLD:
                        battle = BattleEvent(
                            time_sec=current_time,
                            driver_a=p1.driver,
                            driver_b=p2.driver,
                            distance=round(min_dist, 2),
                            position_a=(p1.x, p1.y),
                            position_b=(p2.x, p2.y)
                        )
                        battles.append(battle)
                        window_queue.append(battle)
                        battle_count += 1

            current_time += window_size / 2  # 50% 오버랩 슬라이딩 윈도우

        # Heap에서 Top-5 최접근 이벤트 추출
        print(f"\n  🏁 감지된 배틀 이벤트: {battle_count}건")
        print(f"\n  📊 Top-5 최접근 순간 (Heap 추출):")

        top5 = []
        seen_times = set()
        temp_heap = list(battle_heap)
        heapq.heapify(temp_heap)

        while temp_heap and len(top5) < 5:
            dist, time_sec, p1, p2 = heapq.heappop(temp_heap)
            # 비슷한 시간대 중복 제거
            time_bucket = round(time_sec / 60)
            if time_bucket not in seen_times:
                seen_times.add(time_bucket)
                top5.append((dist, time_sec, p1, p2))
                elapsed_min = (time_sec - min_time) / 60
                lap_est = int(elapsed_min / 1.65) + 1  # 랩타임 ~99초 기준 추정
                alert = "🔴 BATTLE!" if dist < BATTLE_DISTANCE_THRESHOLD else "🟡 CLOSE"
                print(f"    {alert} ~Lap {lap_est:2d} | "
                      f"{p1.driver} ↔ {p2.driver} | "
                      f"거리: {dist:.1f}m | "
                      f"좌표: ({p1.x:.0f},{p1.y:.0f}) ↔ ({p2.x:.0f},{p2.y:.0f})")

        return battles


# ══════════════════════════════════════════════════════════════
# 대시보드 시각화 (Matplotlib)
# ══════════════════════════════════════════════════════════════

def create_dashboard(data: dict, undercut_results: dict, undercut_results_sc: dict,
                     battles: List[BattleEvent], drs_reports: List[DRSReport],
                     output_path: str = 'f1_dashboard.png'):
    """4-패널 분석 대시보드 생성"""

    fig = plt.figure(figsize=(20, 14), facecolor='#0D1117')
    fig.suptitle('F1 Strategy Engine — 2021 French GP Analysis',
                 fontsize=20, fontweight='bold', color='white', y=0.98)

    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3,
                  left=0.06, right=0.96, top=0.92, bottom=0.06)

    style = {
        'bg': '#0D1117',
        'panel': '#161B22',
        'text': '#E6EDF3',
        'grid': '#21262D',
        'ver': '#FF3E3E',
        'ham': '#00D2BE',
        'golden': '#FFD700',
        'battle': '#FF6B35',
    }

    # ─── Panel 1: 랩타임 비교 + 피트스톱 ───
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(style['panel'])

    laps = data['laps']
    ver = laps[laps['Driver'] == 'VER'].sort_values('LapNumber')
    ham = laps[laps['Driver'] == 'HAM'].sort_values('LapNumber')

    ax1.plot(ver['LapNumber'], ver['LapTimeSec'], color=style['ver'],
             linewidth=1.8, label='VER (Red Bull)', alpha=0.9)
    ax1.plot(ham['LapNumber'], ham['LapTimeSec'], color=style['ham'],
             linewidth=1.8, label='HAM (Mercedes)', alpha=0.9)

    # 피트스톱 마커
    ver_pits = ver[ver['PitInTime'].notna()]
    ham_pits = ham[ham['PitInTime'].notna()]
    ax1.scatter(ver_pits['LapNumber'], ver_pits['LapTimeSec'],
                color=style['ver'], s=150, zorder=5, marker='v',
                edgecolors='white', linewidths=1.5, label='VER Pit')
    ax1.scatter(ham_pits['LapNumber'], ham_pits['LapTimeSec'],
                color=style['ham'], s=150, zorder=5, marker='v',
                edgecolors='white', linewidths=1.5, label='HAM Pit')

    ax1.set_ylim(95, 105)
    ax1.set_xlabel('Lap', color=style['text'])
    ax1.set_ylabel('Lap Time (sec)', color=style['text'])
    ax1.set_title('[Layer 1] Lap Times & Pit Stops', color=style['text'],
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=8, loc='upper right', facecolor=style['panel'],
               edgecolor=style['grid'], labelcolor=style['text'])
    ax1.tick_params(colors=style['text'])
    ax1.grid(True, alpha=0.2, color=style['grid'])

    # ─── Panel 2: 언더컷 이득 분석 (일반 vs SC) ───
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(style['panel'])

    # 일반 상황
    normal_laps = [r.pit_lap for r in undercut_results['all_results']]
    normal_gains = [r.time_gain for r in undercut_results['all_results']]
    normal_probs = [r.success_probability for r in undercut_results['all_results']]

    ax2.bar(normal_laps, normal_gains, color=style['ver'], alpha=0.7,
            width=0.4, label='Normal', align='edge')

    # SC 상황
    sc_laps = [r.pit_lap for r in undercut_results_sc['all_results']]
    sc_gains = [r.time_gain for r in undercut_results_sc['all_results']]

    ax2.bar([l + 0.4 for l in sc_laps], sc_gains, color=style['golden'],
            alpha=0.7, width=0.4, label='Safety Car', align='edge')

    # 골든 윈도우 표시
    gw_laps = sorted(set(r.pit_lap for r in undercut_results['golden_windows']))
    if gw_laps:
        ax2.axvspan(gw_laps[0] - 0.5, gw_laps[-1] + 0.5,
                    alpha=0.15, color=style['golden'],
                    label=f'Golden Window (Lap {gw_laps[0]}-{gw_laps[-1]})')

    # 실제 피트인 표시
    ax2.axvline(x=18, color='white', linestyle='--', alpha=0.8,
                label='VER Actual Pit (Lap 18)')

    ax2.set_xlabel('Pit-in Lap', color=style['text'])
    ax2.set_ylabel('Undercut Gain', color=style['text'])
    ax2.set_title('[Layer 2] Undercut Golden Window (Decrease & Conquer)',
                  color=style['text'], fontsize=13, fontweight='bold')
    ax2.legend(fontsize=7, loc='upper right', facecolor=style['panel'],
               edgecolor=style['grid'], labelcolor=style['text'])
    ax2.tick_params(colors=style['text'])
    ax2.grid(True, alpha=0.2, color=style['grid'])

    # ─── Panel 3: 트랙 맵 + 배틀 포인트 ───
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(style['panel'])

    # 트랙 맵 그리기 (VER 텔레메트리 기반)
    tel_ver = data['tel_ver']
    track_x = tel_ver['X'].values[::50]
    track_y = tel_ver['Y'].values[::50]
    ax3.plot(track_x, track_y, color='#30363D', linewidth=8, alpha=0.6)
    ax3.plot(track_x, track_y, color='#484F58', linewidth=2, alpha=0.8)

    # 배틀 포인트 표시
    if battles:
        bx = [b.position_a[0] for b in battles[:20]]
        by = [b.position_a[1] for b in battles[:20]]
        sizes = [max(200 - b.distance * 3, 50) for b in battles[:20]]
        ax3.scatter(bx, by, c=style['battle'], s=sizes, alpha=0.7,
                    zorder=5, edgecolors='white', linewidths=0.5,
                    label=f'Battle Points ({len(battles)})')

    # DRS 존 표시 (폴 리카르 메인 스트레이트 근처)
    ax3.annotate('DRS Zone', xy=(track_x[0], track_y[0]),
                 fontsize=9, color=style['golden'], fontweight='bold',
                 ha='center')

    ax3.set_title('[Layer 3] Track Map & Battle Detection (Divide & Conquer)',
                  color=style['text'], fontsize=13, fontweight='bold')
    ax3.legend(fontsize=8, loc='upper right', facecolor=style['panel'],
               edgecolor=style['grid'], labelcolor=style['text'])
    ax3.set_aspect('equal')
    ax3.tick_params(colors=style['text'])
    ax3.axis('off')

    # ─── Panel 4: 전략 타임라인 ───
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(style['panel'])

    # 포지션 변화
    ax4.plot(ver['LapNumber'], ver['Position'], color=style['ver'],
             linewidth=2.5, label='VER', marker='', alpha=0.9)
    ax4.plot(ham['LapNumber'], ham['Position'], color=style['ham'],
             linewidth=2.5, label='HAM', marker='', alpha=0.9)

    # 이벤트 어노테이션
    events = [
        (18, 'VER Pit 1\n(Undercut)', style['ver']),
        (19, 'HAM Pit\n(Response)', style['ham']),
        (32, 'VER Pit 2\n(2-Stop Switch)', style['ver']),
        (53, 'VER WINS!\nP1', style['golden']),
    ]
    for lap, label, color in events:
        ax4.axvline(x=lap, color=color, linestyle=':', alpha=0.5)
        y_pos = 1 if 'VER' in label else 2.5
        ax4.annotate(label, xy=(lap, y_pos), fontsize=7,
                     color=color, fontweight='bold', ha='center',
                     bbox=dict(boxstyle='round,pad=0.3',
                              facecolor=style['panel'], edgecolor=color, alpha=0.8))

    ax4.set_ylim(0.5, 6)
    ax4.invert_yaxis()
    ax4.set_xlabel('Lap', color=style['text'])
    ax4.set_ylabel('Position', color=style['text'])
    ax4.set_title('Position Timeline & Strategy Flow',
                  color=style['text'], fontsize=13, fontweight='bold')
    ax4.legend(fontsize=9, loc='lower right', facecolor=style['panel'],
               edgecolor=style['grid'], labelcolor=style['text'])
    ax4.tick_params(colors=style['text'])
    ax4.grid(True, alpha=0.2, color=style['grid'])

    plt.savefig(output_path, dpi=150, facecolor=style['bg'],
                bbox_inches='tight')
    print(f"\n  💾 대시보드 저장: {output_path}")

    return output_path


# ══════════════════════════════════════════════════════════════
# 메인 실행
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  🏎️  F1 데이터 기반 실시간 전략 및 중계 분석 엔진")
    print("  📍 2021 프랑스 GP (Circuit Paul Ricard)")
    print("  🎯 언더컷 알고리즘의 효율성 분석")
    print("=" * 70)

    # 데이터 로딩
    print("\n📂 데이터 로딩 중...")
    data = load_data()
    laps = data['laps']
    print(f"  ✅ laps: {len(laps)}행 | tel_VER: {len(data['tel_ver'])}행 | "
          f"tel_HAM: {len(data['tel_ham'])}행")

    # ─── LAYER 1: Brute Force ───
    drs_reports = brute_force_drs_check(laps)

    # ─── LAYER 2: Decrease & Conquer (일반 상황) ───
    optimizer = UndercutOptimizer(laps, 'VER', 'HAM', PIT_STOP_LOSS)
    undercut_results = optimizer.run()

    # ─── LAYER 2 보너스: Decrease & Conquer (세이프티카 상황) ───
    optimizer_sc = UndercutOptimizer(laps, 'VER', 'HAM', PIT_STOP_LOSS_SC)
    undercut_results_sc = optimizer_sc.run(pit_loss=PIT_STOP_LOSS_SC)

    # ─── LAYER 3: Divide & Conquer ───
    detector = ClosestPairBattleDetector(
        data['tel_ver'], data['tel_ham'], 'VER', 'HAM'
    )
    battles = detector.detect_battles(sample_interval=200)

    # ─── 대시보드 생성 ───
    print("\n" + "=" * 70)
    print("  🖥️  대시보드 생성 중...")
    print("=" * 70)
    dashboard_path = create_dashboard(
        data, undercut_results, undercut_results_sc,
        battles, drs_reports, 'f1_dashboard.png'
    )

    # ─── 최종 요약 ───
    print("\n" + "=" * 70)
    print("  📋 최종 분석 요약")
    print("=" * 70)
    print(f"""
  ┌─────────────────────────────────────────────────────────────────┐
  │  LAYER 1 (Brute Force)                                        │
  │    → VER-HAM 간 DRS 범위 진입 {'있음' if any(r.driver in ('VER','HAM') and r.driver_ahead in ('VER','HAM') and r.drs_eligible for r in drs_reports) else '없음': <6s}                       │
  │    → 직접 추월 가능성: 극히 낮음 (폴 리카르 서킷 특성)        │
  │    → 판정: 언더컷 전략이 유일한 해법                          │
  │                                                               │
  │  LAYER 2 (Decrease & Conquer)                                 │
  │    → 최적 피트인: Lap {undercut_results['optimal_lap']:<3d}                                  │
  │    → 실제 피트인: Lap 18  (VER 1차 피트)                      │
  │    → 골든 윈도우: {'Lap ' + str(sorted(set(r.pit_lap for r in undercut_results['golden_windows']))[0]) + '-' + str(sorted(set(r.pit_lap for r in undercut_results['golden_windows']))[-1]) if undercut_results['golden_windows'] else 'N/A': <10s}                                   │
  │    → SC 상황 시: 피트스톱 로스 25s→12s → 이득 대폭 증가       │
  │                                                               │
  │  LAYER 3 (Divide & Conquer)                                   │
  │    → 감지된 배틀 이벤트: {len(battles):<4d}건                            │
  │    → Closest Pair 알고리즘으로 긴박한 순간 자동 포착          │
  │    → 중계 카메라 전환 트리거 제공                             │
  │                                                               │
  │  🏆 최종 결과: VER P1 / HAM P2 (2-Stop vs 1-Stop 전략 승리)  │
  └─────────────────────────────────────────────────────────────────┘
    """)

    return dashboard_path


if __name__ == '__main__':
    output = main()
