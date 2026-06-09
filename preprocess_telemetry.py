"""
F1 2021 French GP — Full Telemetry Preprocessor
모든 드라이버의 텔레메트리를 FastF1 캐시에서 추출해
data/telemetry_full.json 으로 저장합니다.

실행: python preprocess_telemetry.py
"""

import os, json, math
import fastf1
import pandas as pd
import numpy as np

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE, 'f1_cache')
OUT_PATH  = os.path.join(BASE, 'data', 'telemetry_full.json')

TEAM_COLORS = {
    'Red Bull Racing':'#FF3E3E','Mercedes':'#00D2BE','McLaren':'#FF8700',
    'Ferrari':'#DC0000','Alpine':'#0090FF','AlphaTauri':'#4E7C9B',
    'Aston Martin':'#006F62','Williams':'#005AFF',
    'Alfa Romeo Racing':'#900000','Haas F1 Team':'#787878'
}
TEAM_SHORT = {
    'Red Bull Racing':'RBR','Mercedes':'MER','McLaren':'MCL','Ferrari':'FER',
    'Alpine':'ALP','AlphaTauri':'ATR','Aston Martin':'AMR','Williams':'WIL',
    'Alfa Romeo Racing':'ALF','Haas F1 Team':'HAS'
}

# ── FastF1 세션 로드 ──────────────────────────────────────────────────────────
print("▶ FastF1 캐시 로드 중...")
fastf1.Cache.enable_cache(CACHE_DIR)
session = fastf1.get_session(2021, 'French Grand Prix', 'R')
session.load(telemetry=True, weather=False, messages=False)
print(f"  드라이버 {len(session.drivers)}명 로드 완료")

# ── 트랙 좌표 추출 (VER 3랩 기준) ───────────────────────────────────────────
print("▶ 트랙 좌표 추출 중...")
ver_laps = session.laps.pick_driver('VER').sort_values('LapNumber')
lap3     = ver_laps[ver_laps['LapNumber'] == 3].iloc[0]
tel3     = lap3.get_telemetry()
# 300 포인트로 샘플링
step  = max(1, len(tel3) // 300)
track = [[round(float(r.X), 1), round(float(r.Y), 1)]
         for _, r in tel3.iloc[::step].iterrows()]

# 누적 거리 → 정규화
cd = [0.0]
for i in range(1, len(track)):
    dx = track[i][0] - track[i-1][0]
    dy = track[i][1] - track[i-1][1]
    cd.append(cd[-1] + math.sqrt(dx*dx + dy*dy))
ndist = [round(d / cd[-1], 6) for d in cd]

# ── 세션 기준 시간 ────────────────────────────────────────────────────────────
all_laps = session.laps.copy()
def td_sec(td):
    try:
        return td.total_seconds() if pd.notna(td) else None
    except:
        return None

all_laps['LapStartSec']   = all_laps['LapStartTime'].apply(td_sec)
all_laps['SessionTimeSec']= all_laps['Time'].apply(td_sec)
all_laps['LapTimeSec']    = all_laps['LapTime'].apply(td_sec)
session_start = all_laps['LapStartSec'].min()
session_end   = all_laps['SessionTimeSec'].max()

# ── 드라이버별 처리 ───────────────────────────────────────────────────────────
drivers_info = []
driver_laps  = {}
driver_history = {}
driver_telemetry = {}   # ★ 새로 추가: 샘플링된 텔레메트리

driver_abbrs = sorted(session.drivers)

for drv_num in driver_abbrs:
    try:
        abbr = session.get_driver(drv_num)['Abbreviation']
    except:
        abbr = drv_num

    drv_laps = all_laps[all_laps['Driver'] == abbr].sort_values('LapNumber')
    if drv_laps.empty:
        drv_laps = all_laps[all_laps['DriverNumber'] == drv_num].sort_values('LapNumber')
    if drv_laps.empty:
        print(f"  ⚠  {abbr} 랩 데이터 없음 — 스킵")
        continue

    team = drv_laps.iloc[0].get('Team', '')
    drivers_info.append({
        'name':     abbr,
        'team':     TEAM_SHORT.get(team, team[:3] if team else '???'),
        'color':    TEAM_COLORS.get(team, '#787878'),
        'fullTeam': team
    })

    # ── 랩 테이블 ──────────────────────────────────────────────────────────
    ll = []
    for _, r in drv_laps.iterrows():
        s = r['LapStartSec']; e = r['SessionTimeSec']
        if s is None or e is None: continue
        comp = r.get('Compound', 'M')
        comp = (comp[0] if isinstance(comp, str) and comp else 'M')
        ll.append([
            round(s, 2), round(e, 2),
            int(r['LapNumber']),
            int(r['Position']) if pd.notna(r.get('Position')) else 20,
            comp,
            int(r['TyreLife'])  if pd.notna(r.get('TyreLife'))  else 0,
            round(float(r['SpeedST']), 0) if pd.notna(r.get('SpeedST')) else 280,
            1 if pd.notna(r.get('PitInTime')) else 0,
            round(r['LapTimeSec'], 3) if r['LapTimeSec'] and r['LapTimeSec'] < 130 else 0
        ])
    driver_laps[abbr] = ll

    # ── 히스토리 ──────────────────────────────────────────────────────────
    h = []
    for _, r in drv_laps.iterrows():
        lt = r['LapTimeSec']
        comp = r.get('Compound', 'M')
        comp = (comp[0] if isinstance(comp, str) and comp else 'M')
        h.append({
            'l': int(r['LapNumber']),
            't': round(lt, 3) if lt and lt < 130 else 0,
            'p': int(r['Position']) if pd.notna(r.get('Position')) else 20,
            'c': comp,
            'tl': int(r['TyreLife']) if pd.notna(r.get('TyreLife')) else 0,
            's': round(float(r['SpeedST']), 0) if pd.notna(r.get('SpeedST')) else 0,
            'pit': 1 if pd.notna(r.get('PitInTime')) else 0
        })
    driver_history[abbr] = h

    # ── ★ 텔레메트리 (Speed, Throttle, Brake, Gear, X, Y, SessionTime) ──
    print(f"  텔레메트리 처리 중: {abbr}", end=' ', flush=True)
    tel_rows = []
    try:
        for _, lap_row in drv_laps.iterrows():
            try:
                tel = lap_row.get_telemetry()
            except Exception:
                continue
            if tel is None or tel.empty:
                continue

            # 4Hz → 약 0.25초 간격으로 샘플링 (데이터 크기 감소)
            SAMPLE_EVERY = 2   # 원본 ~14Hz → ~7Hz
            tel = tel.iloc[::SAMPLE_EVERY].copy()

            for _, t in tel.iterrows():
                st = t['SessionTime']
                if pd.isna(st): continue
                sec = st.total_seconds() if hasattr(st, 'total_seconds') else float(st)
                tel_rows.append([
                    round(sec, 3),
                    round(float(t['X']), 1),
                    round(float(t['Y']), 1),
                    int(round(float(t['Speed']))),
                    int(round(float(t['Throttle']))),
                    1 if bool(t['Brake']) else 0,
                    int(t['nGear']) if pd.notna(t['nGear']) else 1,
                    int(t['DRS'])   if pd.notna(t.get('DRS')) else 0,
                ])

        print(f"→ {len(tel_rows)} 샘플")
    except Exception as e:
        print(f"→ 실패: {e}")

    driver_telemetry[abbr] = tel_rows

# ── JSON 직렬화 & 저장 ────────────────────────────────────────────────────────
print("▶ JSON 저장 중...")
out = {
    'track':           track,
    'ndist':           ndist,
    'drivers':         drivers_info,
    'driverLaps':      driver_laps,
    'driverHistory':   driver_history,
    'driverTelemetry': driver_telemetry,   # ★
    'sessionStart':    round(session_start, 2),
    'sessionEnd':      round(session_end,   2),
}

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(out, f, separators=(',', ':'))

size_mb = os.path.getsize(OUT_PATH) / 1024 / 1024
print(f"✅ 저장 완료: {OUT_PATH}  ({size_mb:.1f} MB)")
print(f"   드라이버: {len(drivers_info)}명")
print(f"   트랙 포인트: {len(track)}개")
for d, rows in driver_telemetry.items():
    print(f"   {d}: {len(rows)} 샘플")
