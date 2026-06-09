"""
F1 2021 Abu Dhabi GP — Full Telemetry Preprocessor
세이프티카 이벤트 포함
실행: python preprocess_abu_dhabi.py
"""

import os, json, math
import fastf1
import pandas as pd
import numpy as np

BASE     = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE, 'f1_cache')
OUT_PATH  = os.path.join(BASE, 'data', 'telemetry_abu2021.json')

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

print("▶ FastF1 캐시 로드 중...")
fastf1.Cache.enable_cache(CACHE_DIR)
session = fastf1.get_session(2021, 'Abu Dhabi Grand Prix', 'R')
session.load(telemetry=True, weather=False, messages=True)
print(f"  드라이버 {len(session.drivers)}명 로드 완료")

# ── 트랙 좌표 (VER 3랩 기준) ─────────────────────────────────────────────
print("▶ 트랙 좌표 추출 중...")
ver_laps = session.laps.pick_driver('VER').sort_values('LapNumber')
lap3     = ver_laps[ver_laps['LapNumber'] == 3].iloc[0]
tel3     = lap3.get_telemetry()
step     = max(1, len(tel3) // 300)
track    = [[round(float(r.X), 1), round(float(r.Y), 1)]
            for _, r in tel3.iloc[::step].iterrows()]

cd = [0.0]
for i in range(1, len(track)):
    dx = track[i][0] - track[i-1][0]
    dy = track[i][1] - track[i-1][1]
    cd.append(cd[-1] + math.sqrt(dx*dx + dy*dy))
ndist = [round(d / cd[-1], 6) for d in cd]

# ── 세션 기준 시간 ───────────────────────────────────────────────────────
all_laps = session.laps.copy()
def td_sec(td):
    try: return td.total_seconds() if pd.notna(td) else None
    except: return None

all_laps['LapStartSec']    = all_laps['LapStartTime'].apply(td_sec)
all_laps['SessionTimeSec'] = all_laps['Time'].apply(td_sec)
all_laps['LapTimeSec']     = all_laps['LapTime'].apply(td_sec)
session_start = all_laps['LapStartSec'].min()
session_end   = all_laps['SessionTimeSec'].max()

# ── 세이프티카 이벤트 추출 ───────────────────────────────────────────────
print(">> 세이프티카 이벤트 추출 중...")
sc_events = []

def to_sec(val):
    """timedelta / Timestamp / float 모두 세션 초 단위로 변환."""
    import datetime
    if val is None or (hasattr(val, '__class__') and 'NaT' in type(val).__name__):
        return None
    if isinstance(val, datetime.timedelta):
        return val.total_seconds()
    if hasattr(val, 'total_seconds'):
        return val.total_seconds()
    try:
        return float(val)
    except Exception:
        return None

try:
    msgs = session.race_control_messages
    if msgs is not None and not msgs.empty:
        for _, m in msgs.iterrows():
            msg  = str(m.get('Message', '')).upper()
            flag = str(m.get('Flag',    '')).upper()
            if 'SAFETY CAR' in msg or 'VIRTUAL SAFETY CAR' in msg or flag in ('SC', 'VSC'):
                sec = to_sec(m.get('Time'))
                if sec is None:
                    continue
                sc_type = 'VSC' if 'VIRTUAL' in msg else 'SC'
                action  = 'END' if ('WITHDRAWN' in msg or 'ENDING' in msg or 'IN THIS LAP' in msg) else 'DEPLOY'
                sc_events.append({'t': round(sec, 2), 'type': sc_type, 'action': action, 'msg': str(m.get('Message', ''))})
                print(f"  {sc_type} {action} @ {sec:.1f}s")
except Exception as e:
    print(f"  [!] race_control_messages 실패: {e}")

# TrackStatus 백업: 랩별 트랙 상태로 SC 구간 보완
try:
    sc_laps = []
    for _, lap in all_laps.drop_duplicates('LapNumber').sort_values('LapNumber').iterrows():
        ts = str(lap.get('TrackStatus', ''))
        if '4' in ts or '6' in ts:   # 4=SC, 6=Red
            sc_laps.append(int(lap['LapNumber']))
    if sc_laps and not sc_events:
        print(f"  TrackStatus SC 랩 (백업): {sc_laps}")
        for lap_n in sc_laps:
            rows = all_laps[all_laps['LapNumber'] == lap_n]
            if rows.empty: continue
            sec = rows.iloc[0]['LapStartSec']
            if sec: sc_events.append({'t': round(sec, 2), 'type': 'SC', 'action': 'DEPLOY', 'msg': f'SC lap {lap_n}'})
except Exception as e:
    print(f"  [!] TrackStatus 백업 실패: {e}")

print(f"  SC 이벤트 {len(sc_events)}개 추출 완료")

# ── 드라이버별 처리 ──────────────────────────────────────────────────────
drivers_info     = []
driver_laps      = {}
driver_history   = {}
driver_telemetry = {}

for drv_num in sorted(session.drivers):
    try:    abbr = session.get_driver(drv_num)['Abbreviation']
    except: abbr = drv_num

    drv_laps = all_laps[all_laps['Driver'] == abbr].sort_values('LapNumber')
    if drv_laps.empty:
        drv_laps = all_laps[all_laps['DriverNumber'] == drv_num].sort_values('LapNumber')
    if drv_laps.empty:
        print(f"  [!] {abbr} 랩 데이터 없음 - 스킵")
        continue

    team = drv_laps.iloc[0].get('Team', '')
    drivers_info.append({
        'name':     abbr,
        'team':     TEAM_SHORT.get(team, team[:3] if team else '???'),
        'color':    TEAM_COLORS.get(team, '#787878'),
        'fullTeam': team
    })

    ll = []
    for _, r in drv_laps.iterrows():
        s = r['LapStartSec']; e = r['SessionTimeSec']
        if s is None or e is None: continue
        comp = r.get('Compound', 'M')
        comp = comp[0] if isinstance(comp, str) and comp else 'M'
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

    h = []
    for _, r in drv_laps.iterrows():
        lt   = r['LapTimeSec']
        comp = r.get('Compound', 'M')
        comp = comp[0] if isinstance(comp, str) and comp else 'M'
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

    print(f"  텔레메트리 처리 중: {abbr}", end=' ', flush=True)
    tel_rows = []
    try:
        for _, lap_row in drv_laps.iterrows():
            try:    tel = lap_row.get_telemetry()
            except: continue
            if tel is None or tel.empty: continue
            tel = tel.iloc[::2].copy()
            for _, t in tel.iterrows():
                st = t['SessionTime']
                if pd.isna(st): continue
                sec = st.total_seconds() if hasattr(st, 'total_seconds') else float(st)
                tel_rows.append([
                    round(sec, 3),
                    round(float(t['X']), 1), round(float(t['Y']), 1),
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

# ── 저장 ────────────────────────────────────────────────────────────────
print("▶ JSON 저장 중...")
out = {
    'track':           track,
    'ndist':           ndist,
    'drivers':         drivers_info,
    'driverLaps':      driver_laps,
    'driverHistory':   driver_history,
    'driverTelemetry': driver_telemetry,
    'sessionStart':    round(session_start, 2),
    'sessionEnd':      round(session_end,   2),
    'scEvents':        sc_events,   # 세이프티카 이벤트
    'raceMeta': {
        'year': 2021, 'round': 22,
        'name': '2021 ABU DHABI GRAND PRIX',
        'circuit': 'Yas Marina Circuit',
        'totalLaps': 58,
    },
}

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(out, f, separators=(',', ':'))

size_mb = os.path.getsize(OUT_PATH) / 1024 / 1024
print(f"[OK] 저장 완료: {OUT_PATH}  ({size_mb:.1f} MB)")
print(f"   드라이버: {len(drivers_info)}명  |  SC 이벤트: {len(sc_events)}개")
