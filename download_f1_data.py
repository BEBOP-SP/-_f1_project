import fastf1
import os

# 캐시 폴더 먼저 생성
os.makedirs('f1_cache', exist_ok=True)
fastf1.Cache.enable_cache('f1_cache')

# 2021 프랑스 GP 레이스 세션 로드
session = fastf1.get_session(2021, 'France', 'R')
session.load()

# 저장할 폴더
os.makedirs('data', exist_ok=True)

# 1) 전체 랩 데이터 (드라이버별 랩타임, 타이어, 피트 여부 등)
laps = session.laps
laps.to_csv('data/laps.csv', index=False)

# 2) 레이스 결과
results = session.results
results.to_csv('data/results.csv', index=False)

# 3) 텔레메트리 (VER, HAM만 — 용량 커서 주요 드라이버만)
for driver in ['VER', 'HAM']:
    tel = laps.pick_drivers(driver).get_telemetry()
    tel.to_csv(f'data/telemetry_{driver}.csv', index=False)

print("완료! data 폴더 확인해봐")