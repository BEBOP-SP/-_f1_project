"""
Playwright verification script for 2단계/3단계 algorithm features.
France 2021 GP and Abu Dhabi 2021 GP.
"""
import asyncio, json, time
from playwright.async_api import async_playwright

BASE = "http://127.0.0.1:8000/f1_viewer.html"
TIMEOUT = 15000

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport={"width":1600,"height":900})
        page = await ctx.new_page()

        results = {}

        for race_key, race_label in [("france2021", "France 2021"), ("abudhabi2021", "Abu Dhabi 2021")]:
            print(f"\n{'='*60}")
            print(f"  {race_label}")
            print(f"{'='*60}")

            await page.goto(BASE, wait_until="networkidle", timeout=30000)
            await page.wait_for_timeout(2000)

            # Switch race if Abu Dhabi
            if race_key == "abudhabi2021":
                btn = page.locator("#race-btn")
                if await btn.count():
                    await btn.click()
                    await page.wait_for_timeout(500)
                    menu_item = page.locator(f"[data-race='{race_key}'], #race-menu [onclick*='{race_key}']")
                    if await menu_item.count():
                        await menu_item.first.click()
                        await page.wait_for_timeout(3000)
                    else:
                        # Try text-based click
                        await page.get_by_text("ABU DHABI", exact=False).first.click()
                        await page.wait_for_timeout(3000)

            # Play the race
            play_btn = page.locator("#play-btn, button[onclick*='togglePlay'], .play-btn")
            if await play_btn.count():
                await play_btn.first.click()
                print("  ▶ Play started")
            else:
                # Try keyboard space
                await page.keyboard.press("Space")
                print("  ▶ Play via Space")

            # Speed up simulation (5x or 10x)
            speed_btn = page.locator("#speed-btn, [onclick*='speed'], .speed-btn")
            for _ in range(3):
                if await speed_btn.count():
                    await speed_btn.first.click()
                    await page.wait_for_timeout(200)

            print("  [wait] 12s for race to progress...")
            await page.wait_for_timeout(12000)

            # ── Check 1: Battle banner (분할 정복) ─────────────────────
            banner = page.locator("#battle-trigger-banner")
            banner_visible = await banner.is_visible() if await banner.count() else False
            banner_text = (await banner.text_content() or "").strip() if await banner.count() else ""
            print(f"\n  [분할정복] battle-trigger-banner:")
            print(f"    visible={banner_visible}  text='{banner_text}'")

            # ── Check 2: DRS tag (억지 기법) ────────────────────────────
            battles_list = page.locator("#battles-list")
            battles_html = await battles_list.inner_html() if await battles_list.count() else ""
            drs_present = "DRS" in battles_html
            print(f"\n  [억지기법] battles-list DRS tag: {drs_present}")
            print(f"    list html snippet: {battles_html[:200]}")

            # ── Check 3: DP ETA (동적 계획법) ───────────────────────────
            dp_rows = page.locator("#dp-eta-rows")
            dp_html = await dp_rows.inner_html() if await dp_rows.count() else ""
            dp_has_data = bool(dp_html.strip()) and dp_html.strip() != "—" and "VER" in dp_html or "HAM" in dp_html or "LAP" in dp_html.upper() or len(dp_html.strip()) > 10
            print(f"\n  [동적계획법] dp-eta-rows:")
            print(f"    html='{dp_html[:300]}'")

            # ── Check 4: B&B pit panel (분기한정) ───────────────────────
            ucp = page.locator("#undercut-panel, .undercut-panel, [id*='ucp']")
            bb_panel = page.locator("#ucp-prob-val, .ucp-prob-val")
            bb_visible = await bb_panel.is_visible() if await bb_panel.count() else False
            bb_text = (await bb_panel.text_content() or "").strip() if await bb_panel.count() else ""

            # Also check the pit golden time label
            golden_lbl = page.locator("#ucp-sc-verdict, [id*='ucp']")
            golden_html = ""
            if await golden_lbl.count():
                golden_html = await golden_lbl.first.inner_html()

            print(f"\n  [분기한정] B&B 피트 골든타임 패널:")
            print(f"    prob-val visible={bb_visible}  text='{bb_text}'")

            # Select VER to trigger BB panel
            await page.evaluate("selectedDriver = 'VER'; if(typeof updateUndercutPanel==='function') updateUndercutPanel('VER');")
            await page.wait_for_timeout(1000)
            bb_text2 = (await bb_panel.text_content() or "").strip() if await bb_panel.count() else ""
            print(f"    after selecting VER: '{bb_text2}'")

            await page.screenshot(path=f"verify_{race_key}.png")
            print(f"\n  📸 Screenshot saved: verify_{race_key}.png")

            results[race_label] = {
                "battle_banner_visible": banner_visible,
                "battle_banner_text": banner_text,
                "drs_tag_present": drs_present,
                "dp_eta_html": dp_html[:200],
                "bb_panel_text": bb_text2,
            }

        await browser.close()

        print(f"\n\n{'='*60}")
        print("  SUMMARY")
        print(f"{'='*60}")
        for race, r in results.items():
            print(f"\n  {race}:")
            print(f"    분할정복 배너: {'✅' if r['battle_banner_visible'] else '❌'}  '{r['battle_banner_text']}'")
            print(f"    억지기법 DRS: {'✅' if r['drs_tag_present'] else '—'}")
            print(f"    동적계획법 ETA: {'✅' if r['dp_eta_html'] else '❌'}")
            print(f"    분기한정 B&B: {'✅' if r['bb_panel_text'] else '❌'}  '{r['bb_panel_text']}'")

asyncio.run(run())
