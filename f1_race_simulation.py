"""
F1 Race Replay helper for the standalone HTML frontend.

The Streamlit frontend was removed. Use f1_viewer.html as the UI and
data/telemetry_full.json as the replay data source.

Examples:
    python f1_race_simulation.py
    python f1_race_simulation.py --serve
"""

from __future__ import annotations

import argparse
import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from f1_algorithms import OUTPUT_FILE as ALGORITHM_FILE
from f1_algorithms import ensure_algorithm_cache


BASE_DIR = Path(__file__).resolve().parent
HTML_FILE = BASE_DIR / "f1_viewer.html"
DATA_FILE = BASE_DIR / "data" / "telemetry_full.json"
DEFAULT_PORT = 8000


def load_data_summary() -> dict[str, object]:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"{DATA_FILE} does not exist. Run preprocess_telemetry.py first."
        )

    with DATA_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)

    drivers = data.get("drivers", [])
    telemetry = data.get("driverTelemetry", {})
    samples = sum(len(rows) for rows in telemetry.values() if isinstance(rows, list))

    return {
        "drivers": len(drivers),
        "telemetry_samples": samples,
        "session_start": data.get("sessionStart"),
        "session_end": data.get("sessionEnd"),
        "data_size_mb": DATA_FILE.stat().st_size / 1024 / 1024,
        "algorithm_cache": ALGORITHM_FILE.exists(),
        "algorithm_size_mb": ALGORITHM_FILE.stat().st_size / 1024 / 1024
        if ALGORITHM_FILE.exists()
        else 0,
    }


def print_status() -> None:
    ensure_algorithm_cache()
    summary = load_data_summary()
    print("Standalone HTML frontend is ready.")
    print(f"HTML: {HTML_FILE}")
    print(f"Data: {DATA_FILE}")
    print(f"Drivers: {summary['drivers']}")
    print(f"Telemetry samples: {summary['telemetry_samples']}")
    print(f"Data size: {summary['data_size_mb']:.1f} MB")
    print(f"Algorithm cache: {ALGORITHM_FILE}")
    print(f"Algorithm cache size: {summary['algorithm_size_mb']:.1f} MB")
    print()
    print("Run with --serve to open it through a local HTTP server.")


def serve(port: int) -> None:
    if not HTML_FILE.exists():
        raise FileNotFoundError(f"{HTML_FILE} does not exist.")

    ensure_algorithm_cache()
    load_data_summary()
    os.chdir(BASE_DIR)

    server = ThreadingHTTPServer(("127.0.0.1", port), SimpleHTTPRequestHandler)
    url = f"http://127.0.0.1:{port}/{HTML_FILE.name}"
    print(f"Serving {BASE_DIR}")
    print(f"Open: {url}")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check or serve the standalone F1 HTML replay frontend."
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="serve f1_viewer.html locally so fetch('./data/telemetry_full.json') works",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"local server port when using --serve (default: {DEFAULT_PORT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.serve:
        serve(args.port)
    else:
        print_status()


if __name__ == "__main__":
    main()
