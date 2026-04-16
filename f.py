import requests
from pathlib import Path
from time import sleep
import re

BASE_URL = "http://88.198.65.184/RECORDINGS/MP3/"
SAVE_DIR = Path("AMDDetection/data/recordings-mva2-3")

SAVE_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "text/html,application/xhtml+xml",
    "Connection": "keep-alive",
}

print("[INFO] Fetching directory listing...")
resp = requests.get(BASE_URL, headers=HEADERS, timeout=30)

if resp.status_code != 200:
    raise RuntimeError(f"Failed to access directory: HTTP {resp.status_code}")

html = resp.text



files = sorted(set(re.findall(r'href="([^"]+\.mp3)"', html)))

if not files:
    raise RuntimeError("No MP3 files found (directory listing may be disabled)")

print(f"[INFO] Found {len(files)} MP3 files")



for i, fname in enumerate(files, 1):
    url = BASE_URL + fname
    out_path = SAVE_DIR / fname

    if out_path.exists():
        print(f"[SKIP] {fname}")
        continue

    try:
        r = requests.get(url, headers=HEADERS, timeout=60)
        if r.status_code == 200 and len(r.content) > 0:
            with open(out_path, "wb") as f:
                f.write(r.content)
            print(f"[{i}/{len(files)}] Downloaded {fname}")
            sleep(1)  # polite throttle
        else:
            print(f"[FAIL] {fname} HTTP {r.status_code}")
    except Exception as e:
        print(f"[ERROR] {fname} → {e}")

print("ALL DOWNLOADS COMPLETE")
