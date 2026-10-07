"""UI smoke check (M1.2 / M2.9): every page at phone / tablet / desktop size.

Records horizontal overflow, third-party requests and screenshots (reports/ui/). Works WITHOUT the backend: pages are served
statically, so API calls fail and the pages' error states are what gets checked. Run with the API up for the real flow.
Usage: python tests/e2e/ui_smoke.py [--base http://127.0.0.1:8000] [--serve]   (--serve starts a static server on :8765)
"""
import argparse
import functools
import http.server
import json
import threading
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
PAGES = ["index.html", "case.html", "graph.html", "demo.html", "settings.html", "help.html"]
SIZES = {"phone": (375, 812), "tablet": (768, 1024), "desktop": (1280, 800)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8765")
    ap.add_argument("--serve", action="store_true")
    a = ap.parse_args()
    if a.serve:
        h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(ROOT / "Frontend"))
        h.log_message = lambda *args, **kw: None
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 8765), h)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    host = urlparse(a.base).netloc
    out = ROOT / "reports" / "ui"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with sync_playwright() as p:
        b = p.chromium.launch()
        for size, (w, hgt) in SIZES.items():
            for page_name in PAGES:
                ctx = b.new_context(viewport={"width": w, "height": hgt})
                pg = ctx.new_page()
                third, errors = [], []
                pg.on("request", lambda r: third.append(r.url) if urlparse(r.url).netloc not in (host, "") and not r.url.startswith("data:") else None)
                pg.on("pageerror", lambda e: errors.append(str(e)[:120]))
                pg.goto(f"{a.base}/{page_name}", wait_until="networkidle")
                pg.wait_for_timeout(400)
                m = pg.evaluate("() => ({sw: document.documentElement.scrollWidth, cw: document.documentElement.clientWidth})")
                pg.screenshot(path=str(out / f"{page_name.replace('.html', '')}_{size}.png"), full_page=True)
                rows.append(dict(page=page_name, size=size, overflow_px=max(0, m["sw"] - m["cw"]), third_party=len(third), js_errors=errors))
                ctx.close()
        b.close()
    (out / "ui_smoke.json").write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(f"{r['page']:14}{r['size']:8} overflow={r['overflow_px']:4}px third_party={r['third_party']} js_errors={len(r['js_errors'])}")


if __name__ == "__main__":
    main()
