"""
render_sim.py – Render-compatible replay server (JSON-only, no ML libs).

Reads pre-exported replay_data.json and streams events via WebSocket.
Handles pause/resume commands from the browser.
Memory usage: ~50MB (just aiohttp + json).

Start command:
    python render_sim.py
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path

from aiohttp import web, WSMsgType

BASE       = Path(__file__).parent
STATIC_DIR = BASE / "sim" / "static"
DATA_FILE  = STATIC_DIR / "replay_data.json"
SPEED      = float(os.environ.get("REPLAY_SPEED", "30"))


def _patch_html(html: str) -> str:
    return re.sub(
        r"const WS_URL\s*=\s*['\"]ws://localhost:\d+['\"];",
        "const WS_URL = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`;",
        html,
    )


async def handle_index(request: web.Request) -> web.Response:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return web.Response(text=_patch_html(html), content_type="text/html")


async def handle_static(request: web.Request) -> web.Response:
    filepath = STATIC_DIR / request.match_info["filename"]
    if filepath.exists():
        return web.FileResponse(filepath)
    return web.Response(status=404)


async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    with open(DATA_FILE) as f:
        data = json.load(f)

    events = data["events"]
    delay  = 1.0 / SPEED if SPEED > 0 else 0.0
    paused = False

    # ── Receive pause/resume commands concurrently ────────────────────────────
    async def _recv() -> None:
        nonlocal paused
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    cmd = json.loads(msg.data).get("cmd", "")
                    if cmd == "pause":
                        paused = True
                    elif cmd == "resume":
                        paused = False
                except Exception:
                    pass
            elif msg.type in (WSMsgType.CLOSE, WSMsgType.ERROR):
                break

    recv_task = asyncio.create_task(_recv())

    try:
        current_ts: int | None = None
        buffer: list[dict] = []

        async def flush() -> None:
            for ev in buffer:
                if ws.closed:
                    return
                # Wait while paused
                while paused and not ws.closed:
                    await asyncio.sleep(0.05)
                await ws.send_str(json.dumps(ev, separators=(",", ":")))
            buffer.clear()
            if delay > 0 and not ws.closed:
                while paused and not ws.closed:
                    await asyncio.sleep(0.05)
                await asyncio.sleep(delay)

        for ev in events:
            if ws.closed:
                break

            ts = ev.get("ts")

            if ev["type"] in ("candle", "stats") and ts != current_ts and current_ts is not None:
                await flush()
                current_ts = ts

            if current_ts is None and ts is not None:
                current_ts = ts

            buffer.append(ev)

        if buffer and not ws.closed:
            await flush()

    finally:
        recv_task.cancel()

    return ws


def main() -> None:
    port = int(os.environ.get("PORT", 8080))

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"{DATA_FILE} not found.\n"
            "Run export_replay.py locally first:\n"
            "  python export_replay.py --data Data/BTCUSDT/monthly/2026-01_1m.csv"
        )

    app = web.Application()
    app.router.add_get("/",           handle_index)
    app.router.add_get("/ws",         websocket_handler)
    app.router.add_get("/{filename}", handle_static)

    print(f"[render_sim] Serving {DATA_FILE.name} on port {port}")
    web.run_app(app, host="0.0.0.0", port=port, print=None)


if __name__ == "__main__":
    main()
