"""Live audio streaming server.

Exposes `GET /stream.mp3` — chunked MP3 stream of the live PulseAudio
source. Multiple listeners can connect; ffmpeg only runs while at least
one listener is active. When the last listener disconnects, ffmpeg is
killed to free CPU.

Designed to be reachable via Cloudflare Tunnel (or LAN). Auth is intended
to live in front of this (CF Worker handles password gate, then proxies
to this endpoint). The endpoint itself is unauthenticated — bind it to a
private port + put it behind a tunnel/auth layer.
"""
from __future__ import annotations

import asyncio
import logging

from aiohttp import web

log = logging.getLogger(__name__)


class StreamServer:
    def __init__(
        self,
        pulse_source: str,
        port: int = 8765,
        bitrate_kbps: int = 96,
    ) -> None:
        self.pulse_source = pulse_source
        self.port = port
        self.bitrate_kbps = bitrate_kbps
        self._listeners: list[asyncio.Queue[bytes]] = []
        self._ffmpeg: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self._runner: web.AppRunner | None = None

    async def start(self) -> None:
        app = web.Application()
        app.router.add_get("/stream.mp3", self._stream_handler)
        app.router.add_get("/health", self._health_handler)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "0.0.0.0", self.port)
        await site.start()
        log.info("stream server listening on 0.0.0.0:%d/stream.mp3", self.port)

    async def stop(self) -> None:
        async with self._lock:
            await self._stop_ffmpeg_locked()
        if self._runner:
            await self._runner.cleanup()
            self._runner = None

    async def _stream_handler(self, request: web.Request) -> web.StreamResponse:
        q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=200)
        await self._add_listener(q)
        peer = request.headers.get("X-Forwarded-For") or request.remote
        log.info("stream listener attached: %s (%d total)", peer, len(self._listeners))

        resp = web.StreamResponse(
            headers={
                "Content-Type": "audio/mpeg",
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Connection": "close",
                "Access-Control-Allow-Origin": "*",
            }
        )
        await resp.prepare(request)
        try:
            while True:
                chunk = await q.get()
                if chunk == b"":  # sentinel meaning "stop"
                    break
                await resp.write(chunk)
        except (ConnectionResetError, asyncio.CancelledError, ConnectionError):
            pass
        finally:
            await self._remove_listener(q)
            log.info("stream listener detached: %s (%d remaining)", peer, len(self._listeners))
        return resp

    async def _health_handler(self, request: web.Request) -> web.Response:
        return web.json_response({
            "status": "ok",
            "listeners": len(self._listeners),
            "ffmpeg_running": self._ffmpeg is not None and self._ffmpeg.returncode is None,
            "pulse_source": self.pulse_source,
        })

    async def _add_listener(self, q: asyncio.Queue) -> None:
        async with self._lock:
            self._listeners.append(q)
            if self._ffmpeg is None:
                await self._start_ffmpeg_locked()

    async def _remove_listener(self, q: asyncio.Queue) -> None:
        async with self._lock:
            try:
                self._listeners.remove(q)
            except ValueError:
                pass
            if not self._listeners:
                await self._stop_ffmpeg_locked()

    async def _start_ffmpeg_locked(self) -> None:
        log.info("starting ffmpeg encode (source=%s, %dk MP3)", self.pulse_source, self.bitrate_kbps)
        self._ffmpeg = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-hide_banner", "-loglevel", "error",
            "-f", "pulse", "-i", self.pulse_source,
            "-ac", "1",
            "-c:a", "libmp3lame",
            "-b:a", f"{self.bitrate_kbps}k",
            "-f", "mp3",
            "pipe:1",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._reader_task = asyncio.create_task(self._read_and_fanout())

    async def _stop_ffmpeg_locked(self) -> None:
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
            self._reader_task = None
        if self._ffmpeg:
            try:
                self._ffmpeg.terminate()
                await asyncio.wait_for(self._ffmpeg.wait(), timeout=2.0)
            except (asyncio.TimeoutError, ProcessLookupError):
                try:
                    self._ffmpeg.kill()
                except ProcessLookupError:
                    pass
            self._ffmpeg = None
            log.info("ffmpeg encode stopped (no listeners)")

    async def _read_and_fanout(self) -> None:
        assert self._ffmpeg is not None and self._ffmpeg.stdout is not None
        try:
            while True:
                chunk = await self._ffmpeg.stdout.read(4096)
                if not chunk:
                    break
                # Fan out to all listeners; drop chunks for slow consumers
                for q in list(self._listeners):
                    try:
                        q.put_nowait(chunk)
                    except asyncio.QueueFull:
                        # Listener too slow — best-effort drop, don't block other listeners
                        pass
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("ffmpeg read error: %s", e)
        finally:
            # Wake all listeners so they can clean up
            for q in list(self._listeners):
                try:
                    q.put_nowait(b"")
                except asyncio.QueueFull:
                    pass
