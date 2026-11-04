"""voice-api v1 executable contract — reference stub + wire checks (CI-safe).

Serves a reference implementation of docs/voice-api.md on 127.0.0.1:<ephemeral>
and asserts the wire behaviors the GPU conformance suite (task/voice-api/)
expects: discovery shapes, multipart + raw-WAV transcriptions
(json/text/verbose_json), sse partials on stream=true (partial + final
events), wav/pcm speech with X-Sample-Rate, the 400 paths, and the
optional VOICE_API_TOKEN gate (/health exempt).

Pure stdlib, no docker, no GPU — this is the CI job. Exit 0 = contract holds.
The stub is also the readable reference for integrators; when you change
docs/voice-api.md, change both.
"""

from __future__ import annotations

import io
import json
import math
import struct
import sys
import threading
import urllib.error
import urllib.request
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TOKEN = ""  # set by the token-mode leg; a test fixture, not a secret  # nosec B105
VOICE = "test_voice"
MODEL = "test_model"
SR = 24000


def _sine(seconds: float = 0.2) -> bytes:
    n = int(SR * seconds)
    return struct.pack(
        f"<{n}h", *(int(12000 * math.sin(2 * math.pi * 220 * i / SR)) for i in range(n))
    )


def _wav(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm)
    return buf.getvalue()


class ReferenceHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _auth_ok(self):
        if not TOKEN:
            return True
        # /health is exempt: orchestrator healthchecks cannot send the token
        return (
            self.path.split("?")[0] == "/health"
            or self.headers.get("Authorization", "") == f"Bearer {TOKEN}"
        )

    def _deny(self):
        self._json(
            401,
            {
                "error": {
                    "message": "invalid or missing bearer token",
                    "type": "authentication_error",
                }
            },
        )

    def do_GET(self):
        if not self._auth_ok():
            return self._deny()
        p = self.path.split("?")[0]
        if p == "/health":
            return self._json(
                200, {"status": "ok", "engine": "reference", "version": "1"}
            )
        if p == "/v1/voices":
            return self._json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": VOICE if TASK == "tts" else MODEL,
                            "object": "voice",
                            "task": TASK,
                            "sample_rate": SR,
                            "engine": MODEL,
                            "streaming": True,
                        }
                    ],
                },
            )
        if p == "/v1/models":
            return self._json(
                200,
                {
                    "object": "list",
                    "data": [{"id": MODEL, "object": "model", "owned_by": "reference"}],
                },
            )
        return self._json(404, {"error": {"message": f"no route {p}"}})

    def do_POST(self):
        if not self._auth_ok():
            return self._deny()
        p = self.path.split("?")[0]
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        ctype = self.headers.get("Content-Type", "")

        if p == "/v1/audio/transcriptions":
            return self._transcribe(body, ctype)
        if p == "/v1/audio/speech":
            return self._speech(body, ctype)
        return self._json(404, {"error": {"message": f"no route {p}"}})

    def _transcribe(self, body, ctype):
        path, _, query = self.path.partition("?")
        qv = {
            k: v for k, _, v in (seg.partition("=") for seg in query.split("&") if seg)
        }
        want_stream = qv.get("stream", "").lower() in ("true", "1")
        model = fmt = None
        raw = False
        if ctype.startswith("multipart/form-data"):
            fields = _parse_multipart(body, ctype)
            model = fields.get("model")
            fmt = fields.get("response_format", "json")
            if fields.get("stream", "").lower() in ("true", "1"):
                want_stream = True
            audio = fields.get("file")
        else:
            model = None  # single-model server: the field may be absent on raw
            fmt = "json"
            raw = True
            audio = body
        if model is not None and model != MODEL:
            return self._json(
                400,
                {
                    "error": {
                        "message": f"unknown model {model!r}",
                        "type": "invalid_request_error",
                    }
                },
            )
        if audio is None and not raw:
            return self._json(
                400,
                {
                    "error": {
                        "message": "missing required field 'file'",
                        "type": "invalid_request_error",
                    }
                },
            )
        text = "reference transcript"
        if want_stream:
            # reference = a partial-capable STT (moonshine-streaming variant):
            # SSE partials then one final. Incapable engines accept-and-ignore
            # stream (JSON final) — that path is per-engine, see the roster.
            return self._sse_transcribe(text)
        if fmt == "text":
            payload, ct = text.encode(), "text/plain"
        elif fmt == "verbose_json":
            payload, ct = (
                json.dumps(
                    {"task": "transcribe", "text": text, "duration": 0.2}
                ).encode(),
                "application/json",
            )
        else:
            payload, ct = json.dumps({"text": text}).encode(), "application/json"
        self.send_response(200)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _sse_transcribe(self, text):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def evt(name, obj):
            b = f"event: {name}\ndata: {json.dumps(obj)}\n\n".encode()
            self.wfile.write(f"{len(b):x}\r\n".encode() + b + b"\r\n")

        for partial in ("reference", "reference transcript"):
            evt("partial", {"text": partial})
        evt("final", {"text": text, "duration": 0.2})
        self.wfile.write(b"0\r\n\r\n")

    def _speech(self, body, ctype):
        try:
            req = json.loads(body)
        except Exception:
            return self._json(400, {"error": {"message": "invalid json"}})
        voice = req.get("voice")
        if voice is not None and voice != VOICE:
            return self._json(
                400,
                {
                    "error": {
                        "message": f"unknown voice {voice!r}",
                        "type": "invalid_request_error",
                    }
                },
            )
        if req.get("model") is not None and req["model"] != MODEL:
            return self._json(
                400,
                {
                    "error": {"message": f"unknown model {req['model']!r}"},
                    "type": "invalid_request_error",
                },
            )
        if req.get("response_format") == "sse":
            return self._json(
                400, {"error": {"message": "response_format must be wav or pcm"}}
            )
        if not (req.get("input") or "").strip():
            return self._json(400, {"error": {"message": "input must not be empty"}})
        pcm = _sine()
        if req.get("response_format", "wav") == "pcm":
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("X-Sample-Rate", str(SR))
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for off in range(0, len(pcm), 1024):
                chunk = pcm[off : off + 1024]
                self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")
            return
        wav = _wav(pcm)
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("X-Sample-Rate", str(SR))
        self.send_header("Content-Length", str(len(wav)))
        self.end_headers()
        self.wfile.write(wav)


TASK = "stt"


def _parse_multipart(body, ctype):
    boundary = ctype.split("boundary=", 1)[1].split(";", 1)[0].strip().encode()
    fields = {}
    for part in body.split(b"--" + boundary):
        if b"\r\n\r\n" not in part:
            continue
        head, _, data = part.partition(b"\r\n\r\n")
        data = data.rsplit(b"\r\n", 1)[0]
        name = b'name="'
        if name in head:
            key = head.split(name, 1)[1].split(b'"', 1)[0].decode()
            fields[key] = data.decode(errors="replace")
    return fields


def _parse_sse(body: bytes):
    """[(event, data_str)] from a chunk-decoded text/event-stream body."""
    events = []
    for block in body.decode("utf-8", "replace").split("\n\n"):
        ev, data = None, None
        for line in block.strip().splitlines():
            if line.startswith("event: "):
                ev = line[7:].strip()
            elif line.startswith("data: "):
                data = line[6:]
        if ev and data:
            events.append((ev, data))
    return events


def _get(base, path, token=""):  # nosec B107 - localhost contract test
    req = urllib.request.Request(base + path)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:  # nosec B310 - 127.0.0.1 only
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def _post(base, path, data, ctype, token=""):  # nosec B107 - localhost contract test
    req = urllib.request.Request(base + path, data=data, method="POST")
    req.add_header("Content-Type", ctype)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:  # nosec B310 - 127.0.0.1 only
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def _multipart(fields):
    b = "xXcontractXx"
    body = b""
    for k, v in fields.items():
        body += (
            f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n'.encode()
            + (v if isinstance(v, bytes) else v.encode())
            + b"\r\n"
        )
    return body + f"--{b}--\r\n".encode(), f"multipart/form-data; boundary={b}"


CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn

    return deco


@check("health status ok")
def _(base):
    s, _, b = _get(base, "/health")
    assert s == 200 and json.loads(b)["status"] == "ok", (s, b)


@check("voices shape")
def _(base):
    s, _, b = _get(base, "/v1/voices")
    d = json.loads(b)["data"][0]
    assert s == 200 and d["task"] == TASK and d["sample_rate"] == SR, (s, b)


@check("models alias")
def _(base):
    s, _, b = _get(base, "/v1/models")
    d = json.loads(b)["data"][0]
    assert s == 200 and d["id"] == MODEL and d["object"] == "model", (s, b)


@check("transcribe multipart json")
def _(base):
    body, ct = _multipart({"model": MODEL, "response_format": "json", "file": "x"})
    s, _, b = _post(base, "/v1/audio/transcriptions", body, ct)
    assert s == 200 and json.loads(b)["text"], (s, b)


@check("transcribe multipart text")
def _(base):
    body, ct = _multipart({"model": MODEL, "response_format": "text", "file": "x"})
    s, _, b = _post(base, "/v1/audio/transcriptions", body, ct)
    assert s == 200 and b[:1] != b"{", (s, b[:40])


@check("transcribe multipart verbose_json")
def _(base):
    body, ct = _multipart(
        {"model": MODEL, "response_format": "verbose_json", "file": "x"}
    )
    s, _, b = _post(base, "/v1/audio/transcriptions", body, ct)
    j = json.loads(b)
    assert s == 200 and j["task"] == "transcribe" and "duration" in j, (s, b)


@check("transcribe stream=true sse partials (raw wav + query param)")
def _(base):
    s, h, b = _post(
        base, "/v1/audio/transcriptions?stream=true", _wav(_sine()), "audio/wav"
    )
    assert s == 200 and h.get("Content-Type") == "text/event-stream", (s, h)
    events = _parse_sse(b)
    partials = [json.loads(d) for n, d in events if n == "partial"]
    finals = [json.loads(d) for n, d in events if n == "final"]
    assert partials and finals, events
    assert all("text" in p for p in partials)
    assert finals[0]["text"] == "reference transcript" and finals[0]["duration"] == 0.2


@check("transcribe stream=true sse via multipart field, final matches plain")
def _(base):
    body, ctype = _multipart({"stream": "true", "file": _wav(_sine())})
    s, h, b = _post(base, "/v1/audio/transcriptions", body, ctype)
    assert s == 200 and h.get("Content-Type") == "text/event-stream", (s, h)
    finals = [json.loads(d) for n, d in _parse_sse(b) if n == "final"]
    body2, ctype2 = _multipart({"file": _wav(_sine())})
    s2, _, b2 = _post(base, "/v1/audio/transcriptions", body2, ctype2)
    assert s2 == 200 and json.loads(b2)["text"] == finals[0]["text"], (s2, b2[:80])


@check("transcribe raw octet-stream")
def _(base):
    s, _, b = _post(
        base, "/v1/audio/transcriptions", _wav(_sine()), "application/octet-stream"
    )
    assert s == 200 and json.loads(b)["text"], (s, b[:80])


@check("wrong model -> 400")
def _(base):
    body, ct = _multipart({"model": "bogus", "file": "x"})
    s, _, b = _post(base, "/v1/audio/transcriptions", body, ct)
    assert s == 400 and "invalid_request_error" in b.decode(), (s, b[:80])


@check("missing file -> 400")
def _(base):
    body, ct = _multipart({"model": MODEL})
    s, _, b = _post(base, "/v1/audio/transcriptions", body, ct)
    assert s == 400, (s, b[:80])


@check("speech wav + X-Sample-Rate")
def _(base):
    body = json.dumps(
        {"input": "hello", "voice": VOICE, "model": MODEL, "response_format": "wav"}
    ).encode()
    s, h, b = _post(base, "/v1/audio/speech", body, "application/json")
    assert s == 200 and b[:4] == b"RIFF" and h.get("X-Sample-Rate") == str(SR), (
        s,
        dict(h),
    )


@check("speech pcm chunked, s16le-framed")
def _(base):
    body = json.dumps({"input": "hello", "response_format": "pcm"}).encode()
    s, h, b = _post(base, "/v1/audio/speech", body, "application/json")
    assert s == 200 and h.get("X-Sample-Rate") == str(SR), (s, dict(h))
    assert len(b) > 0 and len(b) % 2 == 0, len(b)


@check("speech bad voice -> 400")
def _(base):
    body = json.dumps({"input": "x", "voice": "bogus"}).encode()
    s, _, b = _post(base, "/v1/audio/speech", body, "application/json")
    assert s == 400, (s, b[:80])


@check("speech sse -> 400")
def _(base):
    body = json.dumps({"input": "x", "response_format": "sse"}).encode()
    s, _, b = _post(base, "/v1/audio/speech", body, "application/json")
    assert s == 400, (s, b[:80])


@check("speech empty input -> 400")
def _(base):
    body = json.dumps({"input": "  "}).encode()
    s, _, b = _post(base, "/v1/audio/speech", body, "application/json")
    assert s == 400, (s, b[:80])


@check("token gate: /health exempt, routes 401/200")
def _(base):
    global TOKEN
    TOKEN = "s3cret"  # nosec B105 - the token-mode fixture
    try:
        s, _, _b = _get(base, "/health")
        assert s == 200, s  # exempt by design
        s, _, _b = _get(base, "/v1/voices")
        assert s == 401, s
        s, _, _b = _get(base, "/v1/voices", token="s3cret")  # nosec B106
        assert s == 200, s
    finally:
        TOKEN = ""  # nosec B105 - reset to the open default


def main() -> int:
    global TOKEN
    srv = ThreadingHTTPServer(("127.0.0.1", 0), ReferenceHandler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    failed = []
    try:
        for name, fn in CHECKS:
            try:
                fn(base)
                print(f"PASS {name}")
            except AssertionError as e:
                failed.append(name)
                print(f"FAIL {name}: {e}")
            except Exception as e:  # noqa: BLE001
                failed.append(name)
                print(f"FAIL {name}: {e!r}")
    finally:
        srv.shutdown()
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} contract checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
