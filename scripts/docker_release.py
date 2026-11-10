#!/usr/bin/env python3
"""Docker Hub release CLI — push versioned images + sync Hub repo metadata.

Builds stay on the Jetson host exactly as today (`./build.sh`); this tool
pushes the resulting local images and keeps the Hub side tidy:

  - docker login (token via stdin/env/docker config — NEVER a CLI argument,
    so it can't leak into shell history or `ps`)
  - auto-create the Hub repository under the namespace when missing
  - PATCH the repo description + full README on EVERY release (the package's
    README.md is the source of truth)
  - docker push <ns>/<name>:<version> and :latest

The NAMESPACE (--docker-user) may be an organisation; the LOGIN identity is
separate (--hub-login-user, or the username stored by `docker login`) because
orgs have no passwords — your user just needs write access to the org.

Usage (from the repo root):

    echo "$TOKEN" | python3 scripts/docker_release.py packages/tts/kokoro-tts \
        --docker-user narandill --token-stdin

    # preview everything without side effects (no token needed):
    python3 scripts/docker_release.py --all --docker-user narandill --dry-run

    # token from env (CI-friendly):
    DOCKER_HUB_TOKEN=... python3 scripts/docker_release.py --all --docker-user narandill

    # token from the repo .env (gitignored; also DOCKER_HUB_USER for the login
    # identity when it differs from the namespace) — just run it, no sourcing:
    #   DOCKER_HUB_TOKEN=dckr_pat_...
    #   DOCKER_HUB_USER=your-username
    python3 scripts/docker_release.py --all --docker-user narandill

Token precedence: --token-stdin > $DOCKER_HUB_TOKEN > .env DOCKER_HUB_TOKEN >
~/.docker/config.json (the base64 auth `docker login` stored — no second
secret needed). The login USER resolves the same way: --hub-login-user >
$DOCKER_HUB_USER > .env DOCKER_HUB_USER > docker-login user > namespace.
Override stale `docker login` credentials by simply setting the env/.env pair —
or `docker logout` to drop them entirely.

Create the token at hub.docker.com -> Account Settings -> Personal access
tokens, scope Read/Write (never use the account password). The login identity
must be the account the token belongs to; the namespace (--docker-user) is
only where images land.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import NoReturn

ROOT = Path(__file__).resolve().parent.parent
HUB = "https://hub.docker.com"
DOCKER_CONFIG = Path.home() / ".docker" / "config.json"
DOTENV = ROOT / ".env"
# env knobs that would override the build.sh defaults the tag parser reads —
# a shell-exported VERSION=0.9.0 would otherwise build 0.9.0 and push 0.8.0
BUILD_ENV_SCRUB = ("VERSION", "DOCKER_REPO", "LLAMA_CPP_VERSION")
MAX_FULL_DESC = 25000  # Hub limit on full_description


def fail(msg: str) -> NoReturn:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def discover_packages(args) -> list[Path]:
    if args.all and args.packages:
        raise ValueError("--all and package paths are mutually exclusive")
    if args.all:
        return sorted(p.parent for p in ROOT.glob("packages/*/*/Dockerfile"))
    return [Path(p) for p in args.packages]


def _version_default(bs: str) -> str | None:
    """Resolve VERSION="..." with arbitrary ${VAR:-default} nesting."""
    for line in bs.splitlines():
        m = re.match(r'\s*(?:export\s+)?VERSION="(.*)"\s*$', line)
        if not m:
            continue
        val = m.group(1)
        while True:
            inner = re.search(r"\$\{[A-Z_]*:-([^\$\{\}]+)\}", val)
            if not inner:
                break
            val = val.replace(inner.group(0), inner.group(1), 1)
        if val and "$" not in val:
            return val.strip()
    return None


def pkg_meta(pkg: Path) -> tuple[str, str, str]:
    """(image name, version, local repo ns) from the package's build.sh."""
    bs = (pkg / "build.sh").read_text()
    version = _version_default(bs)
    if not version:
        raise ValueError("cannot parse VERSION default from build.sh")
    m = re.search(r'REPO="\$\{DOCKER_REPO:-([^"}]+)\}"', bs)
    repo = m.group(1) if m else "narandill"
    return pkg.name, version, repo


def _dotenv() -> dict[str, str]:
    """Tiny .env reader (KEY=VALUE, # comments, optional quotes; no shell
    expansion, no export prefix needed). Missing file = empty dict."""
    if not DOTENV.exists():
        return {}
    out: dict[str, str] = {}
    for line in DOTENV.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip("\"'")
        if key:
            out[key] = val
    return out


def _config_auth() -> tuple[str, str] | None:
    """(user, token) stored by `docker login`, or None."""
    if not DOCKER_CONFIG.exists():
        return None
    try:
        cfg = json.loads(DOCKER_CONFIG.read_text())
        auth = (
            (cfg.get("auths") or {}).get("https://index.docker.io/v1/", {}).get("auth")
        )
        if not auth:
            return None
        user, token = base64.b64decode(auth).decode().split(":", 1)
        return user, token
    except Exception as e:  # corrupt/colonless/unreadable config — degrade
        print(f"[auth] ignoring {DOCKER_CONFIG}: {e}", file=sys.stderr)
        return None


def read_token(args, dot: dict[str, str]) -> tuple[str, str, str]:
    """Returns (login_user, token, config_user). Token never via argv.

    Precedence: stdin > $DOCKER_HUB_TOKEN > .env > docker-login config. The
    login user: --hub-login-user > $DOCKER_HUB_USER > .env > config > ns —
    the pair must belong to the same account (a stale `docker login` user
    with a fresh token is the classic 401).
    """
    cfg = _config_auth()
    config_user, config_token = cfg if cfg else ("", "")
    token = ""  # nosec B105 - placeholder, filled from stdin/env/.env below
    if args.token_stdin:
        if sys.stdin.isatty():
            fail(
                "--token-stdin needs a pipe (echo $TOKEN | ...), not a TTY — "
                "the secret would echo to the screen"
            )
        token = sys.stdin.readline().strip()
        if not token:
            print(
                "[auth] --token-stdin got an empty line — falling back to env/.env/config",
                file=sys.stderr,
            )
    if not token and os.environ.get("DOCKER_HUB_TOKEN"):
        token = os.environ["DOCKER_HUB_TOKEN"].strip()
    if not token and dot.get("DOCKER_HUB_TOKEN"):
        token = dot["DOCKER_HUB_TOKEN"].strip()
        print(f"[auth] token from {DOTENV} (gitignored)")
    if not token and not args.token_stdin and config_token:
        token = config_token
        print(
            f"[auth] reusing docker-login credentials for {config_user} ({DOCKER_CONFIG})"
        )
    if not token:
        fail(
            "no token: pipe one via --token-stdin, set DOCKER_HUB_TOKEN (env or "
            f"{DOTENV}), or `docker login` first (config fallback)"
        )
    # login identity: explicit flag > env > .env > stored docker-login user > ns
    login_user = (
        args.hub_login_user
        or os.environ.get("DOCKER_HUB_USER", "").strip()
        or dot.get("DOCKER_HUB_USER", "").strip()
        or config_user
        or args.docker_user
    )
    return login_user, token, config_user


def _resp_str(resp) -> str:
    return resp.decode(errors="replace") if isinstance(resp, bytes) else str(resp)


def hub_request(
    path: str, method: str = "GET", body: dict | None = None, jwt: str = ""
) -> tuple[int, object]:
    """Hub v2 call. Returns (status, parsed-json-or-text). Transient network
    errors surface as status 0 so per-package failure handling sees them."""
    req = urllib.request.Request(HUB + path, method=method)
    if jwt:
        req.add_header("Authorization", f"JWT {jwt}")
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, data=data, timeout=30) as r:  # nosec B310 - hub API over https
            raw = r.read()
            return r.status, (
                json.loads(raw) if raw and raw[:1] in (b"{", b"[") else raw
            )
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, raw
    except urllib.error.URLError as e:
        return 0, f"network error: {e}"


def hub_login(user: str, token: str) -> str:
    code, resp = hub_request(
        "/v2/users/login/", "POST", {"username": user, "password": token}
    )
    if code != 200 or not isinstance(resp, dict) or "token" not in resp:
        fail(f"hub login failed as {user!r} ({code}): {_resp_str(resp)[:200]}")
    return resp["token"]


class HubSession:
    """Hub v2 session that re-logins on 401, once per call.

    The login JWT is short-lived (15 min observed) while a release push of
    ~90 GB runs much longer — without this, every call after the expiry
    fails with `unauthorized` mid-run. The PAT itself is only used here, at
    re-login; its own expiry setting is irrelevant to the session.
    """

    def __init__(self, user: str, token: str):
        self.user, self.token = user, token
        self.jwt = ""
        self.relogins = 0

    def login(self) -> None:
        self.jwt = hub_login(self.user, self.token)

    def call(
        self, path: str, method: str = "GET", body: dict | None = None
    ) -> tuple[int, object]:
        code, resp = hub_request(path, method, body, jwt=self.jwt)
        if code == 401:  # session expired (or revoked) — refresh once, retry
            self.login()
            self.relogins += 1
            code, resp = hub_request(path, method, body, jwt=self.jwt)
        return code, resp


def sh(args: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, **kw)


def short_desc(readme: str) -> str:
    for line in readme.splitlines():
        s = line.strip()
        if not s or s.startswith(("#", "|", "![", "[")) or s.startswith("* "):
            continue
        return s.replace("**", "").replace("`", "")[:100]
    return (readme.strip().splitlines() or [""])[0][:100]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("packages", nargs="*", help="package dirs (or --all)")
    ap.add_argument(
        "--all", action="store_true", help="every packages/*/* with a Dockerfile"
    )
    ap.add_argument(
        "--docker-user", required=True, help="Hub NAMESPACE (user or org) to push to"
    )
    ap.add_argument(
        "--hub-login-user",
        help="Hub username to authenticate as (defaults to the docker-login user or --docker-user; REQUIRED for org namespaces unless the login user is stored)",
    )
    ap.add_argument(
        "--token-stdin", action="store_true", help="read the Hub token from stdin"
    )
    ap.add_argument(
        "--build",
        action="store_true",
        help="run ./build.sh first (default: push existing local images)",
    )
    ap.add_argument("--no-latest", action="store_true", help="skip the :latest tag")
    ap.add_argument(
        "--dry-run", action="store_true", help="print the plan; no login/push/API"
    )
    args = ap.parse_args()

    try:
        pkgs = discover_packages(args)
    except ValueError as e:
        ap.error(str(e))
    login_user, token, _cfg_user = ("", "", "")
    dot: dict[str, str] = {}
    if not args.dry_run:  # dry-run must not demand (or decode) a token
        dot = _dotenv()
    login_user, token, _ = read_token(args, dot)
    if not args.dry_run and args.docker_user != login_user and not args.hub_login_user:
        print(
            f"[auth] logging in as {login_user!r}, pushing to namespace {args.docker_user!r}"
        )
    print(
        f"[plan] namespace: {args.docker_user} | packages: {len(pkgs)} | dry-run: {args.dry_run}"
    )

    session = None
    if not args.dry_run:
        session = HubSession(login_user, token)
        session.login()
        print(f"[auth] hub api login OK as {login_user}")
        r = sh(["docker", "login", "-u", login_user, "--password-stdin"], input=token)
        if r.returncode != 0:
            fail(f"docker login failed: {r.stderr.strip()[:200]}")
        print("[auth] docker login OK")

    failures: list[str] = []
    skipped: list[str] = []
    try:
        for pkg in pkgs:
            try:
                name, version, local_repo = pkg_meta(pkg)
            except (ValueError, OSError) as e:
                skipped.append(pkg.name)
                print(f"SKIP  {pkg.name}: {e}")
                continue
            readme_path = pkg / "README.md"
            full_desc = ""
            try:
                full_desc = readme_path.read_text()
            except OSError as e:
                print(f"WARN  {name}: no README ({e}) — generic description only")
            if len(full_desc) > MAX_FULL_DESC:
                print(
                    f"WARN  {name}: README {len(full_desc)} chars > Hub limit {MAX_FULL_DESC} — truncating"
                )
                full_desc = full_desc[:MAX_FULL_DESC]
            desc = (
                short_desc(full_desc)
                if full_desc
                else f"{name} — Jetson voice-stack image"
            )
            tags = [f"{args.docker_user}/{name}:{version}"] + (
                [] if args.no_latest else [f"{args.docker_user}/{name}:latest"]
            )
            local_tag = f"{local_repo}/{name}:{version}"

            if args.build:
                print(f"[build] {pkg}")
                if not args.dry_run:
                    env = {
                        k: v for k, v in os.environ.items() if k not in BUILD_ENV_SCRUB
                    }
                    r = sh(["./build.sh"], cwd=pkg, env=env)
                    if r.returncode != 0:
                        failures.append(name)
                        print(f"FAIL  build {name}: {r.stderr.strip()[-200:]}")
                        continue
            else:
                ins = sh(["docker", "image", "inspect", local_tag])
                if ins.returncode != 0:
                    failures.append(name)
                    print(
                        f"FAIL  {name}: local image {local_tag} missing — build first (./build.sh) or use --build"
                    )
                    continue

            # re-tag when the target namespace differs from the build default
            if args.docker_user != local_repo and not args.dry_run:
                for t in tags:
                    r = sh(["docker", "tag", local_tag, t])
                    if r.returncode != 0:
                        failures.append(name)
                        print(f"FAIL  tag {t}: {r.stderr.strip()[-150:]}")
                        break

            print(f"[{name}] {', '.join(tags)}")
            print(
                f"        desc: {desc[:70]}{'…' if len(desc) > 70 else ''} | README: {len(full_desc)} chars"
            )

            if args.dry_run:
                continue

            # ensure the repo exists: GET, then POST (409 on POST = someone
            # created it concurrently — same as exists)
            code, resp = session.call(f"/v2/repositories/{args.docker_user}/{name}/")
            if code == 404:
                code, resp = session.call(
                    f"/v2/repositories/{args.docker_user}/",
                    "POST",
                    {
                        "name": name,
                        "namespace": args.docker_user,
                        "description": desc,
                        "full_description": full_desc,
                        "is_private": False,
                    },
                )
                if code in (200, 201, 409):
                    print("        hub: repository created")
                else:
                    failures.append(name)
                    print(f"FAIL  hub create {name} ({code}): {_resp_str(resp)[:200]}")
                    continue
            elif code != 200:
                failures.append(name)
                print(f"FAIL  hub get {name} ({code}): {_resp_str(resp)[:200]}")
                continue

            # description/README sync on EVERY release
            code, resp = session.call(
                f"/v2/repositories/{args.docker_user}/{name}/",
                "PATCH",
                {"description": desc, "full_description": full_desc},
            )
            if code == 200:
                print("        hub: description + README synced")
            else:
                print(f"WARN  hub patch {name} ({code}) — image push continues")

            for t in tags:
                r = sh(["docker", "push", t])
                if r.returncode == 0:
                    print(f"        pushed {t}")
                else:
                    failures.append(name)
                    print(f"FAIL  push {t}: {r.stderr.strip()[-200:]}")
    except KeyboardInterrupt:
        print(
            "\ninterrupted — registry tags are atomic (manifest last), so no "
            "half-pushed tag exists; re-run to continue",
            file=sys.stderr,
        )
        return 130

    if skipped:
        print(f"SKIPPED (unparseable build.sh): {', '.join(skipped)}", file=sys.stderr)
    if failures:
        print(f"\nFAILED: {', '.join(sorted(set(failures)))}", file=sys.stderr)
        return 1
    print("\nall packages released")
    return 0


if __name__ == "__main__":
    sys.exit(main())
