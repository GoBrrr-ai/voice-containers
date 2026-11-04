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

Usage (from the repo root):

    echo "$TOKEN" | python3 scripts/docker_release.py packages/tts/kokoro-tts \
        --docker-user narandill --token-stdin

    # preview everything without side effects:
    python3 scripts/docker_release.py --all --docker-user narandill --dry-run

    # token from env (CI-friendly):
    DOCKER_HUB_TOKEN=... python3 scripts/docker_release.py --all --docker-user narandill

Token precedence: --token-stdin > $DOCKER_HUB_TOKEN > ~/.docker/config.json
(the base64 auth `docker login` stored — no second secret needed).

Create the token at hub.docker.com -> Account Settings -> Personal access
tokens, scope Read/Write (never use the account password).
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

ROOT = Path(__file__).resolve().parent.parent
HUB = "https://hub.docker.com"
DOCKER_CONFIG = Path.home() / ".docker" / "config.json"


def fail(msg: str) -> "None":
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def discover_packages(args) -> list[Path]:
    if args.all:
        pkgs = sorted(p.parent for p in ROOT.glob("packages/*/*/Dockerfile"))
    else:
        pkgs = [Path(p) for p in args.packages]
    out = []
    for p in pkgs:
        if not (p / "Dockerfile").exists():
            fail(f"{p}: no Dockerfile")
        out.append(p)
    return out


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
        raise ValueError(f"cannot parse VERSION default from build.sh")
    m = re.search(r'REPO="\$\{DOCKER_REPO:-([^"}]+)\}"', bs)
    repo = m.group(1) if m else "narandill"
    return pkg.name, version, repo


def read_token(args) -> tuple[str, str]:
    """Returns (user, token). Never accepts a token via argv."""
    user, token = args.docker_user or "", ""
    if args.token_stdin:
        token = sys.stdin.readline().strip()
    elif os.environ.get("DOCKER_HUB_TOKEN"):
        token = os.environ["DOCKER_HUB_TOKEN"].strip()
    elif DOCKER_CONFIG.exists():
        cfg = json.loads(DOCKER_CONFIG.read_text())
        auth = (
            (cfg.get("auths") or {}).get("https://index.docker.io/v1/", {}).get("auth")
        )
        if auth:
            user_b, token_b = base64.b64decode(auth).decode().split(":", 1)
            user = user or user_b
            token = token_b
            print(
                f"[auth] reusing docker-login credentials for {user} ({DOCKER_CONFIG})"
            )
    if not token:
        fail(
            "no token: pipe one via --token-stdin, set DOCKER_HUB_TOKEN, or "
            "`docker login` first (config.json fallback)"
        )
    if not user:
        fail("no username: pass --docker-user (or have it in ~/.docker/config.json)")
    return user, token


def hub_request(
    path: str, method: str = "GET", body: dict | None = None, jwt: str = ""
) -> tuple[int, dict | bytes]:
    req = urllib.request.Request(HUB + path, method=method)
    if jwt:
        req.add_header("Authorization", f"JWT {jwt}")
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, data=data, timeout=30) as r:
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


def hub_login(user: str, token: str) -> str:
    code, resp = hub_request(
        "/v2/users/login/", "POST", {"username": user, "password": token}
    )
    if code != 200 or not isinstance(resp, dict) or "token" not in resp:
        fail(f"hub login failed ({code}): {str(resp)[:200]}")
    return resp["token"]


def sh(args: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, **kw)


def short_desc(readme: str) -> str:
    for line in readme.splitlines():
        line = line.strip()
        if line and not line.startswith(("#", "|", "!", "[")):
            return line[:100]
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
        "--docker-user", help="Hub namespace (user or org); one flag, no secrets"
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

    pkgs = discover_packages(args)
    user, token = read_token(args)
    print(f"[plan] namespace: {user} | packages: {len(pkgs)} | dry-run: {args.dry_run}")

    jwt = ""
    if not args.dry_run:
        jwt = hub_login(user, token)
        print("[auth] hub api login OK")
        r = sh(["docker", "login", "-u", user, "--password-stdin"], input=token)
        if r.returncode != 0:
            fail(f"docker login failed: {r.stderr.strip()[:200]}")
        print("[auth] docker login OK")

    failures = []
    skipped = []
    for pkg in pkgs:
        try:
            name, version, local_repo = pkg_meta(pkg)
        except ValueError as e:
            skipped.append(pkg.name)
            print(f"SKIP  {pkg.name}: {e}")
            continue
        readme_path = pkg / "README.md"
        full_desc = readme_path.read_text() if readme_path.exists() else ""
        desc = (
            short_desc(full_desc) if full_desc else f"{name} — Jetson voice-stack image"
        )
        tags = [f"{user}/{name}:{version}"] + (
            [] if args.no_latest else [f"{user}/{name}:latest"]
        )
        local_tag = f"{local_repo}/{name}:{version}"

        if args.build:
            print(f"[build] {pkg}")
            if not args.dry_run:
                r = sh(["./build.sh"], cwd=pkg)
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
        if user != local_repo and not args.dry_run:
            for t in tags:
                sh(["docker", "tag", local_tag, t])

        print(f"[{name}] {', '.join(tags)}")
        print(
            f"        desc: {desc[:70]}{'…' if len(desc) > 70 else ''} | README: {len(full_desc)} chars"
        )

        if args.dry_run:
            continue

        # ensure the repo exists (auto-create under the org)
        code, resp = hub_request(f"/v2/repositories/{user}/{name}/", jwt=jwt)
        if code == 404:
            code, resp = hub_request(
                f"/v2/repositories/{user}/{name}/",
                "POST",
                {
                    "name": name,
                    "namespace": user,
                    "description": desc,
                    "full_description": full_desc,
                    "is_private": False,
                },
                jwt=jwt,
            )
            if code in (200, 201):
                print(f"        hub: repository created")
            else:
                failures.append(name)
                print(f"FAIL  hub create {name} ({code}): {str(resp)[:200]}")
                continue
        elif code != 200:
            failures.append(name)
            print(f"FAIL  hub get {name} ({code}): {str(resp)[:200]}")
            continue

        # description/README sync on EVERY release
        code, resp = hub_request(
            f"/v2/repositories/{user}/{name}/",
            "PATCH",
            {"description": desc, "full_description": full_desc},
            jwt=jwt,
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

    if skipped:
        print(f"SKIPPED (unparseable build.sh): {', '.join(skipped)}", file=sys.stderr)
    if failures:
        print(f"\nFAILED: {', '.join(sorted(set(failures)))}", file=sys.stderr)
        return 1
    print("\nall packages released")
    return 0


if __name__ == "__main__":
    sys.exit(main())
