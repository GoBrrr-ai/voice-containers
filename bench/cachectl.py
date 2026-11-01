"""Host page-cache management + the one-time, targeted NOPASSWD sudo grant
that lets the benchmark runner clear it without a password prompt.

The runner clears the host page cache
(``sysctl -w vm.drop_caches=3``) right before each engine measurement so a
warm file cache can't skew the memory numbers. That call needs root, so
``bench grant-permissions`` installs a *targeted* NOPASSWD grant — drop-caches
only, **never full sudo** — as a single sudoers drop-in
(``/etc/sudoers.d/edge-containers-drop-caches``). After that one install (one
sudo password prompt) the cache clear runs passwordless on every run. A grant
installed by another checkout is
detected via ``sudo -n -l`` and reused.

``prepare_bench()`` is the "always-before" hook the runner calls before each
engine container: it clears when a grant is present and silently skips (with
a hint) when it isn't — it never prompts and never fails a run.
"""

from __future__ import annotations

import contextlib
import getpass
import os
import re
import subprocess
import sys
import tempfile

# The command the grant covers. sudoers requires the absolute path; the
# property is the kernel page/dentry/inode cache clear.
SYSCTL = "/usr/sbin/sysctl"
DROP_CACHES = "vm.drop_caches=3"
DROP_CACHES_CMD = f"{SYSCTL} -w {DROP_CACHES}"

# The sudoers drop-in this package installs (one file, one rule).
SUDOERS_DROPIN = "/etc/sudoers.d/edge-containers-drop-caches"

# The rule is recognized in a `sudo -l` listing by the sysctl command + the
# property name — so a grant installed under any drop-in name counts.
_GRANTED_RE = re.compile(r"sysctl.*vm\.drop_caches")


def _user() -> str:
    """The invoking user (honors SUDO_USER when already running under sudo)."""
    return os.environ.get("SUDO_USER") or getpass.getuser()


def _rule() -> str:
    """The single sudoers rule: this user runs the cache clear passwordless —
    nothing else (explicitly *not* full sudo)."""
    return f"{_user()} ALL=(ALL) NOPASSWD: {DROP_CACHES_CMD}\n"


def drop_caches_granted() -> bool:
    """True if the current user can run the cache clear passwordless.

    Checked via the non-interactive ``sudo -n -l`` listing (read-only, no
    password prompt). A missing/locked sudo → False (callers skip the clear
    rather than fail)."""
    try:
        p = subprocess.run(
            ["sudo", "-n", "-l"], capture_output=True, text=True, timeout=15
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return p.returncode == 0 and _GRANTED_RE.search(p.stdout) is not None


def drop_caches() -> int:
    """Clear the page cache passwordless (``sudo -n sysctl -w
    vm.drop_caches=3``). Returns the sysctl exit code (0 = cleared).

    The command is passed as separate argv elements (NOT one joined string)
    so ``sudo`` matches it against the NOPASSWD rule — a single token would
    not match and would fall through to the password-requiring rule. ``-n``
    makes a missing grant fail fast instead of prompting."""
    return subprocess.call(["sudo", "-n", SYSCTL, "-w", DROP_CACHES])


def grant_drop_caches(assume_yes: bool = False) -> int:
    """Install the targeted drop-caches NOPASSWD grant (idempotent; exit code).

    Already granted → no-op (0). Otherwise the single rule is written to a
    temp file, validated with ``visudo -cf`` (a bad rule can never reach
    sudoers), then installed as a 0440 root-owned drop-in via one ``sudo
    install`` — the single password prompt. Adds only the cache-clear
    command, never full sudo."""
    if drop_caches_granted():
        print(
            "[bench] drop-caches grant already present — nothing to do", file=sys.stderr
        )
        return 0
    if not assume_yes:
        try:
            ans = (
                input(
                    f"Install a NOPASSWD sudo grant for '{DROP_CACHES_CMD}' "
                    "(cache clear only — not full sudo)? [y/N] "
                )
                .strip()
                .lower()
            )
        except EOFError:
            ans = ""
        if ans not in ("y", "yes"):
            print("[bench] aborted — no change", file=sys.stderr)
            return 1

    rule = _rule()
    fd, tmp = tempfile.mkstemp(prefix="edge-containers-sudoers-")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(rule)
        chk = subprocess.run(["visudo", "-cf", tmp], capture_output=True, text=True)
        if chk.returncode != 0:
            print(
                f"[bench] visudo rejected the rule:\n{chk.stdout}{chk.stderr}",
                file=sys.stderr,
            )
            return 1
        rc = subprocess.call(
            [
                "sudo",
                "install",
                "-m",
                "0440",
                "-o",
                "root",
                "-g",
                "root",
                tmp,
                SUDOERS_DROPIN,
            ]
        )
        if rc != 0:
            print(f"[bench] failed to install the drop-in (exit {rc})", file=sys.stderr)
            return rc
        print(f"[bench] installed {SUDOERS_DROPIN}", file=sys.stderr)
        return 0
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp)


def prepare_bench(quiet: bool = False) -> None:
    """The "always-before" clean-cache hook. Called before each engine
    container (and once when the clean window opens) so a warm file cache
    can't skew a measurement. Never raises and never prompts — if the grant
    is missing it prints a hint once and proceeds (the work still happens,
    just on a warm cache)."""
    if not drop_caches_granted():
        if not quiet:
            print(
                "[bench] page-cache clear skipped (grant missing) — run "
                "'bench grant-permissions' to enable it",
                file=sys.stderr,
            )
        return
    rc = drop_caches()
    if rc == 0:
        if not quiet:
            print("[bench] page cache cleared (vm.drop_caches=3)", file=sys.stderr)
    elif not quiet:
        print(
            f"[bench] page-cache clear failed (exit {rc}) — proceeding", file=sys.stderr
        )
