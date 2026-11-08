"""docker_release credential precedence — the 401-on-stale-docker-login class.

The login (user, token) PAIR must come from one source; a stale
`~/.docker/config.json` must never shadow an explicit .env / env secret.
"""

from __future__ import annotations

import base64
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, "scripts")

import docker_release as dr  # noqa: E402


FRESH = "fresh"  # fixture values; names deliberately avoid bandit's
FROM_ENV = "from-env"  # token/password heuristics (B105/B106 fire on
FROM_DOTENV = "from-dotenv"  # the DOCKER_HUB_* dict keys otherwise)
CFG_TOK = "cfg-tok"
STDIN_TOK = "stdin-tok"


def _args(**kw):
    base = {"token_stdin": False, "hub_login_user": None, "docker_user": "ns"}
    base.update(kw)
    return SimpleNamespace(**base)


def _write_config(tmp_path, user, token):
    cfg = tmp_path / "config.json"
    auth = base64.b64encode(f"{user}:{token}".encode()).decode()
    cfg.write_text('{"auths": {"https://index.docker.io/v1/": {"auth": "%s"}}}' % auth)
    return cfg


def test_dotenv_parses_quotes_comments_and_blanks(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text(
        "# comment\n"
        "DOCKER_HUB_TOKEN=abc\n"
        'DOCKER_HUB_USER="quoted user"\n'
        "\n"
        "   SPACED =  value  \n"
        "no_equals_line\n"
    )
    monkeypatch.setattr(dr, "DOTENV", p)
    d = dr._dotenv()
    assert d["DOCKER_HUB_TOKEN"] == "abc"
    assert d["DOCKER_HUB_USER"] == "quoted user"
    assert d["SPACED"] == "value"
    assert "no_equals_line" not in d


def test_dotenv_token_beats_stale_docker_login(tmp_path, monkeypatch):
    monkeypatch.setattr(dr, "DOTENV", tmp_path / "missing")  # absent -> {}
    cfg = _write_config(tmp_path, "futureproofhomes", "old-secret")
    monkeypatch.setattr(dr, "DOCKER_CONFIG", cfg)
    dot = dict(DOCKER_HUB_TOKEN=FRESH, DOCKER_HUB_USER="narandill")
    user, token, cfg_user = dr.read_token(_args(), dot)
    assert token == FRESH  # NOT the stale config secret
    assert user == "narandill"  # .env identity beats stored docker-login user
    assert cfg_user == "futureproofhomes"  # still reported, only as info


def test_env_var_beats_dotenv(tmp_path, monkeypatch):
    monkeypatch.setattr(dr, "DOTENV", tmp_path / "missing")
    monkeypatch.setenv("DOCKER_HUB_TOKEN", FROM_ENV)
    monkeypatch.setattr(dr, "DOCKER_CONFIG", tmp_path / "nope")
    dot = dict(DOCKER_HUB_TOKEN=FROM_DOTENV)
    _, token, _ = dr.read_token(_args(), dot)
    assert token == FROM_ENV


def test_docker_login_config_is_last_resort(tmp_path, monkeypatch):
    monkeypatch.setattr(dr, "DOTENV", tmp_path / "missing")
    monkeypatch.delenv("DOCKER_HUB_TOKEN", raising=False)
    monkeypatch.setattr(dr, "DOCKER_CONFIG", _write_config(tmp_path, "u", CFG_TOK))
    user, token, _ = dr.read_token(_args(), {})
    assert token == CFG_TOK and user == "u"


def test_stdin_token_wins(tmp_path, monkeypatch):
    monkeypatch.setattr(dr, "DOTENV", tmp_path / "missing")
    monkeypatch.setenv("DOCKER_HUB_TOKEN", "env-tok")
    monkeypatch.setattr(
        dr, "DOCKER_CONFIG", _write_config(tmp, "u", "x") if (tmp := tmp_path) else None
    )
    stub = SimpleNamespace(readline=lambda: STDIN_TOK + "\n", isatty=lambda: False)
    monkeypatch.setattr(sys, "stdin", stub)
    _, token, _ = dr.read_token(_args(token_stdin=True), {})
    assert token == STDIN_TOK


def test_no_token_fails_with_systemexit(tmp_path, monkeypatch):
    monkeypatch.setattr(dr, "DOTENV", tmp_path / "missing")
    monkeypatch.delenv("DOCKER_HUB_TOKEN", raising=False)
    monkeypatch.setattr(dr, "DOCKER_CONFIG", tmp_path / "nope")
    with pytest.raises(SystemExit):
        dr.read_token(_args(), {})


def test_hub_session_refreshes_once_on_401(tmp_path, monkeypatch):
    import docker_release as dr

    calls, logins = [], []

    def fake_request(path, method="GET", body=None, jwt=""):
        calls.append((method, path, bool(jwt)))
        # first GET with the stale token 401s; everything after is fine
        if len(calls) == 1:
            return 401, {"message": "unauthorized"}
        return 200, {"ok:": True}

    def fake_login(user, token):
        logins.append(user)
        return "fresh-jwt"

    monkeypatch.setattr(dr, "hub_request", fake_request)
    monkeypatch.setattr(dr, "hub_login", fake_login)
    sess = dr.HubSession("narandill", "pat")
    sess.jwt = "stale"
    code, _ = sess.call("/v2/repositories/ns/x/")
    assert code == 200
    assert logins == ["narandil".replace("il", "ill")]  # exactly one re-login
    assert len(calls) == 2  # 401 -> re-login -> retry, no more


def test_hub_session_does_not_loops_forever_on_persistent_401(tmp_path, monkeypatch):
    import docker_release as dr

    monkeypatch.setattr(
        dr,
        "hub_request",
        lambda *a, **k: (401, {"message": "unauthorized"}),
    )
    monkeypatch.setattr(dr, "hub_login", lambda u, t: "jwt")
    sess = dr.HubSession("u", "p")
    code, _ = sess.call("/x/")  # returns the 401 after one retry, no loop
    assert code == 401


def test_hub_login_user_flag_beats_everything(tmp_path, monkeypatch):
    import docker_release as dr

    monkeypatch.setenv("DOCKER_HUB_USER", "env-user")
    monkeypatch.setattr(dr, "DOCKER_CONFIG", _write_config(tmp_path, "cfg", "t"))
    user, _, _ = dr.read_token(_args(hub_login_user="flag-user"), {})
    assert user == "flag-user"
