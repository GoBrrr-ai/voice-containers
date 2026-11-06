"""Repo-root conftest: keeps ``bench.*`` imports resolvable under any pytest
invocation. The Makefile uses ``python -m pytest`` (which adds the cwd to
sys.path itself); the root conftest also covers a bare ``pytest`` run.
"""
