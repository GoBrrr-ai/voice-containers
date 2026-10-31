"""WER/CER + distribution statistics (PRD §6, §6.3). Pure functions over records."""

from __future__ import annotations
import math
from typing import Iterable


def _edit_counts(ref: list[str], hyp: list[str]) -> tuple[int, int, int]:
    """Levenshtein alignment -> (sub, del, ins)."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    # backtrack
    i, j, sub, del_, ins = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and d[i][j] == d[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1)
        ):
            if ref[i - 1] != hyp[j - 1]:
                sub += 1
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            del_ += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return sub, del_, ins


def corpus_wer_cer(refs: Iterable[str], hyps: Iterable[str]) -> dict:
    """Corpus-level WER/CER = summed errors / summed words (PRD 6.1)."""
    S = D = I = Nw = Nc = CERe = 0
    ser = empty = hall = 0
    n = 0
    for ref, hyp in zip(refs, hyps):
        n += 1
        rw, hw = ref.split(), hyp.split()
        s, d, i = _edit_counts(rw, hw)
        S += s
        D += d
        I += i
        Nw += len(rw)
        if not hw and ref.strip():
            empty += 1
        if len(hw) > 1.5 * max(1, len(rw)) or _repeats(hw):
            hall += 1
        # empty hyp on non-empty ref: _edit_counts already yields del=len(ref),
        # i.e. a sentence error — no special case (PRD 6.1)
        ser += 1 if (s + d + i) > 0 else 0
        rc = list(ref.replace(" ", ""))
        hc = list(hyp.replace(" ", ""))
        cs, cd, ci = _edit_counts(rc, hc)
        CERe += cs + cd + ci
        Nc += len(rc)
    return {
        "n": n,
        "wer": (S + D + I) / Nw if Nw else None,
        "cer": CERe / Nc if Nc else None,
        "sub_rate": S / Nw if Nw else None,
        "del_rate": D / Nw if Nw else None,
        "ins_rate": I / Nw if Nw else None,
        "ser": ser / n if n else None,
        "empty_rate": empty / n if n else None,
        "hallucination_rate": hall / n if n else None,
    }


def _repeats(words: list[str]) -> bool:
    for n in (3, 4):
        for i in range(len(words) - 3 * n + 1):
            if (
                words[i : i + n]
                == words[i + n : i + 2 * n]
                == words[i + 2 * n : i + 3 * n]
            ):
                return True
    return False


def dist_stats(values: list[float], key: str) -> dict:
    """n, mean, std, min, p50..p99, max, IQR, MAD (mean abs successive diff). PRD 6.3."""
    v = sorted(x for x in values if x is not None and math.isfinite(x))
    if not v:
        return {key: None, f"{key}_low_confidence": True}

    def pct(p):
        k = (len(v) - 1) * p
        f, c = math.floor(k), math.ceil(k)
        return v[f] if f == c else v[f] + (v[c] - v[f]) * (k - f)

    mad = (
        sum(abs(b - a) for a, b in zip(v, v[1:])) / (len(v) - 1) if len(v) > 1 else 0.0
    )
    mean = sum(v) / len(v)
    std = (sum((x - mean) ** 2 for x in v) / len(v)) ** 0.5
    return {
        key: pct(0.50),
        f"{key}_n": len(v),
        f"{key}_mean": mean,
        f"{key}_std": std,
        f"{key}_min": v[0],
        f"{key}_max": v[-1],
        f"{key}_p90": pct(0.90),
        f"{key}_p95": pct(0.95),
        f"{key}_p99": pct(0.99),
        f"{key}_iqr": pct(0.75) - pct(0.25),
        f"{key}_mad": mad,
        f"{key}_low_confidence": len(v) < 100,
    }
