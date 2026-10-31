"""Common records shared by datasets, engines, runner and metrics (spec §3.1)."""

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass(frozen=True)
class Sample:
    dataset: str  # "librispeech", "libritts_r", "fleurs", "custom"
    dataset_version: str  # HF revision hash or archive checksum
    split: str  # "test-clean", "test", ...
    utt_id: str  # globally unique within dataset
    language: str  # BCP-47, e.g. "en", "pl"
    speaker_id: Optional[str] = None
    gender: Optional[str] = None
    text_raw: str = ""  # original transcript, punctuation/case preserved if available
    text_norm: str = (
        ""  # output of the shared normalizer (filled by harness, not adapter)
    )
    audio_path: Optional[str] = None  # None for TTS-only text sources
    sample_rate: Optional[int] = None
    duration_s: Optional[float] = None
    num_chars: int = 0
    num_words: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SttEvent:
    kind: str  # "partial" | "final"
    text: str
    t_ns: int = 0  # stamped by the runner on return, not by the engine


@dataclass
class SttResult:
    text: str
    words: Optional[list] = None  # (word, start_s, end_s)
    language: Optional[str] = None


def jsonl_write(path, rows) -> None:
    import json

    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def jsonl_read(path):
    import json

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
