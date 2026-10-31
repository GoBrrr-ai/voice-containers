"""Text normalization — one fixed normalizer for every engine (spec §3.5).

Whisper EnglishTextNormalizer for English, BasicTextNormalizer otherwise.
Name + version are recorded in every result.
"""

from functools import lru_cache

NORMALIZER_NAME = "whisper_english_basic"
NORMALIZER_VERSION = "transformers (vendored normalizer logic, spec 3.5)"


@lru_cache(maxsize=2)
def _get(kind: str):
    try:
        from transformers.models.whisper.english_normalizer import (
            BasicTextNormalizer,
            EnglishTextNormalizer,
        )
    except Exception as e:  # pragma: no cover - deps missing on host
        raise RuntimeError(
            "pip install transformers (bench host venv) for the normalizer"
        ) from e
    if kind == "en":
        english_spelling = {
            "colour": "color",
            "centre": "center",
            "organise": "organize",
        }
        return EnglishTextNormalizer(english_spelling)
    return BasicTextNormalizer()


def normalize(text: str, language: str = "en") -> str:
    kind = "en" if (language or "en").startswith("en") else "basic"
    return _get(kind)(text).strip()
