"""names — user-visible identity: image -> project -> family -> display/color.

The config ids (``parakeet_cpp_0``…) must never leak into user-visible
surfaces; everything here derives from the recorded image + model.
"""

from bench import names


def _row(project, **kw):
    row = {
        "image": f"narandill/{project}:0.1.0",
        "model": "models/whatever.gguf",
        "requested_backend": "cuda",
        "opts": {},
    }
    row.update(kw)
    return row


def test_project_comes_from_image_not_engine_id():
    row = _row("kokoro-tts", engine="kokoro_0")
    assert names.project(row) == "kokoro-tts"


def test_project_falls_back_to_engine_id():
    assert names.project({"engine": "parakeet_cpp_0"}) == "parakeet_cpp_0"


def test_every_registered_package_resolves_full_identity():
    for name in names.PACKAGES:
        row = _row(name)
        fam = names.family(row)
        assert fam, name
        assert names.display(row), name
        assert names.color(row).startswith("#"), name
        assert names.EMOJI.get(fam), f"family {fam} has no emoji"


def test_unknown_package_gets_stable_fallbacks():
    row = _row("totally-new-thing")
    assert names.family(row) == "totally-new-thing"
    assert names.color(row) == "#0072B2"  # the documented fallback hue
    assert names.display(row)  # never empty


def test_medal_only_for_the_top3():
    assert names.medal({"rank": 1}) == names.MEDALS[1]
    assert names.medal({"rank": 2}) == names.MEDALS[2]
    assert names.medal({"rank": 3}) == names.MEDALS[3]
    assert names.medal({"rank": 4}) == ""
    assert names.medal({}) == ""
