"""Per-modality capability defaults reach media execution, and never clobber a pin.

ONE STORE, ONE TABLE. The mapping from a media output spec to the capability
default route that holds its default lives in AbstractCore
(`config/capability_defaults.py::_OUTPUT_ROUTE_TABLE`). The Runtime used to keep
a second copy that had drifted from it: it minted `output.voice.tts`,
`input.voice.stt`, `output.music.text_to_music` and `output.sound.text_to_sound`
-- keys `set_capability_default` can never persist, because none of those task
names are in `CAPABILITY_ROUTE_TASKS`. Every one fell through to the broad
modality key, so the table was right only by accident, and it had no `scene3d`
row at all.
"""

from __future__ import annotations

import pytest

from abstractruntime.integrations.abstractcore.llm_client import (
    _output_default_route_keys,
    _with_capability_default_route,
)
from abstractruntime.integrations.abstractcore.output_specs import (
    capability_default_route_keys_for_spec,
)


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ({"modality": "image", "task": "image_generation"}, ("output.image.text_to_image", "output.image")),
        ({"modality": "image", "task": "image_edit"}, ("output.image.image_to_image", "output.image")),
        ({"modality": "image", "task": "image_upscale"}, ("output.image.image_upscale", "output.image")),
        ({"modality": "video", "task": "text_to_video"}, ("output.video.text_to_video", "output.video")),
        ({"modality": "video", "task": "image_to_video"}, ("output.video.image_to_video", "output.video")),
        ({"modality": "voice", "task": "tts"}, ("output.voice", None)),
        ({"modality": "voice", "task": "stt"}, ("input.voice", None)),
        ({"modality": "music", "task": "music_generation"}, ("output.music", None)),
        ({"modality": "sound", "task": ""}, ("output.sound", None)),
        # scene3d had NO row in the runtime's old copy of the table.
        ({"modality": "scene3d", "task": ""}, ("output.scene3d.text_to_scene3d", "output.scene3d")),
    ],
)
def test_route_keys_match_cores_table(spec, expected) -> None:
    assert _output_default_route_keys(spec) == expected
    assert capability_default_route_keys_for_spec(spec) == expected


def test_runtime_no_longer_mints_route_keys_the_store_cannot_hold() -> None:
    """The specific dead keys the drifted copy produced."""

    dead = {
        "output.voice.tts",
        "input.voice.stt",
        "output.music.text_to_music",
        "output.sound.text_to_sound",
    }
    for spec in (
        {"modality": "voice", "task": "tts"},
        {"modality": "voice", "task": "stt"},
        {"modality": "music", "task": "music_generation"},
        {"modality": "sound", "task": "sound_generation"},
    ):
        assert not dead.intersection({k for k in _output_default_route_keys(spec) if k})


@pytest.mark.parametrize(
    ("spec", "route_key"),
    [
        ({"modality": "image", "task": "image_generation"}, "output.image.text_to_image"),
        ({"modality": "video", "task": "text_to_video"}, "output.video.text_to_video"),
        ({"modality": "voice", "task": "tts"}, "output.voice"),
        ({"modality": "music", "task": "music_generation"}, "output.music"),
    ],
)
def test_an_unpinned_media_spec_receives_the_configured_default(spec, route_key) -> None:
    defaults = {
        route_key: {
            "provider": "configured-provider",
            "model": "configured-model",
            "base_url": "http://127.0.0.1:9999/v1",
            "source": "abstractcore.capability_defaults",
        }
    }
    merged = _with_capability_default_route(dict(spec), defaults)
    assert merged["provider"] == "configured-provider"
    assert merged["model"] == "configured-model"
    assert merged["base_url"] == "http://127.0.0.1:9999/v1"


def test_a_task_specific_default_falls_back_to_the_broad_modality_route() -> None:
    defaults = {"output.image": {"provider": "broad", "model": "broad-model"}}
    merged = _with_capability_default_route({"modality": "image", "task": "image_upscale"}, defaults)
    assert (merged["provider"], merged["model"]) == ("broad", "broad-model")


def test_the_task_specific_route_outranks_the_broad_one() -> None:
    defaults = {
        "output.image": {"provider": "broad", "model": "broad-model"},
        "output.image.image_upscale": {"provider": "exact", "model": "exact-model"},
    }
    merged = _with_capability_default_route({"modality": "image", "task": "image_upscale"}, defaults)
    assert (merged["provider"], merged["model"]) == ("exact", "exact-model")


@pytest.mark.parametrize("pin", [{"provider": "authored"}, {"model": "authored-model"}, {"base_url": "http://pin/v1"}])
def test_an_authored_pin_is_never_clobbered_by_a_default(pin) -> None:
    """Tier 1 of the cascade: an explicit pin always wins.

    A flow node's `image_provider`/`tts_provider`/... reaches execution on the
    output spec. The default may only fill an ABSENT one.
    """

    defaults = {"output.image.text_to_image": {"provider": "configured", "model": "configured-model"}}
    spec = {"modality": "image", "task": "image_generation", **pin}
    merged = _with_capability_default_route(dict(spec), defaults)
    for key, value in pin.items():
        assert merged[key] == value
    # Nothing else was filled in either: a partially-pinned spec is the
    # author's, not the default's, to complete.
    assert merged == spec


def test_route_options_are_merged_but_never_overwrite_the_spec() -> None:
    defaults = {
        "output.voice": {
            "provider": "abstractvoice",
            "model": "supertonic-3",
            "options": {"voice": "nova", "speed": 1.0},
        }
    }
    merged = _with_capability_default_route({"modality": "voice", "task": "tts", "voice": "authored"}, defaults)
    assert merged["voice"] == "authored"
    assert merged["speed"] == 1.0
    assert merged["provider"] == "abstractvoice"


def test_an_unconfigured_route_leaves_the_spec_alone() -> None:
    spec = {"modality": "image", "task": "image_generation"}
    assert _with_capability_default_route(dict(spec), {}) == spec
    assert _with_capability_default_route(dict(spec), None) == spec
    # An explicit not_configured row is not a default.
    not_configured = {"output.image.text_to_image": {"key": "output.image.text_to_image", "source": "not_configured"}}
    assert _with_capability_default_route(dict(spec), not_configured) == spec


def test_a_source_image_selects_the_edit_route_for_a_bare_generate() -> None:
    assert _output_default_route_keys(
        {"modality": "image", "task": "image_generation"}, has_source_image=True
    ) == ("output.image.image_to_image", "output.image")


# ---------------------------------------------------------------------------
# THE CROSS-BOUNDARY PARITY GATE
# ---------------------------------------------------------------------------
# The defect class this file exists for was SILENT DRIFT across the repo
# boundary: the Runtime emitted route keys AbstractCore's store could not hold,
# and nothing failed -- every bad key just fell through to the broad modality
# key and looked right. The two tests below are the gate. They read BOTH sides
# from AbstractCore at run time (the output vocabulary and the store's spec
# catalog) and drive the RUNTIME's emitter, so a change to either side that the
# other does not follow fails here rather than in production.


def _core_vocabulary():
    from abstractcore.core.output_specs import (
        OUTPUT_MODALITY_ALIASES,
        OUTPUT_TASK_ALIASES,
        OUTPUT_TASK_MODALITIES,
    )

    return OUTPUT_MODALITY_ALIASES, OUTPUT_TASK_ALIASES, OUTPUT_TASK_MODALITIES


def test_every_key_the_runtime_emits_is_one_the_store_can_hold() -> None:
    """Exhaustive over AbstractCore's whole output vocabulary.

    Every (modality, task) AbstractCore can normalize a request into, times
    with/without a source image, must map to a key that appears in
    `iter_capability_default_specs()` -- the store's own catalog. A key outside
    it is a default that can be written nowhere and read nowhere.
    """

    from abstractcore.config.capability_defaults import (
        CAPABILITY_MODALITIES,
        CAPABILITY_ROUTE_TASKS,
        iter_capability_default_specs,
    )

    modality_aliases, task_aliases, task_modalities = _core_vocabulary()
    known_keys = {spec.key for spec in iter_capability_default_specs()}

    pairs = {(modality, task) for task, modality in task_modalities.items()}
    pairs |= set(modality_aliases.values())
    pairs |= {(task_modalities.get(canonical, ""), alias) for alias, canonical in task_aliases.items()}
    pairs |= {(modality, "") for modality in CAPABILITY_MODALITIES}
    pairs |= {("voice", "stt"), ("sound", "sound_generation")}
    pairs.discard(("", ""))

    for modality, task in sorted(pairs):
        if not modality:
            continue
        for has_source_image in (False, True):
            exact, broad = capability_default_route_keys_for_spec(
                {"modality": modality, "task": task}, has_source_image=has_source_image
            )
            assert exact == _output_default_route_keys(
                {"modality": modality, "task": task}, has_source_image=has_source_image
            )[0]
            if exact is None:
                continue
            assert exact in known_keys, f"{modality}/{task or '<bare>'} emitted unknown route key {exact!r}"
            if broad is not None:
                assert broad in known_keys
                assert exact.startswith(f"{broad}.")
            parts = exact.split(".")
            # A `.task` suffix is only legal for the tasks the store persists.
            if len(parts) == 3:
                assert parts[2] in CAPABILITY_ROUTE_TASKS, f"{exact!r} names a task the store cannot persist"


def test_every_generation_task_core_can_infer_has_a_default_route() -> None:
    """The other half of the gate: no generation task may fall through the table.

    A task in the vocabulary with NO route key silently loses the operator's
    default and hands the call back to the plugin's own env-or-openai fallback.
    `voice_clone` -- the task AbstractCore assigns to `output="voice"` plus
    reference audio -- did exactly that when this table was first written.
    Transcription is the one deliberate None: it is provisioned by the
    `input.voice` INPUT route, not by an output route.
    """

    _, _, task_modalities = _core_vocabulary()

    missing = []
    for task, modality in sorted(task_modalities.items()):
        if task == "transcription":
            assert capability_default_route_keys_for_spec({"modality": modality, "task": task})[0] is None
            continue
        exact, _ = capability_default_route_keys_for_spec({"modality": modality, "task": task})
        if not exact:
            missing.append(f"{modality}/{task}")
    assert not missing, f"generation tasks with no capability default route: {missing}"


# ---------------------------------------------------------------------------
# THE MODALITY ROW IS A PARENT, NOT A REMNANT (operator question 2026-08-01)
# ---------------------------------------------------------------------------


def test_runtime_merge_falls_back_to_the_modality_row() -> None:
    """`output.image` alone must serve generate / edit / upscale.

    This is the shape the fresh-install seed writes -- one `output.image`
    value and no task rows -- so if this merge stopped reading the parent, a
    fresh machine would hand every image call back to the plugin's own
    env-or-openai fallback. The pushed payload enumerates EVERY route, so the
    task rows are PRESENT here carrying `source: "not_configured"`; that
    sentinel must not veto the parent.
    """

    from abstractruntime.integrations.abstractcore.llm_client import _with_capability_default_route

    defaults = {
        "output.image": {"key": "output.image", "provider": "mlx-gen", "model": "one-model"},
        "output.image.text_to_image": {"key": "output.image.text_to_image", "source": "not_configured"},
        "output.image.image_to_image": {"key": "output.image.image_to_image", "source": "not_configured"},
        "output.image.image_upscale": {"key": "output.image.image_upscale", "source": "not_configured"},
    }
    for task in ("text_to_image", "image_edit", "image_upscale"):
        routed = _with_capability_default_route({"modality": "image", "task": task}, defaults)
        assert (routed.get("provider"), routed.get("model")) == ("mlx-gen", "one-model"), task


def test_runtime_merge_prefers_the_task_row_over_its_parent() -> None:
    from abstractruntime.integrations.abstractcore.llm_client import _with_capability_default_route

    defaults = {
        "output.image": {"key": "output.image", "provider": "mlx-gen", "model": "parent"},
        "output.image.image_upscale": {
            "key": "output.image.image_upscale",
            "provider": "mlx-gen",
            "model": "upscaler",
        },
    }
    routed = _with_capability_default_route({"modality": "image", "task": "image_upscale"}, defaults)
    assert routed.get("model") == "upscaler"
    # ...and a task with no row of its own still reaches the parent.
    routed = _with_capability_default_route({"modality": "image", "task": "image_edit"}, defaults)
    assert routed.get("model") == "parent"


def test_runtime_merge_never_reads_the_parent_once_every_task_row_is_set() -> None:
    """Why the grid may say `not needed` instead of `not configured`."""

    from abstractruntime.integrations.abstractcore.llm_client import _with_capability_default_route

    defaults = {
        "output.image": {"key": "output.image", "provider": "mlx-gen", "model": "PARENT-MUST-NOT-BE-READ"},
        "output.image.text_to_image": {"key": "output.image.text_to_image", "provider": "mlx-gen", "model": "t2i"},
        "output.image.image_to_image": {"key": "output.image.image_to_image", "provider": "mlx-gen", "model": "i2i"},
        "output.image.image_upscale": {"key": "output.image.image_upscale", "provider": "mlx-gen", "model": "up"},
    }
    for task in ("", "text_to_image", "image_generation", "image_edit", "i2i", "image_upscale", "upscale"):
        for has_source_image in (False, True):
            spec = {"modality": "image", "task": task}
            exact, _ = capability_default_route_keys_for_spec(spec, has_source_image=has_source_image)
            assert exact != "output.image", f"{task or '<bare>'} resolved to the parent row"
        routed = _with_capability_default_route({"modality": "image", "task": task}, defaults)
        assert routed.get("model") != "PARENT-MUST-NOT-BE-READ", task
