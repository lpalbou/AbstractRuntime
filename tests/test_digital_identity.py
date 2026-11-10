"""The digital identity record (ADR 0052 draft, framework backlog 1119): `<home>/identity.json`.

Every rule of the runtime half has a test here that goes red when the rule is removed: the
record's bounds and its strict reader, the rev rule, the biography row written before the file,
the recorded default (pure, engraved pool order, HSL colour rule), the reference and where it rides
on a run, the fingerprint formula, the one-sentence summon cue and its place in the prelude.
"""

from __future__ import annotations

from abstractruntime.identity.lifecycle_fence import LifecycleFence  # noqa: E402
import colorsys
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pytest

from abstractruntime.identity import digital_identity as di
from abstractruntime.identity.digital_identity import (
    DEFAULT_STYLE_POOL,
    AvatarChoice,
    DigitalIdentity,
    IdentityInvalid,
    IdentityRef,
    IdentityUnreadable,
    VoiceChoice,
    default_identity,
    identity_at_rev,
    identity_change_cue,
    identity_fingerprint,
    identity_ref,
    parse_identity_ref,
    read_identity,
    read_run_identity_ref,
    record_default_identity,
    stamp_run_identity_ref,
    write_identity,
)
from abstractruntime.identity.fingerprint import ActorFingerprint

pytestmark = pytest.mark.basic

CASTOR = "entity:castor"
QWEN = {"provider": "abstractvoice", "model": "qwen3-tts", "voice": "fr-male-rene"}
PIPER = {"provider": "abstractvoice", "model": "piper", "voice": "en-female-amy", "speed": 1.1}
CHOSEN_AVATAR = AvatarChoice(style="watch-vortex", color="#3a7bd5", source="chosen")


def _voice(spec: Dict[str, Any], source: str = "chosen") -> VoiceChoice:
    return VoiceChoice(source=source, **spec)


def _clock(monkeypatch: pytest.MonkeyPatch, iso: str) -> None:
    moment = datetime.fromisoformat(iso)
    monkeypatch.setattr(di, "_utc_now", lambda: moment)


def _rows(home: Path) -> List[Dict[str, Any]]:
    path = home / "state_history.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write(home: Path, **over: Any) -> DigitalIdentity:
    kwargs: Dict[str, Any] = dict(
        avatar=CHOSEN_AVATAR, voice=_voice(QWEN), set_by="user:laurent", cause="chosen", provenance="test"
    )
    kwargs.update(over)
    return write_identity(home, **kwargs)


@pytest.fixture()
def home(tmp_path: Path) -> Path:
    path = tmp_path / "entities" / "castor"
    path.mkdir(parents=True)
    return path


# ----------------------------------------------------------------------------- write, read, rev


def test_round_trip_rev_one_then_two_with_biography_rows(home: Path, monkeypatch) -> None:
    _clock(monkeypatch, "2026-11-04T09:00:00+00:00")
    first = _write(home)
    assert first.rev == 1 and first.set_at == "2026-11-04T09:00:00+00:00"
    assert read_identity(home) == first
    _clock(monkeypatch, "2026-11-04T10:00:00+00:00")
    second = _write(home, voice=_voice(PIPER))
    assert second.rev == 2
    assert read_identity(home) == second

    rows = _rows(home)
    assert [list(r) for r in rows] == [
        ["marker", "rev", "from", "to", "written_by", "cause", "at", "provenance"]
    ] * 2
    assert rows[0]["marker"] == "identity_changed"
    assert rows[0]["from"] is None and rows[0]["to"] == first.to_dict()
    assert rows[1]["from"] == first.to_dict() and rows[1]["to"] == second.to_dict()
    assert rows[1]["written_by"] == "user:laurent" and rows[1]["cause"] == "chosen"
    assert rows[1]["at"] == second.set_at and rows[1]["provenance"] == "test"


def test_file_shape_is_schema_one(home: Path) -> None:
    record = _write(home)
    data = json.loads((home / "identity.json").read_text(encoding="utf-8"))
    assert data == {
        "schema_version": 1,
        "rev": 1,
        "avatar": {"style": "watch-vortex", "color": "#3a7bd5", "source": "chosen"},
        "voice": {**QWEN, "source": "chosen"},
        "set_at": record.set_at,
        "set_by": "user:laurent",
    }


def test_legacy_home_reads_none_and_nothing_is_created(home: Path) -> None:
    (home / "voice.yaml").write_text("provider: x\nmodel: y\nvoice: z\n", encoding="utf-8")
    assert read_identity(home) is None
    assert identity_change_cue(home, since=None) is None
    assert sorted(p.name for p in home.iterdir()) == ["voice.yaml"]


def test_the_biography_row_lands_before_the_file(home: Path, monkeypatch) -> None:
    first = _write(home)

    def refuse(*_a: Any, **_k: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("abstractruntime.utils.atomic_files.atomic_write_text", refuse)
    with pytest.raises(OSError):
        _write(home, voice=_voice(PIPER))
    assert read_identity(home) == first  # the old file stands
    assert [r["rev"] for r in _rows(home)] == [1, 2]  # the row was written first
    monkeypatch.undo()

    # rev 2 was never in effect: the next write is rev 3, never a reused 2,
    # and rev 2 does not resolve.
    third = _write(home, voice=_voice(PIPER))
    assert third.rev == 3
    with pytest.raises(LookupError):
        identity_at_rev(home, 2)
    assert identity_at_rev(home, 1) == first


def test_a_failed_biography_append_writes_nothing(home: Path) -> None:
    (home / "state_history.jsonl").mkdir()  # the append cannot open it
    with pytest.raises(OSError):
        _write(home)
    assert not (home / "identity.json").exists()


def _pausing_check(first_in: "threading.Event", release: "threading.Event"):
    """A check hook that pauses the FIRST writer inside its write until `release` is set (or
    0.5 s pass), so a second writer can try to interleave."""
    import threading

    lock = threading.Lock()
    state = {"paused": False}

    def check(_record: DigitalIdentity) -> None:
        with lock:
            pause = not state["paused"]
            state["paused"] = True
        if pause:
            first_in.set()
            release.wait(0.5)

    return check


def test_racing_writers_never_share_a_rev(home: Path) -> None:
    import threading

    record_default_identity(home, entity_id=CASTOR, gateway_default_voice=QWEN, provenance="birth")
    first_in, second_done = threading.Event(), threading.Event()
    check = _pausing_check(first_in, second_done)
    results: Dict[str, DigitalIdentity] = {}

    def first() -> None:
        results["a"] = _write(home, set_by="user:a", check=check)

    def second() -> None:
        first_in.wait(5)
        results["b"] = _write(home, set_by="user:b", voice=_voice(PIPER), check=check)
        second_done.set()

    threads = [threading.Thread(target=first), threading.Thread(target=second)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)

    # Without the lock the second writer reads rev 1 while the first is paused, and both write
    # rev 2. With it, the second waits: revs 2 then 3, each row moving on from the one before.
    rows = _rows(home)
    assert [r["rev"] for r in rows] == [1, 2, 3]
    assert [r["written_by"] for r in rows] == ["default", "user:a", "user:b"]
    assert rows[2]["from"] == rows[1]["to"]
    assert (results["a"].rev, results["b"].rev) == (2, 3)
    assert read_identity(home) == results["b"]
    assert identity_at_rev(home, 2) == results["a"]
    assert (home / ".identity.lock").exists()


def test_racing_default_passes_write_it_once(home: Path) -> None:
    import threading

    first_in, second_done = threading.Event(), threading.Event()
    check = _pausing_check(first_in, second_done)
    results: List[Any] = []

    def first() -> None:
        results.append(record_default_identity(
            home, entity_id=CASTOR, gateway_default_voice=QWEN, provenance="birth", check=check))

    def second() -> None:
        first_in.wait(5)
        results.append(record_default_identity(
            home, entity_id=CASTOR, gateway_default_voice=QWEN, provenance="boot pass", check=check))
        second_done.set()

    threads = [threading.Thread(target=first), threading.Thread(target=second)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)

    assert [r["rev"] for r in _rows(home)] == [1]
    assert sorted(written for _record, written in results) == [False, True]
    assert read_identity(home).rev == 1


def test_check_hook_refuses_before_anything_is_written(home: Path) -> None:
    seen: List[DigitalIdentity] = []

    def gateway_check(record: DigitalIdentity) -> None:
        seen.append(record)
        raise ValueError("voice not downloaded")

    with pytest.raises(ValueError, match="not downloaded"):
        _write(home, check=gateway_check)
    assert seen and seen[0].rev == 1
    assert not (home / "identity.json").exists() and _rows(home) == []


def test_identity_at_rev(home: Path) -> None:
    first = _write(home)
    second = _write(home, avatar=AvatarChoice(style="watch-halo", color="#112233", source="chosen"))
    assert identity_at_rev(home, 1) == first
    assert identity_at_rev(home, 2) == second
    for bad in (0, 3, -1):
        with pytest.raises(LookupError):
            identity_at_rev(home, bad)


def test_life_sleep_stats_ignores_identity_rows(home: Path) -> None:
    from abstractruntime.identity.life import life_sleep_stats, write_entity_state

    write_entity_state(home, "asleep", written_by="self", fence=LifecycleFence.admit(home))
    before = life_sleep_stats(home)
    _write(home)
    _write(home, voice=_voice(PIPER))
    assert life_sleep_stats(home) == before


# ----------------------------------------------------------------------------- bounds


BOUNDS = [
    ("avatar.style", dict(avatar=AvatarChoice("Watch-halo", "#3a7bd5", "chosen"))),
    ("avatar.style", dict(avatar=AvatarChoice("9halo", "#3a7bd5", "chosen"))),
    ("avatar.style", dict(avatar=AvatarChoice("a" * 65, "#3a7bd5", "chosen"))),
    ("avatar.style", dict(avatar=AvatarChoice("watch_halo", "#3a7bd5", "chosen"))),
    ("avatar.color", dict(avatar=AvatarChoice("watch-halo", "#3A7BD5", "chosen"))),
    ("avatar.color", dict(avatar=AvatarChoice("watch-halo", "3a7bd5", "chosen"))),
    ("avatar.color", dict(avatar=AvatarChoice("watch-halo", "#3a7bd", "chosen"))),
    ("avatar.source", dict(avatar=AvatarChoice("watch-halo", "#3a7bd5", "picked"))),
    ("voice.provider", dict(voice=VoiceChoice("", "m", "v", "chosen"))),
    ("voice.model", dict(voice=VoiceChoice("p", "m" * 129, "v", "chosen"))),
    ("voice.voice", dict(voice=VoiceChoice("p", "m", "v\n", "chosen"))),
    ("voice.speed", dict(voice=VoiceChoice("p", "m", "v", "chosen", speed=0))),
    ("voice.speed", dict(voice=VoiceChoice("p", "m", "v", "chosen", speed=4.01))),
    ("voice.speed", dict(voice=VoiceChoice("p", "m", "v", "chosen", speed=True))),
    ("voice.quality_preset", dict(voice=VoiceChoice("p", "m", "v", "chosen", quality_preset="q" * 33))),
    ("voice.source", dict(voice=VoiceChoice("p", "m", "v", "picked"))),
    ("set_by", dict(set_by="laurent")),
    ("set_by", dict(set_by="user:laurent:x")),
    ("set_by", dict(set_by="user: laurent")),
    ("voice", dict(voice=None)),
    ("voice", dict(voice=None, set_by="default")),  # the avatar is chosen: not the recorded default
    ("cause", dict(cause=" ")),
    ("provenance", dict(provenance="")),
]


@pytest.mark.parametrize("field,over", BOUNDS)
def test_every_bound_refuses_naming_the_field(home: Path, field: str, over: Dict[str, Any]) -> None:
    with pytest.raises(IdentityInvalid) as caught:
        _write(home, **over)
    assert caught.value.field == field
    assert not (home / "identity.json").exists() and _rows(home) == []


def test_the_upper_bounds_themselves_are_accepted(home: Path) -> None:
    record = _write(
        home,
        avatar=AvatarChoice("a" + "b" * 63, "#000000", "chosen"),
        voice=VoiceChoice("p" * 128, "m" * 128, "v" * 128, "chosen", speed=4, quality_preset="q" * 32),
        set_by="operator",
    )
    assert read_identity(home) == record


def test_voice_null_only_on_the_recorded_default(home: Path) -> None:
    record = _write(home, avatar=AvatarChoice("watch-halo", "#3a7bd5", "default"), voice=None, set_by="default")
    assert record.voice is None and read_identity(home) == record


# ----------------------------------------------------------------------------- unreadable files


def _good_dict() -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "rev": 1,
        "avatar": {"style": "watch-halo", "color": "#3a7bd5", "source": "chosen"},
        "voice": {**QWEN, "source": "chosen"},
        "set_at": "2026-11-04T09:00:00+00:00",
        "set_by": "user:laurent",
    }


def _mutated(change) -> str:
    data = _good_dict()
    change(data)
    return json.dumps(data)


UNREADABLE = {
    "newer schema": _mutated(lambda d: d.update(schema_version=2)),
    "schema zero": _mutated(lambda d: d.update(schema_version=0)),
    "unknown top field": _mutated(lambda d: d.update(nickname="cas")),
    "unknown avatar field": _mutated(lambda d: d["avatar"].update(shape="kitty")),
    "unknown voice field": _mutated(lambda d: d["voice"].update(pitch=2)),
    "partial": _mutated(lambda d: d.pop("set_by")),
    "partial voice": _mutated(lambda d: d["voice"].pop("model")),
    "rev zero": _mutated(lambda d: d.update(rev=0)),
    "naive time": _mutated(lambda d: d.update(set_at="2026-11-04T09:00:00")),
    "offset time": _mutated(lambda d: d.update(set_at="2026-11-04T09:00:00+02:00")),
    "corrupt": '{"schema_version": 1, "rev": ',
    "not an object": "[1, 2]",
    "too large": json.dumps(_good_dict()) + " " * 4096,
}


@pytest.mark.parametrize("name", sorted(UNREADABLE))
def test_unreadable_files_refuse_on_read_and_on_write(home: Path, name: str) -> None:
    (home / "identity.json").write_text(UNREADABLE[name], encoding="utf-8")
    with pytest.raises(IdentityUnreadable):
        read_identity(home)
    with pytest.raises(IdentityUnreadable):
        _write(home)
    with pytest.raises(IdentityUnreadable):
        record_default_identity(home, entity_id=CASTOR, gateway_default_voice=None, provenance="boot")
    assert (home / "identity.json").read_text(encoding="utf-8") == UNREADABLE[name]
    assert _rows(home) == []


def test_the_good_dict_itself_reads(home: Path) -> None:
    # The unreadable cases above differ from this one by one change each.
    (home / "identity.json").write_text(json.dumps(_good_dict()), encoding="utf-8")
    assert read_identity(home).rev == 1


def test_newer_schema_names_the_upgrade(home: Path) -> None:
    (home / "identity.json").write_text(UNREADABLE["newer schema"], encoding="utf-8")
    with pytest.raises(IdentityUnreadable, match="is newer than this runtime knows .*upgrade AbstractRuntime"):
        read_identity(home)


# ----------------------------------------------------------------------------- the recorded default


def _expected_default(entity_id: str) -> tuple:
    """The ADR 0052 section 2 rule, written out independently of the module."""
    seed = hashlib.sha256(entity_id.encode("utf-8")).digest()
    pool = ["watch-halo", "watch-vortex", "watch-nebula", "watch-rings",
            "watch-galaxy", "watch-lotus", "watch-crystal", "watch-aurora"]
    hue = (seed[1] * 256 + seed[2]) % 360
    # HSL -> RGB, the textbook formula (not colorsys), rounded half up.
    s, l = 0.62, 0.52
    c = (1 - abs(2 * l - 1)) * s
    x = c * (1 - abs((hue / 60) % 2 - 1))
    m = l - c / 2
    r, g, b = [(c, x, 0), (x, c, 0), (0, c, x), (0, x, c), (x, 0, c), (c, 0, x)][hue // 60]
    color = "#" + "".join(f"{int((v + m) * 255 + 0.5):02x}" for v in (r, g, b))
    return pool[seed[0] % 8], color


def test_default_pool_order_is_engraved() -> None:
    assert DEFAULT_STYLE_POOL == (
        "watch-halo", "watch-vortex", "watch-nebula", "watch-rings",
        "watch-galaxy", "watch-lotus", "watch-crystal", "watch-aurora",
    )


@pytest.mark.parametrize(
    "entity_id",
    [CASTOR, "entity:pollux", "entity:castor@h-001", "entity:castor@h-002", "entity:janus", "entity:hypnos",
     "entity:ariadne", "entity:ephemeral", "entity:a", "entity:z@x"],
)
def test_default_follows_the_written_rule(entity_id: str) -> None:
    avatar, _voice_choice = default_identity(entity_id, None)
    assert (avatar.style, avatar.color) == _expected_default(entity_id)
    assert avatar.source == "default"


def test_default_golden_values() -> None:
    # Pinned once: a change of the seed, the pool, the hue rule or the rounding turns this red.
    assert default_identity(CASTOR, None)[0] == AvatarChoice(*_expected_default(CASTOR), "default")
    assert _expected_default(CASTOR) == GOLDEN_CASTOR


# Worked by hand: seed[0] % 8 = 2; hue = (146 * 256 + 81) % 360 = 17; HSL(17, 62 %, 52 %) = #d06439.
GOLDEN_CASTOR = ("watch-nebula", "#d06439")


def test_default_is_pure() -> None:
    first = default_identity(CASTOR, QWEN)
    assert all(default_identity(CASTOR, QWEN) == first for _ in range(1000))


def test_two_homes_of_one_spark_get_different_colours() -> None:
    one = default_identity("entity:castor@h-001", None)[0]
    two = default_identity("entity:castor@h-002", None)[0]
    assert one.color != two.color


def test_default_voice_is_the_gateway_value_or_null() -> None:
    _avatar, voice = default_identity(CASTOR, PIPER)
    assert voice == VoiceChoice(source="default", **PIPER)
    assert default_identity(CASTOR, None)[1] is None
    with pytest.raises(IdentityInvalid) as caught:
        default_identity(CASTOR, {"provider": "abstractvoice", "model": "piper"})
    assert caught.value.field == "voice.voice"


def test_every_pool_style_is_drawable_in_the_avatar_catalog() -> None:
    abstractavatar = pytest.importorskip(
        "abstractavatar", reason="abstractavatar is not installed: the pool cannot be checked against its catalog"
    )
    catalog = {record["id"]: record for record in abstractavatar.catalog()}
    for style in DEFAULT_STYLE_POOL:
        assert style in catalog, style
        assert catalog[style]["watch_code"], style


def test_record_default_identity_is_idempotent_and_never_overwrites(home: Path) -> None:
    record, written = record_default_identity(home, entity_id=CASTOR, gateway_default_voice=QWEN, provenance="birth")
    assert written is True
    assert record.rev == 1 and record.set_by == "default"
    assert record.avatar.source == "default" and record.voice.source == "default"
    assert _rows(home)[0]["cause"] == "recorded_default" and _rows(home)[0]["written_by"] == "default"
    raw = (home / "identity.json").read_bytes()

    # A second pass, even after the gateway's default voice changed, leaves it untouched.
    again, written_again = record_default_identity(
        home, entity_id=CASTOR, gateway_default_voice=PIPER, provenance="boot pass"
    )
    assert written_again is False and again == record
    assert (home / "identity.json").read_bytes() == raw
    assert len(_rows(home)) == 1


def test_record_default_without_a_gateway_voice(home: Path) -> None:
    record, _ = record_default_identity(home, entity_id=CASTOR, gateway_default_voice=None, provenance="boot pass")
    assert record.voice is None and read_identity(home) == record


# ----------------------------------------------------------------------------- the reference


def test_ref_string_form_round_trips() -> None:
    ref = identity_ref(CASTOR, 3)
    assert str(ref) == "entity:castor#3" and ref.to_dict() == {"of": "entity:castor", "rev": 3}
    assert parse_identity_ref("entity:castor#3") == ref
    assert parse_identity_ref("session:sess_9f2a#1") == IdentityRef("session:sess_9f2a", 1)


@pytest.mark.parametrize(
    "text", ["castor", "entity:castor", "entity:castor#0", "entity:castor#03", "entity:castor#", "#3",
             "entity:#3", ":castor#3", "entity:cas tor#3", "entity:castor#3a", "entity:castor#-1", "castor#3"],
)
def test_ref_parse_refuses(text: str) -> None:
    with pytest.raises(ValueError):
        parse_identity_ref(text)


def test_ref_rides_beside_the_signed_stamp() -> None:
    stamp = {"entity_id": CASTOR, "channel": "visit", "nonce": "n1"}
    run_vars: Dict[str, Any] = {"_runtime": {"entity": dict(stamp)}}
    assert read_run_identity_ref(run_vars) is None  # never fabricated
    stamp_run_identity_ref(run_vars, identity_ref(CASTOR, 2))
    assert run_vars["_runtime"]["identity_ref"] == {"of": "entity:castor", "rev": 2}
    assert run_vars["_runtime"]["entity"] == stamp  # the signed tuple is untouched
    assert read_run_identity_ref(run_vars) == IdentityRef(CASTOR, 2)
    json.dumps(run_vars)  # JSON-safe, the RunState.vars invariant
    with pytest.raises(ValueError):
        read_run_identity_ref({"_runtime": {"identity_ref": "entity:castor#2"}})


# ----------------------------------------------------------------------------- the fingerprint


MANIFEST = {"slug": "castor", "entity_id": "entity:castor@h-001", "home_id": "h-001", "spark_hash": "ab" * 32}


def test_fingerprint_is_the_adr_formula() -> None:
    fp = identity_fingerprint(MANIFEST, book_head="f00d")
    expected = ActorFingerprint.from_metadata(
        kind="entity",
        display_name="castor",
        metadata={"entity_id": "entity:castor@h-001", "home_id": "h-001", "spark_hash": "ab" * 32},
    )
    assert fp.actor_id == expected.actor_id and fp.actor_id.startswith("ar_") and len(fp.actor_id) == 27
    assert fp.to_dict() == {
        "actor_id": expected.actor_id, "kind": "entity", "display_name": "castor",
        "entity_id": "entity:castor@h-001", "home_id": "h-001", "spark_hash": "ab" * 32, "book_head": "f00d",
    }


def test_fingerprint_is_stable_and_names_the_individual() -> None:
    base = identity_fingerprint(MANIFEST, book_head=None).actor_id
    assert identity_fingerprint(dict(MANIFEST), book_head="later").actor_id == base  # the book grows; who stays
    for key, value in (("entity_id", "entity:castor@h-002"), ("home_id", "h-002"), ("spark_hash", "cd" * 32)):
        assert identity_fingerprint({**MANIFEST, key: value}, book_head=None).actor_id != base, key
    with pytest.raises(ValueError):
        identity_fingerprint({k: v for k, v in MANIFEST.items() if k != "home_id"}, book_head=None)


# ----------------------------------------------------------------------------- the summon cue


NOW = datetime(2026, 11, 4, 18, 0, tzinfo=timezone.utc)


def test_cue_for_the_first_record(home: Path, monkeypatch) -> None:
    _clock(monkeypatch, "2026-11-04T09:00:00+00:00")
    record_default_identity(home, entity_id=CASTOR, gateway_default_voice=QWEN, provenance="birth")
    assert identity_change_cue(home, since=None, now=NOW) == "You were given your look and voice today."
    later = datetime(2026, 11, 6, tzinfo=timezone.utc)
    assert identity_change_cue(home, since=None, now=later) == "You were given your look and voice on 4 November."


def test_cue_for_a_first_record_without_voice(home: Path) -> None:
    record_default_identity(home, entity_id=CASTOR, gateway_default_voice=None, provenance="birth")
    assert identity_change_cue(home, since=None) == "You were given your look today; you have no voice yet."


def test_cue_is_one_sentence_newer_than_the_previous_summon_and_none_otherwise(home: Path, monkeypatch) -> None:
    _clock(monkeypatch, "2026-11-01T09:00:00+00:00")
    record_default_identity(home, entity_id=CASTOR, gateway_default_voice=QWEN, provenance="birth")
    _clock(monkeypatch, "2026-11-04T09:00:00+00:00")
    current = read_identity(home)
    _write(home, avatar=current.avatar, voice=_voice({**QWEN, "voice": "fr-male-rene-2"}))

    cue = identity_change_cue(home, since="2026-11-03T12:00:00+00:00", now=datetime(2026, 11, 5, tzinfo=timezone.utc))
    assert cue == "Your voice changed on 4 November (by laurent): now fr-male-rene-2."
    assert identity_change_cue(home, since="2026-11-04T09:00:00+00:00") is None
    assert identity_change_cue(home, since="2026-11-04T10:00:00Z") is None


def test_cue_names_look_and_both(home: Path, monkeypatch) -> None:
    _clock(monkeypatch, "2026-11-04T09:00:00+00:00")
    first = _write(home)
    _write(home, avatar=AvatarChoice("watch-lotus", "#aa3344", "chosen"), set_by="operator")
    assert identity_change_cue(home, since=None, now=NOW) == (
        "Your look changed today (by the operator): now watch-lotus in #aa3344."
    )
    _write(home, avatar=first.avatar, voice=_voice(PIPER))
    assert identity_change_cue(home, since=None, now=NOW) == (
        "Your look and voice changed today (by laurent): now watch-vortex in #3a7bd5, voice en-female-amy."
    )


def test_cue_never_tells_a_change_that_did_not_land(home: Path, monkeypatch) -> None:
    _clock(monkeypatch, "2026-11-01T09:00:00+00:00")
    _write(home)
    _clock(monkeypatch, "2026-11-04T09:00:00+00:00")
    monkeypatch.setattr("abstractruntime.utils.atomic_files.atomic_write_text", lambda *a, **k: (_ for _ in ()).throw(OSError("x")))
    with pytest.raises(OSError):
        _write(home, voice=_voice(PIPER))
    # The newest row (rev 2) never landed: the cue speaks of rev 1, which is older than this summon.
    assert identity_change_cue(home, since="2026-11-02T00:00:00+00:00") is None


def test_cue_refuses_an_unreadable_identity(home: Path) -> None:
    (home / "identity.json").write_text("{", encoding="utf-8")
    with pytest.raises(IdentityUnreadable):
        identity_change_cue(home, since=None)


# ----------------------------------------------------------------------------- the prelude


def _engrammed_home(home_dir: Path, entity_id: str = "entity:castor@home-test") -> Path:
    pytest.importorskip("abstractmemory")
    import yaml
    from abstractmemory import DEFAULT_SPARK_TEMPLATE, MemorySystem, SQLiteJournal, SQLiteTripleStore, engram

    spark = copy.deepcopy(dict(DEFAULT_SPARK_TEMPLATE))
    spark["name"] = "Castor"
    home_dir.mkdir(parents=True, exist_ok=True)
    (home_dir / "spark.yaml").write_text(yaml.safe_dump(spark, sort_keys=False), encoding="utf-8")
    (home_dir / "manifest.json").write_text(json.dumps({"entity_id": entity_id}), encoding="utf-8")
    store = SQLiteTripleStore(home_dir / "memory.sqlite3")
    journal = SQLiteJournal(home_dir / "memory.sqlite3")
    ms = MemorySystem(store=store, journal=journal, task_queue=True)
    assert engram(ms, spark, owner_id=entity_id).created is True
    store.close()
    journal.close()
    return home_dir


def test_prelude_carries_the_cue_in_place(tmp_path: Path) -> None:
    from abstractruntime.identity.chat import open_home
    from abstractruntime.identity.prelude import render_summon_prelude

    home_dir = _engrammed_home(tmp_path / "entities" / "castor")
    record_default_identity(home_dir, entity_id="entity:castor@home-test", gateway_default_voice=QWEN, provenance="birth")
    home = open_home(home_dir, fence=None)
    try:
        seq = home.ms.current_seq()
        plain = render_summon_prelude(home.ms, home.diary, entity_id=home.entity_id, budget=1600, spark=home.spark)
        told = render_summon_prelude(
            home.ms, home.diary, entity_id=home.entity_id, budget=1600, spark=home.spark, home_dir=home_dir
        )
        quiet = render_summon_prelude(
            home.ms, home.diary, entity_id=home.entity_id, budget=1600, spark=home.spark, home_dir=home_dir,
            identity_since="2999-01-01T00:00:00+00:00",
        )
        assert home.ms.current_seq() == seq  # still a pure read
    finally:
        home.close()

    sentence = "You were given your look and voice today."
    assert "identity" not in plain["sections"] and sentence not in plain["text"]
    assert told["sections"]["identity"] == sentence
    assert told["text"].count(sentence) == 1
    lines = told["text"].splitlines()
    at = lines.index(sentence)
    assert at > 0 and lines[at - 1].startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9."))
    assert told["text"].index(sentence) < told["text"].index("</identity_prelude>")
    assert "identity" not in quiet["sections"] and sentence not in quiet["text"]


def test_prelude_names_an_unreadable_identity_and_still_summons(tmp_path: Path) -> None:
    from abstractruntime.identity.chat import open_home
    from abstractruntime.identity.prelude import render_summon_prelude

    home_dir = _engrammed_home(tmp_path / "entities" / "castor")
    (home_dir / "identity.json").write_text("{", encoding="utf-8")
    home = open_home(home_dir, fence=None)
    try:
        out = render_summon_prelude(
            home.ms, home.diary, entity_id=home.entity_id, budget=1600, spark=home.spark, home_dir=home_dir
        )
    finally:
        home.close()
    assert out["refused"] is False
    assert any("identity cue unavailable" in w for w in out["warnings"])


class _LLM:
    def generate(self, *, messages, system_prompt):  # pragma: no cover - never called here
        raise AssertionError("no model call while opening a session")


def test_chat_session_summon_carries_the_cue(tmp_path: Path) -> None:
    from abstractruntime.identity.chat import ChatSession, open_home

    home_dir = _engrammed_home(tmp_path / "entities" / "castor")
    record_default_identity(home_dir, entity_id="entity:castor@home-test", gateway_default_voice=QWEN, provenance="birth")
    for since, expected in ((None, True), ("2999-01-01T00:00:00+00:00", False)):
        home = open_home(home_dir, fence=None)
        try:
            session = ChatSession(
                home, _LLM(), participants=["person:albou"], session_id="s1", context_window=32768,
                out=lambda _s: None, identity_since=since, fence=None,
            )
            assert ("You were given your look and voice today." in session.system_base) is expected
        finally:
            home.close()


def test_visit_summon_carries_the_cue_and_the_door_seed_wins(tmp_path: Path) -> None:
    from abstractruntime.core.models import EffectType
    from abstractruntime.core.runtime import EffectOutcome
    from abstractruntime.identity.entity_runtime import open_entity_runtime
    from abstractruntime.identity.visit_workflow import build_visit_workflow

    home_dir = _engrammed_home(tmp_path / "entities" / "castor")
    record_default_identity(home_dir, entity_id="entity:castor@home-test", gateway_default_voice=QWEN, provenance="birth")
    ert = open_entity_runtime(
        home_dir, extra_handlers={EffectType.LLM_CALL: lambda *a, **k: EffectOutcome.completed({"content": "-"})}, fence=LifecycleFence.admit(home_dir)
    )
    try:
        wf = build_visit_workflow(ert.home, participants=["person:albou"], idle_seconds=3600)
        bases = []
        for seeded in ({}, {"_visit": {"identity_since": "2999-01-01T00:00:00+00:00"}}):
            run_id = ert.runtime.start(workflow=wf, vars=copy.deepcopy(seeded), session_id=f"visit-{len(bases)}")
            state = ert.runtime.tick(workflow=wf, run_id=run_id, max_steps=5)
            bases.append(state.vars["_visit"]["system_base"])
    finally:
        ert.close()
    sentence = "You were given your look and voice today."
    assert sentence in bases[0]
    assert sentence not in bases[1]
