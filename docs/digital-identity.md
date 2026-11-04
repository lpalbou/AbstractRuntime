# Digital identity (an entity's look and voice)

Every entity has one digital identity: the avatar design the apps draw for it, the colour of that
design, and the voice it speaks with. AbstractRuntime keeps it as one small file in the entity's
home, `identity.json`, and owns its format and its writer
(`abstractruntime.identity.digital_identity`). AbstractGateway checks what only it can check (the
design is in the avatar catalog it loaded, the voice is offered and downloaded, the home is not
held for maintenance) and then calls the runtime writer. Apps read the identity through the
gateway. This page documents the runtime half; the decision record is the framework's ADR 0052.

Identity says WHO the entity is. What it is doing right now (activity, poses) is a separate
subject.

## The record

```json
{
  "schema_version": 1,
  "rev": 2,
  "avatar": {"style": "watch-vortex", "color": "#3a7bd5", "source": "chosen"},
  "voice": {"provider": "abstractvoice", "model": "qwen3-tts", "voice": "fr-male-rene", "source": "chosen"},
  "set_at": "2026-11-04T09:00:00+00:00",
  "set_by": "user:laurent"
}
```

| Field | Rule |
|---|---|
| `schema_version` | exactly 1 |
| `rev` | 1 for the first record, one more for each write, never reused |
| `avatar.style` | an abstractavatar design id: a lowercase letter, then up to 63 lowercase letters, digits or dashes |
| `avatar.color` | `#rrggbb`, lowercase hex |
| `avatar.source`, `voice.source` | `default` (the recorded default) or `chosen` |
| `voice` | `provider`, `model` and `voice` together (each non-empty, at most 128 printable characters), optional `speed` (above 0, at most 4) and `quality_preset` (at most 32 characters); or `null` |
| `set_at` | UTC time with its offset, set by the writer's clock |
| `set_by` | `namespace:name` (for example `user:laurent`), `operator`, or `default` |

`voice: null` is allowed only on the recorded default of a gateway that had no default voice
(`set_by: default`, `avatar.source: default`). Such an entity has no voice and says so; it never
borrows one. The whole file is at most 4096 bytes.

A file the runtime cannot read in full raises `IdentityUnreadable`, on read and on write: a newer
`schema_version`, broken JSON, a missing field, an unknown field, or a value out of bounds. It is
never treated as "no identity", and the writer never overwrites it. A home without
`identity.json` reads as `None`: its older `voice.yaml` and `appearance.json` are the gateway's to
read, and the runtime never touches them.

## Writing it

```python
from abstractruntime.identity import AvatarChoice, VoiceChoice, write_identity

record = write_identity(
    home_dir,
    avatar=AvatarChoice(style="watch-vortex", color="#3a7bd5", source="chosen"),
    voice=VoiceChoice(provider="abstractvoice", model="qwen3-tts", voice="fr-male-rene", source="chosen"),
    set_by="user:laurent",
    cause="chosen",
    provenance="PUT /entities/castor/identity",
    check=gateway_checks,  # optional: raise to refuse (catalog, offered voice, downloaded model)
)
```

In this order: the current file is read; the new record gets the next `rev` and `set_at`; every
bound is checked (`IdentityInvalid`, whose `field` names the bad value, for example
`voice.speed`); `check(record)` runs; one row is appended to the home's biography; then the file
is replaced atomically. A refusal before the row writes nothing.

The biography is `<home>/state_history.jsonl`, the same append-only file and field names as the
`phase_changed` rows, so one reader handles both:

```json
{"marker": "identity_changed", "rev": 2, "from": {...rev 1...}, "to": {...rev 2...},
 "written_by": "user:laurent", "cause": "chosen", "at": "2026-11-04T09:00:00+00:00",
 "provenance": "PUT /entities/castor/identity"}
```

The row is on disk before the file moves. If the file then fails to move, the old identity stays,
the row stays, and the next write takes the rev after it: a rev is never given twice.
`identity_at_rev(home_dir, n)` answers what the identity was at rev `n` from the current file or
from these rows (a rev whose file never landed raises `LookupError`), so a copied home knows every
look and voice it ever had. `life_sleep_stats` skips these rows like every other marker row.

One write at a time per home: every write holds `<home>/.identity.lock` from its first read to
the file replace, and a second writer (another thread, another process, the gateway's start-up
pass racing a change from an app) waits its turn. Two writers never read the same rev, so a rev
is never given twice, and two default passes racing on one home write it once. The check hook
runs under the lock: keep it short. The lock file is never removed.

## The recorded default

Every entity has a look and, when the gateway has a default voice, a voice from day one. The
default is recorded, never computed again at read time: once written it does not change when the
gateway's default voice changes.

`default_identity(entity_id, gateway_default_voice)` returns `(avatar, voice)`; it is pure (no
clock, no randomness), so anyone can recompute it:

- seed = SHA-256 of the entity id exactly as the manifest records it (`entity:castor`, or the
  older `entity:castor@<home_id>`);
- design = `DEFAULT_STYLE_POOL[seed[0] % 8]`, the pool being, in this order: `watch-halo`,
  `watch-vortex`, `watch-nebula`, `watch-rings`, `watch-galaxy`, `watch-lotus`, `watch-crystal`,
  `watch-aurora` (all eight have a Watch drawing in abstractavatar 0.1.1);
- colour = HSL(hue = (seed[1] × 256 + seed[2]) mod 360, saturation 62 %, lightness 52 %), each RGB
  channel = floor(c × 255 + 0.5), written `#rrggbb`;
- voice = the gateway's default voice `{provider, model, voice, speed?, quality_preset?}` as a
  value, or `None` when it has none;
- `source: default` on both parts.

Worked example: for `entity:castor` the seed starts `0x1a 0x92 0x51`, so the design is
`26 % 8 = 2` → `watch-nebula` and the hue is `(146 × 256 + 81) % 360 = 17` → `#d06439`.

`record_default_identity(home_dir, entity_id=..., gateway_default_voice=..., provenance=...)`
writes it as the home's first record (`set_by: default`, biography `cause: recorded_default`) and
returns `(record, True)`; on a home that already has a record it returns `(record, False)` and
writes nothing. The gateway calls it at birth, once per existing home when it starts, and on the
first read of a home that start skipped.

## The reference

Apps name an identity by reference, never by a copy of its values: `{"of": <subject>, "rev": <n>}`,
string form `<subject>#<n>`.

```python
from abstractruntime.identity import identity_ref, parse_identity_ref

ref = identity_ref("entity:castor", 3)
str(ref)                                 # "entity:castor#3"
parse_identity_ref("session:sess_9f2a#1")  # a regular agent's identity rides its session
```

`parse_identity_ref` refuses anything else (`castor`, `entity:castor`, `entity:castor#0`,
`entity:castor#03`). On a run the reference rides `vars["_runtime"]["identity_ref"]` as
`{"of", "rev"}`, beside the signed entity stamp (`vars["_runtime"]["entity"]`) and never inside
it: presentation is not authority. `stamp_run_identity_ref(vars, ref)` puts it there;
`read_run_identity_ref(vars)` returns it, or `None` when the run carries none (a reference is never
made up). An old rev resolves through `identity_at_rev`. The replay view copies it onto the run's public
updates (`identity_ref`, see [replay.md](replay.md#public-conversation-messages)).

## The fingerprint

`identity_fingerprint(manifest, book_head=diary.head())` names the individual:
`actor_id = ActorFingerprint.from_metadata(kind="entity", display_name=<slug>,
metadata={entity_id, home_id, spark_hash}).actor_id` (`ar_` + 24 hex), served with the continuity
receipts the home already has: `spark_hash` (the seed), `home_id` (the individual) and `book_head`
(the diary's chain head). It is derived on every read and stored nowhere. A copy of a home with a
new `home_id` is a different individual with a different `actor_id`. There are no keys and no
signatures.

## The summon cue

At a summon the entity is told, in one sentence, about the change that made its current identity:

- "You were given your look and voice today." (the first record)
- "You were given your look today; you have no voice yet." (a first record without a voice)
- "Your voice changed on 4 November (by laurent): now fr-male-rene."
- "Your look changed today (by the operator): now watch-lotus in #aa3344."
- "Your look and voice changed today (by laurent): now watch-vortex in #3a7bd5, voice en-female-amy."

`identity_change_cue(home_dir, since=...)` returns that sentence, or `None`. It speaks of the
biography row of the current rev (never of a row whose file did not land), and only when that row
is newer than `since`, the previous summon's time; with `since=None` (the caller knows no previous
summon) it is always told. Dates are UTC.

`render_summon_prelude(..., home_dir=..., identity_since=...)` puts the sentence after the
identity core and before the diary tail, as a section that may be dropped to fit the budget (it is
fitted first). The prelude stays a pure read. The chat driver (`ChatSession(...,
identity_since=...)`) and the durable visit workflow (`build_visit_workflow(...,
identity_since=...)`, or `_visit.identity_since` seeded by the door at run creation, which wins)
pass the summon's home. An unreadable identity is named in the prelude's warnings; it does not
stop the summon.

The diary stays the entity's own record: nothing here writes a diary entry. The entity may write
one itself after hearing the cue.

## Tests

`tests/test_digital_identity.py` (`-m basic`): the round trip and the row shape, every bound
naming its field, unreadable files refused on read and on write, the row before the file, the rev
never reused, the default rule against an independent HSL computation and a worked value, the pool
order, purity over 1000 calls, two homes of one spark getting different colours, the pool checked
against abstractavatar's catalog (skipped when the package is not installed), idempotent default
writes, the reference format and its place beside the stamp, the fingerprint formula, the cue
sentences and the `since` rule, and the cue in place in the prelude, the chat driver and the visit
workflow.
