# Backlog (maintainers)

This README currently serves as the backlog overview for this repository.

This folder contains a structured backlog used during development. Items are grouped as:
- `completed/` — implemented work items (what shipped, with implementation pointers)
- `planned/` — committed future work items that still match current runtime reality
- `proposed/` — uncommitted ideas and follow-on risks worth preserving
- `deprecated/` — historical backlog notes (superseded)

If you are new to the project, start with `../README.md` and `../architecture.md` instead.

## Counts

- Planned: 20 (8 top-level + 12 in the `runtime_systemic_reliability/` track)
- Proposed: 11 (3 top-level + 8 in the `runtime_systemic_reliability/` track)
- Completed: 33
- Deprecated: 13
- Recurrent: 0

## Topic tracks

- `planned/runtime_systemic_reliability/` + `proposed/runtime_systemic_reliability/`
  — the 2026-07-12 maintainer-requested two-adversary meta-analysis of the
  whole package, split into executable items: converting reliability from
  artisanal per-feature rigor into systemic guarantees before the 24/7
  resident fleet and before 1.0. Source record: `planned/runtime_systemic_reliability/0044_meta_analysis_record.md`.
  Keystone item (do first): `0045_crash_replay_harness_and_ordering_invariant.md`.

## Next recommended work

1. `planned/runtime_systemic_reliability/0045_crash_replay_harness_and_ordering_invariant.md`
   The keystone: kill-and-replay harness + ONE written crash-ordering
   invariant + the two ordering-inversion fixes. Every other track item is a
   consumer.
2. `planned/runtime_systemic_reliability/0046_per_run_driver_exclusion.md`
   Correctness-grade: two drivers can race one run over aliased state today.
3. `planned/runtime_systemic_reliability/0047_indexed_idempotency_lookup.md`
   The deterministic 24/7 scale cliff (O(ledger) scan per effect step).
4. `planned/runtime_systemic_reliability/0048_durable_io_unification.md` +
   `0049_llm_client_mechanical_split.md`
   Round out the P0 band (fsync discipline; the 11.8k-line module).
5. `planned/018_workspace_access_policy_for_media_and_tools.md`
   Keep workspace and tool policy explicit while Gateway extracts its local
   workspace helpers.
6. `planned/014_remote_tool_worker_executor.md`
   The public ToolExecutor path is still the larger follow-on after the current
   Gateway boundary cleanup.

## Completed

| ID | Item |
|----|------|
| 001 | `completed/001_runtime_kernel.md` |
| 002 | `completed/002_persistence_and_ledger.md` |
| 003 | `completed/003_wait_primitives.md` |
| 004 | `completed/004_scheduler_driver.md` |
| 005 | `completed/005_abstractcore_integration.md` |
| 006 | `completed/006_snapshots_bookmarks.md` |
| 007 | `completed/007_provenance_hash_chain.md` |
| 009 | `completed/009_artifact_store.md` |
| 010 | `completed/010_examples_and_composition.md` |
| 011 | `completed/011_subworkflow_support.md` |
| 012 | `completed/012_run_store_query_and_scheduler_support.md` |
| 013 | `completed/013_effect_retries_and_idempotency.md` |
| 016 | `completed/016_runtime_aware_parameters.md` |
| 019 | `completed/019_runtime_host_facade_for_core_operator_surfaces.md` |
| 020 | `completed/020_runtime_gateway_install_boundary.md` |
| 021 | `completed/021_runtime_gateway_env_namespace_cleanup.md` |
| 022 | `completed/022_model_residency_control_plane.md` |
| 023 | `completed/023_truthful_local_media_residency_boundaries.md` |
| 024 | `completed/024_runtime_owned_run_scoped_media_execution.md` |
| 026 | `completed/026_runtime_host_discovery_facade_for_core_catalogs.md` |
| 027 | `completed/027_runtime_durable_bloc_prompt_cache_facade.md` |
| 028 | `completed/028_runtime_bloc_kv_lifecycle_and_pruning.md` |
| 029 | `completed/029_runtime_music_generation_and_discovery_via_abstractcore.md` |
| 0030 | `completed/0030_runtime_host_facades_for_comms_telegram_and_tool_specs.md` |
| 0032 | `completed/0032_runtime_durable_outbound_comms_truth.md` |
| 0033 | `completed/0033_runtime_host_local_prompt_cache_export_import_surface.md` |
| 0035 | `completed/0035_model_residency_provider_truth_for_local_http_clients.md` |
| 0037 | `completed/0037_visualflow_generate_music_node_compiler_parity.md` |
| 0039 | `completed/0039_runtime_music_structure_prompt_bool_contract.md` |
| 0040 | `completed/0040_task_agnostic_local_residency_listing.md` |
| 0041 | `completed/0041_runtime_hardware_extras_avoid_nonpermissive_document_stacks.md` |
| 0042 | `completed/0042_core_vision_upscale_and_parameter_surface.md` |
| 0043 | `completed/0043_runtime_vision_adapter_and_batch_surface.md` |

## Planned

| ID | Item |
|----|------|
| 008 | `planned/008_signatures_and_keys.md` |
| 014 | `planned/014_remote_tool_worker_executor.md` |
| 017 | `planned/017_limit_warnings_and_observability.md` |
| 018 | `planned/018_workspace_access_policy_for_media_and_tools.md` |
| 025 | `planned/025_runtime_retention_and_purge_contract.md` |
| 026* | `planned/026_context_checkpoint_self_compression.md` |
| 027* | `planned/027_quote_provenance_discipline.md` |
| 028* | `planned/028_selective_artifact_rehydration.md` |
| 0044 | `planned/runtime_systemic_reliability/0044_meta_analysis_record.md` (track source record) |
| 0045 | `planned/runtime_systemic_reliability/0045_crash_replay_harness_and_ordering_invariant.md` |
| 0046 | `planned/runtime_systemic_reliability/0046_per_run_driver_exclusion.md` |
| 0047 | `planned/runtime_systemic_reliability/0047_indexed_idempotency_lookup.md` |
| 0048 | `planned/runtime_systemic_reliability/0048_durable_io_unification.md` |
| 0049 | `planned/runtime_systemic_reliability/0049_llm_client_mechanical_split.md` |
| 0050 | `planned/runtime_systemic_reliability/0050_control_sidecar_all_writers.md` |
| 0051 | `planned/runtime_systemic_reliability/0051_runtime_owned_durable_event_mailbox.md` |
| 0052 | `planned/runtime_systemic_reliability/0052_sync_store_event_loop_tripwire.md` |
| 0053 | `planned/runtime_systemic_reliability/0053_bounded_run_vars_growth.md` |
| 0054 | `planned/runtime_systemic_reliability/0054_runtime_health_counters.md` |
| 0055 | `planned/runtime_systemic_reliability/0055_jsonl_store_honesty.md` |

\* 026/027/028 numerically collide with completed items of the same prefix
(legacy three-digit numbering; they predate the four-digit convention).
Flagged for the next hygiene pass — renumber the PLANNED trio into the
four-digit space rather than the completed records.

## Proposed

| ID | Item |
|----|------|
| 0031 | `proposed/0031_runtime_tool_spec_adapters_for_gateway_and_mcp.md` |
| 0036 | `proposed/0036_local_media_residency_bridge_to_core_residency.md` |
| 0038 | `proposed/0038_core_server_pool_residency_affinity.md` |
| 0056 | `proposed/runtime_systemic_reliability/0056_schema_version_and_quarantine_loading.md` |
| 0057 | `proposed/runtime_systemic_reliability/0057_trust_boundary_stamp_registry.md` |
| 0058 | `proposed/runtime_systemic_reliability/0058_terminal_run_archival_helper.md` |
| 0059 | `proposed/runtime_systemic_reliability/0059_memory_span_handler_extraction.md` |
| 0060 | `proposed/runtime_systemic_reliability/0060_visualflow_compiler_decomposition.md` |
| 0061 | `proposed/runtime_systemic_reliability/0061_root_api_deprecation_machinery.md` |
| 0062 | `proposed/runtime_systemic_reliability/0062_dependency_honesty_extras.md` |
| 0063 | `proposed/runtime_systemic_reliability/0063_b3_envelope_lane_unification.md` |

## Deprecated

See `deprecated/DEPRECATED_README.md` for context on the deprecated backlog set.
Recent deprecation:
- `deprecated/0034_agent_runtime_convenience_constructor.md`

## Planning notes

- 2026-07-12: the `runtime_systemic_reliability` track landed from the
  maintainer-requested two-adversary meta-analysis (planned 0044-0055,
  proposed 0056-0063). The analysis record was originally committed as
  `planned/029_runtime_meta_analysis_2026_07_12.md` and was renumbered to
  `0044_meta_analysis_record.md` in the same pass (029 collides with a
  completed item; dates do not belong in filenames). The track README
  carries the unanimous DO-NOT-BUILD list (no async kernel, no tick-loop
  decomposition, no new eventing abstraction, no ledger auto-pruning, no
  client-class merge) — treat it as binding unless the maintainer overrules.
