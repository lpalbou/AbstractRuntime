# Backlog (maintainers)

This README currently serves as the backlog overview for this repository.

This folder contains a structured backlog used during development. Items are grouped as:
- `completed/` — implemented work items (what shipped, with implementation pointers)
- `planned/` — committed future work items that still match current runtime reality
- `proposed/` — uncommitted ideas and follow-on risks worth preserving
- `deprecated/` — historical backlog notes (superseded)

If you are new to the project, start with `../README.md` and `../architecture.md` instead.

## Counts

- Planned: 5
- Proposed: 3
- Completed: 33
- Deprecated: 13
- Recurrent: 0

## Next recommended work

1. `planned/018_workspace_access_policy_for_media_and_tools.md`
   Keep workspace and tool policy explicit while Gateway extracts its local
   workspace helpers.
2. `planned/014_remote_tool_worker_executor.md`
   The public ToolExecutor path is still the larger follow-on after the current
   Gateway boundary cleanup.
3. `proposed/0036_local_media_residency_bridge_to_core_residency.md`
   Local Runtime media residency should relay live Core-owned truth for image/TTS/STT or stay unsupported when no
   Core-owned local media residency facade is wired.
4. `proposed/0031_runtime_tool_spec_adapters_for_gateway_and_mcp.md`
   The comms/Telegram boundary cleanup is now complete; the remaining lower
   pressure follow-up is a Runtime-owned tool-spec surface if Gateway or the
   MCP worker still need one after adoption.
5. `proposed/0038_core_server_pool_residency_affinity.md`
   Future-only topology item for multiple AbstractCore servers, server identity, and residency routing affinity.

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

## Proposed

| ID | Item |
|----|------|
| 0031 | `proposed/0031_runtime_tool_spec_adapters_for_gateway_and_mcp.md` |
| 0036 | `proposed/0036_local_media_residency_bridge_to_core_residency.md` |
| 0038 | `proposed/0038_core_server_pool_residency_affinity.md` |

## Deprecated

See `deprecated/DEPRECATED_README.md` for context on the deprecated backlog set.
Recent deprecation:
- `deprecated/0034_agent_runtime_convenience_constructor.md`
