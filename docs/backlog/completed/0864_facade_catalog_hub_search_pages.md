# 0864 — The config facade passes Hugging Face search pages through to AbstractCore

**Status**: completed (2026-10-29, unreleased) · **Priority**: P1 · **Created**: 2026-10-29
**Cross-package**: AbstractCore backlog 0864 (PR lpalbou/AbstractCore#14: `catalog(hub_page=, hub_page_size=)`,
`hub.search`). AbstractGateway's console Models page pages through the results with it.

## Why
The gateway reaches AbstractCore's catalog through `config_facade.model_catalog`, which had no paging arguments. The
console could not ask for a second page of Hugging Face results.

## What shipped
- `config_facade.model_catalog(..., hub_page=None, hub_page_size=None)`: the arguments are passed only when given.
  The facade checks the installed `catalog` signature first. A catalog without them raises `AbstractCoreTooOld`
  ("Hugging Face search pages requires an AbstractCore whose catalog accepts hub_page") rather than silently
  answering page 1 again.
- `AbstractCoreTooOld(..., required=)`: a description replaces the floor when no released version carries the
  feature yet.
- Release order: AbstractCore with 0864, then this, then the gateway that sends `hub_page`. Each consumer's floor is
  raised at release time.

## Evidence
- `tests/test_config_facade_models_engines.py`, both tests mutation-checked:
  - `test_reads_pass_arguments_through_unchanged`: an AbstractCore without `hub_page` makes the facade refuse in
    words. Red without the signature check.
  - `test_catalog_pages_pass_through_to_an_abstractcore_that_has_them`: page 3 of size 10 is passed through, and no
    page asked leaves AbstractCore's default. Red when the paging kwargs are dropped.
- With the AbstractCore branch on the path, the real `catalog` signature has `hub_page`.
