### Added

- **`EntityRuntime.readmit(fence)` / `ChatHome.readmit(fence)`: the host's Resume path for a home it keeps open**
  (Fable review F3). An opened home now acts under one `lifecycle_fence.FenceHolder` that its handler table, memory
  commit guard, diary and artifact guards read at every check; `readmit` swaps the fence inside the home state lock
  and refuses one that is not the home's current admission (`StaleLifecycleGeneration`). The gateway's cached
  EntityRuntime per slug no longer refuses every effect forever after Stop -> Resume, and needs no reopen; readmit it
  under the per-slug lock once no run of the old generation is still acting. `ChatHome.fence` reads the holder.
