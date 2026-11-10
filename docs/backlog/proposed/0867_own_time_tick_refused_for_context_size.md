# 0867 — An own-time tick refused for the endpoint's context size stops the loop after three identical failures

> Package: AbstractRuntime (identity/life.py own-time loop; prompt composition for the tick), AbstractCore (endpoint context size, if it can be read)
> Type: bug
> Status: proposed (TO PROMOTE to planned at the next runtime backlog sweep: the operator confirmed it 2026-11-10; the move is a git mv this seat may not run)
> Created: 2026-11-10
> Priority: P2
> Labels: entities, own-time, context, honest-failure

## Summary

2026-11-10 18:03–18:04 (a live home on an LM Studio endpoint): three ticks in a row were refused with HTTP 400
"request (17261 tokens) exceeds the available context size (8192 tokens)" and the loop stopped by failures ("Stopped after
three failed ticks"). The operator pressed Retry twice; each retry failed the same way, because nothing changes between
retries: the prompt is the same size and the endpoint's context is the same. The loop treated a deterministic refusal
as a transient failure.

Operator, 2026-11-10 18:45: "the problem was on me, I increased the context size. but indeed, it would have been nice to
have a more informative message so I understand the issue." For the next runtime cut after 0.19.2. Until then the card
shows the provider's own error sentence (runtime 0.19.2 loop note: "Stopped after three failed ticks: <reason>").

## Scope

1. Recognise the context-size refusal (the provider's structured error: type exceed_context_size_error, n_prompt_tokens,
   n_ctx) and stop after ONE with a note that names both numbers and the remedy ("his tick needs 17,261 tokens; the
   model on <endpoint> serves 8,192: raise the model's context length, or lower his window in Mind"), never three
   identical retries.
2. Before the first tick, when the endpoint can tell its context size (AbstractCore: read it from the server where the
   API exposes it), compare it with the composed prompt and refuse to start with the same note.
3. Do NOT trim the prompt silently (ADR-0026: no truncation; stored text and history are whole).

## Validation

A fake endpoint answering the LM Studio 400 body: the loop stops after one tick with the note naming both numbers; a
retry with the same endpoint answers the same note at once; a retry after the endpoint's n_ctx grows runs.
