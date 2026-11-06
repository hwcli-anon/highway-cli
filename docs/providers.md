# Model providers

Highway-CLI supports built-in mock and replay providers, Jev and Decision-1
through the Decisions interface, and configuration-driven OpenAI-compatible
chat endpoints.

## Structured contract

In `intent-hierarchical` mode the adapter requests:

1. `{"selected_intent":"<listed intent id>"}`
2. `{"selected_action":"<listed compatible action>"}`

The second request is skipped when only one compatible action exists. Responses
are checked against the candidates supplied in the request. A format retry is
reported separately from transport retries, and an exhausted semantic failure
becomes an auditable fallback rather than a hidden rerun.

## Secret-free configuration

YAML files name an environment variable such as `OPENROUTER_API_KEY`; they never
contain the credential. The loader rejects embedded `api_key`, `token`, and
`secret` fields. Local endpoints may set `api_key_required: false`.

Model availability, identifiers, prices, and service behavior can change. Pin a
specific model identifier for an experiment and record the access date. Treat
the included configurations as examples, not guarantees of current access.
