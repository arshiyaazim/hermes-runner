# Fazle-AI routing policy (hermes-runner)

This directory holds the **version-controlled, non-secret** AI routing
policy for the Hermes-Runner workload. It is the single source of truth
for "what models/providers are allowed, what the fallback chain is, and
how the runner behaves on transient/permanent failures."

## Layout

```
config/fazle-ai/
  policy.yaml                          # Global non-secret policy (deny lists, allowlist, timeouts, retry matrix)
  workloads/
    hermes-runner.yaml                 # Workload-specific override (currently only Hermes-Runner)
  examples/
    runtime.env.example                # Non-secret runtime settings (copy -> runtime.env, edit per env)
    secrets.env.example                # SECRET REFERENCE SCHEMA ONLY (copy -> secrets.env, populate per env)
config/fazle_ai/
  contracts.py                         # Phase 3 model-independent request/route/audit contract
  policy_loader.py                     # Pure-Python loader/validator (rejects auto/*, etc.)
  retry_policy.py                      # Legacy bounded executor; runtime adapter migration is Phase 3B
```

## What goes here, what doesn't

| In tracked files | In `runtime.env` (gitignored) | In `secrets.env` (gitignored) |
|---|---|---|
| model names, allow/deny globs, timeouts, retry policy, secret_refs (`vault://...`), workload names | host:port, log level, feature flags, timeout overrides, policy file path | actual key values, actual DB URLs, actual cookies |

**NEVER put a real key, token, password, or connection string in any
tracked file in this directory.** If you do, the loader's `REDACT` rule
will print `[REDACTED]` in any error message, but the leak still
happens — fix it before committing.

## How the runner uses this today

`POLICIES_ENABLED` defaults to `false`. With it `false`, the runner
behaves **exactly as before** — purely env-driven, no policy file
consulted. This is the backward-compatible mode that keeps production
running while this directory rolls out.

Phase 3A adds only the pure semantic contract in
`config/fazle_ai/contracts.py`. It performs no I/O and changes no runtime
selection. Business code names a workload and requirements; policy adapters
later resolve compatible routes. The existing retry executor remains available
as a legacy adapter input until Phase 3B unifies its older failure names with
the Phase 3 taxonomy.

To turn it on (after Owner sign-off):
1. Copy `examples/runtime.env.example` to a location of your choice
   (e.g. `/etc/fazle-ai/runtime.env`).
2. Copy `examples/secrets.env.example` to a separate location
   (e.g. `/etc/fazle-ai/secrets.env`).
3. Fill `secrets.env` with real `_REF=vault://...` values (today this is
   `KEY=value`, no Vault yet — see "Current limitations" below).
4. Set `POLICIES_ENABLED=true`, `POLICY_FILE=/etc/fazle-ai/policy.yaml`,
   `WORKLOAD_CONFIG_DIR=/etc/fazle-ai/workloads`,
   `SECRETS_FILE=/etc/fazle-ai/secrets.env`.
5. Restart hermes-runner. The loader validates the policy and refuses
   to start if `auto/*` is reachable.

## Current limitations (intentional, per task scope)

- **No Vault installed.** `_REF` placeholders are honored as plain
  `KEY=value` lookups until a Vault deployment is approved separately.
- **No core cutover in this task.** `fazle-core` continues to use its own
  per-provider modules under `core/app/` and `core/.env`. The
  `workloads/fazle-core.yaml`, `open-webui.yaml`, `media-processor.yaml`
  are deliberately NOT yet created.
- **No secrets moved.** No `.env` file moved, renamed, or read.

## How `auto/*` is blocked

`config/policy_loader.py::_validate_model()` walks every model string
found in any loaded YAML and matches it against the deny_patterns list
using `fnmatch.fnmatch` (so `auto/*` matches `auto/best-fast`,
`auto/claude-opus`, etc.). On a hit, it raises `PolicyValidationError`
with a message that lists **only the policy key names and the offending
model string** — never any secret value, never any auth header. The
runner is expected to surface that error at startup, before any request
is served.

Tests in `tests/test_policy_loader.py` and `tests/test_retry_policy.py`
cover the deny-pattern behavior exhaustively.
