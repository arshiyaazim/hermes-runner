# hermes-runner

A tiny local HTTP shim that lets `assistant-backend` (which runs inside
Docker) invoke Hermes Agent, a host-level CLI with real terminal, file,
and `code_execution` access. This is exactly why this shim runs on the
bare host and not inside a container.

The shim is bound to the loopback interface only and is reached through
an nginx-side IP-restricted location on the public vhost — it is never
exposed directly. A shared-secret bearer token is required on top of
that as defense in depth.

The authoritative description of every route, request shape, and
non-obvious behavior lives in `server.py`'s top-level docstring. This
README is a concise orientation map; when the two disagree, the code
wins.

## Endpoints

All endpoints return JSON. Auth is `Authorization: Bearer <secret>` on
every endpoint except `GET /health`.

| Method | Path     | Auth             | Purpose                                           |
| ------ | -------- | ---------------- | ------------------------------------------------- |
| GET    | /health  | none             | Liveness probe — `{"healthy": true}` only.        |
| GET    | /mode    | `RUNNER_SECRET`  | Current mode state + available modes list.        |
| POST   | /mode    | `RUNNER_SECRET`  | Change mode (with optional `ttl_seconds`).       |
| POST   | /audit   | `RUNNER_SECRET`  | Run one of the read-only `audit_tools.*` functions on the host's filesystem (admin-only, gated upstream in `chat.js`). |
| POST   | /run     | dual-secret      | Invoke `hermes chat` once and stream the reply.   |

> Note: there is no `/dashboard` route. The closest equivalent is
> `GET /mode`, which is the closest thing to a runtime state view this
> service exposes (mode, TTL, switch history, available modes).

### `POST /run`

Request body:

```
{
  "hermes_session_id": str | null,
  "message": str,
  "read_only": bool (optional),
  "readonly_key": str | null (optional),
  "caller_scope": str | null (optional, "customer" only)
}
```

Response: `{"reply": str, "hermes_session_id": str}` or `{"error": str}`.

- `hermes_session_id` is Hermes's own auto-generated id (Hermes generates
  it on the first call; the shim returns it; pass it back on every
  subsequent call to `--resume` the same conversation).
- Requests for the same `hermes_session_id` are serialized via a
  per-id lock, so two overlapping messages to the same Hermes
  conversation cannot race.
- `readonly_key` is only consulted when `read_only=true` or
  `caller_scope="customer"`. It picks which lock bucket this call
  serializes against, so independent read-only calls (and different
  customers in the customer scope) do not block each other.
- A read-only call may now pass a `hermes_session_id`, but only one
  this server itself already returned from a prior read-only call. The
  in-memory allowlist is process-local and resets on restart.

### `POST /audit`

```
{"tool": str, "args": dict}
```

`tool` must be one of `audit_tools.AUDIT_TOOLS`'s keys — a closed
allowlist, not arbitrary code execution. The same bearer secret gates
this endpoint, but that secret alone does NOT distinguish an admin chat
user from a non-admin one. The actual admin-only boundary is enforced
upstream in `chat.js`, where audit tool definitions are only added to
the model's tool list and `execute_audit_tool` is only called when
`req.user.role === 'admin'`.

### `POST /mode`

```
{"mode": str, "ttl_seconds": int | null, "scope": str | null}
```

`mode` must be in the allowlist. `ttl_seconds` defaults to 30 minutes
when an elevation is requested without one — it never defaults to
permanent. The new state is written to `~/hermes-runner/current_mode.txt`
and an audit row is appended to `~/hermes-runner/mode_audit.log`.

## Mode system

`server.py` defines a `MODE_TOOLSETS` dict that maps each mode name to
its toolset. The mode is the safety boundary for every `/run` call:

- The persistent mode is read from `current_mode.txt` on disk.
- `POST /mode` may elevate (or change) the persistent mode.
- Mode changes are auditable in `mode_audit.log`.
- When a `/run` request carries `force_mode`, the request uses that
  mode's toolset instead of the persistent one — the persistent state
  is not touched. `force_mode` is only set by:
  - `read_only=true` (relay/audit traffic) → `force_mode="READ"`.
  - `caller_scope="customer"` (fazle-core customer WhatsApp) →
    `force_mode="CUSTOMER"`.

The `CUSTOMER` mode and its toolset are deliberately NOT in
`MODE_TOOLSETS`'s value space that `/mode` exposes — it is only ever
reached via `caller_scope="customer"` on `/run`. This is the mode
where the wrong toolset or preamble would cause the most harm, so it
is the least selectable.

## Dual-secret auth

There are two independent bearer secrets, deliberately not reused:

- `HERMES_RUNNER_SECRET` (env: `RUNNER_SECRET` in code) — for the admin
  Assistant-Platform web UI, the WhatsApp HERMES relay, and any
  `/mode` / `/audit` caller. Falls to `RUNNER_SECRET` empty → reject
  EVERY request and the process still starts; the existing admin path
  is therefore broken until it is configured, which is the desired
  fail-closed default.
- `HERMES_RUNNER_CUSTOMER_SECRET` (env: `RUNNER_CUSTOMER_SECRET`) — for
  fazle-core's `modules.hermes_dispatch` customer WhatsApp handler.
  Empty by default (unset in `hermes-runner/.env`); a
  `caller_scope="customer"` request is rejected outright while this
  is empty, same fail-closed contract — but an empty value here does
  NOT stop the process from starting, because the admin path must
  keep working even before this is configured.

The two secrets are strictly non-interchangeable. A
`caller_scope="customer"` request authenticated with `RUNNER_SECRET`
is rejected, and an admin-scope request authenticated with
`RUNNER_CUSTOMER_SECRET` is rejected. The body must therefore be
parsed before the auth check on `/run` (unlike `/mode` and `/audit`,
which only accept `RUNNER_SECRET`).

## Customer-scope lock

When `caller_scope="customer"` arrives on `/run`:

- It is authenticated with `RUNNER_CUSTOMER_SECRET`, never
  `RUNNER_SECRET`.
- `force_mode` is hard-locked to `CUSTOMER` — the minimal toolset in
  `MODE_TOOLSETS` that excludes `fazle-core`, `file`, `code_execution`,
  `terminal`, and `web`.
- The system preamble is swapped to `CUSTOMER_SYSTEM_PREAMBLE`, not
  the admin `SYSTEM_PREAMBLE`. `NEW_CONVERSATION_GREETING` is never
  applied either.
- A customer-facing reply is per-message stateless and un-personified:
  a `hermes_session_id` is rejected, and a non-default persona is
  rejected. A customer call never resumes, never continues, and never
  touches `current_mode.txt`.
- The customer lock is selected by `readonly_key` (same
  bucket-selection logic as the admin read-only path), so different
  customers do not serialize behind each other or behind the admin
  read-only bucket.

The customer path exists so that a customer-facing WhatsApp reply
cannot, under any combination of auth and code paths, look anything
like the admin Assistant-Platform conversation — different toolset,
different preamble, different secret, no resumability, no shared
state.

## Environment

| Variable                            | Purpose                                     |
| ----------------------------------- | ------------------------------------------- |
| `HERMES_RUNNER_SECRET`              | Admin bearer secret (required to serve).    |
| `HERMES_RUNNER_CUSTOMER_SECRET`     | Customer-scope bearer secret (optional start). |
| `HERMES_RUNNER_PORT`                | Port to bind on the loopback interface.     |
| `HERMES_BIN`                        | Path to the `hermes` CLI executable.        |
| `HERMES_MODE_AUDIT_LOG`             | Path to the mode-change audit log.          |
| `HERMES_RUNNER_MODEL` / `_PROVIDER` | Per-process model/provider override.       |

None of these values are hardcoded in this repository. The shim reads
them from the environment on each request/process start.

## Running

```
python3 server.py
```

The process binds to the loopback interface on the configured port.
Expose it through nginx's IP-restricted `/hermes-internal/` location —
never publish the port directly.

## Source of truth

The module-level docstring at the top of `server.py` is the
authoritative description of every endpoint, every request shape, and
every non-obvious behavior (TTL defaults, lock-bucket selection, the
allowlist for read-only session ids, the customer-path hard-locks).
When this README and that docstring disagree, the docstring is the
source.
