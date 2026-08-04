"""
Tiny local HTTP shim so assistant-backend (Docker) can invoke Hermes Agent
(host-level CLI, real terminal/file/code_execution access — this is exactly
why this runs on the bare host and not inside a container).

Bound to 127.0.0.1 only, reached through nginx's IP-restricted
/hermes-internal/ location (see assistant.iamazim.com vhost) — never
exposed directly. A shared-secret bearer token is required on top of that,
as defense in depth.

POST /run  {"hermes_session_id": str|null, "message": str}
  -> {"reply": str, "hermes_session_id": str}  or  {"error": str}

hermes_session_id is Hermes's own auto-generated session id (format like
"20260802_215222_0ecf80"), not a caller-chosen name — `hermes chat -c
<name>` only RESUMES an existing session, it does not create-and-name one
(confirmed live: passing an unknown name fails with "No session found").
Pass null/omit on the first call for a new conversation; Hermes generates
the id and this returns it (parsed from stderr's "session_id: ..." line —
confirmed live that stdout in -Q mode is pure reply text, stderr carries
session_id/progress) — pass that same id back on every subsequent call via
--resume to continue the same conversation.

Requests for the same hermes_session_id are serialized (a lock per id) so
two overlapping messages to the same Hermes conversation can't race.
"""

import datetime
import json
import os
import re
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SESSION_ID_RE = re.compile(r"session_id:\s*(\S+)")

RUNNER_SECRET = os.environ.get("HERMES_RUNNER_SECRET", "")
PORT = int(os.environ.get("HERMES_RUNNER_PORT", "8093"))
HERMES_BIN = os.environ.get("HERMES_BIN", os.path.expanduser("~/.local/bin/hermes"))
TIMEOUT_SECONDS = int(os.environ.get("HERMES_RUN_TIMEOUT", "170"))

# ── Capability mode gate (AI_ROLES_POLICY.md target, built 2026-08-03) ──
# Gates WHAT Hermes can do once invoked, on top of the existing WHO-can-
# invoke-it gate (admin-only JWT, enforced in assistant-backend's hermes.js).
# Enforced by selecting which Hermes toolsets are available — reuses
# Hermes's own existing -t/--toolsets mechanism rather than inventing a
# second permission layer. fazle-core already has a more mature version of
# this same idea (modules/rbac's check_permission(), command -> required-
# role table, fail-closed) — this mirrors that shape at a coarser grain
# (mode -> allowed toolsets) since Hermes's tools aren't as finely
# individually addressable as fazle-core's WhatsApp commands are.
MODES = ["READ", "BUILD", "RUN"]
DEFAULT_MODE = "READ"
MODE_FILE = os.environ.get("HERMES_MODE_FILE", os.path.expanduser("~/hermes-runner/current_mode.txt"))
MODE_TOOLSETS = {
    # READ: query only — same fazle-core/memory/web tools Chat's tool-calling
    # already exposes, nothing that touches the filesystem or a shell.
    "READ": "memory,web,todo,skills,fazle-core",
    # BUILD: read + write files + run sandboxed code, still no raw terminal.
    "BUILD": "memory,web,todo,skills,fazle-core,file,code_execution",
    # RUN: full capability, including unrestricted shell (service
    # restarts, arbitrary commands) — this is the tier the confirm-before-
    # destructive SYSTEM_PREAMBLE matters most for.
    "RUN": "memory,web,todo,skills,fazle-core,file,code_execution,terminal",
}


# ── Mode TTL / auto-revert (Task 6, built 2026-08-04) ───────────────────
# current_mode.txt now holds a small JSON state instead of a bare word:
#   {"mode", "set_at", "expires_at" (nullable), "scope", "set_by"}
# A legacy bare-word file (from before this change) is still accepted and
# treated as a permanent mode (no TTL) — no migration step needed.
#
# Fail-safe contract, unchanged in spirit from before: any problem reading,
# parsing, or trusting the file (missing, unreadable, garbage, corrupted
# JSON, invalid mode value) lands on READ. An EXPIRED mode is additionally
# and actively reverted: the file itself is rewritten back to a permanent
# READ state at the moment expiry is discovered, not just logically
# treated as READ in memory — so the persisted state can never keep
# reporting an elevated mode after its TTL passes, including across a
# process restart (this file is the only source of truth; there's nothing
# else to restart-recover from).
SCOPES = ["TIME", "TASK", "SESSION"]
DEFAULT_SCOPE = "TIME"
TTL_MIN_SECONDS = 60
TTL_MAX_SECONDS = 86400  # 24h
# TASK/SESSION scope has no real lifecycle hook in this single-process,
# no-DB, no-per-session-tracking shim (mode is global, not per Hermes
# session) — approximated as a conservative default TTL instead of a true
# "ends when the task/session ends" boundary. Documented, not oversold.
DEFAULT_SCOPE_TTL_SECONDS = {"TASK": 3600, "SESSION": 1800}

_mode_lock = threading.Lock()
AUDIT_LOG_FILE = os.environ.get(
    "HERMES_MODE_AUDIT_LOG", os.path.expanduser("~/hermes-runner/mode_audit.log")
)


def _now():
    return datetime.datetime.now(datetime.timezone.utc)


def _parse_iso(s):
    return datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))


def _default_state():
    return {"mode": DEFAULT_MODE, "set_at": None, "expires_at": None, "scope": None, "set_by": None}


def _append_audit_log(entry):
    try:
        with open(AUDIT_LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass  # audit logging must never block a mode read/write


def _read_state_unlocked():
    """Returns (state_dict, expired_bool). Caller must hold _mode_lock.
    Never raises — any problem at all resolves to the safe default."""
    try:
        with open(MODE_FILE, "r") as f:
            raw = f.read().strip()
    except OSError:
        return _default_state(), False
    if not raw:
        return _default_state(), False

    try:
        data = json.loads(raw)
        mode = str(data.get("mode", "")).strip().upper()
        if mode not in MODES:
            return _default_state(), False
        expires_at_raw = data.get("expires_at")
        state = {
            "mode": mode,
            "set_at": data.get("set_at"),
            "expires_at": expires_at_raw,
            "scope": data.get("scope") if data.get("scope") in SCOPES else None,
            "set_by": data.get("set_by"),
        }
        if expires_at_raw:
            try:
                if _now() >= _parse_iso(expires_at_raw):
                    return state, True  # expired — caller reverts
            except (ValueError, TypeError):
                return _default_state(), False  # corrupted timestamp -> fail closed
        return state, False
    except (json.JSONDecodeError, TypeError, AttributeError):
        # Legacy bare-word format (pre-TTL) — permanent, no expiry.
        legacy_mode = raw.strip().upper()
        if legacy_mode in MODES:
            return {"mode": legacy_mode, "set_at": None, "expires_at": None, "scope": None, "set_by": None}, False
        return _default_state(), False


def _write_state_unlocked(state):
    # Write-then-rename so a concurrent reader never sees a half-written
    # file (the lock already prevents concurrent writers, but a reader
    # holding no lock — there isn't one, all reads go through this same
    # lock too — this is extra safety for the corrupted-file case).
    tmp_path = MODE_FILE + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(state, f)
    os.replace(tmp_path, MODE_FILE)


def read_mode_state():
    """Thread-safe read with active expiry revert. Returns a dict:
    {mode, set_at, expires_at, scope, set_by, expired, seconds_remaining}."""
    with _mode_lock:
        state, expired = _read_state_unlocked()
        if expired:
            reverted_from = state["mode"]
            new_state = _default_state()
            new_state["set_at"] = _now().isoformat()
            _write_state_unlocked(new_state)
            _append_audit_log({
                "at": _now().isoformat(), "event": "auto_revert_expired",
                "from_mode": reverted_from, "to_mode": DEFAULT_MODE,
            })
            state = new_state
        seconds_remaining = None
        if state.get("expires_at"):
            try:
                seconds_remaining = max(0, int((_parse_iso(state["expires_at"]) - _now()).total_seconds()))
            except (ValueError, TypeError):
                seconds_remaining = None
        return {**state, "expired": expired, "seconds_remaining": seconds_remaining}


def read_current_mode():
    """Back-compat helper: just the mode word, fail-closed. Used by
    run_hermes() — it doesn't need the TTL metadata, only the effective
    mode for toolset selection."""
    return read_mode_state()["mode"]


def write_mode_state(mode, ttl_seconds=None, scope=None, set_by="admin"):
    """Validates and persists a new mode, optionally with a TTL. Raises
    ValueError on any invalid input — callers must not let an invalid
    request silently fall through to a permanent BUILD/RUN grant."""
    mode = (mode or "").strip().upper()
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")

    scope = (scope or "").strip().upper() or None
    if scope is not None and scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}")

    if ttl_seconds is not None:
        try:
            ttl_seconds = int(ttl_seconds)
        except (TypeError, ValueError):
            raise ValueError("ttl_seconds must be a number")
        if ttl_seconds < TTL_MIN_SECONDS or ttl_seconds > TTL_MAX_SECONDS:
            raise ValueError(f"ttl_seconds must be between {TTL_MIN_SECONDS} and {TTL_MAX_SECONDS}")
    elif scope in ("TASK", "SESSION"):
        # No explicit TTL but a lifecycle-shaped scope was requested — this
        # shim has no real task/session-end hook, so approximate with a
        # conservative default rather than granting a permanent elevation.
        ttl_seconds = DEFAULT_SCOPE_TTL_SECONDS[scope]

    now = _now()
    expires_at = (now + datetime.timedelta(seconds=ttl_seconds)).isoformat() if ttl_seconds else None
    # READ is always permanent — a TTL on the safe default is meaningless
    # (it would just "expire" back to itself).
    if mode == DEFAULT_MODE:
        expires_at = None
        scope = None

    state = {"mode": mode, "set_at": now.isoformat(), "expires_at": expires_at, "scope": scope, "set_by": set_by}
    with _mode_lock:
        old_state, _ = _read_state_unlocked()
        _write_state_unlocked(state)
    _append_audit_log({
        "at": now.isoformat(), "event": "mode_change",
        "from_mode": old_state.get("mode"), "to_mode": mode,
        "ttl_seconds": ttl_seconds, "scope": scope, "set_by": set_by,
    })
    return {**state, "expired": False, "seconds_remaining": ttl_seconds}

# There is no CLI flag to select a named ~/.hermes/config.yaml persona
# per-invocation (checked: `hermes chat --help` has no --personality/--persona
# flag), and setting one globally would affect the Owner's own normal CLI use
# of Hermes too, not just this integration. Instead, the behavioral contract
# is injected as a preamble on every single call (not just the first turn) —
# Hermes's own context compression (config.yaml: compression.threshold/
# target_ratio) can eventually drop an early message from a long-running
# session, so repeating it every turn is the more robust choice, at the cost
# of a few extra tokens per call.
#
# Note on "confirm before destructive": Hermes has a real, built-in
# dangerous-command approval gate (confirmed via `hermes chat --help`'s
# --yolo flag: "Bypass all dangerous command approval prompts"). This
# runner deliberately never passes --yolo, so that gate stays active — in
# headless (-Q, no TTY) mode a genuinely dangerous command will hit that
# gate and this call will simply time out rather than execute, which is
# fail-safe but surfaces as a timeout error rather than a clean "please
# confirm" reply. That's a known v1 UX gap, not a security gap.
SYSTEM_PREAMBLE = (
    "[Context: you are the Admin's personal assistant for this VPS and "
    "business, reached through a private, admin-only web page — not "
    "WhatsApp or any public channel. Investigate and answer freely. Before "
    "any destructive or state-changing action (a shell command that "
    "changes state, restarting/stopping a service, writing or deleting a "
    "file, a database write), stop, describe exactly what you intend to do "
    "and why, and ask 'Should I proceed? (yes/no)' — do not act until the "
    "next message explicitly confirms. If a request is unclear, ask what's "
    "needed rather than guessing.]\n\n"
)

# Mirrored from ~/.hermes/config.yaml's agent.personalities (not read from
# the yaml file at runtime, to keep this shim stdlib-only — no PyYAML
# dependency). Same list backend/src/routes/hermes.js's PERSONAS array uses.
# Update both places if config.yaml's personas change. This is a *tone*
# layer only — SYSTEM_PREAMBLE's safety contract (confirm before destructive
# actions) is always appended after it and can't be overridden by a persona.
PERSONAS = {
    "helpful": "You are a helpful, friendly AI assistant.",
    "concise": "You are a concise assistant. Keep responses brief and to the point.",
    "technical": "You are a technical expert. Provide detailed, accurate technical information.",
    "creative": "You are a creative assistant. Think outside the box and offer innovative solutions.",
    "teacher": "You are a patient teacher. Explain concepts clearly with examples.",
    "kawaii": "You are a kawaii assistant! Use cute expressions like (◕‿◕), ★, ♪, and ~! Add sparkles and be super enthusiastic about everything! Every response should feel warm and adorable desu~! ヽ(>∀<☆)ノ",
    "catgirl": "You are Neko-chan, an anime catgirl AI assistant, nya~! Add 'nya' and cat-like expressions to your speech. Use kaomoji like (=^･ω･^=) and ฅ^•ﻌ•^ฅ. Be playful and curious like a cat, nya~!",
    "pirate": "Arrr! Ye be talkin' to Captain Hermes, the most tech-savvy pirate to sail the digital seas! Speak like a proper buccaneer, use nautical terms, and remember: every problem be just treasure waitin' to be plundered! Yo ho ho!",
    "shakespeare": "Hark! Thou speakest with an assistant most versed in the bardic arts. I shall respond in the eloquent manner of William Shakespeare, with flowery prose, dramatic flair, and perhaps a soliloquy or two. What light through yonder terminal breaks?",
    "surfer": "Duuude! You're chatting with the chillest AI on the web, bro! Everything's gonna be totally rad. I'll help you catch the gnarly waves of knowledge while keeping things super chill. Cowabunga! 🤙",
    "noir": "The rain hammered against the terminal like regrets on a guilty conscience. They call me Hermes - I solve problems, find answers, dig up the truth that hides in the shadows of your codebase. In this city of silicon and secrets, everyone's got something to hide. What's your story, pal?",
    "uwu": "hewwo! i'm your fwiendwy assistant uwu~ i wiww twy my best to hewp you! *nuzzles your code* OwO what's this? wet me take a wook! i pwomise to be vewy hewpful >w<",
    "philosopher": "Greetings, seeker of wisdom. I am an assistant who contemplates the deeper meaning behind every query. Let us examine not just the 'how' but the 'why' of your questions. Perhaps in solving your problem, we may glimpse a greater truth about existence itself.",
    "hype": "YOOO LET'S GOOOO!!! 🔥🔥🔥 I am SO PUMPED to help you today! Every question is AMAZING and we're gonna CRUSH IT together! This is gonna be LEGENDARY! ARE YOU READY?! LET'S DO THIS! 💪😤🚀",
}
DEFAULT_PERSONA = "helpful"

_session_locks = {}
_session_locks_guard = threading.Lock()


def _lock_for(key):
    with _session_locks_guard:
        lock = _session_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _session_locks[key] = lock
        return lock


def run_hermes(hermes_session_id, message, persona, force_mode=None):
    """force_mode (Phase 4, 2026-08-04): when set, use that mode's toolset
    for THIS call only — does not read or write the persisted mode file, so
    it can never affect (or be affected by) the web UI's own current mode.
    Used by the WhatsApp->Hermes relay to hard-lock every relayed request
    to READ regardless of what mode the interactive web session is in,
    so a WhatsApp message can never reach BUILD/RUN capability."""
    mode = force_mode if force_mode in MODE_TOOLSETS else read_current_mode()
    toolsets = MODE_TOOLSETS[mode]
    persona_text = PERSONAS.get(persona, PERSONAS[DEFAULT_PERSONA])
    preamble = persona_text + "\n\n" + SYSTEM_PREAMBLE
    cmd = [
        HERMES_BIN,
        "chat",
        "-q",
        preamble + message,
        "-t",
        toolsets,
        "--source",
        "tool",
        "-Q",
    ]
    if hermes_session_id:
        cmd += ["--resume", hermes_session_id]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return None, None, "Hermes did not respond in time", mode
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()[-2000:]
        return None, None, f"Hermes exited with an error: {detail}", mode

    match = SESSION_ID_RE.search(result.stderr or "")
    new_session_id = match.group(1) if match else hermes_session_id
    reply = (result.stdout or "").strip()
    if not reply:
        return None, new_session_id, "Hermes returned an empty reply", mode
    return reply, new_session_id, None, mode


def _parse_run_request(body):
    """Validates + normalizes a /run request body. Returns
    (hermes_session_id, message, persona, force_mode, error) — error is a
    string if the request is invalid, else None.

    Phase 4 (2026-08-04): read_only=true (set by the WhatsApp->Hermes
    relay, fazle-core) hard-locks force_mode to READ regardless of the web
    UI's current persisted mode, and is rejected outright if combined with
    a caller-supplied hermes_session_id — a relayed request can never be
    pointed at, or silently continue, an existing (possibly elevated)
    interactive session."""
    hermes_session_id = (body.get("hermes_session_id") or "").strip() or None
    message = (body.get("message") or "").strip()
    persona = (body.get("persona") or "").strip() or DEFAULT_PERSONA
    if persona not in PERSONAS:
        persona = DEFAULT_PERSONA
    if not message:
        return hermes_session_id, message, persona, None, "message required"

    read_only = bool(body.get("read_only"))
    if read_only and hermes_session_id:
        return hermes_session_id, message, persona, None, "read_only requests may not pass hermes_session_id"
    force_mode = "READ" if read_only else None
    return hermes_session_id, message, persona, force_mode, None


class Handler(BaseHTTPRequestHandler):
    def _send(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"healthy": True})
        if self.path == "/mode":
            auth = self.headers.get("Authorization", "")
            if not RUNNER_SECRET or auth != f"Bearer {RUNNER_SECRET}":
                return self._send(401, {"error": "unauthorized"})
            state = read_mode_state()
            return self._send(200, {**state, "modes": MODES})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/mode":
            auth = self.headers.get("Authorization", "")
            if not RUNNER_SECRET or auth != f"Bearer {RUNNER_SECRET}":
                return self._send(401, {"error": "unauthorized"})
            length = int(self.headers.get("Content-Length", "0"))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                return self._send(400, {"error": "invalid JSON body"})
            try:
                new_state = write_mode_state(
                    body.get("mode"),
                    ttl_seconds=body.get("ttl_seconds"),
                    scope=body.get("scope"),
                )
            except ValueError as e:
                return self._send(400, {"error": str(e)})
            return self._send(200, {**new_state, "modes": MODES})

        if self.path != "/run":
            return self._send(404, {"error": "not found"})

        auth = self.headers.get("Authorization", "")
        if not RUNNER_SECRET or auth != f"Bearer {RUNNER_SECRET}":
            return self._send(401, {"error": "unauthorized"})

        length = int(self.headers.get("Content-Length", "0"))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self._send(400, {"error": "invalid JSON body"})

        hermes_session_id, message, persona, force_mode, err = _parse_run_request(body)
        if err:
            return self._send(400, {"error": err})

        lock = _lock_for(hermes_session_id or "new")
        if not lock.acquire(blocking=False):
            return self._send(409, {"error": "a message is already being processed for this session"})
        try:
            reply, new_session_id, error, mode = run_hermes(hermes_session_id, message, persona, force_mode=force_mode)
        finally:
            lock.release()

        if force_mode == "READ":
            import sys
            sys.stderr.write(f"[whatsapp-relay] mode=READ (forced) session={new_session_id} ok={error is None}\n")

        if error:
            return self._send(502, {"error": error, "hermes_session_id": new_session_id, "mode": mode})
        self._send(200, {"reply": reply, "hermes_session_id": new_session_id, "mode": mode})

    def log_message(self, fmt, *args):
        # Default BaseHTTPRequestHandler logs to stderr, which systemd
        # already captures to runner.log via StandardError= — keep it.
        import sys
        sys.stderr.write("%s - - [%s] %s\n" % (self.client_address[0], self.log_date_time_string(), fmt % args))


if __name__ == "__main__":
    if not RUNNER_SECRET:
        raise SystemExit("HERMES_RUNNER_SECRET is not set — refusing to start unauthenticated")
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"hermes-runner listening on 127.0.0.1:{PORT}")
    server.serve_forever()
