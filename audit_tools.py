"""
Read-only audit toolkit, exposed over HTTP by server.py's `/audit` route so
Chat (assistant-backend, Docker container) can reach it too.

Ported from fazle-mcp/audit_tools.py's filesystem/git-facing tools (2026-08-04,
Chat audit toolkit wiring, Owner-approved "7 read-only audit tools" scope --
see assistant-platform/proposal_chat_audit_toolkit_20260804.md). Deliberately
duplicated rather than cross-imported: fazle-mcp and hermes-runner are
separate deployable units with separate venvs, matching this project's
existing convention of duplicating shared logic across process boundaries
(e.g. the PERSONAS dict already duplicated between here and
assistant-platform/backend/src/routes/hermes.js). Keep both copies in sync if
either changes.

Runs here specifically because this process is host-level (like
fazle-mcp/hermes CLI itself), unlike assistant-backend, which runs inside a
Docker container with no bind-mounted access to /home/azim/assistant-platform
or /home/azim/core and no Docker socket -- a straight JS port living inside
the container would have nothing to search. This module stays stdlib-only
(os, re, subprocess), matching server.py's own no-external-dependencies
convention.

Every function here is read-only, rooted to an explicit allowlist, and uses
argv-list subprocess calls (shell=False) -- no string ever reaches a shell.

Deliberately excluded from this port: `audit_lookup_whatsapp_messages` (queries
fazle-core's DB via an authenticated HTTP client, not the filesystem -- out of
the approved 7-tool scope; Chat already has its own DB-backed tools in
fazleTools.js).
"""

import os
import re
import subprocess

AUDIT_ROOTS = {
    "assistant-platform": "/home/azim/assistant-platform",
    "fazle-core": "/home/azim/core",
}
KB_ROOT = "/home/azim/core/knowledge_base"

# Substring match on the resolved absolute path -- deny if any of these
# appear anywhere in it, not just as a filename.
DENY_PATH_SUBSTRINGS = [
    ".env", "secrets/", "secret/", "private_key", ".pem", ".key",
    "id_rsa", "id_ed25519", "credentials", ".git/",
]

EXCLUDE_DIRS = {
    ".git", "node_modules", "venv", "__pycache__", ".pytest_cache", "dist", "build",
    "coverage_html", "htmlcov",
}

SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|password|secret|token)\s*[:=]\s*\S+"),
    re.compile(r"-----BEGIN [A-Z ]+-----"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._-]{10,}"),
]

_PHONE_RE = re.compile(r"(\+?880|0)1[0-9]{9}\b")

MAX_FILE_BYTES = 2 * 1024 * 1024

LOG_SOURCES = {
    "backend": {"type": "docker", "service": "assistant-backend", "cwd": "/home/azim/assistant-platform"},
    "fazle-core": {"type": "file", "path": "/home/azim/core/logs/fazle-core.log"},
    "hermes-runner": {"type": "journal", "unit": "hermes-runner.service"},
}


def _redact_line(line):
    for pat in SECRET_PATTERNS:
        line = pat.sub("[REDACTED]", line)
    return _PHONE_RE.sub(lambda m: m.group(0)[:1] + "X" * (len(m.group(0)) - 5) + m.group(0)[-4:], line)


def _resolve_in_root(root_key, rel_or_abs_path=""):
    """Resolve a path and verify it's genuinely inside the named allowlisted
    root (realpath, so no ../ or symlink can escape it). Returns (abs_path,
    error) -- error is a string if the path isn't allowed, else None."""
    if root_key not in AUDIT_ROOTS:
        return None, f"unknown root: {root_key!r}. allowed: {list(AUDIT_ROOTS)}"
    root = os.path.realpath(AUDIT_ROOTS[root_key])
    candidate = rel_or_abs_path or root
    if not os.path.isabs(candidate):
        candidate = os.path.join(root, candidate)
    resolved = os.path.realpath(candidate)
    if resolved != root and not resolved.startswith(root + os.sep):
        return None, "path escapes the allowed root"
    for bad in DENY_PATH_SUBSTRINGS:
        if bad in resolved:
            return None, "path is denylisted (looks like a secret/credential location)"
    return resolved, None


def _run(argv, timeout=10, cwd=None):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, shell=False, cwd=cwd)
    except subprocess.TimeoutExpired:
        return None, "command timed out"
    except OSError as e:
        return None, f"failed to run command: {e}"
    return result, None


def audit_search_code(query: str, root: str = "assistant-platform", max_results: int = 20) -> dict:
    """Search source code for a literal string (grep -rn), rooted to an
    approved repo. root: 'assistant-platform' or 'fazle-core'."""
    return _grep(query, root, max_results, docs_only=False)


def audit_search_docs(query: str, root: str = "assistant-platform", max_results: int = 20) -> dict:
    """Search markdown docs for a literal string, rooted to an approved repo."""
    return _grep(query, root, max_results, docs_only=True)


def audit_search_kb(query: str, max_results: int = 20) -> dict:
    """Search fazle-core's knowledge_base/ directory for a literal string."""
    if not query or not isinstance(query, str):
        return {"error": "query required"}
    max_results = min(max(int(max_results or 20), 1), 100)
    exclude_args = []
    for d in EXCLUDE_DIRS:
        exclude_args += ["--exclude-dir", d]
    argv = ["grep", "-rn", *exclude_args, "--", query, KB_ROOT]
    result, err = _run(argv)
    if err:
        return {"error": err}
    if result.returncode not in (0, 1):
        return {"error": f"search failed: {(result.stderr or '').strip()[:500]}"}
    lines = (result.stdout or "").splitlines()[:max_results]
    return {"matches": lines, "truncated": len((result.stdout or "").splitlines()) > max_results}


def _grep(query, root, max_results, docs_only):
    if not query or not isinstance(query, str):
        return {"error": "query required"}
    root_path, err = _resolve_in_root(root)
    if err:
        return {"error": err}
    max_results = min(max(int(max_results or 20), 1), 100)
    exclude_args = []
    for d in EXCLUDE_DIRS:
        exclude_args += ["--exclude-dir", d]
    argv = ["grep", "-rn", *exclude_args]
    if docs_only:
        argv += ["--include=*.md"]
    argv += ["--", query, root_path]
    result, err = _run(argv)
    if err:
        return {"error": err}
    if result.returncode not in (0, 1):
        return {"error": f"search failed: {(result.stderr or '').strip()[:500]}"}
    all_lines = (result.stdout or "").splitlines()
    return {"matches": all_lines[:max_results], "truncated": len(all_lines) > max_results}


def audit_search_logs(query: str = "", log: str = "backend", max_lines: int = 100) -> dict:
    """Search a known, allowlisted log source (log: 'backend' -- Docker,
    'fazle-core' -- file, or 'hermes-runner' -- systemd journal) for a
    literal string (or tail it if query is empty). Output has secrets and
    phone numbers redacted before being returned."""
    if log not in LOG_SOURCES:
        return {"error": f"unknown log: {log!r}. allowed: {list(LOG_SOURCES)}"}
    source = LOG_SOURCES[log]
    max_lines = min(max(int(max_lines or 100), 1), 500)

    if source["type"] == "file":
        path = source["path"]
        if not os.path.isfile(path):
            return {"matches": [], "note": "log file does not exist (nothing logged yet, or path changed)"}
        argv = ["grep", "-n", "--", query, path] if query else ["tail", "-n", str(max_lines), path]
        result, err = _run(argv)
        if err:
            return {"error": err}
        if result.returncode not in (0, 1):
            return {"error": f"log search failed: {(result.stderr or '').strip()[:500]}"}
        lines = (result.stdout or "").splitlines()[:max_lines]
        return {"matches": [_redact_line(l) for l in lines]}

    if source["type"] == "docker":
        argv = ["docker", "compose", "logs", "--no-color", "--tail", str(max_lines), source["service"]]
        result, err = _run(argv, timeout=15, cwd=source["cwd"])
        if err:
            return {"error": err}
        if result.returncode != 0:
            return {"error": f"docker logs failed: {(result.stderr or '').strip()[:500]}"}
        lines = (result.stdout or "").splitlines()
        if query:
            lines = [l for l in lines if query.lower() in l.lower()]
        return {"matches": [_redact_line(l) for l in lines[:max_lines]]}

    if source["type"] == "journal":
        argv = ["journalctl", "--user", "-u", source["unit"], "-n", str(max_lines), "--no-pager"]
        if query:
            argv += ["-g", query]
        result, err = _run(argv, timeout=15)
        if err:
            return {"error": err}
        if result.returncode != 0:
            return {"error": f"journalctl failed: {(result.stderr or '').strip()[:500]}"}
        lines = (result.stdout or "").splitlines()[:max_lines]
        return {"matches": [_redact_line(l) for l in lines]}

    return {"error": f"unsupported log source type: {source['type']!r}"}


def audit_read_file(path: str, root: str = "assistant-platform", start_line: int = None, end_line: int = None, max_lines: int = 500) -> dict:
    """Read a file (optionally a bounded line range) from an approved root.
    Denies secrets/credential-shaped paths and enforces a size cap."""
    resolved, err = _resolve_in_root(root, path)
    if err:
        return {"error": err}
    if not os.path.isfile(resolved):
        return {"error": "not a file (or does not exist)"}
    size = os.path.getsize(resolved)
    max_lines = min(max(int(max_lines or 500), 1), 2000)
    if size > MAX_FILE_BYTES and start_line is None and end_line is None:
        return {"error": f"file too large ({size} bytes) — pass start_line/end_line for a bounded read"}
    try:
        with open(resolved, "r", errors="replace") as f:
            all_lines = f.readlines()
    except OSError as e:
        return {"error": f"could not read file: {e}"}
    start = max((start_line or 1) - 1, 0)
    end = min(end_line, len(all_lines)) if end_line else min(start + max_lines, len(all_lines))
    return {
        "path": resolved,
        "total_lines": len(all_lines),
        "start_line": start + 1,
        "end_line": end,
        "content": "".join(all_lines[start:end]),
    }


def audit_git_status(repo: str = "assistant-platform") -> dict:
    """Read-only `git status --porcelain` for an approved repo."""
    root_path, err = _resolve_in_root(repo)
    if err:
        return {"error": err}
    result, err = _run(["git", "-C", root_path, "status", "--porcelain"])
    if err:
        return {"error": err}
    if result.returncode != 0:
        return {"error": f"git status failed: {(result.stderr or '').strip()[:500]}"}
    return {"changes": (result.stdout or "").splitlines()}


def audit_recent_commits(repo: str = "assistant-platform", limit: int = 10) -> dict:
    """Read-only recent commit log for an approved repo (git log --oneline)."""
    root_path, err = _resolve_in_root(repo)
    if err:
        return {"error": err}
    limit = min(max(int(limit or 10), 1), 100)
    result, err = _run(["git", "-C", root_path, "log", "-n", str(limit), "--oneline"])
    if err:
        return {"error": err}
    if result.returncode != 0:
        return {"error": f"git log failed: {(result.stderr or '').strip()[:500]}"}
    return {"commits": (result.stdout or "").splitlines()}


# Name -> callable allowlist for server.py's /audit dispatcher. A closed set,
# not "whatever's a public function in this module" -- adding a function here
# is a deliberate act, not an accident of not underscore-prefixing something.
AUDIT_TOOLS = {
    "audit_search_code": audit_search_code,
    "audit_search_docs": audit_search_docs,
    "audit_search_kb": audit_search_kb,
    "audit_search_logs": audit_search_logs,
    "audit_read_file": audit_read_file,
    "audit_git_status": audit_git_status,
    "audit_recent_commits": audit_recent_commits,
}
