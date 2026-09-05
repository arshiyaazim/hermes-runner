#!/usr/bin/env bash
# hermes-runner :: add OPENROUTER_API_KEY to this process's own .env
#
#   bash scripts/deploy_openrouter_key.sh
#
# Owner directive (2026-09-05): the fazle-core WhatsApp Admin route
# (Bridge1/Bridge2 -> fazle-core -> hermes-runner:8093) moves off MiniMax-M3
# (credit exhausted) to provider=openrouter, model=deepseek/deepseek-v4-pro-0813.
# hermes-runner's own .env has no OPENROUTER_API_KEY at all yet (confirmed
# by hash comparison this session -- it is genuinely absent here, not just
# different from the observer's copy).
#
# This is the SAME credential already deployed to, and sanitized-probed as
# valid (HTTP 200, real account, real usage) from, fazle-observer's own
# .secrets/openrouter.env -- the owner's own personal OpenRouter
# subscription, used for the isolated-Hermes shadow-pilot qualification.
# NOT a new key, NOT a rotation -- this script only makes an ALREADY-
# verified value additionally available to this second, separate Hermes
# process, exactly once, without ever echoing it. Modeled directly on
# fazle-observer/scripts/deploy_hermes_env.sh's own pattern (read from one
# named source file, append one line, chmod, verify by structure/hash only,
# print nothing sensitive) rather than an ad-hoc `cat >> .env` one-liner.
#
# Does NOT touch HERMES_RUNNER_WHATSAPP_ADMIN_MODEL/_PROVIDER (the actual
# model/provider switch) -- that is a separate, reviewable one-line .env
# edit, deliberately kept out of this script so a credential-deployment run
# and a model-routing change are two distinct, independently-auditable
# actions.
set -euo pipefail
umask 077

SOURCE_ENV="${SOURCE_ENV:-/home/azim/fazle-observer/.secrets/openrouter.env}"
DEST_ENV="${DEST_ENV:-/home/azim/hermes-runner/.env}"

[ -r "$SOURCE_ENV" ] || { echo "cannot read $SOURCE_ENV"; exit 1; }
[ -f "$DEST_ENV" ] || { echo "$DEST_ENV does not exist -- refusing to create it from scratch"; exit 1; }

trim() { sed -E 's/^[^=]+=//; s/^"//; s/"$//; s/^'\''//; s/'\''$//'; }

if grep -qE '^OPENROUTER_API_KEY=' "$DEST_ENV"; then
    echo "OPENROUTER_API_KEY already present in $DEST_ENV -- not overwriting. Remove the existing line first if you intend to replace it."
    exit 0
fi

ORKEY="$(grep -E '^OPENROUTER_API_KEY=' "$SOURCE_ENV" | head -1 | trim || true)"
[ -n "${ORKEY:-}" ] || { echo "OPENROUTER_API_KEY not found in $SOURCE_ENV"; exit 1; }

printf 'OPENROUTER_API_KEY=%s\n' "$ORKEY" >> "$DEST_ENV"
chmod 0600 "$DEST_ENV"

echo "appended OPENROUTER_API_KEY to $DEST_ENV"
echo "keys now present: $(grep -oE '^[A-Z_]+' "$DEST_ENV" | tr '\n' ' ')"

# Verify by hash only -- confirms the SAME already-probed value landed here,
# without ever printing it.
_dest_val="$(grep -E '^OPENROUTER_API_KEY=' "$DEST_ENV" | head -1 | cut -d= -f2-)"
_src_val="$ORKEY"
if [ "$_dest_val" = "$_src_val" ]; then
    echo "verified: sha256=$(printf '%s' "$_dest_val" | sha256sum | cut -c1-16)... matches the source credential"
else
    echo "!! FAIL: value written does not match source after append"; exit 1
fi
unset _dest_val _src_val ORKEY

echo "NOTE: hermes-runner's own process must be restarted to pick this up (it inherits os.environ once at its own startup, not per-request). Not done by this script."
