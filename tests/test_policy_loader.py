"""tests/test_policy_loader.py — unit tests for config.fazle_ai.policy_loader.

Covers the validator rules from task #22:
  1. No tracked configuration file contains a real secret value.
  2. runtime.env.example contains no credential-like setting.
  3. secrets.env.example contains only _REF references.
  4. auto/* is rejected by the validator.
  5. A valid allowlisted model is accepted.
  6. Cross-validation between policy and workload.
  7. Backward-compat: missing file -> load_policy returns None.
  8. Hard invariants: workload cannot disable the no-secret logging rules.
"""

from __future__ import annotations

import os
import re
import unittest
from pathlib import Path

from config.fazle_ai.policy_loader import (
    Policy,
    PolicyValidationError,
    WorkloadConfig,
    load_policy,
    load_workload,
    redact_for_log,
    validate_workload_against_policy,
)


POLICY_PATH = '/home/azim/hermes-runner/config/fazle-ai/policy.yaml'
WORKLOAD_PATH = '/home/azim/hermes-runner/config/fazle-ai/workloads/hermes-runner.yaml'
RUNTIME_EXAMPLE = '/home/azim/hermes-runner/config/fazle-ai/examples/runtime.env.example'
SECRETS_EXAMPLE = '/home/azim/hermes-runner/config/fazle-ai/examples/secrets.env.example'
CONFIG_DIR = Path('/home/azim/hermes-runner/config/fazle-ai')


def _load_policy_or_skip() -> Policy:
    p = load_policy(POLICY_PATH)
    if p is None:
        raise unittest.SkipTest(f'policy.yaml missing at {POLICY_PATH}')
    return p


def _load_workload_or_skip() -> WorkloadConfig:
    return load_workload('hermes-runner', WORKLOAD_PATH)


# ── Test 1: secret safety ──────────────────────────────────────────────────


class NoTrackedSecretTest(unittest.TestCase):
    """No tracked configuration file contains a real secret value."""

    def test_tracked_yaml_files_have_no_credential_like_values(self):
        # Heuristic: lines that look like key=value pairs where the value
        # matches a credential pattern. Real workloads use vault:// URIs.
        credential_patterns = [
            re.compile(r'^sk-[A-Za-z0-9_\-]{16,}$'),         # OpenAI / Anthropic sk-*
            re.compile(r'^sk-ant-[A-Za-z0-9_\-]{16,}$'),    # Anthropic specific
            re.compile(r'^sk-ov-[A-Za-z0-9_\-]{16,}$'),     # OAuth
            re.compile(r'^gho_[A-Za-z0-9]{20,}$'),          # GitHub OAuth
            re.compile(r'^ghp_[A-Za-z0-9]{20,}$'),          # GitHub PAT
            re.compile(r'^gsk_[A-Za-z0-9]{20,}$'),          # Groq
            re.compile(r'^xai-[A-Za-z0-9]{20,}$'),          # xAI
            re.compile(r'^AIza[A-Za-z0-9_\-]{30,}$'),       # Google
            re.compile(r'^hf_[A-Za-z0-9]{20,}$'),          # HF
            re.compile(r'^[A-Fa-f0-9]{32,}$'),              # raw hex
        ]
        scanned_files = [POLICY_PATH, WORKLOAD_PATH, RUNTIME_EXAMPLE, SECRETS_EXAMPLE]
        for path in scanned_files:
            with open(path) as f:
                content = f.read()
            for line in content.splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith('#'):
                    continue
                if '=' not in stripped:
                    continue
                key, _, val = stripped.partition('=')
                val = val.strip().strip('"').strip("'")
                for pat in credential_patterns:
                    if pat.match(val):
                        self.fail(
                            f'{path}: key {key!r} has a value matching credential pattern '
                            f'(value redacted, length={len(val)}). Real secrets must NEVER '
                            f'appear in tracked files.'
                        )

    def test_no_env_files_in_tracked_tree(self):
        # Walk config/fazle-ai and assert no file is named exactly '.env'.
        for root, dirs, files in os.walk(CONFIG_DIR):
            for fn in files:
                self.assertNotEqual(
                    fn, '.env',
                    f'tracked .env found at {os.path.join(root, fn)}',
                )

    def test_redact_for_log_returns_safe_placeholder(self):
        # redact_for_log must never echo the input back.
        for val in ['hunter2', 'sk-1234', 'very-secret-value', None, '', 42]:
            out = redact_for_log(val)
            self.assertEqual(out, '[REDACTED]')


# ── Test 2: runtime.env.example has no credentials ─────────────────────────


class RuntimeEnvExampleTest(unittest.TestCase):

    def test_runtime_env_example_has_no_credential_like_values(self):
        credential_substrings = ['sk-', 'sk_', 'AIza', 'gsk_', 'gho_', 'ghp_', 'huggingface_', 'xai-']
        with open(RUNTIME_EXAMPLE) as f:
            content = f.read()
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith('#') or '=' not in stripped:
                continue
            key, _, val = stripped.partition('=')
            val = val.strip().strip('"').strip("'")
            for sub in credential_substrings:
                if sub in val:
                    self.fail(
                        f'runtime.env.example key {key!r} contains credential-like substring {sub!r}'
                    )


# ── Test 3: secrets.env.example is _REF-only ─────────────────────────────────


class SecretsEnvExampleTest(unittest.TestCase):

    def test_secrets_env_example_only_has_ref_keys(self):
        violations: list[str] = []
        with open(SECRETS_EXAMPLE) as f:
            content = f.read()
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith('#') or '=' not in stripped:
                continue
            key, _, val = stripped.partition('=')
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if not key.endswith('_REF'):
                violations.append(f'key {key!r} does not end in _REF (val redacted, len={len(val)})')
            if not (val.startswith('vault://') or val.startswith('env://')):
                violations.append(f'key {key!r} value is not a vault:// or env:// URI (val redacted)')
        if violations:
            self.fail('secrets.env.example violations:\n  ' + '\n  '.join(violations))

    def test_secrets_env_example_has_no_actual_secret_values(self):
        # Defense-in-depth: ensure no value in this file looks like a real secret.
        real_secret_patterns = [
            re.compile(r'^sk-[A-Za-z0-9_\-]{16,}$'),
            re.compile(r'^sk-ant-[A-Za-z0-9_\-]{16,}$'),
            re.compile(r'^AIza[A-Za-z0-9_\-]{20,}$'),
            re.compile(r'^gsk_[A-Za-z0-9]{20,}$'),
            re.compile(r'^gh[oprsu]_[A-Za-z0-9]{20,}$'),
            re.compile(r'^[A-Fa-f0-9]{32,}$'),
        ]
        with open(SECRETS_EXAMPLE) as f:
            content = f.read()
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith('#') or '=' not in stripped:
                continue
            _, _, val = stripped.partition('=')
            val = val.strip().strip('"').strip("'")
            for pat in real_secret_patterns:
                if pat.match(val):
                    self.fail(f'secrets.env.example has a real-secret-shaped value (redacted, len={len(val)})')


# ── Test 4: auto/* is rejected ─────────────────────────────────────────────


class AutoDenyTest(unittest.TestCase):

    def _write_temp_policy(self, deny_patterns: list[str], allow_models: list[str]) -> str:
        import tempfile
        text = (
            'version: "1.0"\n'
            'environment: production\n'
            'defaults:\n'
            '  request_timeout_s: 30\n'
            '  connect_timeout_s: 5\n'
            '  read_budget_total_s: 300\n'
            'retry_policy: {}\n'
            'model_policy:\n'
            '  mode: allowlist\n'
            f'  deny_patterns: {deny_patterns!r}\n'
            f'  allow_models: {allow_models!r}\n'
            'provider_limits: {}\n'
            'shared: {}\n'
        )
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        f.write(text)
        f.close()
        return f.name

    def test_policy_global_allowlist_rejects_auto_wildcard(self):
        for m in ['auto/best-fast', 'auto/claude-opus', 'auto/anything']:
            with self.subTest(model=m):
                tmp = self._write_temp_policy(['auto/*'], [m])
                try:
                    with self.assertRaises(PolicyValidationError):
                        load_policy(tmp)
                finally:
                    os.unlink(tmp)

    def test_workload_fallback_chain_with_auto_is_rejected(self):
        # The actual hermes-runner.yaml denylist contains "auto/*" at the
        # workload level. Use that to test that the WORKLOAD-level deny
        # list rejects the auto model on its own (without needing the
        # cross-validation step).
        with tempfile_workload_with_model_and_deny(
            'auto/best-fast', deny_patterns=['auto/*']
        ) as path:
            with self.assertRaises(PolicyValidationError) as cm:
                load_workload('hermes-runner', path)
            msg = str(cm.exception)
            self.assertIn('auto/*', msg)
            self.assertIn('auto/best-fast', msg)

    def test_cross_validation_rejects_auto_via_global_policy(self):
        # Even if a workload has NO local deny patterns, cross-validation
        # against a global policy that denylists "auto/*" must reject.
        p = _load_policy_or_skip()
        with tempfile_workload_with_model('auto/best-fast') as path:
            wl = load_workload('hermes-runner', path)
            with self.assertRaises(PolicyValidationError) as cm:
                from config.fazle_ai.policy_loader import validate_workload_against_policy
                validate_workload_against_policy(wl, p)
            msg = str(cm.exception)
            self.assertIn('auto/*', msg)

    def test_workload_deny_pattern_rejects_auto_pattern_via_validate(self):
        p = _load_policy_or_skip()
        p_deny = set(p.deny_patterns)
        self.assertIn('auto/*', p_deny)


# ── Test 5: valid allowlisted model is accepted ────────────────────────────


class AcceptValidModelTest(unittest.TestCase):

    def test_known_allowlisted_models_validate(self):
        p = _load_policy_or_skip()
        for m in [
            'claude/claude-opus-4-6',
            'gemini/gemini-3.1-flash-lite',
            'moonshot/kimi-k2.7-code',
            'groq/llama-3.3-70b-versatile',
            'ollama-local/qwen3:8b',
        ]:
            self.assertIn(m, p.allow_models, f'{m!r} must be in allow_models')

    def test_workload_loads_with_all_allowlisted_models(self):
        wl = _load_workload_or_skip()
        p = _load_policy_or_skip()
        validate_workload_against_policy(wl, p)

    def test_redact_does_not_echo_input(self):
        # Belt + suspenders: redact_for_log must not return the original value.
        self.assertNotEqual(redact_for_log('hunter2'), 'hunter2')
        self.assertEqual(redact_for_log('hunter2'), '[REDACTED]')


# ── Test 6: backward compatibility (legacy env-only config) ────────────────


class BackwardCompatTest(unittest.TestCase):

    def test_load_policy_returns_none_when_file_missing(self):
        self.assertIsNone(load_policy('/nonexistent/policy.yaml'))


# ── Test 7: hard invariants ────────────────────────────────────────────────


class HardInvariantsTest(unittest.TestCase):

    def test_workload_cannot_enable_secret_logging(self):
        # Try to construct a workload YAML that tries to set
        # log_request_authorization_headers=true. The loader must reject it.
        import tempfile
        yaml_text = '''version: "1.0"
environment: production
workload:
  name: bad-workload
  enabled: true
fallback_chain:
  - provider: p
    model: m-a
model_deny_patterns: []
logging:
  log_request_authorization_headers: true
  log_secret_values: false
secret_refs: {}
backward_compat: {}
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_text)
            tmp = f.name
        try:
            with self.assertRaises(PolicyValidationError) as cm:
                load_workload('bad-workload', tmp)
            self.assertIn('log_request_authorization_headers', str(cm.exception))
        finally:
            os.unlink(tmp)

    def test_workload_cannot_enable_secret_value_logging(self):
        import tempfile
        yaml_text = '''version: "1.0"
environment: production
workload:
  name: bad-workload
  enabled: true
fallback_chain:
  - provider: p
    model: m-a
model_deny_patterns: []
logging:
  log_request_authorization_headers: false
  log_secret_values: true
secret_refs: {}
backward_compat: {}
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_text)
            tmp = f.name
        try:
            with self.assertRaises(PolicyValidationError) as cm:
                load_workload('bad-workload', tmp)
            self.assertIn('log_secret_values', str(cm.exception))
        finally:
            os.unlink(tmp)

    def test_secret_refs_must_be_uri_shaped(self):
        import tempfile
        yaml_text = '''version: "1.0"
environment: production
workload:
  name: bad-workload
  enabled: true
fallback_chain:
  - provider: p
    model: m-a
model_deny_patterns: []
logging:
  log_request_authorization_headers: false
  log_secret_values: false
secret_refs:
  some_key: "not-a-uri"
backward_compat: {}
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_text)
            tmp = f.name
        try:
            with self.assertRaises(PolicyValidationError) as cm:
                load_workload('bad-workload', tmp)
            self.assertIn('vault://', str(cm.exception))
        finally:
            os.unlink(tmp)


# ── Helpers ────────────────────────────────────────────────────────────────


import contextlib

@contextlib.contextmanager
def tempfile_workload_with_model(model: str):
    """Write a minimal workload YAML whose fallback_chain contains `model`."""
    import tempfile
    text = f'''version: "1.0"
environment: production
workload:
  name: hermes-runner
  enabled: true
fallback_chain:
  - provider: p
    model: "{model}"
model_deny_patterns: []
logging:
  log_request_authorization_headers: false
  log_secret_values: false
secret_refs: {{}}
backward_compat: {{}}
'''
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(text)
        tmp = f.name
    try:
        yield tmp
    finally:
        os.unlink(tmp)


@contextlib.contextmanager
def tempfile_workload_with_model_and_deny(model: str, deny_patterns: list[str]):
    """Write a workload YAML with the given model in fallback_chain AND
    the given deny_patterns in the workload's model_deny_patterns."""
    import tempfile
    deny_yaml = '\n'.join(f'  - "{p}"' for p in deny_patterns)
    text = f'''version: "1.0"
environment: production
workload:
  name: hermes-runner
  enabled: true
fallback_chain:
  - provider: p
    model: "{model}"
model_deny_patterns:
{deny_yaml}
logging:
  log_request_authorization_headers: false
  log_secret_values: false
secret_refs: {{}}
backward_compat: {{}}
'''
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(text)
        tmp = f.name
    try:
        yield tmp
    finally:
        os.unlink(tmp)


# ── Test 8: provider_limits strict allowlist ─────────────────────────────


class ProviderLimitsAllowlistTest(unittest.TestCase):
    """provider_limits must reject any field not in the explicit allowlist.

    Secret-shaped keys like api_key, token, password, secret, authorization,
    credential must NEVER be accepted, even if the value is an int. The
    rejection message must be actionable and must NEVER echo the value.
    """

    ALLOWLIST_BASE = (
        'version: "1.0"\n'
        'environment: production\n'
        'defaults:\n'
        '  request_timeout_s: 30\n'
        '  connect_timeout_s: 5\n'
        '  read_budget_total_s: 300\n'
        'retry_policy:\n'
        '  network_timeout_or_transient_5xx: {max_attempts_same_target: 2, backoff_s: [2]}\n'
        '  rate_limited: {max_attempts_same_target: 2, backoff_s: [2]}\n'
        '  quota_exhausted: {max_attempts_same_target: 0}\n'
        '  provider_or_model_unavailable: {max_attempts_same_target: 0}\n'
        '  context_too_large: {max_attempts_same_target: 1, backoff_s: [1]}\n'
        '  auth_or_authorization_failed: {max_attempts_same_target: 0}\n'
        '  invalid_request_or_policy_or_safety_refusal: {max_attempts_same_target: 0}\n'
        'rate_limit_policy: {retry_after_safe_maximum_s: 30, one_shot: true}\n'
        'provider_limits:\n'
        '  omniroute: {requests_per_minute: 600}\n'
        'model_policy:\n'
        '  mode: allowlist\n'
        '  deny_patterns: ["auto/*"]\n'
        '  allow_models: ["claude/claude-opus-4-6"]\n'
        'shared: {}\n'
    )

    def _with_provider_limits(self, limits_block):
        prefix, suffix = self.ALLOWLIST_BASE.split('provider_limits:\n', 1)
        keep = 'model_policy:\n' + suffix.split('model_policy:\n', 1)[1]
        return prefix + 'provider_limits:\n' + limits_block + keep

    def _write_policy(self, yaml_text):
        import tempfile
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        f.write(yaml_text)
        f.close()
        return f.name

    def test_allowlisted_requests_per_minute_loads(self):
        path = self._write_policy(self.ALLOWLIST_BASE)
        try:
            pol = load_policy(path)
            self.assertIsNotNone(pol)
            self.assertEqual(
                pol.provider_limits.get('omniroute', {}).get('requests_per_minute'),
                600,
            )
        finally:
            os.unlink(path)

    def test_allowlisted_max_concurrent_loads(self):
        yaml_text = self._with_provider_limits('  omniroute: {max_concurrent: 4}\n')
        path = self._write_policy(yaml_text)
        try:
            pol = load_policy(path)
            self.assertIsNotNone(pol)
            self.assertEqual(
                pol.provider_limits.get('omniroute', {}).get('max_concurrent'),
                4,
            )
        finally:
            os.unlink(path)

    def test_unknown_field_api_key_rejected_with_redacted_message(self):
        SECRET = 'AKIA1234567890ABCDEF-DO-NOT-LEAK'
        yaml_text = self._with_provider_limits(
            f'  omniroute: {{api_key: "{SECRET}"}}\n'
        )
        path = self._write_policy(yaml_text)
        try:
            with self.assertRaises(PolicyValidationError) as cm:
                load_policy(path)
            msg = str(cm.exception)
            self.assertIn('api_key', msg)
            self.assertIn('not an allowed configuration field', msg)
            self.assertNotIn(SECRET, msg)
            self.assertNotIn('AKIA', msg)
        finally:
            os.unlink(path)

    def test_unknown_field_token_rejected_with_redacted_message(self):
        SECRET = 'gho_secret1234567890abcdefghij'
        yaml_text = self._with_provider_limits(
            f'  omniroute: {{token: "{SECRET}"}}\n'
        )
        path = self._write_policy(yaml_text)
        try:
            with self.assertRaises(PolicyValidationError) as cm:
                load_policy(path)
            msg = str(cm.exception)
            self.assertIn('token', msg)
            self.assertIn('not an allowed configuration field', msg)
            self.assertNotIn(SECRET, msg)
        finally:
            os.unlink(path)

    def test_arbitrary_unknown_field_rejected(self):
        for bad in ('auth', 'credential', 'foobar', 'banana', 'totally_made_up'):
            with self.subTest(bad=bad):
                yaml_text = self._with_provider_limits(
                    f'  omniroute: {{{bad}: 42}}\n'
                )
                path = self._write_policy(yaml_text)
                try:
                    with self.assertRaises(PolicyValidationError) as cm:
                        load_policy(path)
                    msg = str(cm.exception)
                    self.assertIn(bad, msg)
                    self.assertIn('not an allowed configuration field', msg)
                finally:
                    os.unlink(path)

    def test_allowlisted_field_with_string_value_rejected(self):
        SECRET = 's3cret-AKIA1234567890ABCDEF-DO-NOT-LEAK'
        # Even when the FIELD is allowlisted, the VALUE must be int.
        yaml_text = self._with_provider_limits(
            f'  omniroute: {{requests_per_minute: "{SECRET}"}}\n'
        )
        path = self._write_policy(yaml_text)
        try:
            with self.assertRaises(PolicyValidationError) as cm:
                load_policy(path)
            msg = str(cm.exception)
            self.assertNotIn(SECRET, msg)
        finally:
            os.unlink(path)

    def test_unknown_field_rejected_for_every_provider(self):
        for prov in ('omniroute', 'groq', 'moonshot', 'ollama-local', 'github-models'):
            with self.subTest(prov=prov):
                yaml_text = self._with_provider_limits(
                    f'  {prov}: {{secret_value: 42}}\n'
                )
                path = self._write_policy(yaml_text)
                try:
                    with self.assertRaises(PolicyValidationError):
                        load_policy(path)
                finally:
                    os.unlink(path)

    def test_negative_value_rejected(self):
        yaml_text = self._with_provider_limits(
            '  omniroute: {requests_per_minute: -1}\n'
        )
        path = self._write_policy(yaml_text)
        try:
            with self.assertRaises(PolicyValidationError) as cm:
                load_policy(path)
            self.assertIn('non-negative', str(cm.exception))
        finally:
            os.unlink(path)

    def test_allowlist_is_exactly_two_supported_fields(self):
        # Defense-in-depth: the allowlist equals the two supported fields.
        from config.fazle_ai import policy_loader as _pl
        self.assertEqual(
            _pl._ALLOWED_PROVIDER_LIMITS_KEYS,
            frozenset({'requests_per_minute', 'max_concurrent'}),
        )
