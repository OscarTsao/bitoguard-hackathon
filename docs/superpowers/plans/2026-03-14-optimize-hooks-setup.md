# Hook Config Optimization Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix and optimize the Claude Code hooks setup so the team-mode security system actually works, dangerous-operation warnings fire correctly, and no wasteful/conflicting hooks run.

**Architecture:** Rewrite both custom Python hook scripts to use Claude Code's stdin-JSON protocol; register them in `~/.claude/settings.json` under a `hooks` key; remove `skipDangerousModePermissionPrompt`; disable one conflicting output-style plugin; and either wire a useful hookify rule or disable its overhead.

**Tech Stack:** Python 3, Claude Code hooks API (`~/.claude/settings.json`), JSON stdin/stdout hook protocol.

---

## Background: What Is Broken and Why

| Issue | Root Cause | Impact |
|-------|-----------|--------|
| `keyword-route.py` never fires | No `hooks` key in `settings.json` | Auto-escalation dead |
| `mode-guard.py` never fires | Same: not registered | Strict-mode enforcement dead |
| Both scripts use `sys.argv` parsing | Claude Code sends JSON on **stdin**, not argv | Would crash even if registered |
| `skipDangerousModePermissionPrompt: true` | Disables built-in safety prompt | No safety net exists |
| Conflicting `SessionStart` hooks | Both `explanatory-output-style` and `learning-output-style` enabled | Competing injections |
| hookify overhead | Runs Python on every tool call with zero `.local.md` rules | Pure latency, zero benefit |

**Current state file**: `~/.claude/team_mode_state.json` has `mode: strict` from a prior manual activation — this is orphaned state that nothing enforces.

---

## Claude Code Hook Protocol (Reference)

All hook scripts must follow this protocol or they silently fail:

```
STDIN  → JSON object (varies by hook type)
STDOUT → JSON object (optional, controls behavior)
EXIT 0 → allow / continue
EXIT 2 → block (output becomes feedback to Claude)
EXIT 1 → error (logged, operation continues)
```

**`UserPromptSubmit`** stdin shape:
```json
{"prompt": "user text", "session_id": "...", "cwd": "/path"}
```
Output `{"systemMessage": "..."}` to inject advisory context into Claude's view.

**`PreToolUse`** stdin shape:
```json
{"tool_name": "Bash", "tool_input": {"command": "..."}, "tool_use_id": "...", "session_id": "...", "cwd": "/path"}
```
Output `{"decision": "block", "reason": "..."}` + exit 2 to block. Output `{"systemMessage": "..."}` + exit 0 to warn.

---

## File Map

| Action | Path | Responsibility |
|--------|------|---------------|
| **Modify** | `~/.claude/settings.json` | Add `hooks` section; remove `skipDangerousModePermissionPrompt`; disable `learning-output-style` plugin |
| **Rewrite** | `~/.claude/hooks/keyword-route.py` | `UserPromptSubmit` hook: warn on truly dangerous prompt patterns via systemMessage |
| **Rewrite** | `~/.claude/hooks/mode-guard.py` | `PreToolUse` hook: block Write/Edit/Bash when `team_mode_state.json` has `mode: strict` |
| **Create** | `~/.claude/hooks/test_hooks.py` | Unit tests for both hook scripts (no Claude Code needed, uses subprocess + JSON) |

---

## Chunk 1: Settings Wiring + Plugin Cleanup

### Task 1: Remove `skipDangerousModePermissionPrompt` and register hooks

**Files:**
- Modify: `~/.claude/settings.json`

- [ ] **Step 1: Read the current settings file**

Verify current content (already done in review — reproduced here for reference):
```json
{
  "model": "opusplan",
  "skipDangerousModePermissionPrompt": true,
  ...
}
```

- [ ] **Step 2: Add the `hooks` section and remove the dangerous flag**

Edit `~/.claude/settings.json` to add after `"effortLevel": "high"`:

```json
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python3 /Users/oscartsao/.claude/hooks/keyword-route.py",
            "timeout": 5
          }
        ]
      }
    ],
    "PreToolUse": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python3 /Users/oscartsao/.claude/hooks/mode-guard.py",
            "timeout": 5
          }
        ],
        "matcher": "Bash|Edit|Write|NotebookEdit|MultiEdit"
      }
    ]
  }
```

And remove the line:
```json
"skipDangerousModePermissionPrompt": true,
```

- [ ] **Step 3: Disable `learning-output-style` plugin** (keeps `explanatory-output-style` which is already providing combined behavior per the session-reminder)

In `enabledPlugins`, change:
```json
"learning-output-style@claude-plugins-official": true
```
to:
```json
"learning-output-style@claude-plugins-official": false
```

> **Why keep explanatory, disable learning?** The Opus review noted both SessionStart hooks fire. The current session-reminder shows the system already operates in explanatory mode. Disabling learning removes the conflict without changing the active behavior you're already experiencing.

- [ ] **Step 4: Disable hookify plugin** (runs 4 Python scripts per tool call with zero active rules)

Change:
```json
"hookify@claude-plugins-official": true
```
to:
```json
"hookify@claude-plugins-official": false
```

> If you later want hookify rules, re-enable it and add `.local.md` rule files to `~/.claude/`.

- [ ] **Step 5: Verify JSON is valid**

```bash
python3 -c "import json; json.load(open('/Users/oscartsao/.claude/settings.json')); print('✅ Valid JSON')"
```
Expected: `✅ Valid JSON`

---

## Chunk 2: Rewrite `keyword-route.py`

### Task 2: UserPromptSubmit advisory hook for dangerous prompt patterns

**Files:**
- Rewrite: `~/.claude/hooks/keyword-route.py`
- Create: `~/.claude/hooks/test_hooks.py` (partial — keyword-route tests)

The redesign philosophy: **advisory-only, never blocks**. Focus on patterns that are _actually_ dangerous (destructive shell commands, SQL mass-delete), not common dev terms like "deploy" or "migration" that would cause constant false-positive noise.

- [ ] **Step 1: Write the failing test for keyword-route**

Create `~/.claude/hooks/test_hooks.py`:

```python
#!/usr/bin/env python3
"""Unit tests for keyword-route.py and mode-guard.py hook scripts."""
import json
import subprocess
import sys
from pathlib import Path

HOOKS_DIR = Path.home() / '.claude' / 'hooks'


def run_hook(script_name, stdin_data: dict) -> tuple[int, dict | None]:
    """Run a hook script with JSON stdin, return (exit_code, stdout_json)."""
    result = subprocess.run(
        [sys.executable, str(HOOKS_DIR / script_name)],
        input=json.dumps(stdin_data),
        capture_output=True,
        text=True,
    )
    stdout_json = None
    if result.stdout.strip():
        try:
            stdout_json = json.loads(result.stdout)
        except json.JSONDecodeError:
            pass
    return result.returncode, stdout_json


# ─── keyword-route tests ───────────────────────────────────────────────────────

def test_safe_prompt_exits_0():
    code, out = run_hook('keyword-route.py', {'prompt': 'add a new feature to the API', 'session_id': 'x', 'cwd': '/tmp'})
    assert code == 0, f"Expected 0, got {code}"
    assert out is None or 'systemMessage' not in out

def test_rm_rf_triggers_warning():
    code, out = run_hook('keyword-route.py', {'prompt': 'run rm -rf /tmp/test', 'session_id': 'x', 'cwd': '/tmp'})
    assert code == 0, "keyword-route should never block (exit 0 always)"
    assert out is not None and 'systemMessage' in out
    assert 'DANGEROUS' in out['systemMessage'].upper()

def test_drop_table_triggers_warning():
    code, out = run_hook('keyword-route.py', {'prompt': 'execute DROP TABLE users', 'session_id': 'x', 'cwd': '/tmp'})
    assert code == 0
    assert out is not None and 'systemMessage' in out

def test_force_push_triggers_warning():
    code, out = run_hook('keyword-route.py', {'prompt': 'git push --force origin main', 'session_id': 'x', 'cwd': '/tmp'})
    assert code == 0
    assert out is not None and 'systemMessage' in out

def test_common_dev_terms_do_not_trigger():
    """Words like 'deploy', 'schema', 'migration', 'production' must NOT fire."""
    safe_prompts = [
        'deploy the app to staging',
        'update the database schema',
        'create a new migration',
        'review the production config',
        'remove the unused import',
        'alter the function signature',
    ]
    for prompt in safe_prompts:
        code, out = run_hook('keyword-route.py', {'prompt': prompt, 'session_id': 'x', 'cwd': '/tmp'})
        assert code == 0, f"Unexpected non-zero exit for: {prompt!r}"
        assert out is None or 'systemMessage' not in out, f"False positive for: {prompt!r}"

def test_word_boundary_prevents_false_positive():
    """'Gibraltar' contains 'alter' — must NOT trigger."""
    code, out = run_hook('keyword-route.py', {'prompt': 'The Gibraltar deployment', 'session_id': 'x', 'cwd': '/tmp'})
    assert code == 0
    assert out is None or 'systemMessage' not in out

def test_malformed_stdin_does_not_crash():
    result = subprocess.run(
        [sys.executable, str(HOOKS_DIR / 'keyword-route.py')],
        input='not valid json{{',
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, "Must exit 0 on malformed input (fail-open)"


if __name__ == '__main__':
    tests = [v for k, v in globals().items() if k.startswith('test_')]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f'  ✅ {t.__name__}')
            passed += 1
        except AssertionError as e:
            print(f'  ❌ {t.__name__}: {e}')
            failed += 1
    print(f'\n{passed} passed, {failed} failed')
    sys.exit(1 if failed else 0)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 ~/.claude/hooks/test_hooks.py
```
Expected: `❌` failures for `test_rm_rf_triggers_warning`, `test_drop_table_triggers_warning`, `test_force_push_triggers_warning` (script not rewritten yet). `test_malformed_stdin_does_not_crash` may also fail.

- [ ] **Step 3: Rewrite `keyword-route.py`**

Replace the entire content of `~/.claude/hooks/keyword-route.py` with:

```python
#!/usr/bin/env python3
"""
UserPromptSubmit hook — warns on genuinely destructive prompt patterns.

Claude Code hook protocol:
  stdin : {"prompt": "...", "session_id": "...", "cwd": "..."}
  stdout: {"systemMessage": "..."} — injected as advisory context (optional)
  exit 0: always (advisory only, never blocks user prompts)
"""
import sys
import json
import re
from datetime import datetime, timezone
from pathlib import Path

LOG_FILE = Path.home() / '.claude' / 'logs' / 'mode-guard.log'

# Each entry: (regex_pattern, human_readable_description)
# Use word-boundary (\b) anchors to prevent substring false positives.
# ONLY include patterns for operations with irreversible destructive impact.
DANGEROUS_PATTERNS = [
    (r'\brm\s+-rf\b',                               'recursive force delete (rm -rf)'),
    (r'\bDROP\s+(?:TABLE|DATABASE|SCHEMA)\b',       'SQL DROP TABLE/DATABASE/SCHEMA'),
    (r'\bTRUNCATE\s+TABLE\b',                       'SQL TRUNCATE TABLE'),
    (r'\bDELETE\s+FROM\b(?!.*\bWHERE\b)',           'SQL DELETE FROM without WHERE clause'),
    (r'\bgit\s+push\s+(?:--force|-f)\b',            'force git push (rewrites remote history)'),
    (r'\bgit\s+reset\s+--hard\b',                   'hard git reset (discards uncommitted work)'),
    (r'\bgit\s+clean\s+.*-f\b',                     'git clean -f (deletes untracked files)'),
    (r'\bdd\s+if=',                                  'dd (raw disk write)'),
    (r'\bchmod\s+-R\s+777\b',                       'chmod -R 777 (world-writable recursively)'),
    (r'\bdrop_all\s*\(',                             'SQLAlchemy drop_all() (drops all tables)'),
]


def _log(msg: str) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    with open(LOG_FILE, 'a') as f:
        f.write(f"[{ts}] keyword-route: {msg}\n")


def scan(prompt: str) -> list[str]:
    """Return descriptions of all dangerous patterns found in prompt."""
    found = []
    for pattern, description in DANGEROUS_PATTERNS:
        if re.search(pattern, prompt, re.IGNORECASE):
            found.append(description)
    return found


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError, ValueError):
        # Fail-open: malformed input must never block Claude
        sys.exit(0)

    prompt = data.get('prompt', '')
    matched = scan(prompt)

    if matched:
        _log(f"Dangerous patterns detected: {matched}")
        warning = (
            "⚠️ DANGEROUS OPERATION in prompt — "
            + "; ".join(matched)
            + ". Confirm explicitly before executing. Prefer reversible alternatives."
        )
        print(json.dumps({"systemMessage": warning}))

    sys.exit(0)  # Advisory only — never block


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run tests — verify all pass**

```bash
python3 ~/.claude/hooks/test_hooks.py
```
Expected output:
```
  ✅ test_safe_prompt_exits_0
  ✅ test_rm_rf_triggers_warning
  ✅ test_drop_table_triggers_warning
  ✅ test_force_push_triggers_warning
  ✅ test_common_dev_terms_do_not_trigger
  ✅ test_word_boundary_prevents_false_positive
  ✅ test_malformed_stdin_does_not_crash

7 passed, 0 failed
```

- [ ] **Step 5: Verify script syntax**

```bash
python3 -m py_compile ~/.claude/hooks/keyword-route.py && echo "✅ syntax ok"
```

---

## Chunk 3: Rewrite `mode-guard.py`

### Task 3: PreToolUse enforcement hook for team-mode state

**Files:**
- Rewrite: `~/.claude/hooks/mode-guard.py`
- Modify: `~/.claude/hooks/test_hooks.py` (add mode-guard tests)

Design note: The original script tried to distinguish between agents by name (`sys.argv[1]`). Claude Code's hook API has **no agent-name field** in PreToolUse stdin — this distinction is not possible. The revised design enforces restrictions uniformly when `mode: strict` is set. The "executor-codex bypass" concept only works at the orchestrator level (the team-mode skill), not at the hook level.

- [ ] **Step 1: Add mode-guard tests to `test_hooks.py`**

Append to `~/.claude/hooks/test_hooks.py` (before `if __name__ == '__main__':`):

```python
# ─── mode-guard tests ─────────────────────────────────────────────────────────

import tempfile
import os

STATE_FILE = Path.home() / '.claude' / 'team_mode_state.json'


def _set_mode(mode: str, expires_at: str | None = None):
    state = {'mode': mode, 'expires_at': expires_at}
    STATE_FILE.write_text(json.dumps(state))


def _clear_state():
    if STATE_FILE.exists():
        # Reset to off rather than delete, to avoid confusing other sessions
        STATE_FILE.write_text(json.dumps({'mode': 'off'}))


def test_mode_off_allows_bash():
    _set_mode('off')
    code, out = run_hook('mode-guard.py', {
        'tool_name': 'Bash', 'tool_input': {'command': 'ls'}, 'tool_use_id': 'x', 'session_id': 'x', 'cwd': '/tmp'
    })
    assert code == 0, f"Expected 0 (allow), got {code}"
    _clear_state()


def test_strict_mode_blocks_bash():
    _set_mode('strict')
    code, out = run_hook('mode-guard.py', {
        'tool_name': 'Bash', 'tool_input': {'command': 'echo hi'}, 'tool_use_id': 'x', 'session_id': 'x', 'cwd': '/tmp'
    })
    assert code == 2, f"Expected 2 (block), got {code}"
    assert out is not None and out.get('decision') == 'block'
    _clear_state()


def test_strict_mode_blocks_edit():
    _set_mode('strict')
    code, out = run_hook('mode-guard.py', {
        'tool_name': 'Edit', 'tool_input': {}, 'tool_use_id': 'x', 'session_id': 'x', 'cwd': '/tmp'
    })
    assert code == 2
    _clear_state()


def test_strict_mode_blocks_write():
    _set_mode('strict')
    code, out = run_hook('mode-guard.py', {
        'tool_name': 'Write', 'tool_input': {}, 'tool_use_id': 'x', 'session_id': 'x', 'cwd': '/tmp'
    })
    assert code == 2
    _clear_state()


def test_strict_mode_does_not_block_read():
    """Read, Glob, Grep are not write tools — must be allowed in strict mode."""
    _set_mode('strict')
    for tool in ['Read', 'Glob', 'Grep']:
        code, out = run_hook('mode-guard.py', {
            'tool_name': tool, 'tool_input': {}, 'tool_use_id': 'x', 'session_id': 'x', 'cwd': '/tmp'
        })
        assert code == 0, f"{tool} should be allowed in strict mode, got exit {code}"
    _clear_state()


def test_expired_strict_mode_allows_bash():
    """An expired strict mode should auto-reset and allow operations."""
    _set_mode('strict', expires_at='2020-01-01T00:00:00Z')
    code, out = run_hook('mode-guard.py', {
        'tool_name': 'Bash', 'tool_input': {'command': 'echo hi'}, 'tool_use_id': 'x', 'session_id': 'x', 'cwd': '/tmp'
    })
    assert code == 0, f"Expired strict mode should allow, got {code}"
    # Verify state was reset to off
    state = json.loads(STATE_FILE.read_text())
    assert state['mode'] == 'off'
    _clear_state()


def test_semi_strict_warns_but_allows():
    _set_mode('semi-strict')
    code, out = run_hook('mode-guard.py', {
        'tool_name': 'Bash', 'tool_input': {'command': 'echo hi'}, 'tool_use_id': 'x', 'session_id': 'x', 'cwd': '/tmp'
    })
    assert code == 0, "semi-strict should allow (exit 0)"
    assert out is not None and 'systemMessage' in out
    _clear_state()


def test_malformed_state_file_fails_open():
    STATE_FILE.write_text('{invalid json{{')
    code, out = run_hook('mode-guard.py', {
        'tool_name': 'Bash', 'tool_input': {'command': 'ls'}, 'tool_use_id': 'x', 'session_id': 'x', 'cwd': '/tmp'
    })
    assert code == 0, "Malformed state file must fail-open"
    _clear_state()


def test_malformed_stdin_does_not_crash_mode_guard():
    result = subprocess.run(
        [sys.executable, str(HOOKS_DIR / 'mode-guard.py')],
        input='{{bad json',
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
```

- [ ] **Step 2: Run tests to verify mode-guard tests fail**

```bash
python3 ~/.claude/hooks/test_hooks.py
```
Expected: mode-guard tests fail (old script uses `sys.argv`, crashes with `IndexError`).

- [ ] **Step 3: Rewrite `mode-guard.py`**

Replace the entire content of `~/.claude/hooks/mode-guard.py` with:

```python
#!/usr/bin/env python3
"""
PreToolUse hook — blocks write operations when team mode is 'strict'.

Claude Code hook protocol:
  stdin : {"tool_name": "...", "tool_input": {...}, "tool_use_id": "...", "session_id": "...", "cwd": "..."}
  stdout: {"decision": "block", "reason": "..."} — to block (requires exit 2)
          {"systemMessage": "..."}               — advisory (exit 0)
  exit 0: allow
  exit 2: block tool call
"""
import sys
import json
from datetime import datetime, timezone
from pathlib import Path

STATE_FILE = Path.home() / '.claude' / 'team_mode_state.json'
LOG_FILE   = Path.home() / '.claude' / 'logs' / 'mode-guard.log'

# Tools that can modify the filesystem or execute code
GUARDED_TOOLS = {'Write', 'Edit', 'MultiEdit', 'Bash', 'NotebookEdit'}


def _log(msg: str) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    with open(LOG_FILE, 'a') as f:
        f.write(f"[{ts}] mode-guard: {msg}\n")


def _read_state() -> dict:
    try:
        if STATE_FILE.exists():
            return json.loads(STATE_FILE.read_text())
    except (json.JSONDecodeError, PermissionError, OSError):
        pass
    return {'mode': 'off'}


def _write_state(state: dict) -> None:
    try:
        STATE_FILE.write_text(json.dumps(state, indent=2))
    except (PermissionError, OSError) as e:
        _log(f"Failed to write state: {e}")


def _check_expiry(state: dict) -> dict:
    """If mode has an expiry that has passed, reset to off and return updated state."""
    expires_at = state.get('expires_at')
    if not expires_at:
        return state
    try:
        exp = datetime.fromisoformat(expires_at.replace('Z', '+00:00'))
        if datetime.now(timezone.utc) > exp:
            _log(f"Mode '{state.get('mode')}' expired at {expires_at}, resetting to off")
            state = {'mode': 'off', 'expires_at': None, 'reason': 'expired'}
            _write_state(state)
    except (ValueError, AttributeError):
        pass
    return state


def main() -> None:
    # Parse stdin — fail-open on any error
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError, ValueError):
        sys.exit(0)

    tool_name = data.get('tool_name', '')

    # Only gate write-capable tools; read-only tools are always allowed
    if tool_name not in GUARDED_TOOLS:
        sys.exit(0)

    state = _read_state()
    state = _check_expiry(state)
    mode  = state.get('mode', 'off')

    if mode == 'strict':
        _log(f"BLOCKED {tool_name} (mode=strict)")
        print(json.dumps({
            "decision": "block",
            "reason": (
                f"🔒 Team mode STRICT: {tool_name} is blocked. "
                "Only the executor-codex subagent may perform write operations. "
                f"To disable strict mode: echo '{{\"mode\":\"off\"}}' > {STATE_FILE}"
            )
        }))
        sys.exit(2)

    if mode == 'semi-strict':
        _log(f"WARNING: {tool_name} in semi-strict mode")
        print(json.dumps({
            "systemMessage": (
                f"⚠️ Team mode SEMI-STRICT: {tool_name} requested. "
                "Consider delegating write operations to the executor-codex subagent."
            )
        }))
        sys.exit(0)

    # mode == 'off' or any unknown mode: allow
    sys.exit(0)


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run all tests — verify all pass**

```bash
python3 ~/.claude/hooks/test_hooks.py
```
Expected:
```
  ✅ test_safe_prompt_exits_0
  ✅ test_rm_rf_triggers_warning
  ✅ test_drop_table_triggers_warning
  ✅ test_force_push_triggers_warning
  ✅ test_common_dev_terms_do_not_trigger
  ✅ test_word_boundary_prevents_false_positive
  ✅ test_malformed_stdin_does_not_crash
  ✅ test_mode_off_allows_bash
  ✅ test_strict_mode_blocks_bash
  ✅ test_strict_mode_blocks_edit
  ✅ test_strict_mode_blocks_write
  ✅ test_strict_mode_does_not_block_read
  ✅ test_expired_strict_mode_allows_bash
  ✅ test_semi_strict_warns_but_allows
  ✅ test_malformed_state_file_fails_open
  ✅ test_malformed_stdin_does_not_crash_mode_guard

16 passed, 0 failed
```

- [ ] **Step 5: Verify script syntax**

```bash
python3 -m py_compile ~/.claude/hooks/mode-guard.py && echo "✅ syntax ok"
```

- [ ] **Step 6: Reset orphaned state file to 'off'**

The state file currently has `mode: strict` from a prior manual activation. Since strict mode was activated without enforcement, reset it cleanly:

```bash
echo '{"mode": "off", "reason": "reset_after_hook_fix"}' > ~/.claude/team_mode_state.json
cat ~/.claude/team_mode_state.json
```
Expected: the file shows `mode: off`.

---

## Chunk 4: End-to-End Verification

### Task 4: Verify hooks fire in a live Claude Code session

These are manual verification steps (cannot be unit-tested without a live session).

- [ ] **Step 1: Verify `keyword-route.py` fires on prompt submission**

In a new Claude Code session, type a prompt containing a dangerous pattern:
```
Can you run rm -rf /tmp/test_delete_me to clean up?
```
Expected: Before Claude responds, a system message appears in Claude's context warning about "recursive force delete (rm -rf)". Claude's response should acknowledge the warning.

- [ ] **Step 2: Verify `keyword-route.py` does NOT fire on normal prompts**

Type:
```
Let me deploy the new feature to staging and update the schema migration.
```
Expected: No warning injected. Normal response.

- [ ] **Step 3: Verify `mode-guard.py` blocks in strict mode**

Activate strict mode manually:
```bash
echo '{"mode": "strict"}' > ~/.claude/team_mode_state.json
```

Then ask Claude to edit a file. Expected: Claude's Edit/Write/Bash attempts are blocked with the "Team mode STRICT" message. Claude should report it cannot perform write operations.

- [ ] **Step 4: Deactivate strict mode**

```bash
echo '{"mode": "off"}' > ~/.claude/team_mode_state.json
```

Verify Claude can edit files normally again.

- [ ] **Step 5: Check logs were written**

```bash
cat ~/.claude/logs/mode-guard.log | tail -20
```
Expected: Log entries with timestamps showing the hook invocations.

- [ ] **Step 6: Verify conflicting SessionStart hooks are resolved**

In a new session, the session-reminder should show only one output-style mode injection (explanatory), not both.

---

## Summary of Changes

| What | Before | After |
|------|--------|-------|
| `keyword-route.py` wiring | Not registered (dead) | `UserPromptSubmit` hook in `settings.json` |
| `mode-guard.py` wiring | Not registered (dead) | `PreToolUse` hook, matcher: write tools |
| Hook input protocol | `sys.argv` (wrong) | `json.load(sys.stdin)` (correct) |
| Dangerous-operation warning | Silent | `systemMessage` advisory injected |
| Strict-mode enforcement | Unenforced | Blocks Write/Edit/Bash with exit 2 |
| `skipDangerousModePermissionPrompt` | `true` (disabled built-in safety) | Removed (built-in prompt restored) |
| False-positive keywords | Substring match on "deploy", "schema" etc. | Regex word-boundary match on truly destructive ops only |
| Timezone handling | `datetime.utcnow()` (naive, fragile) | `datetime.now(timezone.utc)` (correct) |
| Malformed state/stdin | Unhandled crash | `try/except` fail-open |
| Conflicting output-style plugins | Both firing | `learning-output-style` disabled |
| hookify overhead | 4 Python scripts/tool call, 0 rules | Plugin disabled |
| State file | Orphaned `mode: strict` | Reset to `off` |
