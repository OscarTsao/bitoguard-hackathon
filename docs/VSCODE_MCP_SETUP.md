# VS Code MCP Setup for BitoGuard

This workspace already enables Kiro MCP configuration via `.vscode/settings.json`.

The repository now ships a ready-to-merge MCP definition in `mcp.json` for running Codex CLI as a local stdio MCP server:

```json
{
  "mcpServers": {
    "codex-cli": {
      "command": "codex",
      "args": ["mcp-server"]
    }
  }
}
```

## Codex CLI MCP Server

Use `codex-cli` when you want a coding agent exposed as an MCP tool provider inside another MCP-capable client.

1. Verify Codex CLI is installed: `codex --version`
2. Verify you are logged in: `codex login`
3. Merge the `codex-cli` block from `mcp.json` into your client config.

### Kiro / VS Code

Merge the server block into `~/.kiro/settings/mcp.json`.

### Cursor

Merge the server block into `~/.cursor/mcp.json`.

## Recommended MCP Servers

Use only servers you trust and keep API keys in your shell profile or VS Code secrets.

1. `filesystem` (scoped to this repo)
2. `github` (issues/PR workflows)
3. `duckdb` (read-only SQL exploration for `bitoguard_core/artifacts/bitoguard.duckdb`)
4. `fetch` (docs/spec lookups)
5. `codex-cli` (local coding agent over stdio via `codex mcp-server`)

## Suggested Guardrails

1. Prefer read-only MCP servers by default.
2. Restrict filesystem roots to this repository path only.
3. Do not grant write/delete tools to MCP servers unless you need them.
4. Keep long-running commands in VS Code tasks, not MCP tools.

## Practical Tool Routing in This Repo

1. Use VS Code tasks for pipeline and tests (`make test`, `make train`, `make serve`).
2. Use Python + Ruff + Pylance for backend edits.
3. Use ESLint + Tailwind CSS tooling for frontend edits.
4. Use MCP only for cross-system integrations (GitHub, external docs, database exploration).
