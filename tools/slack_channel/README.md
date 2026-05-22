# Slack channel for Claude Code

A two-way bridge between Slack DMs and a local `claude` CLI session. Sits on
Claude Code's experimental [Channels](https://code.claude.com/docs/en/channels.md)
feature, so messages arrive in CC as `<channel source="slack" ...>` tags and
CC replies through a built-in `reply` tool.

Use case: DM the bot from any Slack client (desktop, web, mobile) and the
message lands in your running Nemo CC session with all five MCP servers
attached. Replaces the mobile-app + Remote-Control path with something you
can use from Slack channels too.

## Architecture

```
   Slack workspace
         |  (Socket Mode WebSocket, app token xapp-...)
         v
 tools/slack_channel/server.ts        <- Bun, MCP server
   |                  ^
   | notifications/   | reply tool call
   | claude/channel   |
   v                  |
 claude CLI (launched by scripts/nemo-launch.ps1)
   |
   v
 nemo_alpaca, nemo_financial, nemo_finnhub, nemo_fred, nemo_web
```

The plugin runs as a CC subprocess over stdio (same pattern as every other
Nemo MCP server). It holds a long-lived Socket Mode WebSocket to Slack and
pumps messages in both directions.

## Files

| Path | Purpose |
|---|---|
| `package.json` | Bun deps (`@modelcontextprotocol/sdk`, `@slack/bolt`) and start script |
| `server.ts` | Orchestrator: env load, MCP server, Bolt app, message handlers, shutdown |
| `lib.ts` | Pure helpers (env loader, allowlist gate, chunker, meta builder) |
| `test/lib.test.ts` | `bun test` coverage of the pure helpers |
| `.env.example` | Documents the three required env vars |

Secrets and allowlist live **outside** the repo at:

| Path | Purpose |
|---|---|
| `~/.claude/channels/slack/.env` | `SLACK_BOT_TOKEN`, `SLACK_APP_TOKEN`, `SLACK_ALLOWED_USER_ID` |
| `~/.claude/channels/slack/access.json` | `{ "allowFrom": ["U..."] }` multi-user allowlist |

## Slack app setup

1. https://api.slack.com/apps -> **Create New App** -> **From scratch**.
   Name `Nemo` (or whatever), pick your workspace.
2. **Socket Mode** -> toggle on. Create an app-level token named `nemo-socket`
   with scope `connections:write`. Save the `xapp-...` token.
3. **OAuth & Permissions** -> Bot Token Scopes: add `chat:write`, `im:history`,
   `im:read`, `im:write`, `app_mentions:read`.
4. **Event Subscriptions** -> Enable Events -> Subscribe to bot events ->
   add `message.im`. Save changes.
5. **App Home** -> Show Tabs -> tick **Messages Tab** and **Allow users to
   send Slash commands and messages from the messages tab**.
6. **Install App** -> Install to Workspace. Copy the `xoxb-...` Bot User
   OAuth Token.
7. In Slack itself: your profile -> kebab menu -> **Copy member ID**
   (format `U...`).

## Local secrets

Create `~/.claude/channels/slack/.env`:

```
SLACK_BOT_TOKEN=xoxb-...
SLACK_APP_TOKEN=xapp-...
SLACK_ALLOWED_USER_ID=U...
```

Create `~/.claude/channels/slack/access.json` (one or more user IDs):

```json
{ "allowFrom": ["U0123ABCDEF"] }
```

The plugin reads `SLACK_ALLOWED_USER_ID` as a single-user fast path and
falls back to `access.json` if the env var is unset. Either is fine; both
together is fine (env wins for that one ID, file is checked for others).

## Install and register

```powershell
cd tools/slack_channel
bun install

# Register with CC at user scope (matches the rest of Nemo's MCP setup)
claude mcp add --scope user slack -- `
  "$env:USERPROFILE\.bun\bin\bun.exe" run `
  --cwd "$PWD" --shell=bun --silent start
```

Verify:

```powershell
claude mcp list
# slack: ... bun.exe run --cwd ... start - Connected
```

## Launch

`scripts/nemo-launch.ps1` already passes
`--dangerously-load-development-channels server:slack`. Just double-click
`nemo.bat` as normal. Watch for these stderr lines on startup:

```
slack channel: MCP stdio connected
slack channel: Bolt Socket Mode connected (allowlist source: env)
```

## Smoke test

1. From any Slack client, DM the bot: `time?`. CC should respond inline.
2. Verify the bot drops non-allowlisted senders: temporarily remove your ID
   from `access.json` (or unset `SLACK_ALLOWED_USER_ID`), DM again, watch
   `slack channel: dropped message from <U...> (not allowlisted)` on stderr.
3. Nemo round-trip: `pull AAPL revenue base and reply with the number`.
   Expect CC to call `mcp__nemo_web__get_revenue_base`, then call the
   `reply` tool. The number lands in Slack.

## Tests

```powershell
cd tools/slack_channel
bun test
```

Covers the pure helpers (env loader, allowlist gate, chunker, meta builder).
`server.ts` itself is integration-tested by running CC and DMing the bot.

## Trading discipline

Unchanged. The Slack frontend doesn't bypass `risk_check_proposed_trade` --
that gate is server-side in `tools/alpaca/server.py` and runs the same way
whether the user prompt arrives from the terminal, the mobile app, or
Slack. See the CLAUDE.md analyst playbook for the rules.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `claude mcp list` shows `slack: ... - x Failed to connect` | Missing env file or bad token. Check `~/.claude/channels/slack/.env`. Look at `~/.claude/debug/<session>.txt` for the bun stderr trace. |
| Slack DMs to the bot get no response | Verify `message.im` is subscribed under Event Subscriptions, and that `Allow users to send messages from the messages tab` is checked in App Home. |
| Bot replies in Slack but CC doesn't see the inbound | The allowlist dropped your user ID. Check `slack channel: dropped message from ...` on stderr. |
| `Socket Mode connection failure` on startup | App-level token is wrong, or `connections:write` scope is missing. Regenerate the `xapp-` token. |
| Zombie `bun.exe` after Ctrl-C | The stdin EOF handler should catch this; check Task Manager. Open issue if it recurs. |

## Not yet built (v2+)

- Multi-CC-session support (channel-per-session routing)
- Pairing-flow skill for managing the allowlist without editing JSON
- `claude/channel/permission` capability for remote tool-approval buttons
  (not needed while running with `--dangerously-skip-permissions`)
- File attachments (Slack `files.upload_v2`)
- Streaming output via `chat.update` edits
- Publishing as a proper plugin to a marketplace
