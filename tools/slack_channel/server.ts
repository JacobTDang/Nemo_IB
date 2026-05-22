// Slack channel for Claude Code.
//
// Bridges Slack DMs to the local CC session via the experimental
// claude/channel capability. Inbound: bolt receives a DM, we gate it
// against the allowlist, then emit notifications/claude/channel. Outbound:
// CC calls the reply tool, which posts to chat.postMessage.
//
// Env file at ~/.claude/channels/slack/.env (outside the repo) supplies
// SLACK_BOT_TOKEN, SLACK_APP_TOKEN, SLACK_ALLOWED_USER_ID. Allowlist file
// at ~/.claude/channels/slack/access.json supplies multi-user allowlists.

import { Server } from '@modelcontextprotocol/sdk/server/index.js'
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js'
import { CallToolRequestSchema, ListToolsRequestSchema } from '@modelcontextprotocol/sdk/types.js'
import { App, LogLevel, type Logger } from '@slack/bolt'
import { homedir } from 'os'
import { join } from 'path'

import { buildMeta, chunkText, isAllowed, loadEnvFile } from './lib.ts'

const CHANNEL_HOME = join(homedir(), '.claude', 'channels', 'slack')
const ENV_FILE     = join(CHANNEL_HOME, '.env')
const ACCESS_FILE  = join(CHANNEL_HOME, 'access.json')

loadEnvFile(ENV_FILE)

const BOT_TOKEN = process.env.SLACK_BOT_TOKEN
const APP_TOKEN = process.env.SLACK_APP_TOKEN
if (!BOT_TOKEN || !APP_TOKEN) {
  process.stderr.write(
    `slack channel: missing SLACK_BOT_TOKEN or SLACK_APP_TOKEN\n` +
    `               expected in ${ENV_FILE}\n`,
  )
  process.exit(1)
}

// Bolt's default logger writes to stdout — that corrupts MCP stdio. Route
// everything through stderr instead.
// Bolt's debug stream is high-volume (ping/pong, WSS handshakes); silence it.
// Info/warn/error still surface to stderr, which stays out of MCP stdio.
const stderrLogger: Logger = {
  debug: () => {},
  info:  (...m) => process.stderr.write(`[bolt info ] ${m.join(' ')}\n`),
  warn:  (...m) => process.stderr.write(`[bolt warn ] ${m.join(' ')}\n`),
  error: (...m) => process.stderr.write(`[bolt error] ${m.join(' ')}\n`),
  setLevel: () => {},
  getLevel: () => LogLevel.WARN,
  setName:  () => {},
}

const mcp = new Server(
  { name: 'slack', version: '0.1.0' },
  {
    capabilities: {
      experimental: { 'claude/channel': {} },
      tools: {},
    },
    instructions: [
      'Messages from Slack arrive as <channel source="slack" channel="..." user="..." user_id="..." thread_ts="..." ts="...">.',
      'Reply by calling the reply tool. Always pass back the channel attribute from the inbound message.',
      'Pass thread_ts from the inbound tag to keep the reply threaded under the user\'s message; omit thread_ts only when starting a new top-level conversation.',
      'For proactive notifications (no inbound message — Sentry findings, alerts, briefs), use post_notification. It opens a DM with SLACK_ALLOWED_USER_ID and posts there.',
      'mrkdwn=true (default) renders Slack-flavored markdown (bold, code blocks, etc.).',
      'Very long replies are chunked at 40000 characters across multiple messages — keep responses focused.',
    ].join('\n'),
  },
)

mcp.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [{
    name: 'reply',
    description: 'Send a message back to Slack. Pass the channel from the inbound message.',
    inputSchema: {
      type: 'object',
      required: ['channel', 'text'],
      properties: {
        channel:   { type: 'string', description: 'Slack channel ID from the inbound channel attribute.' },
        text:      { type: 'string', description: 'Message body. Chunked at 40000 chars.' },
        thread_ts: { type: 'string', description: 'Inbound thread_ts to keep the reply threaded.' },
        mrkdwn:    { type: 'boolean', description: 'Render Slack-flavored markdown. Default true.' },
      },
    },
  }, {
    name: 'post_notification',
    description: 'Proactively send a Slack DM to the allowed user. Use for autonomous notifications (Sentry findings, kill-switch alerts, morning briefs) where there is no inbound message to reply to. Opens a DM channel with SLACK_ALLOWED_USER_ID and posts there.',
    inputSchema: {
      type: 'object',
      required: ['text'],
      properties: {
        text:   { type: 'string', description: 'Message body. Chunked at 40000 chars. mrkdwn supported.' },
        mrkdwn: { type: 'boolean', description: 'Render Slack-flavored markdown. Default true.' },
        user:   { type: 'string', description: 'Override target user id. Defaults to SLACK_ALLOWED_USER_ID env var.' },
      },
    },
  }],
}))

mcp.setRequestHandler(CallToolRequestSchema, async req => {
  if (req.params.name === 'reply') {
    const args = (req.params.arguments ?? {}) as {
      channel?: string
      text?: string
      thread_ts?: string
      mrkdwn?: boolean
    }
    if (!args.channel || !args.text) {
      throw new Error('reply requires channel and text')
    }
    const parts = chunkText(args.text, 40000)
    for (const part of parts) {
      await app.client.chat.postMessage({
        channel: args.channel,
        text: part,
        thread_ts: args.thread_ts,
        mrkdwn: args.mrkdwn ?? true,
      })
    }
    return { content: [{ type: 'text', text: `sent ${parts.length} message(s)` }] }
  }

  if (req.params.name === 'post_notification') {
    const args = (req.params.arguments ?? {}) as {
      text?: string
      mrkdwn?: boolean
      user?: string
    }
    if (!args.text) {
      throw new Error('post_notification requires text')
    }
    const targetUser = args.user ?? process.env.SLACK_ALLOWED_USER_ID
    if (!targetUser) {
      throw new Error('post_notification requires user arg or SLACK_ALLOWED_USER_ID env var')
    }
    // Open (or fetch existing) DM channel with the target user, then post.
    // conversations.open is idempotent and returns the same channel id on
    // repeated calls. Requires the bot to have im:write scope.
    const openResp = await app.client.conversations.open({ users: targetUser })
    const dmChannel = openResp.channel?.id
    if (!dmChannel) {
      throw new Error(`could not open DM channel with user ${targetUser}: ${JSON.stringify(openResp)}`)
    }
    const parts = chunkText(args.text, 40000)
    for (const part of parts) {
      await app.client.chat.postMessage({
        channel: dmChannel,
        text: part,
        mrkdwn: args.mrkdwn ?? true,
      })
    }
    return { content: [{ type: 'text', text: `posted ${parts.length} notification message(s) to ${targetUser}` }] }
  }

  throw new Error(`unknown tool: ${req.params.name}`)
})

const app = new App({
  token: BOT_TOKEN,
  appToken: APP_TOKEN,
  socketMode: true,
  logger: stderrLogger,
  logLevel: LogLevel.WARN,
})

app.message(async ({ message }) => {
  // The MessageEvent union has many subtypes; cast once after narrowing.
  if ('subtype' in message && message.subtype) return
  if ('bot_id' in message && message.bot_id) return

  const m = message as {
    user?: string
    channel?: string
    channel_type?: string
    text?: string
    ts?: string
    thread_ts?: string
  }

  // v1 scope: DMs only.
  if (m.channel_type !== 'im') return
  if (!m.text || !m.user || !m.channel) return

  if (!isAllowed(m.user, ACCESS_FILE, process.env.SLACK_ALLOWED_USER_ID)) {
    process.stderr.write(`slack channel: dropped message from ${m.user} (not allowlisted)\n`)
    return
  }

  await mcp.notification({
    method: 'notifications/claude/channel',
    params: {
      content: m.text,
      meta: buildMeta({
        channel:   m.channel,
        user:      m.user,
        ts:        m.ts,
        thread_ts: m.thread_ts,
      }),
    },
  })
})

await mcp.connect(new StdioServerTransport())
process.stderr.write('slack channel: MCP stdio connected\n')

await app.start()
const allowSource = process.env.SLACK_ALLOWED_USER_ID ? 'env' : 'access.json'
process.stderr.write(`slack channel: Bolt Socket Mode connected (allowlist source: ${allowSource})\n`)

process.stdin.on('end', async () => {
  process.stderr.write('slack channel: stdin EOF, shutting down\n')
  try { await app.stop() } catch {}
  process.exit(0)
})
