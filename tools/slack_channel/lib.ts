// Pure helpers used by server.ts. Split out so tests can import without
// triggering the Slack/MCP startup side effects in server.ts.

import { existsSync, readFileSync } from 'fs'

export function loadEnvFile(path: string, env: NodeJS.ProcessEnv = process.env): void {
  if (!existsSync(path)) return
  for (const raw of readFileSync(path, 'utf8').split(/\r?\n/)) {
    const line = raw.trim()
    if (!line || line.startsWith('#')) continue
    const eq = line.indexOf('=')
    if (eq < 0) continue
    const key = line.slice(0, eq).trim()
    const val = line.slice(eq + 1).trim()
    if (!(key in env)) env[key] = val
  }
}

export function chunkText(text: string, max: number): string[] {
  if (text.length <= max) return [text]
  const out: string[] = []
  let i = 0
  while (i < text.length) {
    out.push(text.slice(i, i + max))
    i += max
  }
  return out
}

export interface AccessFile {
  allowFrom: string[]
}

// isAllowed re-reads access.json on every call so the user can edit the
// allowlist without restarting CC. A parse error fails closed.
export function isAllowed(
  userId: string | undefined,
  accessFilePath: string,
  envAllowedId: string | undefined,
): boolean {
  if (!userId) return false
  if (envAllowedId && userId === envAllowedId) return true
  try {
    if (!existsSync(accessFilePath)) return false
    const parsed = JSON.parse(readFileSync(accessFilePath, 'utf8')) as Partial<AccessFile>
    const list = Array.isArray(parsed.allowFrom) ? parsed.allowFrom : []
    return list.includes(userId)
  } catch {
    return false
  }
}

export interface InboundMeta {
  channel: string
  user: string
  user_id: string
  thread_ts: string
  ts: string
}

// Build the meta dict for notifications/claude/channel. Keeps the field
// shape consistent and ignores undefined inputs so the schema stays clean.
export function buildMeta(input: {
  channel: string
  user: string
  ts?: string
  thread_ts?: string
}): InboundMeta {
  return {
    channel:   input.channel,
    user:      input.user,
    user_id:   input.user,
    thread_ts: input.thread_ts ?? input.ts ?? '',
    ts:        input.ts ?? '',
  }
}
