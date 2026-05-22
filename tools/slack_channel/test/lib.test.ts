// Tests for the pure helpers in ../lib.ts. server.ts itself imports @slack/bolt
// and starts side effects at module load, so we keep its testable logic in
// lib.ts and target this file.

import { afterEach, beforeEach, describe, expect, test } from 'bun:test'
import { mkdtempSync, rmSync, writeFileSync } from 'fs'
import { tmpdir } from 'os'
import { join } from 'path'

import { buildMeta, chunkText, isAllowed, loadEnvFile } from '../lib.ts'

let tmp: string

beforeEach(() => { tmp = mkdtempSync(join(tmpdir(), 'slack-channel-test-')) })
afterEach(() => { rmSync(tmp, { recursive: true, force: true }) })

describe('loadEnvFile', () => {
  test('reads KEY=VALUE pairs into the env', () => {
    const path = join(tmp, '.env')
    writeFileSync(path, 'FOO=bar\nBAZ=qux\n')
    const env: NodeJS.ProcessEnv = {}
    loadEnvFile(path, env)
    expect(env.FOO).toBe('bar')
    expect(env.BAZ).toBe('qux')
  })

  test('skips blank lines and comments', () => {
    const path = join(tmp, '.env')
    writeFileSync(path, '# a comment\n\nFOO=bar\n   \n# trailing\n')
    const env: NodeJS.ProcessEnv = {}
    loadEnvFile(path, env)
    expect(env.FOO).toBe('bar')
    expect(Object.keys(env)).toEqual(['FOO'])
  })

  test('does not override values already set in env', () => {
    const path = join(tmp, '.env')
    writeFileSync(path, 'FOO=from-file\n')
    const env: NodeJS.ProcessEnv = { FOO: 'from-shell' }
    loadEnvFile(path, env)
    expect(env.FOO).toBe('from-shell')
  })

  test('is a no-op when the file is missing', () => {
    const env: NodeJS.ProcessEnv = {}
    loadEnvFile(join(tmp, 'does-not-exist'), env)
    expect(Object.keys(env)).toHaveLength(0)
  })

  test('handles CRLF line endings', () => {
    const path = join(tmp, '.env')
    writeFileSync(path, 'FOO=bar\r\nBAZ=qux\r\n')
    const env: NodeJS.ProcessEnv = {}
    loadEnvFile(path, env)
    expect(env.FOO).toBe('bar')
    expect(env.BAZ).toBe('qux')
  })
})

describe('chunkText', () => {
  test('returns a single chunk when under the limit', () => {
    expect(chunkText('hello', 40000)).toEqual(['hello'])
  })

  test('returns a single chunk when exactly at the limit', () => {
    const s = 'x'.repeat(40000)
    expect(chunkText(s, 40000)).toEqual([s])
  })

  test('splits when over the limit', () => {
    const s = 'a'.repeat(45000)
    const out = chunkText(s, 40000)
    expect(out).toHaveLength(2)
    expect(out[0].length).toBe(40000)
    expect(out[1].length).toBe(5000)
    expect(out.join('')).toBe(s)
  })

  test('handles empty string', () => {
    expect(chunkText('', 40000)).toEqual([''])
  })
})

describe('isAllowed', () => {
  test('returns false when userId is undefined', () => {
    expect(isAllowed(undefined, join(tmp, 'access.json'), undefined)).toBe(false)
  })

  test('env allowlist short-circuits before reading file', () => {
    // No file exists; env var alone authorizes.
    expect(isAllowed('U1', join(tmp, 'access.json'), 'U1')).toBe(true)
  })

  test('reads access.json allowFrom', () => {
    const path = join(tmp, 'access.json')
    writeFileSync(path, JSON.stringify({ allowFrom: ['U1', 'U2'] }))
    expect(isAllowed('U2', path, undefined)).toBe(true)
    expect(isAllowed('U3', path, undefined)).toBe(false)
  })

  test('returns false when access.json is malformed', () => {
    const path = join(tmp, 'access.json')
    writeFileSync(path, 'not json {{{')
    expect(isAllowed('U1', path, undefined)).toBe(false)
  })

  test('returns false when allowFrom is missing or wrong type', () => {
    const path = join(tmp, 'access.json')
    writeFileSync(path, JSON.stringify({ allowFrom: 'U1' }))    // wrong type
    expect(isAllowed('U1', path, undefined)).toBe(false)
  })

  test('returns false when file is missing', () => {
    expect(isAllowed('U1', join(tmp, 'missing.json'), undefined)).toBe(false)
  })
})

describe('buildMeta', () => {
  test('mirrors user into user_id and defaults thread_ts to ts', () => {
    const meta = buildMeta({ channel: 'C1', user: 'U1', ts: '123.456' })
    expect(meta).toEqual({
      channel:   'C1',
      user:      'U1',
      user_id:   'U1',
      thread_ts: '123.456',
      ts:        '123.456',
    })
  })

  test('preserves an explicit thread_ts', () => {
    const meta = buildMeta({
      channel:   'C1',
      user:      'U1',
      ts:        '200.000',
      thread_ts: '100.000',
    })
    expect(meta.thread_ts).toBe('100.000')
    expect(meta.ts).toBe('200.000')
  })

  test('handles missing ts without throwing', () => {
    const meta = buildMeta({ channel: 'C1', user: 'U1' })
    expect(meta.thread_ts).toBe('')
    expect(meta.ts).toBe('')
  })
})
