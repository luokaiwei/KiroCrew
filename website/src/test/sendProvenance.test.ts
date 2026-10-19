import { describe, expect, it } from 'vitest'

import { queueEntrySendProvenance, readSendProvenance, SEND_PROVENANCE_META_KEY, shortSessionKey } from '../pages/chat/sendProvenance'

const MID = { mid: 'm-1' }
/** A write time before the record existed: the only rows the text fallback may read. */
const OLD = '2026-10-01T12:00:00+00:00'

const ENV = '[sent by session chat-316-1792176321 via session_send]'
const record = (extra: Record<string, unknown> = {}) => ({
  [SEND_PROVENANCE_META_KEY]: { from: 'chat-316-1792176321', title: 'Release conductor', via: 'session_send', mode: 'queued', ...extra },
})

describe('readSendProvenance', () => {
  it('reads the structured record and strips the envelope line', () => {
    const out = readSendProvenance(`${ENV}\n\nrebase onto main`, record())
    expect(out?.provenance).toEqual({ from: 'chat-316-1792176321', title: 'Release conductor', via: 'session_send', mode: 'queued' })
    expect(out?.body).toBe('rebase onto main')
  })

  it('keeps a multi-line body intact below the envelope', () => {
    const body = 'line one\n\n- item\n- item\n\n```\ncode\n```'
    expect(readSendProvenance(`${ENV}\n\n${body}`, record())?.body).toBe(body)
  })

  it('reads a broadcast count and a steer mode', () => {
    const b = readSendProvenance('[sent by session chat-1 via session_broadcast]\n\nx', {
      [SEND_PROVENANCE_META_KEY]: { from: 'chat-1', via: 'session_broadcast', mode: 'queued', count: 3 },
    })
    expect(b?.provenance.count).toBe(3)
    expect(b?.body).toBe('x')
    const s = readSendProvenance(`${ENV}\n\nstop`, record({ mode: 'steered' }))
    expect(s?.provenance.mode).toBe('steered')
  })

  it('falls back to the text envelope on a legacy row', () => {
    const out = readSendProvenance(`${ENV}\n\nold row`, { mid: 'm-1' }, OLD)
    expect(out?.provenance).toEqual({ from: 'chat-316-1792176321', via: 'session_send', legacy: true })
    expect(out?.body).toBe('old row')
  })

  it('legacy: an envelope with an empty body after it yields an empty body', () => {
    expect(readSendProvenance(ENV, MID, OLD)?.body).toBe('')
    expect(readSendProvenance(`${ENV}\n\n`, MID, OLD)?.body).toBe('')
  })

  it('never labels a row a person typed, whatever its text or meta says', () => {
    expect(readSendProvenance(`${ENV}\n\nhi`, { ...MID, human: true }, OLD)).toBeNull()
    expect(readSendProvenance(`${ENV}\n\nhi`, { ...MID, ...record(), human: true }, OLD)).toBeNull()
    expect(readSendProvenance(`${ENV}\n\nhi`, { ...MID, turnActor: 'app' }, OLD)).toBeNull()
  })

  it('never labels a bubble this client minted', () => {
    expect(readSendProvenance(`${ENV}\n\nhi`, { ...MID, sendId: 's-1' }, OLD)).toBeNull()
    expect(readSendProvenance(`${ENV}\n\nhi`, { ...MID, optimistic: true, steer: true }, OLD)).toBeNull()
    expect(readSendProvenance(`${ENV}\n\nhi`, { ...MID, sendIds: ['a', 'b'] }, OLD)).toBeNull()
  })

  it('does not treat the neutralized quoted form as an envelope', () => {
    expect(readSendProvenance('[quoted: sent by session chat-316-1792176321 via session_send]\n\nhi', MID, OLD)).toBeNull()
  })

  it('legacy: refuses an envelope that is not the opening line, or an unknown verb', () => {
    expect(readSendProvenance(`hello\n${ENV}\n\nhi`, MID, OLD)).toBeNull()
    expect(readSendProvenance(' [sent by session chat-1 via session_send]\n\nhi', MID, OLD)).toBeNull()
    expect(readSendProvenance('[sent by session chat-1 via session_create]\n\nhi', MID, OLD)).toBeNull()
    expect(readSendProvenance('[sent by session chat 1 via session_send]\n\nhi', MID, OLD)).toBeNull()
  })

  it('legacy: refuses a merged row that carries two envelopes', () => {
    const merged = `${ENV}\n\nfirst\n\n[sent by session chat-9 via session_send]\n\nsecond`
    expect(readSendProvenance(merged, MID, OLD)).toBeNull()
  })

  it('ignores a malformed structured record and falls back to the text', () => {
    const out = readSendProvenance(`${ENV}\n\nx`, { ...MID, [SEND_PROVENANCE_META_KEY]: { from: '', via: 'session_send' } }, OLD)
    expect(out?.provenance.legacy).toBe(true)
    expect(readSendProvenance('plain text', { [SEND_PROVENANCE_META_KEY]: { from: 'chat-1', via: 'nope' } })).toBeNull()
  })

  it('legacy: a row with no gateway message id is never read from its text', () => {
    // A row this client rebuilt from a live frame (queue_pop, edit-resend) has no
    // mid; a channel author's text could otherwise pose as a peer's delivery.
    expect(readSendProvenance(`${ENV}\n\nhi`, undefined)).toBeNull()
    expect(readSendProvenance(`${ENV}\n\nhi`, { files: [] })).toBeNull()
    expect(readSendProvenance(`${ENV}\n\nhi`, { mid: '' }, OLD)).toBeNull()
    // The structured record does not need one: only the gateway writes it.
    expect(readSendProvenance(`${ENV}\n\nhi`, record())?.provenance.from).toBe('chat-316-1792176321')
  })

  it('legacy: a row rebuilt from a restored queue entry is never read from its text', () => {
    // A restart drops the entry's actor (an app's text included), so its author
    // cannot be vouched for.
    expect(readSendProvenance(`${ENV}\n\nhi`, { ...MID, turnProvenanceRestored: true }, OLD)).toBeNull()
  })

  it('legacy: only a row written before the record existed is read from its text', () => {
    // A gateway running this code stamps the record on every delivery, so a
    // newer envelope with no record is not a delivery, whatever path wrote it.
    expect(readSendProvenance(`${ENV}\n\nhi`, MID, '2026-10-20T00:00:00Z')).toBeNull()
    expect(readSendProvenance(`${ENV}\n\nhi`, MID, '2027-01-01T00:00:00Z')).toBeNull()
    expect(readSendProvenance(`${ENV}\n\nhi`, MID)).toBeNull()
    expect(readSendProvenance(`${ENV}\n\nhi`, MID, 'not a time')).toBeNull()
    expect(readSendProvenance(`${ENV}\n\nhi`, MID, '2026-10-19T23:59:59Z')?.provenance.legacy).toBe(true)
    // The structured record is not time-bound.
    expect(readSendProvenance(`${ENV}\n\nhi`, record(), '2027-01-01T00:00:00Z')?.provenance.from).toBe('chat-316-1792176321')
  })

  it('returns null for an ordinary row', () => {
    expect(readSendProvenance('just a message', undefined)).toBeNull()
  })
})

describe('shortSessionKey', () => {
  it('drops the creation stamp and cuts a long key', () => {
    expect(shortSessionKey('chat-316-1792176321')).toBe('chat-316')
    expect(shortSessionKey('cron-nightly')).toBe('cron-nightly')
    expect(shortSessionKey('x'.repeat(40))).toBe(`${'x'.repeat(23)}…`)
  })
})

describe('queueEntrySendProvenance', () => {
  it('carries a valid record and drops anything else', () => {
    expect(queueEntrySendProvenance(record())).toEqual({ sendProvenance: { from: 'chat-316-1792176321', title: 'Release conductor', via: 'session_send', mode: 'queued' } })
    expect(queueEntrySendProvenance({ [SEND_PROVENANCE_META_KEY]: { from: 'x', via: 'evil' } })).toEqual({})
    expect(queueEntrySendProvenance({ [SEND_PROVENANCE_META_KEY]: 'chat-1' })).toEqual({})
    expect(queueEntrySendProvenance(undefined)).toEqual({})
  })
})
