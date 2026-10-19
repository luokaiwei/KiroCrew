import { Archive, CircleCheck, Code, ExternalLink } from 'lucide-react'
import { i18nT } from '../i18n/t'
import type { CodeChannelInfo } from '../types'
import { ChannelBrandIcon } from './ChannelBrandIcon'

/**
 * The "Slack code channel" badge for a session that IS a Slack code channel's
 * (`slot.code_channel`, set by the backend only for a bound session while code
 * channels are on, so every other session renders nothing here).
 *
 * One component, two placements: `row` is the compact chip on a sidebar row's
 * meta line (beside the agent name, so the row keeps its fixed height), `header`
 * the chip beside the session title. Both are a real anchor to the channel
 * (Slack's `app_redirect` URL, which opens the desktop app when it is installed
 * and the web client otherwise) in a new tab, marked with the external-link
 * glyph. When the backend could not build a link (a malformed record) the badge
 * still marks the session, as plain text, so it never offers a dead click.
 *
 * The accessible name always starts with "Slack code channel" (visible in the
 * header, an sr-only span on the compact row chip) so a screen reader hears what
 * the chip is before the channel's name. An archived or wrapped-up channel gets
 * its own glyph (an archive box, a check) beside the state word, so the two read
 * differently even on a meta line too narrow for the word, and the tooltip says
 * what each state means.
 */
export function codeChannelName(info: CodeChannelInfo): string {
  return info.channel_name ? `#${info.channel_name}` : ''
}

function statusLabel(status: CodeChannelInfo['status']): string {
  if (status === 'archived') return i18nT('components.codeChannelBadge.archived')
  if (status === 'wrapped_up') return i18nT('components.codeChannelBadge.wrapped_up')
  return ''
}

function statusHint(status: CodeChannelInfo['status']): string {
  if (status === 'archived') return i18nT('components.codeChannelBadge.archived_hint')
  if (status === 'wrapped_up') return i18nT('components.codeChannelBadge.wrapped_up_hint')
  return ''
}

/** Where the click goes. */
export function codeChannelOpenHint(info: CodeChannelInfo): string {
  const name = codeChannelName(info)
  return name
    ? i18nT('components.codeChannelBadge.open_named', { channel: name })
    : i18nT('components.codeChannelBadge.open_unnamed')
}

/** The tooltip: what a code channel is, what its state means, where the click goes.
 *  One sentence per line, so no locale's punctuation has to join them. */
function codeChannelTitle(info: CodeChannelInfo): string {
  return [
    i18nT('components.codeChannelBadge.about'),
    statusHint(info.status),
    info.web_url ? codeChannelOpenHint(info) : '',
  ].filter(Boolean).join('\n')
}

const CHIP_BASE =
  'inline-flex items-center gap-1 rounded-[4px] border no-underline transition-colors motion-reduce:transition-none'

/**
 * How the row chip gives up width on a short meta line (agent name, this chip,
 * the time). The time never shrinks and the agent name yields first (see
 * `SessionRow`); then this chip shrinks, down to a floor of exactly its border,
 * padding and Slack mark, so the mark is never cut.
 *
 * Inside the chip the text (the channel name, or the state word) has no width
 * of its own (`basis-0 grow`) and a floor of "#" plus a few characters. The chip
 * wraps onto a second line that its fixed height clips away, so text that cannot
 * keep that floor leaves whole, with the link glyph after it, instead of
 * shrinking to a sliver; the name stays in the tooltip and the accessible name.
 *
 * Under 10rem of meta line (a phone) the padding and gaps tighten, the link
 * glyph drops (the chip is still the link, and says so to a screen reader) and
 * a state word leaves for its glyph, so "#" plus a few characters fit beside a
 * long "Yesterday 05:55 AM". Under 8.5rem (the sidebar at its narrowest) the
 * padding drops to 1px so a mark-only chip and that time still fit together.
 * Those container-query classes sit inline on the elements (`ROW_CHIP_NARROW`
 * etc. would be ALL-CAPS constants the strict i18n gate reads as copy).
 */
const ROW_CHIP_FIT =
  'h-4 max-w-[55%] min-w-[20px] shrink flex-wrap content-start overflow-hidden px-1'
const ROW_TEXT_FIT = 'basis-0 grow min-w-[4.5ch] truncate'
/** The agent name beside a row chip: yields first (by a shrink weight no other
 *  item comes near). `SessionRow` also hides it on a meta line under 10rem,
 *  where it could only show "k…" and its gap is the room "#" plus a few
 *  characters of the channel need. Lives here so the chip's own fit and the
 *  order it depends on are read together. */
export const CODE_CHANNEL_AGENT_FIT = 'min-w-0 shrink-[1000]'

/** The phone top bar's marker: the Slack mark with a code glyph, so it cannot
 *  be taken for the plain Slack mark an ordinary Slack-linked session wears.
 *  A marker, not a link: the bar's title is already the session menu's
 *  trigger, and the sidebar row carries the link. */
export function CodeChannelMark({ size = 12 }: { size?: number }) {
  const label = i18nT('components.codeChannelBadge.label')
  return (
    <span className="inline-flex shrink-0 items-center gap-px" role="img" aria-label={label} title={label} data-testid="mobile-code-channel-glyph">
      <ChannelBrandIcon channel="slack" size={size} />
      <Code size={size - 2} className="shrink-0 text-muted" aria-hidden="true" />
    </span>
  )
}

export function CodeChannelBadge({ info, variant }: { info: CodeChannelInfo; variant: 'row' | 'header' }) {
  const name = codeChannelName(info)
  const state = statusLabel(info.status)
  const header = variant === 'header'
  const size = header ? 11 : 10
  const sizing = header
    ? 'min-w-0 max-w-full px-2 py-0.5 text-[11px] rounded-md pointer-events-auto'
    : `${ROW_CHIP_FIT} py-0 text-[10px] leading-[14px] font-normal`
  const tone = info.status === 'archived'
    ? 'border-border bg-bg-elevated/60 text-muted'
    : 'border-border bg-accent-subtle text-text'
  const StateIcon = info.status === 'archived' ? Archive : info.status === 'wrapped_up' ? CircleCheck : null
  const stateTone = info.status === 'wrapped_up' ? 'text-ok' : 'text-muted'
  const body = (
    <>
      {/* The Slack mark: a fixed box that never shrinks, first in the chip so
          a wrap never takes it. */}
      <span className="inline-flex shrink-0" style={{ width: size, height: size }} aria-hidden="true" data-testid="code-channel-slack-mark">
        <ChannelBrandIcon channel="slack" size={size} />
      </span>
      {header ? (
        <span className="shrink-0 whitespace-nowrap">{i18nT('components.codeChannelBadge.label')}</span>
      ) : (
        <span className="sr-only">{i18nT('components.codeChannelBadge.label')}</span>
      )}
      {/* The row's meta line has room for one word: an active channel shows its
          name, an archived or wrapped-up one its state instead (the name stays
          in the accessible name and the tooltip). */}
      {name ? (
        <span
          className={header ? 'truncate min-w-0 max-w-[32ch] text-muted' : state ? 'sr-only' : ROW_TEXT_FIT}
          dir="auto"
          data-testid="code-channel-name"
        >{name}</span>
      ) : !header ? (
        // No name: keep saying what the chip is, beside any state word. On a
        // meta line too narrow for both (a phone), the state word wins: the
        // row's `@container/rowmeta` decides, and the tooltip still says it.
        <span className={`${state ? 'hidden shrink-0 whitespace-nowrap @[13rem]/rowmeta:inline' : ROW_TEXT_FIT}`} aria-hidden="true">{i18nT('components.codeChannelBadge.label_short')}</span>
      ) : null}
      {StateIcon && (
        // Two tones as well as two glyphs: an archived channel is greyed, a
        // finished task reads as done. On the row the group's box dissolves
        // (`contents`), so its glyph stays with the mark and only its word
        // takes the text slot's floor and wrap.
        <span className={header ? `inline-flex min-w-0 items-center gap-0.5 ${stateTone}` : `contents ${stateTone}`} data-testid="code-channel-status">
          <StateIcon size={size - 1} className="shrink-0" aria-hidden="true" />
          <span className={header ? 'truncate min-w-0 whitespace-nowrap' : `${ROW_TEXT_FIT} @max-[10rem]/rowmeta:sr-only`}>{state}</span>
        </span>
      )}
      {info.web_url && <ExternalLink size={header ? 10 : 9} className={`shrink-0 text-muted ${header ? '' : '@max-[10rem]/rowmeta:hidden'}`} aria-hidden="true" />}
    </>
  )
  if (!info.web_url) {
    return (
      <span
        className={`${CHIP_BASE} ${sizing} ${tone} ${header ? '' : '@max-[10rem]/rowmeta:gap-0.5 @max-[10rem]/rowmeta:px-0.5 @max-[8.5rem]/rowmeta:min-w-[14px] @max-[8.5rem]/rowmeta:px-px'}`}
        data-testid={`code-channel-badge-${variant}`}
        title={codeChannelTitle(info)}
      >
        {body}
      </span>
    )
  }
  return (
    <a
      href={info.web_url}
      target="_blank"
      rel="noopener noreferrer"
      draggable={false}
      data-testid={`code-channel-badge-${variant}`}
      // The sidebar row is a click-to-switch button and opens the SESSION in a
      // background tab on a middle click: opening the channel must do neither.
      // Pointer focus is dropped so the row's :focus-visible action strip is
      // not pinned open; Enter keeps focus.
      onClick={e => {
        e.stopPropagation()
        if (e.detail > 0) e.currentTarget.blur()
      }}
      onAuxClick={e => e.stopPropagation()}
      className={`${CHIP_BASE} ${sizing} ${tone} ${header ? '' : '@max-[10rem]/rowmeta:gap-0.5 @max-[10rem]/rowmeta:px-0.5 @max-[8.5rem]/rowmeta:min-w-[14px] @max-[8.5rem]/rowmeta:px-px'} hover:text-text hover:border-accent focus-ring`}
      title={codeChannelTitle(info)}
    >
      {body}
      <span className="sr-only">{i18nT('components.codeChannelBadge.opens_in_slack')}</span>
    </a>
  )
}
