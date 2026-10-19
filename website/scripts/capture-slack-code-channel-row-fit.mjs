/**
 * Geometry harness for a Slack code channel row's meta line at narrow widths.
 *
 * The meta line reads: agent name, the code-channel chip, the time. When it is
 * short of room the agent name must give way first, then the channel name, and
 * the Slack mark and the time must never clip. jsdom computes no layout, so the
 * unit test can only pin the classes; this loads the REAL built SPA with canned
 * slots (a "Code channels" folder with a long agent name and long channel
 * names, one row from today and one from yesterday) and MEASURES, at the
 * default sidebar width (260px), the narrowest one (180px) and a 320px phone:
 *
 *   - the Slack mark is drawn at its full size and lies wholly inside the chip;
 *   - the chip ends before the time starts, and the time is wholly inside its
 *     line (nothing clips it) and not ellipsized;
 *   - the channel name still shows "#" plus a few characters;
 *   - when the channel name is cut, the agent name is cut too (it yields first).
 *
 * Usage: node scripts/capture-slack-code-channel-row-fit.mjs [outDir] [--expect-bug]
 *   --expect-bug  exit 0 only if some check FAILS (run it against a build
 *                 without the fix, to prove the harness can see the bug).
 */
import { chromium } from 'playwright'
import { mkdirSync } from 'node:fs'
import { join } from 'node:path'
import { serveDist } from './lib/serve-dist.mjs'
import { logPageProblems, stubDashboardApi, json } from './lib/stub-dashboard-api.mjs'

const args = process.argv.slice(2)
const EXPECT_BUG = args.includes('--expect-bug')
const OUT = args.find(a => !a.startsWith('--')) || '../temp-screenshots/slack-code-channel-row-fit'
mkdirSync(OUT, { recursive: true })

const now = new Date()
const yesterday = new Date(now.getFullYear(), now.getMonth(), now.getDate() - 1, 5, 55).toISOString()
const channel = (id, name, status = 'active') => ({
  channel_name: name,
  status,
  web_url: `https://slack.com/app_redirect?channel=${id}&team=T0TEAM1`,
})
const FOLDER = 'f-code'
const slackLink = { channel: 'slack', label: 'Slack', target: '…0001', binding: 'b', direction: 'out', drives_session: true, live: true }
const row = (key, title, ts, code) => ({
  key, title, running: false, messages: 4, agent: 'kirocrew-conductor', last_ts: ts, last_turn_ts: ts,
  folder_id: FOLDER, last_message: 'Pushed the fix and opened the pull request.', source_links: [], source_links_total: 0,
  links: [slackLink], code_channel: code,
})
const ROWS = [
  ['Fix the login redirect loop', row('chat-today', 'Fix the login redirect loop', new Date(now.getTime() - 60_000).toISOString(), channel('C0CODE1', 'fix-login-redirect-loop'))],
  ['Bump the Node runtime', row('chat-yday', 'Bump the Node runtime', yesterday, channel('C0CODE2', 'bump-node-runtime-and-lockfile'))],
  ['Draft the release notes', row('chat-arch', 'Draft the release notes', yesterday, channel('C0CODE3', 'release-notes', 'archived'))],
]
const SLOTS = ROWS.map(([, s]) => s)
const FOLDERS = [{ id: FOLDER, name: 'Code channels', order: 0, default_agent: 'kirocrew-conductor' }]

const detail = { running: false, has_more: false, total: 0, queue: [], messages: [] }
const extra = (path, route) => {
  if (path.startsWith('/api/chat/slots/')) return json(route, detail), true
  return false
}

const failures = []
function check(label, cond, detailText = '') {
  if (!cond) failures.push(`${label}${detailText ? ` (${detailText})` : ''}`)
}

/** Every measurement for one row's meta line, taken in the page. */
async function measure(page, title) {
  return page.locator('.session-row').filter({ hasText: title }).evaluate(el => {
    const line = el.querySelector('.session-agent-label')
    const chip = line.querySelector('[data-testid="code-channel-badge-row"]')
    const mark = chip.querySelector('[data-testid="code-channel-slack-mark"]') || chip.querySelector('svg, img')
    const agent = line.querySelector('[data-testid="session-row-agent"]') || line.firstElementChild
    const name = chip.querySelector('[data-testid="code-channel-name"]') || chip.querySelector('span[dir="auto"]')
    const time = line.querySelector('[data-testid="session-row-time"]')
    const box = n => { const r = n.getBoundingClientRect(); return { x: r.x, right: r.right, w: r.width } }
    let floor = 0
    if (name) {
      // Width of "#" plus three characters in the name's own font.
      const probe = name.cloneNode(false)
      probe.textContent = (name.textContent || '').slice(0, 4)
      probe.style.cssText = 'position:absolute;visibility:hidden;max-width:none;min-width:0;width:auto;overflow:visible'
      name.parentElement.appendChild(probe)
      floor = probe.getBoundingClientRect().width
      probe.remove()
    }
    const nb = name ? name.getBoundingClientRect() : null
    const cb = chip.getBoundingClientRect()
    // Right edge of what the chip actually draws on its one visible line.
    let inkRight = cb.x
    for (const n of chip.querySelectorAll('*')) {
      const r = n.getBoundingClientRect()
      if (r.width < 1 || r.top >= cb.top + 8 || getComputedStyle(n).position === 'absolute') continue
      inkRight = Math.max(inkRight, Math.min(r.right, cb.right))
    }
    return {
      line: box(line), chip: box(chip), mark: box(mark), time: box(time), inkRight,
      timeCut: time.scrollWidth > time.clientWidth + 0.5,
      agentCut: agent.scrollWidth > agent.clientWidth + 0.5,
      agentW: agent.getBoundingClientRect().width,
      name: name ? {
        w: nb.width, cut: name.scrollWidth > name.clientWidth + 0.5, floor,
        // An sr-only name, or one the chip wrapped onto its clipped second line.
        visible: getComputedStyle(name).position !== 'absolute',
        // Not on the chip's first line: the chip's fixed height clips it away
        // (wholly or, worse, in half).
        wrapped: nb.top > cb.top + 3,
      } : null,
    }
  })
}

async function checkRow(page, title, where, { nameMustShow = false } = {}) {
  const m = await measure(page, title)
  const at = `${where} / ${title}`
  if (process.env.ROWFIT_DEBUG) console.log(at, JSON.stringify(m))
  if (nameMustShow && m.name && m.name.visible) {
    check(`${at}: channel name stays on the chip at this width`, !m.name.wrapped)
  }
  check(`${at}: Slack mark keeps its full size`, m.mark.w >= 9.5, `width ${m.mark.w.toFixed(1)}`)
  check(`${at}: Slack mark lies inside the chip`, m.mark.x >= m.chip.x - 0.5 && m.mark.right <= m.chip.right + 0.5,
    `mark ${m.mark.x.toFixed(1)}-${m.mark.right.toFixed(1)} chip ${m.chip.x.toFixed(1)}-${m.chip.right.toFixed(1)}`)
  check(`${at}: chip ends before the time`, m.chip.right <= m.time.x + 0.5, `chip right ${m.chip.right.toFixed(1)} time x ${m.time.x.toFixed(1)}`)
  check(`${at}: time is not clipped by its line`, m.time.right <= m.line.right + 0.5 && m.time.x >= m.line.x - 0.5,
    `time ${m.time.x.toFixed(1)}-${m.time.right.toFixed(1)} line ${m.line.x.toFixed(1)}-${m.line.right.toFixed(1)}`)
  check(`${at}: time is not ellipsized`, !m.timeCut)
  check(`${at}: chip has no blank slot`, m.chip.right - m.inkRight <= 6, `blank ${(m.chip.right - m.inkRight).toFixed(1)}px`)
  if (m.name && m.name.visible) {
    // The name may leave the chip only once the agent name has gone entirely:
    // that is the line being too short for the mark, "#abc" and the time at once.
    if (m.name.wrapped) {
      check(`${at}: channel name leaves only after the agent name has gone`, m.agentW < 1, `agent still ${m.agentW.toFixed(1)}px`)
    } else {
      check(`${at}: channel name keeps "#" and a few characters`, m.name.w + 0.5 >= m.name.floor, `shown ${m.name.w.toFixed(1)} need ${m.name.floor.toFixed(1)}`)
      if (m.name.cut) check(`${at}: agent name yields before the channel name`, m.agentCut || m.agentW < 1)
    }
  }
  return m
}

async function shootFolder(page, path) {
  const boxes = await page.locator('.session-row').evaluateAll(els => els.map(el => el.getBoundingClientRect().toJSON()))
  const head = await page.getByText('Code channels', { exact: true }).first().boundingBox()
  const x = Math.max(0, Math.floor(Math.min(...boxes.map(b => b.x)) - 8))
  const y = Math.max(0, Math.floor(Math.min(head ? head.y : Infinity, ...boxes.map(b => b.y)) - 8))
  const right = Math.ceil(Math.max(...boxes.map(b => b.right)) + 8)
  const bottom = Math.ceil(Math.max(...boxes.map(b => b.bottom)) + 8)
  await page.screenshot({ path, clip: { x, y, width: right - x, height: bottom - y } })
}

async function desktop(browser, base, theme, width) {
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 2 })
  const page = await context.newPage()
  logPageProblems(page)
  await stubDashboardApi(page, { slots: SLOTS, folders: FOLDERS, theme, extra })
  await page.addInitScript(w => {
    localStorage.setItem('mc-active-slot', 'chat-today')
    localStorage.setItem('mc-privacy-notice-v1', '1')
    localStorage.setItem('mc-sidebar-pinned', 'true')
    localStorage.setItem('mc-session-stale-collapse-ms', '0')
    localStorage.setItem('mc-sidebar-width', String(w))
  }, width)
  await page.goto(base + '/chat', { waitUntil: 'domcontentloaded' })
  await page.locator('.session-row').first().waitFor({ state: 'visible', timeout: 20_000 })
  await page.waitForTimeout(1500)
  await page.mouse.move(1000, 890)
  for (const [title] of ROWS) await checkRow(page, title, `${width}px ${theme}`, { nameMustShow: width >= 260 })
  await shootFolder(page, join(OUT, `sidebar-${width}px-${theme}.png`))
  await context.close()
}

async function phone(browser, base, theme) {
  const context = await browser.newContext({ viewport: { width: 320, height: 640 }, deviceScaleFactor: 3, hasTouch: true, isMobile: true })
  const page = await context.newPage()
  logPageProblems(page)
  await stubDashboardApi(page, { slots: SLOTS, folders: FOLDERS, theme, extra })
  await page.addInitScript(() => {
    localStorage.setItem('mc-active-slot', 'chat-today')
    localStorage.setItem('mc-privacy-notice-v1', '1')
  })
  await page.goto(base + '/chat', { waitUntil: 'domcontentloaded' })
  await page.getByTestId('mobile-topbar-title').waitFor({ state: 'visible', timeout: 20_000 })
  await page.waitForTimeout(1200)
  await page.getByTestId('mobile-topbar-sessions-toggle').tap()
  await page.locator('.session-row').first().waitFor({ state: 'visible', timeout: 10_000 })
  await page.waitForTimeout(800)
  for (const [title] of ROWS) await checkRow(page, title, `320px phone ${theme}`, { nameMustShow: true })
  await shootFolder(page, join(OUT, `sidebar-320px-phone-${theme}.png`))
  await context.close()
}

async function main() {
  const { srv, base } = await serveDist()
  const browser = await chromium.launch()
  try {
    for (const theme of ['dark', 'light']) {
      for (const width of [260, 180]) await desktop(browser, base, theme, width)
      await phone(browser, base, theme)
    }
  } finally {
    await browser.close()
    srv.close()
  }
  for (const f of failures) console.log(`FAIL ${f}`)
  if (EXPECT_BUG) {
    if (failures.length === 0) {
      console.error('expected the bug to show, but every check passed')
      process.exit(1)
    }
    console.log(`\nOK (expect-bug) — ${failures.length} check(s) failed as expected; screenshots in ${OUT}`)
    return
  }
  if (failures.length) {
    console.error(`\n${failures.length} check(s) failed`)
    process.exit(1)
  }
  console.log(`\nOK — every row fits; screenshots in ${OUT}`)
}

main().catch(err => {
  console.error(err)
  process.exit(1)
})
