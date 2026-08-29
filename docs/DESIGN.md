# Design

The console's visual system, recorded from the built interface in `web/`.

## World

The conventional developer-chat canon, played straight — Codex and ChatGPT are the
named references and their craft level is the bar. Neutral near-black, system font
stack, hairline borders, no second accent, no metaphor layer.

This replaced an earlier metaphor-driven design. The reason was structural, not
cosmetic: that version arranged instruments *around* the conversation — a sidebar
of readouts plus a tab bar of views — and that arrangement is what made the tool
read as someone else's, whatever colours it wore. The fix was to make the
conversation the page and collapse everything else behind a toggle.

**Governing rule: chrome earns its place or hides.** Nothing permanent sits around
the conversation except a 48px bar. Both side columns are exactly 0px when closed,
so the chat re-centres rather than shifting.

## Colour

Defined once on `:root` in `web/src/styles.css`.

| Token | Value | Role |
|---|---|---|
| `--bg` | `#1b1b1b` | page ground |
| `--bg-elev` | `#222` | composer, hover fills |
| `--panel` | `#161616` | inspector and rail, recessed |
| `--user` | `#2f2f2f` | the user's message bubble |
| `--fg` | `#ececec` | primary text; also the send button's fill |
| `--fg-dim` | `#a6a6a6` | secondary text, labels |
| `--fg-faint` | `#737373` | metadata, placeholders, disabled |
| `--line` | `rgba(255,255,255,.09)` | visible hairline |
| `--line-soft` | `rgba(255,255,255,.055)` | structural divider |
| `--good` / `--warn` / `--bad` | `#55a877` / `#c9974a` / `#d2564b` | status only — never decoration |

Strategy is **restrained**: neutrals carry the whole surface and the only saturated
colour is state. The primary action inverts (near-white fill, dark glyph) rather
than taking an accent hue. `color-scheme: dark` is declared, so browser-drawn
surfaces match rather than flashing white.

## Type

System stack — `ui-sans-serif, -apple-system, "Segoe UI", Roboto…` — at 15px/1.6.
No webfont is loaded, so the console has no external font dependency and no
flash-of-unstyled-text.

Monospace (`ui-monospace, "Cascadia Mono", Consolas`) is reserved strictly for
measurement: throughput, token counts, expert counts, sizes. Every numeric readout
carries `font-variant-numeric: tabular-nums` so figures do not jitter as they
update.

## Structure

`.app` is a three-column grid — rail, conversation, inspector — over two rows, the
first being the 48px bar. Both side tracks animate between `0` and their width
(`--rail: 248px`, `--inspector: 320px`) on the same 220ms ease, so opening either
one never moves the other. The centre track is `minmax(0, 1fr)`, so content can
never force the page wider than the viewport.

- **Top bar** — mark and name, model picker, connection status, new chat, rail and
  inspector toggles. Nothing else ever lives here.
- **Rail** — saved conversations, newest first, grouped as Today / Yesterday /
  Previous 7 days / older. Sessions live in `localStorage` under one key; there is
  no server-side history.
- **Conversation** — a 46rem centred column. The user's turn is a right-aligned
  bubble; the assistant's turn has **no container at all**, because the page is its
  container. Reasoning collapses into a `<details>` above the answer, open while
  the answer is still empty. An empty thread offers four suggestion cards.
- **Composer** — a 24px-radius field pinned to the bottom of the column, with a
  round send button that becomes a stop square while streaming.
- **Inspector** — four panes, `Session` / `Experts` / `Profiling` / `Setup`,
  holding everything that used to compete with the conversation.

## Motion

One authored moment: **the caret** — a 7px block that trails the stream and blinks
on a 2-step timing function, stopping the instant the turn settles. Everything else
is state feedback: 120–130ms colour transitions, and the 220ms grid-column ease on
the two side columns.

No layout properties are animated. The whole system collapses under
`prefers-reduced-motion: reduce`.

## Instruments

- **Scheduler slots** — one box per capacity slot, green when busy, amber when
  queued, hairline when free.
- **Weight placement** — a single stacked meter over a definition list of counts
  and sizes: how many of the model's routed experts are on the card, how many are
  in host memory, and what each is costing.
- **Expert map** — a canvas at exactly one device pixel per expert, upscaled by CSS
  with smoothing off. Green is GPU-resident, grey is host RAM, dark is cold;
  brightness is routing frequency and a white flash marks experts routed on the
  last token. Decoding happens once per poll, and the animation loop idles when
  nothing is firing.
- **Profiling** — phase shares as thin bars, plus the recent-runs list.
- **Setup** — the API key, then the system prompt and its presets. The key is
  held in this browser and sent as a bearer token on every request; the field is
  a password input, so a stored key is never rendered in the clear.

## Honesty rules

The engine does not own its own forward pass yet, and the interface must not paper
over that.

- A persistent line under the composer states that Strata runs the model directly
  and that the expert map is simulated because the router is not observable yet.
- The expert map carries a `simulated` tag whenever `/health` reports it, with a
  sentence explaining there is no router to observe.
- When every profiling phase timer reads zero, the panel says so instead of drawing
  a convincing breakdown of nothing.

Each of these tracks a real server field. None of them is decoration, and none of
them should be removed before the thing it discloses is fixed.

## Browser surfaces

Selection, scrollbars (WebKit and Firefox), caret colour and focus rings are all
themed. Focus is a 2px `#7c7c7c` ring at 2px offset, never removed.

## Responsive

One breakpoint at 860px. Below it the grid collapses to a single column: the rail
is removed outright, and the inspector moves from a right column to a bottom row at
46vh, defaulting to closed so the conversation keeps the screen. Suggestion cards
drop from two columns to one at 640px. Verified at 390px and 820px with no
horizontal overflow.
