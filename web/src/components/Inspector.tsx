import { useEffect, useMemo, useRef, useState } from "react"

import {
  readCortex,
  readProfile,
  apiKey,
  setApiKey,
  setModelLoaded,
  type Compaction,
  type Cortex,
  type Health,
  type ModelState,
  type ProfileTurn,
} from "../api"
import { PRESETS } from "../store"

type Pane = "session" | "experts" | "profiling" | "settings"

/** Tier → colour. Green is resident on the GPU, grey is host RAM, dark is cold. */
const TIER: Record<number, [number, number, number]> = {
  2: [85, 168, 119],
  1: [122, 122, 122],
  0: [58, 58, 58],
}

const PHASES = [
  { key: "expert_wait_s", name: "I/O wait" },
  { key: "expert_matmul_s", name: "Expert matmul" },
  { key: "attention_s", name: "Attention" },
  { key: "lm_head_s", name: "LM head" },
  { key: "other_s", name: "Unattributed" },
] as const

const gb = (v: number | undefined) => (v == null ? "—" : `${v.toFixed(1)} GB`)
const tokens = (v: number | undefined) => (v == null ? "—" : v.toLocaleString())
const secs = (v: number) => (v >= 10 ? v.toFixed(1) : v.toFixed(2)) + "s"

export function Inspector({
  health,
  connected,
  system,
  onSystem,
}: {
  health: Health | null
  connected: boolean
  system: string
  onSystem: (value: string) => void
}) {
  const [pane, setPane] = useState<Pane>("session")
  return (
    <aside className="inspector">
      <div className="inspector-tabs" role="tablist" aria-label="Inspector">
        {(["session", "experts", "profiling", "settings"] as Pane[]).map((id) => (
          <button
            key={id}
            role="tab"
            aria-selected={pane === id}
            className="inspector-tab"
            onClick={() => setPane(id)}
          >
            {id === "session"
              ? "Session"
              : id === "experts"
                ? "Experts"
                : id === "profiling"
                  ? "Profiling"
                  : "Setup"}
          </button>
        ))}
      </div>
      <div className="inspector-body">
        {pane === "session" ? (
          <Session health={health} />
        ) : pane === "experts" ? (
          <Experts connected={connected} simulated={!!health?.simulated} />
        ) : pane === "profiling" ? (
          <Profiling connected={connected} />
        ) : (
          <Setup system={system} onSystem={onSystem} />
        )}
      </div>
    </aside>
  )
}

function Session({ health }: { health: Health | null }) {
  const s = health?.scheduler
  const hw = health?.hwinfo
  const t = health?.tiers
  const c = health?.compaction
  const m = health?.model
  const slots = s?.capacity ?? health?.kv_slots ?? 4
  const lost = s ? s.rejected + s.timed_out + s.cancelled : 0

  if (!health) return <p className="empty-note">Not connected to the engine.</p>

  const total = t ? t.vram + t.ram + t.disk : 0

  return (
    <>
      <section className="group">
        <h3 className="group-title">Scheduler</h3>
        <div className="slots">
          {Array.from({ length: slots }, (_, i) => {
            const busy = i < (s?.active ?? 0)
            const queued = !busy && i < (s?.active ?? 0) + (s?.queued ?? 0)
            return (
              <div key={i} className="slot" data-on={busy ? "busy" : queued ? "queued" : "free"}>
                {i + 1}
              </div>
            )
          })}
        </div>
        <dl style={{ margin: "12px 0 0" }}>
          <div className="row">
            <dt>Completed</dt>
            <dd>{s?.completed ?? 0}</dd>
          </div>
          <div className="row">
            <dt>Queued</dt>
            <dd>
              {s?.queued ?? 0} <em>/ {s?.max_queue ?? 0}</em>
            </dd>
          </div>
          <div className="row">
            <dt>Cancelled or refused</dt>
            <dd>{lost}</dd>
          </div>
        </dl>
      </section>

      {m?.owned ? <Model model={m} vramGb={hw?.vram_total_gb} /> : null}

      {c ? <Context compaction={c} /> : null}

      {t ? (
        <section className="group">
          <h3 className="group-title">Weight placement</h3>
          <div className="meter" aria-hidden="true">
            <i style={{ width: `${(100 * t.vram) / (total || 1)}%`, background: "#55a877" }} />
            <i style={{ width: `${(100 * t.ram) / (total || 1)}%`, background: "#7a7a7a" }} />
            <i style={{ width: `${(100 * t.disk) / (total || 1)}%`, background: "#3a3a3a" }} />
          </div>
          <dl style={{ margin: "10px 0 0" }}>
            <div className="row">
              <dt>VRAM</dt>
              <dd>
                {t.vram.toLocaleString()} <em>· {gb(t.vram_gb)}</em>
              </dd>
            </div>
            <div className="row">
              <dt>RAM</dt>
              <dd>
                {t.ram.toLocaleString()} <em>· {gb(t.ram_gb)}</em>
              </dd>
            </div>
            <div className="row">
              <dt>Disk</dt>
              <dd>{t.disk.toLocaleString()}</dd>
            </div>
          </dl>
        </section>
      ) : null}

      {hw ? (
        <section className="group">
          <h3 className="group-title">Hardware</h3>
          <dl>
            <div className="row">
              <dt>GPU</dt>
              <dd>{hw.gpu}</dd>
            </div>
            <div className="row">
              <dt>VRAM</dt>
              <dd>{gb(hw.vram_total_gb)}</dd>
            </div>
            <div className="row">
              <dt>CPU</dt>
              <dd>
                {hw.cpu} <em>· {hw.cores} threads</em>
              </dd>
            </div>
            <div className="row">
              <dt>Memory</dt>
              <dd>
                {gb(hw.ram_total_gb)} <em>· {hw.ram_avail_gb.toFixed(1)} free</em>
              </dd>
            </div>
          </dl>
        </section>
      ) : null}
    </>
  )
}

/**
 * The API key, for an engine started with one.
 *
 * A server on loopback needs none, so this stays empty and unused on a normal
 * local setup. It lives in this browser's storage rather than in the bundle,
 * because the same bundle is served to everyone the engine is shared with.
 */
function ApiKey() {
  const [value, setValue] = useState(apiKey())
  const [saved, setSaved] = useState(false)

  const save = () => {
    setApiKey(value)
    setSaved(true)
    window.setTimeout(() => setSaved(false), 1500)
  }

  return (
    <section className="group">
      <h3 className="group-title">API key</h3>
      <div className="model-row">
        <input
          className="key-box"
          type="password"
          value={value}
          placeholder="only if the engine was started with one"
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") save()
          }}
          spellCheck={false}
        />
        <button type="button" className="model-action" onClick={save}>
          {saved ? "Saved" : "Save"}
        </button>
      </div>
      <p className="model-note">
        Sent as a bearer token with every request. An engine listening only on
        localhost does not use one.
      </p>
    </section>
  )
}

/**
 * Whether the weights are resident, and the button to change that.
 *
 * The model is the machine's whole GPU while it is loaded - VRAM nothing else
 * can touch - so there has to be a way to hand it back without stopping the
 * server. Hidden entirely under `--upstream`, where the process is not ours.
 */
function Model({ model, vramGb }: { model: ModelState; vramGb?: number }) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  // Health is polled every few seconds, so the request returning is not the
  // same as the panel knowing. Show what was asked for until a poll actually
  // reports it, otherwise the button snaps back to its old label and reads as
  // though the click did nothing.
  const [pending, setPending] = useState<boolean | null>(null)
  const loaded = pending ?? model.loaded
  useEffect(() => {
    if (pending !== null && model.loaded === pending) setPending(null)
  }, [model.loaded, pending])

  const toggle = async () => {
    const next = !model.loaded
    setBusy(true)
    setError(null)
    setPending(next)
    try {
      await setModelLoaded(next)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setPending(null)
    } finally {
      setBusy(false)
    }
  }

  return (
    <section className="group">
      <h3 className="group-title">Model</h3>
      <div className="model-row">
        <span className="model-name" title={model.name}>
          {model.name}
        </span>
        <button
          type="button"
          className="model-action"
          data-state={loaded ? "loaded" : "unloaded"}
          disabled={busy}
          onClick={toggle}
        >
          {busy ? (loaded ? "Loading…" : "Unloading…") : loaded ? "Unload" : "Load"}
        </button>
      </div>
      {error ? (
        <p className="model-note" data-tone="bad">
          {error}
        </p>
      ) : (
        <p className="model-note">
          {loaded
            ? `Resident${vramGb ? ` in ${vramGb.toFixed(0)} GB of VRAM` : ""}. Unloading stops the compute process and gives the memory back; turns already running finish first.`
            : "Unloaded — the GPU is free. Loading reads the weights back and takes a few seconds."}
        </p>
      )}
    </section>
  )
}

/**
 * How full the model's context window is, and what compaction has had to do
 * about it. `summarised` is the count that costs a forward pass; `reused` is
 * the count that costs nothing, so the ratio between them is the honest
 * measure of whether compaction is in the way.
 */
function Context({ compaction: c }: { compaction: Compaction }) {
  const budget = c.budget || c.ctx
  const fill = Math.min(100, (100 * c.prompt_tokens) / (budget || 1))
  // Compaction triggers at 80% of the budget; warn as that approaches.
  const colour = fill >= 80 ? "#c9a227" : "#55a877"

  return (
    <section className="group">
      <h3 className="group-title">Context</h3>
      <div className="meter" aria-hidden="true">
        <i style={{ width: `${fill}%`, background: colour }} />
      </div>
      <dl style={{ margin: "10px 0 0" }}>
        <div className="row">
          <dt>In the window</dt>
          <dd>
            {tokens(c.prompt_tokens)} <em>/ {tokens(budget)} tokens</em>
          </dd>
        </div>
        <div className="row">
          <dt>Loaded context</dt>
          <dd>{tokens(c.ctx)}</dd>
        </div>
        {c.enabled ? (
          <>
            <div className="row">
              <dt>Compactions</dt>
              <dd>
                {c.summarised} <em>· {c.reused} reused</em>
              </dd>
            </div>
            <div className="row">
              <dt>Tokens reclaimed</dt>
              <dd>{tokens(c.tokens_reclaimed)}</dd>
            </div>
            {c.last ? (
              <div className="row">
                <dt>Last</dt>
                <dd>
                  {tokens(c.last.before_tokens)} → {tokens(c.last.after_tokens)}{" "}
                  <em>· {c.last.summarised_messages} messages summarised</em>
                </dd>
              </div>
            ) : null}
          </>
        ) : (
          <div className="row">
            <dt>Compaction</dt>
            <dd>
              off <em>· a longer conversation will overflow</em>
            </dd>
          </div>
        )}
      </dl>
    </section>
  )
}

function Experts({ connected, simulated }: { connected: boolean; simulated: boolean }) {
  const canvas = useRef<HTMLCanvasElement>(null)
  const pulses = useRef<Float32Array | null>(null)
  const frame = useRef(0)
  const lastSeq = useRef(-1)
  const [sheet, setSheet] = useState<Cortex | null>(null)

  useEffect(() => {
    if (!connected) return
    let stopped = false
    const poll = async () => {
      try {
        const next = await readCortex()
        if (stopped || !next.rows) return
        setSheet(next)
        if (next.seq !== lastSeq.current && next.hits) {
          lastSeq.current = next.seq
          const n = next.rows * next.cols
          if (!pulses.current || pulses.current.length !== n) pulses.current = new Float32Array(n)
          for (let i = 0; i < n; i++) {
            const byte = parseInt(next.hits.substr((i >> 3) * 2, 2), 16) || 0
            if (byte & (1 << (i & 7))) pulses.current[i] = 1
          }
        }
      } catch {
        /* keep the last frame */
      }
    }
    void poll()
    const timer = window.setInterval(() => {
      if (document.visibilityState !== "hidden") void poll()
    }, 1500)
    return () => {
      stopped = true
      window.clearInterval(timer)
    }
  }, [connected])

  // Decode once per poll rather than once per frame.
  const cells = useMemo(() => {
    if (!sheet) return null
    const n = sheet.rows * sheet.cols
    const out = new Uint8Array(n)
    for (let i = 0; i < n; i++) out[i] = parseInt(sheet.map.substr(i * 2, 2), 16) || 0
    return out
  }, [sheet])

  useEffect(() => {
    const el = canvas.current
    if (!el || !sheet || !cells) return
    const ctx = el.getContext("2d")
    if (!ctx) return
    // One device pixel per expert; CSS scales it up with smoothing off.
    el.width = sheet.cols
    el.height = sheet.rows
    const image = ctx.createImageData(sheet.cols, sheet.rows)

    const draw = () => {
      const pulse = pulses.current
      let alive = false
      for (let i = 0; i < cells.length; i++) {
        const byte = cells[i]
        const [r, g, b] = TIER[byte >> 6] ?? TIER[0]
        const level = 0.28 + ((byte & 63) / 63) * 0.72
        let cr = r * level
        let cg = g * level
        let cb = b * level
        const hot = pulse ? pulse[i] : 0
        if (hot > 0.02) {
          alive = true
          cr += (236 - cr) * hot
          cg += (236 - cg) * hot
          cb += (236 - cb) * hot
          pulse![i] = hot * 0.88
        }
        image.data[i * 4] = cr
        image.data[i * 4 + 1] = cg
        image.data[i * 4 + 2] = cb
        image.data[i * 4 + 3] = 255
      }
      ctx.putImageData(image, 0, 0)
      frame.current = alive ? requestAnimationFrame(draw) : 0
    }
    frame.current = requestAnimationFrame(draw)
    return () => {
      if (frame.current) cancelAnimationFrame(frame.current)
    }
  }, [sheet, cells])

  if (!connected) return <p className="empty-note">Not connected to the engine.</p>
  if (!sheet) return <p className="empty-note">Loading expert map…</p>

  return (
    <section className="group">
      <h3 className="group-title">
        Expert map {simulated ? <span className="tag" style={{ marginLeft: 6 }}>simulated</span> : null}
      </h3>
      <div className="grid-frame">
        <canvas ref={canvas} style={{ aspectRatio: `${sheet.cols} / ${sheet.rows}` }} />
      </div>
      <div className="legend">
        <span>
          <i style={{ background: "#55a877" }} /> VRAM
        </span>
        <span>
          <i style={{ background: "#7a7a7a" }} /> RAM
        </span>
        <span>
          <i style={{ background: "#3a3a3a" }} /> Disk
        </span>
        <span>
          <i style={{ background: "#ececec" }} /> Routed
        </span>
      </div>
      <p className="note">
        {sheet.rows} layers × {sheet.cols} experts. Brightness is routing frequency.
        {simulated
          ? " The engine reports this map as synthetic — there is no router to observe while generation is proxied."
          : ""}
      </p>
    </section>
  )
}

function Profiling({ connected }: { connected: boolean }) {
  const [rows, setRows] = useState<ProfileTurn[]>([])

  useEffect(() => {
    if (!connected) return
    let stopped = false
    const poll = async () => {
      try {
        const turns = await readProfile()
        if (!stopped) setRows(turns)
      } catch {
        /* keep the last snapshot */
      }
    }
    void poll()
    const timer = window.setInterval(() => {
      if (document.visibilityState !== "hidden") void poll()
    }, 2500)
    return () => {
      stopped = true
      window.clearInterval(timer)
    }
  }, [connected])

  if (!connected) return <p className="empty-note">Not connected to the engine.</p>
  if (!rows.length) return <p className="empty-note">No runs recorded yet.</p>

  const wall = rows.reduce((s, r) => s + r.wall_s, 0)
  const other = rows.reduce(
    (s, r) =>
      s + Math.max(0, r.wall_s - r.expert_wait_s - r.expert_matmul_s - r.attention_s - r.lm_head_s),
    0,
  )
  const value = (key: string) =>
    key === "other_s" ? other : rows.reduce((s, r) => s + ((r as never)[key] as number), 0)
  const latest = rows[rows.length - 1]
  const rate = latest.wall_s > 0 ? latest.completion_tokens / latest.wall_s : 0

  return (
    <>
      <section className="group">
        <h3 className="group-title">Last run</h3>
        <dl>
          <div className="row">
            <dt>Throughput</dt>
            <dd>{rate.toFixed(1)} tok/s</dd>
          </div>
          <div className="row">
            <dt>Wall time</dt>
            <dd>{secs(latest.wall_s)}</dd>
          </div>
          <div className="row">
            <dt>Tokens</dt>
            <dd>
              {latest.prompt_tokens} <em>in ·</em> {latest.completion_tokens} <em>out</em>
            </dd>
          </div>
        </dl>
      </section>

      <section className="group">
        <h3 className="group-title">Time by phase</h3>
        <div className="bars">
          {PHASES.map((p) => {
            const v = value(p.key)
            const share = wall > 0 ? v / wall : 0
            return (
              <div className="bar-row" key={p.key}>
                <span>{p.name}</span>
                <span className="bar-val">{(100 * share).toFixed(0)}%</span>
                <span className="bar-track" style={{ gridColumn: "1 / -1" }}>
                  <i style={{ width: `${100 * share}%` }} />
                </span>
              </div>
            )
          })}
        </div>
        {wall > 0 && other / wall > 0.98 ? (
          <p className="note">
            Every phase timer reads zero, so all of it lands in Unattributed. The engine reports
            per-phase timings only once its own kernels run the forward pass.
          </p>
        ) : null}
      </section>

      <section className="group">
        <h3 className="group-title">Recent runs</h3>
        <dl>
          {[...rows]
            .reverse()
            .slice(0, 10)
            .map((r, i) => (
              <div className="row" key={rows.length - i}>
                <dt>{String(rows.length - i).padStart(2, "0")}</dt>
                <dd>
                  {(r.wall_s > 0 ? r.completion_tokens / r.wall_s : 0).toFixed(1)} tok/s{" "}
                  <em>· {secs(r.wall_s)}</em>
                </dd>
              </div>
            ))}
        </dl>
      </section>
    </>
  )
}

function Setup({ system, onSystem }: { system: string; onSystem: (v: string) => void }) {
  return (
    <>
      <ApiKey />
      <section className="group">
      <h3 className="group-title">System prompt</h3>
      <textarea
        className="system-box"
        value={system}
        placeholder="Standing instructions sent with every message — your stack, conventions, how you want answers shaped."
        onChange={(e) => onSystem(e.target.value)}
        spellCheck={false}
      />
      <div className="presets">
        {PRESETS.map((p) => (
          <button key={p.name} type="button" className="preset" onClick={() => onSystem(p.text)}>
            {p.name}
          </button>
        ))}
        {system ? (
          <button type="button" className="preset" onClick={() => onSystem("")}>
            Clear
          </button>
        ) : null}
      </div>
      <p className="note">
        Saved in this browser and prepended to every request. It counts against the context window,
        so keep it short.
      </p>
      </section>
    </>
  )
}
