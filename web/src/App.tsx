import { useCallback, useEffect, useRef, useState } from "react"

import { readHealth, readModels, send, switchModel, type Health, type RunResult, type Turn } from "./api"
import { Inspector } from "./components/Inspector"
import { ArrowUp, Chevron, Mark, PanelLeft, PanelRight, Plus, Stop } from "./components/icons"
import { Sidebar } from "./components/Sidebar"
import {
  loadSessions,
  loadSystemPrompt,
  newId as sid,
  saveSessions,
  saveSystemPrompt,
  titleFor,
  type Session,
} from "./store"
import { Markdown } from "./markdown"

const SUGGESTIONS = [
  "Explain how expert routing decides which weights to load.",
  "Write a Rust function that memory-maps a safetensors shard.",
  "Why is a 35B MoE slower than its active parameter count suggests?",
  "Compare paged KV cache designs for multi-session serving.",
]

const newId = () => {
  try {
    return crypto.randomUUID()
  } catch {
    return `t${Date.now()}${Math.random().toString(16).slice(2)}`
  }
}

export default function App() {
  const [health, setHealth] = useState<Health | null>(null)
  const [connected, setConnected] = useState(false)
  const [models, setModels] = useState<string[]>([])
  const [model, setModel] = useState("")
  const [turns, setTurns] = useState<Turn[]>([])
  const [draft, setDraft] = useState("")
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState("")
  const [last, setLast] = useState<RunResult | null>(null)
  const [rate, setRate] = useState<number | null>(null)
  const [sessions, setSessions] = useState<Session[]>(() => loadSessions())
  const [currentId, setCurrentId] = useState<string | null>(null)
  const [system, setSystem] = useState(() => loadSystemPrompt())
  const [showRail, setShowRail] = useState(
    () => typeof window === "undefined" || window.innerWidth > 1100,
  )
  // On a narrow screen the inspector stacks under the chat, so start it closed.
  const [showInspector, setShowInspector] = useState(
    () => typeof window === "undefined" || window.innerWidth > 860,
  )
  const [think, setThink] = useState(false)
  const [format, setFormat] = useState("")
  const [keepAlive, setKeepAlive] = useState("")
  const [image, setImage] = useState<string | undefined>()
  const [reasoningEffort, setReasoningEffort] = useState("")
  const abort = useRef<AbortController | null>(null)
  const field = useRef<HTMLTextAreaElement>(null)
  const foot = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const controller = new AbortController()
    ;(async () => {
      try {
        const [found, state] = await Promise.all([
          readModels(controller.signal),
          readHealth(controller.signal),
        ])
        setModels(found)
        setModel((m) => m || found[0] || "")
        setHealth(state)
        setConnected(true)
      } catch (cause) {
        if (!controller.signal.aborted) {
          setConnected(false)
          setError(cause instanceof Error ? cause.message : "The engine did not answer.")
        }
      }
    })()
    return () => controller.abort()
  }, [])

  useEffect(() => {
    if (!connected) return
    let stopped = false
    const timer = window.setInterval(async () => {
      if (document.visibilityState === "hidden") return
      try {
        const state = await readHealth()
        if (!stopped) setHealth(state)
      } catch {
        /* a missed poll is not a disconnection */
      }
    }, 5000)
    return () => {
      stopped = true
      window.clearInterval(timer)
    }
  }, [connected])

  useEffect(() => () => abort.current?.abort(), [])

  // Persist once a turn has settled rather than on every streamed token.
  useEffect(() => {
    if (!turns.length || turns.some((t) => t.role === "assistant" && !t.fixed)) return
    const id = currentId ?? sid()
    if (!currentId) setCurrentId(id)
    setSessions((current) => {
      const now = Date.now()
      const existing = current.find((c) => c.id === id)
      const next: Session = {
        id,
        title: titleFor(turns),
        turns,
        created: existing?.created ?? now,
        updated: now,
      }
      const merged = [next, ...current.filter((c) => c.id !== id)].sort(
        (a, b) => b.updated - a.updated,
      )
      return saveSessions(merged)
    })
  }, [turns, currentId])

  useEffect(() => saveSystemPrompt(system), [system])
  useEffect(() => {
    foot.current?.scrollIntoView({ behavior: "smooth", block: "end" })
  }, [turns])

  const submit = useCallback(async () => {
    const prompt = draft.trim()
    if (!prompt || busy) return

    const question: Turn = { id: newId(), role: "user", text: prompt, fixed: true }
    const answer: Turn = { id: newId(), role: "assistant", text: "", fixed: false }
    const history = [...turns, question]

    setTurns([...history, answer])
    setDraft("")
    setImage(undefined)
    setError("")
    setBusy(true)
    setRate(null)
    setLast(null)
    if (field.current) field.current.style.height = "auto"

    const controller = new AbortController()
    abort.current = controller
    const started = performance.now()
    const patch = (change: (t: Turn) => Turn) =>
      setTurns((current) => current.map((t) => (t.id === answer.id ? change(t) : t)))

    try {
      const result = await send({
        model,
        history,
        system,
        temperature: 0.7,
        maxTokens: 4096,
        reasoning: think,
        format: format || undefined,
        keepAlive: keepAlive || undefined,
        reasoningEffort: reasoningEffort || undefined,
        image,
        signal: controller.signal,
        onText: (chunk) => patch((t) => ({ ...t, text: t.text + chunk })),
        onThinking: (chunk) => patch((t) => ({ ...t, thinking: (t.thinking ?? "") + chunk })),
      })
      setLast(result)
      const seconds = (performance.now() - started) / 1000
      if (result.usage && seconds > 0) setRate(result.usage.completion_tokens / seconds)
    } catch (cause) {
      const aborted = cause instanceof DOMException && cause.name === "AbortError"
      if (!aborted) setError(cause instanceof Error ? cause.message : "The request failed.")
      else patch((t) => ({ ...t, text: t.text || "Stopped." }))
    } finally {
      patch((t) => ({ ...t, fixed: true }))
      setBusy(false)
      abort.current = null
    }
  }, [draft, busy, turns, model, system, think, format, keepAlive, reasoningEffort, image])

  const startNew = () => {
    abort.current?.abort()
    setTurns([])
    setCurrentId(null)
    setError("")
    setLast(null)
    setRate(null)
    field.current?.focus()
  }

  const openSession = (id: string) => {
    const found = sessions.find((s) => s.id === id)
    if (!found) return
    abort.current?.abort()
    setTurns(found.turns)
    setCurrentId(id)
    setError("")
    setLast(null)
    setRate(null)
  }

  const deleteSession = (id: string) => {
    setSessions((current) => saveSessions(current.filter((s) => s.id !== id)))
    if (id === currentId) startNew()
  }

  const canSend = draft.trim().length > 0 && !busy

  return (
    <div className="app" data-inspector={showInspector} data-rail={showRail}>
      <header className="topbar">
        <button
          type="button"
          className="icon-btn"
          aria-pressed={showRail}
          title={showRail ? "Hide chats" : "Show chats"}
          onClick={() => setShowRail((v) => !v)}
        >
          <PanelLeft />
        </button>
        <div className="wordmark">
          <Mark />
          <span>Strata</span>
        </div>
        <select
          className="model-pick"
          value={model}
          onChange={(e) => {
            const next = e.target.value
            setModel(next)
            // The compute process serves one set of weights, so picking a name
            // has to load it. Without this the picker only relabelled the next
            // turn and the model never changed.
            switchModel(next).catch((err) => setError(String(err)))
          }}
          aria-label="Model"
        >
          {models.length ? models.map((id) => <option key={id}>{id}</option>) : <option>{model || "—"}</option>}
        </select>

        <div className="topbar-right">
          <span className="status" data-state={connected ? "up" : error ? "down" : "wait"}>
            <i />
            {connected ? "Connected" : error ? "No engine" : "Connecting…"}
          </span>
          {turns.length ? (
            <button type="button" className="icon-btn" title="New chat" onClick={startNew}>
              <Plus />
            </button>
          ) : null}
          <button
            type="button"
            className="icon-btn"
            aria-pressed={showInspector}
            title={showInspector ? "Hide details" : "Show details"}
            onClick={() => setShowInspector((v) => !v)}
          >
            <PanelRight />
          </button>
        </div>
      </header>

      <Sidebar
        sessions={sessions}
        currentId={currentId}
        onSelect={openSession}
        onNew={startNew}
        onDelete={deleteSession}
      />

      <main className="main">
        <div className="thread">
          {turns.length === 0 ? (
            <div className="blank">
              <h1>What are we working on?</h1>
              <div className="suggests">
                {SUGGESTIONS.map((s) => (
                  <button
                    key={s}
                    type="button"
                    className="suggest"
                    onClick={() => {
                      setDraft(s)
                      field.current?.focus()
                    }}
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="thread-inner">
              {turns.map((turn) =>
                turn.role === "user" ? (
                  <div className="turn-user" key={turn.id}>
                    <div>{turn.text}</div>
                  </div>
                ) : (
                  <div className="turn-model" key={turn.id}>
                    {turn.thinking ? (
                      <details className="reasoning" open={!turn.text}>
                        <summary>
                          <Chevron /> Thinking
                        </summary>
                        <div className="reasoning-body">{turn.thinking}</div>
                      </details>
                    ) : null}
                    <Markdown source={turn.text} caret={!turn.fixed} />
                    {turn.fixed && last?.usage ? (
                      <div className="turn-meta">
                        {rate != null ? <span>{rate.toFixed(1)} tok/s</span> : null}
                        <span>
                          {last.usage.prompt_tokens} in · {last.usage.completion_tokens} out
                        </span>
                        {last.stopReason === "length" ? <span>hit token limit</span> : null}
                      </div>
                    ) : null}
                  </div>
                ),
              )}
              <div ref={foot} />
            </div>
          )}
        </div>

        <div className="composer-wrap">
          <div className="composer-inner">
            {error ? (
              <div className="alert" role="alert">
                {error}
              </div>
            ) : null}
            <div className="composer-options">
              <label title="Attach an image for vision models">
                <input
                  type="file"
                  accept="image/*"
                  onChange={(e) => {
                    const file = e.target.files?.[0]
                    if (!file) { setImage(undefined); return }
                    const reader = new FileReader()
                    reader.onload = () => setImage(String(reader.result))
                    reader.readAsDataURL(file)
                  }}
                />
              </label>
              <label title="Let the model reason before answering">
                <input type="checkbox" checked={think} onChange={(e) => setThink(e.target.checked)} />
                Think
              </label>
              <select
                aria-label="Reasoning effort"
                value={reasoningEffort}
                onChange={(e) => setReasoningEffort(e.target.value)}
                title="Reasoning effort"
              >
                <option value="">effort: default</option>
                <option value="low">low</option>
                <option value="medium">medium</option>
                <option value="xhigh">xhigh</option>
              </select>
              <select
                aria-label="Response format"
                value={format}
                onChange={(e) => setFormat(e.target.value)}
                title="Response format"
              >
                <option value="">text</option>
                <option value="json">JSON</option>
              </select>
              <select
                aria-label="Keep alive"
                value={keepAlive}
                onChange={(e) => setKeepAlive(e.target.value)}
                title="How long the model stays resident"
              >
                <option value="">keep: default</option>
                <option value="5m">5m</option>
                <option value="30m">30m</option>
                <option value="-1">forever</option>
                <option value="0">unload after</option>
              </select>
            </div>
            <div className="composer">
              <textarea
                ref={field}
                rows={1}
                value={draft}
                placeholder="Message Strata…"
                onChange={(e) => {
                  setDraft(e.target.value)
                  e.target.style.height = "auto"
                  e.target.style.height = `${Math.min(e.target.scrollHeight, 200)}px`
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                    e.preventDefault()
                    if (canSend) void submit()
                  }
                }}
              />
              {busy ? (
                <button
                  type="button"
                  className="send"
                  title="Stop"
                  onClick={() => abort.current?.abort()}
                >
                  <Stop />
                </button>
              ) : (
                <button
                  type="button"
                  className="send"
                  disabled={!canSend}
                  title="Send"
                  onClick={() => void submit()}
                >
                  <ArrowUp />
                </button>
              )}
            </div>
            <p className="composer-note">
              Strata runs the model directly. The expert map is simulated — the router is not
              observable yet.
            </p>
          </div>
        </div>
      </main>

      <Inspector health={health} connected={connected} system={system} onSystem={setSystem} />
    </div>
  )
}
