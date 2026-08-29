/** Client for the Strata engine's HTTP surface (see src/server.rs). */

export type Role = "user" | "assistant"

export interface Turn {
  id: string
  role: Role
  /** Prose the model committed to. */
  text: string
  /** Reasoning stream, kept apart so it is never fed back as history. */
  thinking?: string
  /** False until the turn stops streaming — drives the caret. */
  fixed?: boolean
}

export interface Scheduler {
  active: number
  capacity: number
  queued: number
  max_queue: number
  queue_timeout_seconds: number
  admitted: number
  completed: number
  rejected: number
  timed_out: number
  cancelled: number
}

export interface Tiers {
  vram: number
  ram: number
  disk: number
  vram_gb: number
  ram_gb: number
}

export interface Hardware {
  cores: number
  ram_total_gb: number
  ram_avail_gb: number
  gpus: number
  vram_total_gb: number
  cpu: string
  gpu: string
}

/** What context compaction has done, from GET /compact (also inlined in /health). */
export interface Compaction {
  enabled: boolean
  /** Context the compute process was loaded with. */
  ctx: number
  /** Size of the most recent prompt, after any compaction. */
  prompt_tokens: number
  /** Tokens the conversation may occupy once the reply is reserved for. */
  budget: number
  /** Summaries actually generated - each one costs a forward pass. */
  summarised: number
  /** Turns that reused an existing summary instead of writing a new one. */
  reused: number
  tokens_reclaimed: number
  last: {
    before_tokens: number
    after_tokens: number
    summarised_messages: number
    kept_messages: number
    reused: boolean
    took_ms: number
  } | null
}

/** Whether the weights are resident, and whether they are ours to unload. */
export interface ModelState {
  /** False under --upstream: the process belongs to someone else. */
  owned: boolean
  loaded: boolean
  name: string
}

export interface Health {
  status: string
  simulated?: boolean
  scheduler?: Scheduler
  kv_slots?: number
  model?: ModelState
  compaction?: Compaction
  tiers?: Tiers
  hwinfo?: Hardware
}

export interface ProfileTurn {
  wall_s: number
  prompt_tokens: number
  completion_tokens: number
  expert_disk_s: number
  expert_wait_s: number
  expert_matmul_s: number
  attention_s: number
  lm_head_s: number
  forwards: number
}

export interface Cortex {
  rows: number
  cols: number
  /** One hex byte per expert: tier<<6 | density(0..63). */
  map: string
  /** Bitset, one bit per expert, set when routed on the last token. */
  hits: string
  seq: number
  simulated?: boolean
}

export interface Usage {
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
}

export interface RunResult {
  usage: Usage | null
  stopReason: string | null
  queueWaitMs: number | null
}

/** The engine serves the UI, so it is its own origin unless told otherwise. */
export const engineOrigin = () => window.location.origin

const api = (path: string) => `${engineOrigin()}${path}`

/**
 * The API key, when the server was started with one.
 *
 * Held per browser rather than baked into the bundle, because the same build is
 * served to everyone the engine is shared with. A server on loopback has no key
 * and none of this applies.
 */
const KEY_STORAGE = "strata.apiKey"

export function apiKey(): string {
  try {
    return localStorage.getItem(KEY_STORAGE) ?? ""
  } catch {
    return ""
  }
}

export function setApiKey(key: string) {
  try {
    if (key.trim()) localStorage.setItem(KEY_STORAGE, key.trim())
    else localStorage.removeItem(KEY_STORAGE)
  } catch {
    /* private browsing; the key simply will not persist */
  }
}

/** Headers every call carries. */
function auth(extra?: Record<string, string>): Record<string, string> {
  const key = apiKey()
  return { ...(extra ?? {}), ...(key ? { Authorization: `Bearer ${key}` } : {}) }
}

async function failure(response: Response) {
  if (response.status === 401) {
    return "This engine requires an API key. Add it under Setup in the inspector."
  }
  const fallback = `${response.status} ${response.statusText}`
  try {
    const body = await response.json()
    return body?.error?.message || body?.error || fallback
  } catch {
    return fallback
  }
}

export async function readHealth(signal?: AbortSignal): Promise<Health> {
  const response = await fetch(api("/health"), { signal, headers: auth() })
  if (!response.ok) throw new Error(await failure(response))
  return response.json()
}

export async function readProfile(signal?: AbortSignal): Promise<ProfileTurn[]> {
  const response = await fetch(api("/profile"), { signal, headers: auth() })
  if (!response.ok) throw new Error(await failure(response))
  const body = await response.json()
  return body.turns ?? []
}

export async function readCortex(signal?: AbortSignal): Promise<Cortex> {
  const response = await fetch(api("/experts"), { signal, headers: auth() })
  if (!response.ok) throw new Error(await failure(response))
  return response.json()
}

export async function readModels(signal?: AbortSignal): Promise<string[]> {
  const response = await fetch(api("/v1/models"), { signal, headers: auth() })
  if (!response.ok) throw new Error(await failure(response))
  const body = await response.json()
  return (body.data ?? []).map((m: { id: string }) => m.id)
}

/**
 * Stop the compute process and give its VRAM back, or start it again.
 *
 * Unloading waits for in-flight turns to finish, so this can take a moment;
 * loading reads ~20 GB of weights and takes longer still.
 */
export async function setModelLoaded(loaded: boolean): Promise<void> {
  const response = await fetch(api(loaded ? "/model/load" : "/model/unload"), {
    method: "POST",
    headers: auth(),
  })
  if (!response.ok) throw new Error(await failure(response))
}

export interface SendOptions {
  model: string
  history: Turn[]
  /** Prepended as a system message when non-empty. */
  system?: string
  temperature: number
  maxTokens: number
  reasoning: boolean
  signal: AbortSignal
  onText: (chunk: string) => void
  onThinking: (chunk: string) => void
}

/**
 * Stream one completion. The engine emits OpenAI chunks over SSE; frames are
 * separated by a blank line and the stream closes with a literal [DONE].
 */
export async function send(options: SendOptions): Promise<RunResult> {
  const response = await fetch(api("/v1/chat/completions"), {
    method: "POST",
    headers: auth({ "Content-Type": "application/json" }),
    signal: options.signal,
    body: JSON.stringify({
      model: options.model,
      messages: [
        ...(options.system?.trim() ? [{ role: "system", content: options.system.trim() }] : []),
        ...options.history.map(({ role, text }) => ({ role, content: text })),
      ],
      temperature: options.temperature,
      max_completion_tokens: options.maxTokens,
      enable_thinking: options.reasoning,
      stream: true,
      stream_options: { include_usage: true },
    }),
  })
  if (!response.ok) throw new Error(await failure(response))
  if (!response.body) throw new Error("The engine returned an empty stream.")

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let pending = ""
  let usage: Usage | null = null
  let stopReason: string | null = null

  const absorb = (payload: string) => {
    if (payload === "[DONE]") return
    let frame: any
    try {
      frame = JSON.parse(payload)
    } catch {
      return
    }
    const choice = frame.choices?.[0]
    if (choice?.delta?.content) options.onText(choice.delta.content)
    if (choice?.delta?.reasoning_content) options.onThinking(choice.delta.reasoning_content)
    if (choice?.finish_reason) stopReason = choice.finish_reason
    if (frame.usage) usage = frame.usage
  }

  for (;;) {
    const { value, done } = await reader.read()
    pending += decoder.decode(value, { stream: !done })
    const frames = pending.split(/\r?\n\r?\n/)
    pending = frames.pop() ?? ""
    for (const frame of frames) {
      for (const line of frame.split(/\r?\n/)) {
        if (line.startsWith("data:")) absorb(line.slice(5).trimStart())
      }
    }
    if (done) break
  }

  const waited = Number(response.headers.get("x-strata-queue-wait-ms"))
  return {
    usage,
    stopReason,
    queueWaitMs: Number.isFinite(waited) ? waited : null,
  }
}
