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
  /** Routed experts placed in each tier. Zero for a dense model. */
  vram: number
  ram: number
  disk: number
  /** The VRAM and RAM budgets the plan was solved against. */
  vram_gb: number
  ram_gb: number
  /** Dense weights, KV cache, and the whole file, in GiB. */
  dense_gb: number
  kv_gb: number
  total_gb: number
  expert_count: number
  layers: number
  attention_layers: number
  /** "placement" when solved from the GGUF, "unknown" under --upstream. */
  source: "placement" | "unknown"
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

/**
 * Load a different model in place of the one that is resident.
 *
 * Selecting a name in the picker is a request to run that model, not just to
 * label the next turn with it: the compute process is serving exactly one set
 * of weights, so the choice has to reach the server. The server rolls back to
 * the previous model if the new one fails to load.
 */
export async function switchModel(model: string): Promise<void> {
  const response = await fetch(api("/api/models/load"), {
    method: "POST",
    headers: { ...auth(), "Content-Type": "application/json" },
    body: JSON.stringify({ model }),
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
  /** "json" or a JSON schema; sent as OpenAI response_format. */
  format?: string
  /** Ollama keep_alive duration, e.g. "5m" or "0". */
  keepAlive?: string
  /** low / medium / xhigh for reasoning-capable templates. */
  reasoningEffort?: string
  /** data:image/...;base64 URL for vision models. */
  image?: string
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
      messages: (() => {
        const messages: any[] = [
          ...(options.system?.trim() ? [{ role: "system", content: options.system.trim() }] : []),
          ...options.history.map(({ role, text }) => ({ role, content: text })),
        ]
        if (options.image) {
          for (let i = messages.length - 1; i >= 0; i--) {
            if (messages[i].role === "user") {
              messages[i].content = [
                { type: "text", text: messages[i].content },
                { type: "image_url", image_url: { url: options.image } },
              ]
              break
            }
          }
        }
        return messages
      })(),
      temperature: options.temperature,
      max_completion_tokens: options.maxTokens,
      enable_thinking: options.reasoning,
      ...(options.format ? { response_format: options.format === "json" ? { type: "json_object" } : JSON.parse(options.format) } : {}),
      ...(options.keepAlive ? { keep_alive: options.keepAlive } : {}),
      ...(options.reasoningEffort ? { reasoning_effort: options.reasoningEffort } : {}),
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

// ---------------------------------------------------------------- model management

export interface InstalledModel {
  name: string
  model: string
  modified_at: string
  size: number
  digest: string
  details: Record<string, unknown>
}

export interface RunningModel extends InstalledModel {
  expires_at?: string | null
}

export interface ShowResponse {
  modelfile?: string
  template?: string
  system?: string | null
  parameters?: string
  details?: Record<string, unknown>
  model_info?: Record<string, unknown>
  capabilities?: string[]
}

async function jsonRequest(path: string, init: RequestInit): Promise<any> {
  const response = await fetch(api(path), {
    ...init,
    headers: { ...auth(), "Content-Type": "application/json", ...(init.headers || {}) },
  })
  if (!response.ok) throw new Error(await failure(response))
  return response.json()
}

export async function listInstalled(): Promise<InstalledModel[]> {
  const body = await jsonRequest("/api/tags", { method: "GET" })
  return body.models ?? []
}

export async function listRunning(): Promise<RunningModel[]> {
  const body = await jsonRequest("/api/ps", { method: "GET" })
  return body.models ?? []
}

export async function showModel(model: string, verbose = false): Promise<ShowResponse> {
  return jsonRequest("/api/show", { method: "POST", body: JSON.stringify({ model, verbose }) })
}

export async function copyModel(source: string, destination: string): Promise<void> {
  await jsonRequest("/api/copy", { method: "POST", body: JSON.stringify({ source, destination }) })
}

export async function deleteModel(model: string): Promise<void> {
  await jsonRequest("/api/delete", { method: "DELETE", body: JSON.stringify({ model }) })
}

export async function pullModel(model: string): Promise<string> {
  const response = await fetch(api("/api/pull"), {
    method: "POST",
    headers: { ...auth(), "Content-Type": "application/json" },
    body: JSON.stringify({ model }),
  })
  if (!response.ok) throw new Error(await failure(response))
  return response.text()
}

export async function pushModel(model: string): Promise<void> {
  await jsonRequest("/api/push", { method: "POST", body: JSON.stringify({ model }) })
}

export async function createModel(name: string, body: Record<string, unknown>): Promise<void> {
  await jsonRequest("/api/create", { method: "POST", body: JSON.stringify({ model: name, ...body }) })
}

export async function embed(model: string, input: string[]): Promise<number[][]> {
  const body = await jsonRequest("/api/embed", {
    method: "POST",
    body: JSON.stringify({ model, input }),
  })
  return body.embeddings ?? []
}
