//! Conversation storage.
//!
//! Everything lives in localStorage under one key. Writes happen when a turn
//! settles rather than on every token, and a quota failure drops the oldest
//! sessions instead of throwing away the one being written.

import type { Turn } from "./api"

const SESSIONS_KEY = "strata.sessions"
const SYSTEM_KEY = "strata.system"
const MAX_SESSIONS = 60

export interface Session {
  id: string
  title: string
  turns: Turn[]
  created: number
  updated: number
}

export const newId = () => {
  try {
    return crypto.randomUUID()
  } catch {
    return `s${Date.now()}${Math.random().toString(16).slice(2, 8)}`
  }
}

/** First line of the opening prompt, trimmed to something that fits a rail. */
export function titleFor(turns: Turn[]): string {
  const first = turns.find((t) => t.role === "user" && t.text.trim())
  if (!first) return "New chat"
  const line = first.text.trim().split("\n")[0].replace(/\s+/g, " ")
  return line.length > 48 ? `${line.slice(0, 47)}…` : line
}

export function loadSessions(): Session[] {
  try {
    const raw = localStorage.getItem(SESSIONS_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    if (!Array.isArray(parsed)) return []
    return parsed
      .filter((s): s is Session => !!s && typeof s.id === "string" && Array.isArray(s.turns))
      .sort((a, b) => b.updated - a.updated)
  } catch {
    // Corrupt or unreadable storage should not take the app down with it.
    return []
  }
}

export function saveSessions(sessions: Session[]): Session[] {
  let keep = sessions.slice(0, MAX_SESSIONS)
  for (;;) {
    try {
      localStorage.setItem(SESSIONS_KEY, JSON.stringify(keep))
      return keep
    } catch {
      // Over quota: shed the oldest half rather than lose the newest.
      if (keep.length <= 1) {
        try {
          localStorage.removeItem(SESSIONS_KEY)
        } catch {
          /* storage is unusable; carry on in memory */
        }
        return keep
      }
      keep = keep.slice(0, Math.floor(keep.length / 2))
    }
  }
}

export function loadSystemPrompt(): string {
  try {
    return localStorage.getItem(SYSTEM_KEY) ?? ""
  } catch {
    return ""
  }
}

export function saveSystemPrompt(value: string) {
  try {
    if (value.trim()) localStorage.setItem(SYSTEM_KEY, value)
    else localStorage.removeItem(SYSTEM_KEY)
  } catch {
    /* storage unavailable */
  }
}

export const PRESETS: { name: string; text: string }[] = [
  {
    name: "Rust",
    text: "You are a Rust engineer. Answer with code first and prose after. Prefer the standard library, avoid unwrap outside tests, and keep error handling explicit. Do not explain what the code obviously does.",
  },
  {
    name: "Terse",
    text: "Answer in as few words as possible. Code only unless asked to explain. No preamble, no summary.",
  },
  {
    name: "Reviewer",
    text: "Review the code you are given. Name concrete defects with the input that triggers them. Do not praise, do not restate the code, do not suggest style changes unless they cause bugs.",
  },
]
