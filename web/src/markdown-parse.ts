//! Block-level markdown parsing. No React, no highlighting — pure text in,
//! structure out, so it can be tested on its own.

/** Only the shapes a model actually emits. */
export type Block =
  | { kind: "code"; lang: string; text: string; open: boolean }
  | { kind: "heading"; level: number; text: string }
  | { kind: "list"; ordered: boolean; start: number; items: string[] }
  | { kind: "quote"; lines: string[] }
  | { kind: "rule" }
  | { kind: "para"; text: string }

const FENCE = /^(\s*)(```+|~~~+)\s*([^\s`]*)/
const HEADING = /^(#{1,6})\s+(.*)$/
const RULE = /^\s*([-*_])(\s*\1){2,}\s*$/
const QUOTE = /^\s*>\s?(.*)$/
const BULLET = /^(\s*)[-*+]\s+(.*)$/
const NUMBER = /^(\s*)(\d+)[.)]\s+(.*)$/

/**
 * Blocks from markdown text.
 *
 * Written for streaming: the source is usually a partial document, so a fence
 * that never closes is reported as an open code block rather than swallowing
 * the rest of the message or flickering back to prose on every token.
 */
export function parseBlocks(src: string): Block[] {
  const lines = src.split("\n")
  const out: Block[] = []
  let i = 0

  while (i < lines.length) {
    const line = lines[i]

    const fence = line.match(FENCE)
    if (fence) {
      const marker = fence[2][0]
      const width = fence[2].length
      const lang = fence[3] || ""
      const body: string[] = []
      i++
      let closed = false
      while (i < lines.length) {
        const candidate = lines[i].trim()
        if (candidate.startsWith(marker.repeat(width)) && /^[`~]+\s*$/.test(candidate)) {
          closed = true
          i++
          break
        }
        body.push(lines[i])
        i++
      }
      out.push({ kind: "code", lang, text: body.join("\n"), open: !closed })
      continue
    }

    if (!line.trim()) {
      i++
      continue
    }

    const rule = line.match(RULE)
    if (rule) {
      out.push({ kind: "rule" })
      i++
      continue
    }

    const heading = line.match(HEADING)
    if (heading) {
      out.push({ kind: "heading", level: heading[1].length, text: heading[2] })
      i++
      continue
    }

    if (QUOTE.test(line)) {
      const quoted: string[] = []
      while (i < lines.length && QUOTE.test(lines[i])) {
        quoted.push(lines[i].match(QUOTE)![1])
        i++
      }
      out.push({ kind: "quote", lines: quoted })
      continue
    }

    if (BULLET.test(line) || NUMBER.test(line)) {
      const ordered = NUMBER.test(line)
      const start = ordered ? Number(line.match(NUMBER)![2]) : 1
      const items: string[] = []
      while (i < lines.length) {
        const bullet = lines[i].match(ordered ? NUMBER : BULLET)
        if (bullet) {
          items.push(ordered ? bullet[3] : bullet[2])
          i++
          // continuation lines belong to the item above
          while (i < lines.length && /^\s{2,}\S/.test(lines[i]) && !FENCE.test(lines[i])) {
            items[items.length - 1] += " " + lines[i].trim()
            i++
          }
          continue
        }
        break
      }
      out.push({ kind: "list", ordered, start, items })
      continue
    }

    const paragraph: string[] = []
    while (i < lines.length && lines[i].trim() && !FENCE.test(lines[i]) && !HEADING.test(lines[i]) && !QUOTE.test(lines[i]) && !BULLET.test(lines[i]) && !NUMBER.test(lines[i]) && !RULE.test(lines[i])) {
      paragraph.push(lines[i])
      i++
    }
    if (paragraph.length) {
      out.push({ kind: "para", text: paragraph.join("\n") })
    } else {
      i++
    }
  }
  return out
}

