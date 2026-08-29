import hljs from "highlight.js/lib/core"
import bash from "highlight.js/lib/languages/bash"
import c from "highlight.js/lib/languages/c"
import cpp from "highlight.js/lib/languages/cpp"
import csharp from "highlight.js/lib/languages/csharp"
import css from "highlight.js/lib/languages/css"
import go from "highlight.js/lib/languages/go"
import java from "highlight.js/lib/languages/java"
import js from "highlight.js/lib/languages/javascript"
import json from "highlight.js/lib/languages/json"
import python from "highlight.js/lib/languages/python"
import rust from "highlight.js/lib/languages/rust"
import sql from "highlight.js/lib/languages/sql"
import toml from "highlight.js/lib/languages/ini"
import ts from "highlight.js/lib/languages/typescript"
import xml from "highlight.js/lib/languages/xml"
import yaml from "highlight.js/lib/languages/yaml"
import { useEffect, useMemo, useRef, useState } from "react"

import { Check, Copy } from "./icons"

// Only the languages worth carrying; the full bundle is ten times the size.
const LANGS: Record<string, unknown> = {
  bash,
  c,
  cpp,
  csharp,
  css,
  go,
  java,
  javascript: js,
  json,
  python,
  rust,
  sql,
  toml,
  typescript: ts,
  xml,
  yaml,
}
for (const [name, def] of Object.entries(LANGS)) {
  hljs.registerLanguage(name, def as never)
}
const ALIAS: Record<string, string> = {
  rs: "rust",
  ts: "typescript",
  tsx: "typescript",
  js: "javascript",
  jsx: "javascript",
  py: "python",
  sh: "bash",
  shell: "bash",
  zsh: "bash",
  console: "bash",
  ps1: "bash",
  powershell: "bash",
  yml: "yaml",
  html: "xml",
  ini: "toml",
  "c++": "cpp",
  cs: "csharp",
  golang: "go",
}

const resolve = (lang: string) => {
  const key = lang.trim().toLowerCase()
  const named = ALIAS[key] ?? key
  return hljs.getLanguage(named) ? named : ""
}

export function CodeBlock({
  lang,
  text,
  streaming,
}: {
  lang: string
  text: string
  streaming?: boolean
}) {
  const [copied, setCopied] = useState(false)
  const timer = useRef<number | undefined>(undefined)

  useEffect(() => () => window.clearTimeout(timer.current), [])

  const { html, label } = useMemo(() => {
    const named = resolve(lang)
    try {
      if (named) {
        return { html: hljs.highlight(text, { language: named }).value, label: named }
      }
      // No fence language: let highlight.js guess, but only trust a confident guess.
      const auto = hljs.highlightAuto(text, ["rust", "typescript", "python", "bash", "json"])
      if (auto.relevance > 8 && auto.language) {
        return { html: auto.value, label: auto.language }
      }
    } catch {
      /* fall through to plain text */
    }
    return { html: null, label: lang.trim().toLowerCase() }
  }, [lang, text])

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      window.clearTimeout(timer.current)
      timer.current = window.setTimeout(() => setCopied(false), 1400)
    } catch {
      /* clipboard blocked; the selection still works */
    }
  }

  return (
    <figure className="code">
      <figcaption>
        <span>{label || "text"}</span>
        <button type="button" onClick={copy} aria-label="Copy code" data-copied={copied}>
          {copied ? <Check size={13} /> : <Copy size={13} />}
          {copied ? "Copied" : "Copy"}
        </button>
      </figcaption>
      <pre>
        {html ? (
          <code className="hljs" dangerouslySetInnerHTML={{ __html: html }} />
        ) : (
          <code className="hljs">{text}</code>
        )}
        {streaming ? <span className="caret" /> : null}
      </pre>
    </figure>
  )
}
