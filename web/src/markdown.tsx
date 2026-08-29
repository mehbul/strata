import { Fragment, type ReactNode } from "react"

import { CodeBlock } from "./components/CodeBlock"
import { parseBlocks, type Block } from "./markdown-parse"

export { parseBlocks, type Block }

const INLINE = /(`+)([\s\S]*?)\1|\*\*([\s\S]+?)\*\*|__([\s\S]+?)__|(?<![\w*])\*([^*\n]+?)\*(?![\w*])|\[([^\]]+)\]\(([^)\s]+)[^)]*\)/g

/** Inline spans, rendered as elements — never as raw HTML. */
export function inline(text: string, keyPrefix = ""): ReactNode[] {
  const nodes: ReactNode[] = []
  let last = 0
  let match: RegExpExecArray | null
  let n = 0
  INLINE.lastIndex = 0

  while ((match = INLINE.exec(text))) {
    if (match.index > last) {
      nodes.push(text.slice(last, match.index))
    }
    const key = `${keyPrefix}i${n++}`
    if (match[2] !== undefined) {
      nodes.push(
        <code className="inline-code" key={key}>
          {match[2]}
        </code>,
      )
    } else if (match[3] || match[4]) {
      nodes.push(<strong key={key}>{match[3] ?? match[4]}</strong>)
    } else if (match[5]) {
      nodes.push(<em key={key}>{match[5]}</em>)
    } else if (match[6]) {
      nodes.push(
        <a href={match[7]} key={key} target="_blank" rel="noreferrer noopener">
          {match[6]}
        </a>,
      )
    }
    last = match.index + match[0].length
  }
  if (last < text.length) {
    nodes.push(text.slice(last))
  }
  return nodes
}

export function Markdown({ source, caret }: { source: string; caret?: boolean }) {
  const blocks = parseBlocks(source)
  return (
    <div className="md">
      {blocks.map((block, index) => {
        const key = `b${index}`
        const isLast = index === blocks.length - 1
        switch (block.kind) {
          case "code":
            return <CodeBlock key={key} lang={block.lang} text={block.text} streaming={block.open} />
          case "heading": {
            const Tag = `h${Math.min(block.level + 2, 6)}` as "h3"
            return <Tag key={key}>{inline(block.text, key)}</Tag>
          }
          case "rule":
            return <hr key={key} />
          case "quote":
            return (
              <blockquote key={key}>
                {inline(block.lines.join("\n"), key)}
              </blockquote>
            )
          case "list":
            return block.ordered ? (
              <ol key={key} start={block.start}>
                {block.items.map((item, j) => (
                  <li key={j}>{inline(item, `${key}l${j}`)}</li>
                ))}
              </ol>
            ) : (
              <ul key={key}>
                {block.items.map((item, j) => (
                  <li key={j}>{inline(item, `${key}l${j}`)}</li>
                ))}
              </ul>
            )
          default:
            return (
              <p key={key}>
                {inline(block.text, key)}
                {caret && isLast ? <span className="caret" /> : null}
              </p>
            )
        }
      })}
      {caret && blocks.length === 0 ? <span className="caret" /> : null}
      {caret && blocks.length > 0 && blocks[blocks.length - 1].kind !== "para" ? (
        <span className="caret" />
      ) : null}
    </div>
  )
}

export const Md = Fragment
