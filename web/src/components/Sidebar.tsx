import type { Session } from "../store"
import { Plus, Trash } from "./icons"

interface Props {
  sessions: Session[]
  currentId: string | null
  onSelect: (id: string) => void
  onNew: () => void
  onDelete: (id: string) => void
}

/** Sessions, newest first, grouped by how recently they were touched. */
function bucket(updated: number): string {
  const day = 24 * 60 * 60 * 1000
  const age = Date.now() - updated
  if (age < day) return "Today"
  if (age < 2 * day) return "Yesterday"
  if (age < 7 * day) return "Previous 7 days"
  return "Older"
}

export function Sidebar({ sessions, currentId, onSelect, onNew, onDelete }: Props) {
  const groups: { label: string; items: Session[] }[] = []
  for (const s of sessions) {
    const label = bucket(s.updated)
    const last = groups[groups.length - 1]
    if (last && last.label === label) last.items.push(s)
    else groups.push({ label, items: [s] })
  }

  return (
    <aside className="rail">
      <button type="button" className="rail-new" onClick={onNew}>
        <Plus size={15} />
        New chat
      </button>

      <div className="rail-list">
        {sessions.length === 0 ? (
          <p className="rail-empty">Nothing saved yet.</p>
        ) : (
          groups.map((group) => (
            <div className="rail-group" key={group.label}>
              <div className="rail-group-title">{group.label}</div>
              {group.items.map((s) => (
                <div
                  key={s.id}
                  className="rail-item"
                  data-current={s.id === currentId}
                  onClick={() => onSelect(s.id)}
                  role="button"
                  tabIndex={0}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault()
                      onSelect(s.id)
                    }
                  }}
                >
                  <span className="rail-title">{s.title}</span>
                  <button
                    type="button"
                    className="rail-del"
                    aria-label={`Delete ${s.title}`}
                    onClick={(e) => {
                      e.stopPropagation()
                      onDelete(s.id)
                    }}
                  >
                    <Trash size={13} />
                  </button>
                </div>
              ))}
            </div>
          ))
        )}
      </div>
    </aside>
  )
}
