/** One stroke weight, one corner treatment. */

const stroke = {
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.6,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
}

const box = (size = 16) => ({ width: size, height: size, viewBox: "0 0 24 24", "aria-hidden": true })

export const PanelRight = ({ size = 17 }: { size?: number }) => (
  <svg {...box(size)} {...stroke}>
    <rect x="3" y="4.5" width="18" height="15" rx="2.5" />
    <path d="M15 4.5v15" />
  </svg>
)

export const ArrowUp = ({ size = 16 }: { size?: number }) => (
  <svg {...box(size)} {...stroke} strokeWidth={2}>
    <path d="M12 19V6M6.5 11.5 12 6l5.5 5.5" />
  </svg>
)

export const Stop = ({ size = 14 }: { size?: number }) => (
  <svg {...box(size)} viewBox="0 0 24 24" aria-hidden="true">
    <rect x="6.5" y="6.5" width="11" height="11" rx="2" fill="currentColor" />
  </svg>
)

export const Chevron = ({ size = 12 }: { size?: number }) => (
  <svg {...box(size)} {...stroke}>
    <path d="M9 5l7 7-7 7" />
  </svg>
)

export const Plus = ({ size = 17 }: { size?: number }) => (
  <svg {...box(size)} {...stroke}>
    <path d="M12 5v14M5 12h14" />
  </svg>
)

/** Three stacked bands — the memory hierarchy the engine is named for. */
export const Mark = ({ size = 18 }: { size?: number }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true">
    <rect x="3" y="4" width="18" height="4" rx="1.2" fill="currentColor" />
    <rect x="3" y="10" width="18" height="4" rx="1.2" fill="currentColor" opacity="0.62" />
    <rect x="3" y="16" width="18" height="4" rx="1.2" fill="currentColor" opacity="0.32" />
  </svg>
)

export const Copy = ({ size = 14 }: { size?: number }) => (
  <svg {...box(size)} {...stroke}>
    <rect x="9" y="9" width="11" height="11" rx="2" />
    <path d="M5 15V6a2 2 0 0 1 2-2h8" />
  </svg>
)

export const Check = ({ size = 14 }: { size?: number }) => (
  <svg {...box(size)} {...stroke} strokeWidth={2}>
    <path d="M5 12.5l4.5 4.5L19 7" />
  </svg>
)

export const Trash = ({ size = 14 }: { size?: number }) => (
  <svg {...box(size)} {...stroke}>
    <path d="M4 7h16M10 11v6M14 11v6" />
    <path d="M6 7l1 12a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-12" />
    <path d="M9 7V5a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2" />
  </svg>
)

export const PanelLeft = ({ size = 17 }: { size?: number }) => (
  <svg {...box(size)} {...stroke}>
    <rect x="3" y="4.5" width="18" height="15" rx="2.5" />
    <path d="M9 4.5v15" />
  </svg>
)
