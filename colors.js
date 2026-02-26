/* ===================================================================
   colors.js — Single source of truth for shared color & font values
   Loads in <head> before body renders. Injects CSS custom properties.
   =================================================================== */

// ---------- Alpha helper (appends alpha byte to hex) ----------
const _r = (hex, a) => hex + Math.round(a * 255).toString(16).padStart(2, '0');

// ---------- Font Constants ----------
const _FONT = Object.freeze({
  display: "'Instrument Serif', Georgia, 'Times New Roman', serif",
  body:    "'Geist', system-ui, -apple-system, sans-serif",
  mono:    "'Geist Mono', 'SF Mono', 'Menlo', monospace",
});

// ---------- Palette ----------
const _PALETTE = Object.freeze({
  // Particle physics colors
  neutral:   '#bdc3c7',       // zero-charge particle fill
  chargePos: 220,             // hue — positive charge (blue)
  chargeNeg: 10,              // hue — negative charge (red)
  spinPos:   160,             // hue — positive spin ring (cyan)
  spinNeg:   30,              // hue — negative spin ring (orange)

  // Mode-independent accent
  accent:      '#FE3B01',
  accentHover: '#FF6B3D',

  // Mode-dependent surfaces & text
  dark: Object.freeze({
    canvas:        '#0C0B09',
    panelSolid:    '#181612',
    text:          '#E8E2D4',
    textSecondary: '#8A8278',
    textMuted:     '#5A544C',
  }),

  light: Object.freeze({
    canvas:        '#F0EDE4',
    panelSolid:    '#FCFAF4',
    text:          '#1A1612',
    textSecondary: '#78706A',
    textMuted:     '#A8A098',
  }),
});

// ---------- CSS Custom Property Injection ----------
(function injectPaletteVars() {
  const P = _PALETTE, D = P.dark, L = P.light;

  const style = document.createElement('style');
  style.id = 'palette-vars';
  style.textContent =
`:root {
  --font-display:   ${_FONT.display};
  --font-body:      ${_FONT.body};
  --font-mono:      ${_FONT.mono};

  --bg:             ${D.canvas};
  --canvas-bg:      ${D.canvas};
  --surface:        ${_r(D.panelSolid, 0.88)};
  --surface-solid:  ${D.panelSolid};
  --bg-hover:       ${_r(D.text, 0.05)};

  --text-1:         ${D.text};
  --text-2:         ${D.textSecondary};
  --text-3:         ${D.textMuted};

  --accent:         ${P.accent};
  --accent-hover:   ${P.accentHover};
  --accent-subtle:  ${_r(P.accent, 0.07)};

  --danger:         ${P.accent};
  --danger-subtle:  ${_r(P.accent, 0.07)};

  --border:         ${_r(D.text, 0.06)};
  --border-strong:  ${_r(D.text, 0.12)};
  --slider-track:   ${_r(D.text, 0.08)};
  --backdrop:       rgba(0, 0, 0, 0.5);
}
body.light-theme {
  --bg:             ${L.canvas};
  --canvas-bg:      ${L.canvas};
  --surface:        ${_r(L.panelSolid, 0.82)};
  --surface-solid:  ${L.panelSolid};
  --bg-hover:       ${_r(L.text, 0.04)};

  --text-1:         ${L.text};
  --text-2:         ${L.textSecondary};
  --text-3:         ${L.textMuted};

  --border:         ${_r(L.text, 0.08)};
  --border-strong:  ${_r(L.text, 0.14)};
  --slider-track:   ${_r(L.text, 0.08)};
  --backdrop:       rgba(0, 0, 0, 0.3);
}`;
  document.head.appendChild(style);
})();

// ---------- Expose for ES modules ----------
window._PALETTE = _PALETTE;
window._FONT = _FONT;
window._r = _r;
