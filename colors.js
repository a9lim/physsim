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
  neutral:   '#847a70',       // zero-charge particle fill
  chargePos: 201,             // hue — positive charge (blue)
  chargeNeg: 7,              // hue — negative charge (red)
  spinPos:   174,             // hue — positive spin ring (cyan)
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
// [css-var, palette-key]                → direct value
// [css-var, palette-key, alpha]         → same alpha both themes
// [css-var, palette-key, lightA, darkA] → per-theme alpha
(function injectPaletteVars() {
  const P = _PALETTE, L = P.light, D = P.dark;

  const themeMap = [
    ['bg',            'canvas'],
    ['canvas-bg',     'canvas'],
    ['surface',       'panelSolid',    0.55,  0.58],
    ['surface-solid', 'panelSolid'],
    ['bg-hover',      'text',          0.039, 0.051],

    ['text-1',        'text'],
    ['text-2',        'textSecondary'],
    ['text-3',        'textMuted'],

    ['border',        'text',          0.078, 0.059],
    ['border-strong', 'text',          0.141, 0.122],
    ['slider-track',  'text',          0.06],
  ];

  const gen = (T, dark) => themeMap.map(([name, key, lA, dA]) => {
    const a = dark ? (dA ?? lA) : lA;
    return `  --${name}: ${a != null ? _r(T[key], a) : T[key]};`;
  }).join('\n');

  const style = document.createElement('style');
  style.id = 'palette-vars';
  style.textContent = `:root {
  --font-display:   ${_FONT.display};
  --font-body:      ${_FONT.body};
  --font-mono:      ${_FONT.mono};

${gen(L, false)}

  --accent:         ${P.accent};
  --accent-hover:   ${P.accentHover};
  --accent-subtle:  ${_r(P.accent, 0.078)};
  --accent-glow:    ${_r(P.accent, 0.18)};
  --accent-light:   ${P.accentHover};
  --intro-warm:     ${_r(P.accentHover, 0.08)};
  --intro-warm-hover: ${_r(P.accentHover, 0.12)};
  --intro-cool:     ${_r('#5898ba', 0.04)};
  --danger:         ${P.accent};
  --danger-subtle:  ${_r(P.accent, 0.078)};

  --shadow-sm:      0 1px 4px #0000000a, 0 0 0 1px #00000005;
  --shadow-md:      0 4px 20px #0000000f, 0 0 0 1px #00000005;
  --shadow-lg:      0 12px 48px #0000001a, 0 0 0 1px #00000005;
  --backdrop:       #0000004d;
}
[data-theme="dark"] {
${gen(D, true)}

  --shadow-sm:      0 1px 4px #00000033, 0 0 0 1px #ffffff08;
  --shadow-md:      0 4px 20px #0000004d, 0 0 0 1px #ffffff08;
  --shadow-lg:      0 12px 48px #00000066, 0 0 0 1px #ffffff08;
  --backdrop:       #00000080;
}`;
  document.head.appendChild(style);
})();

// ---------- Expose for ES modules ----------
window._PALETTE = _PALETTE;
window._FONT = _FONT;
window._r = _r;
