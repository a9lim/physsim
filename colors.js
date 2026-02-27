/* ===================================================================
   colors.js — physsim project-specific tokens
   Extends shared-tokens.js with particle physics colors & CSS vars.
   =================================================================== */

// ---------- Project-specific palette keys ----------
_PALETTE.neutral   = _PALETTE.extended.slate;  // zero-charge particle fill
_PALETTE.chargePos = 201;             // hue — positive charge (blue)
_PALETTE.chargeNeg = 7;              // hue — negative charge (red)
_PALETTE.spinPos   = 174;             // hue — positive spin ring (cyan)
_PALETTE.spinNeg   = 30;              // hue — negative spin ring (orange)

Object.freeze(_PALETTE.extended);
Object.freeze(_PALETTE.light);
Object.freeze(_PALETTE.dark);
Object.freeze(_FONT);
Object.freeze(_PALETTE);

// ---------- Project-specific CSS vars ----------
(function injectProjectVars() {
  const P = _PALETTE;

  const style = document.createElement('style');
  style.id = 'project-vars';
  style.textContent = `:root {
  --danger:           ${P.accent};
  --danger-subtle:    ${_r(P.accent, 0.078)};
}`;
  document.head.appendChild(style);
})();

// ---------- Expose for ES modules ----------
window._PALETTE = _PALETTE;
window._FONT = _FONT;
window._r = _r;
