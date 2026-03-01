/* ===================================================================
   colors.js — physsim project-specific tokens
   Extends shared-tokens.js with particle physics colors & CSS vars.
   =================================================================== */

// ---------- Project-specific palette keys ----------
_PALETTE.neutral   = _PALETTE.extended.slate;  // zero-charge particle fill
_PALETTE.chargePos = 4;             // hue — positive charge (red)
_PALETTE.chargeNeg = 197;              // hue — negative charge (blue)
_PALETTE.spinPos   = 153;             // hue — positive spin ring (cyan)
_PALETTE.spinNeg   = 30;              // hue — negative spin ring (orange)

Object.freeze(_PALETTE.extended);
Object.freeze(_PALETTE.light);
Object.freeze(_PALETTE.dark);
Object.freeze(_FONT);
Object.freeze(_PALETTE);

// ---------- Project-specific CSS vars ----------
// (none currently needed — shared tokens cover all physsim styles)

// ---------- Expose for ES modules ----------
window._PALETTE = _PALETTE;
window._FONT = _FONT;
window._r = _r;
