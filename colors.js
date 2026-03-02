/* ===================================================================
   colors.js — physsim project-specific tokens
   Extends shared-tokens.js with particle physics colors & CSS vars.
   =================================================================== */

// ---------- Project-specific palette keys ----------
// Derive particle hues from extended palette colors
const _hueOf = (hex) => Math.round(_rgb2hsl(..._parseHex(hex))[0]);
_PALETTE.neutral   = _PALETTE.extended.slate;  // zero-charge particle fill
_PALETTE.chargePos = _hueOf(_PALETTE.extended.red);     // hue — positive charge
_PALETTE.chargeNeg = _hueOf(_PALETTE.extended.blue);    // hue — negative charge
_PALETTE.spinPos   = _hueOf(_PALETTE.extended.cyan);    // hue — positive spin ring
_PALETTE.spinNeg   = _hueOf(_PALETTE.extended.orange);  // hue — negative spin ring

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
