/* ===================================================================
   colors.js — geon project-specific tokens
   Extends shared-tokens.js with particle physics colors & CSS vars.
   =================================================================== */

// ---------- Project-specific palette keys ----------
const _hueOf = (hex) => Math.round(_rgb2hsl(..._parseHex(hex))[0]);
_PALETTE.neutral   = _PALETTE.extended.slate;  // zero-charge particle fill
_PALETTE.chargePos = _hueOf(_PALETTE.extended.red);     // hue — positive charge
_PALETTE.chargeNeg = _hueOf(_PALETTE.extended.blue);    // hue — negative charge
_PALETTE.spinPos   = _hueOf(_PALETTE.extended.cyan);    // hue — positive spin ring
_PALETTE.spinNeg   = _hueOf(_PALETTE.extended.orange);  // hue — negative spin ring

_freezeTokens();
