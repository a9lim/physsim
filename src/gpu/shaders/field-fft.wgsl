// ─── 2D FFT via Stockham Auto-Sort (Ping-Pong) ───
// Each pass: one butterfly stage along one axis.
// JS dispatches log2(GRID) passes for rows, then log2(GRID) for columns.
// Two complex buffers (A, B) ping-pong between passes.
// Convention: complex stored as interleaved pairs: [re0, im0, re1, im1, ...]
// Total buffer size: GRID * GRID * 2 floats per buffer.

// Bind group 0: the two complex ping-pong buffers
@group(0) @binding(0) var<storage, read_write> bufA: array<f32>;  // GRID*GRID*2
@group(0) @binding(1) var<storage, read_write> bufB: array<f32>;  // GRID*GRID*2
@group(0) @binding(2) var<uniform> fftParams: FFTParams;

struct FFTParams {
    stageLen: u32,     // butterfly half-length: 1, 2, 4, ..., GRID/2
    direction: i32,    // -1 = forward FFT, +1 = inverse FFT
    axis: u32,         // 0 = row transform, 1 = column transform
    gridSize: u32,     // GRID
    invN: f32,         // 1/GRID (for IFFT normalization on last pass)
    isLastStage: u32,  // 1 if this is the last pass (apply invN)
    _pad0: u32,
    _pad1: u32,
};

// ─── Stockham Butterfly Pass ───
// One thread per element. Reads from bufA, writes to bufB.
// After dispatch, JS swaps the bind group so output becomes input.
@compute @workgroup_size(256)
fn fftButterfly(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let N = fftParams.gridSize;
    let N2 = N * N;
    if (tid >= N2) { return; }

    // Determine which row/column and position within it
    var lineIdx: u32;   // which line (row or column index)
    var posInLine: u32;  // position within that line
    if (fftParams.axis == 0u) {
        // Row transform: tid = row * N + col
        lineIdx = tid / N;
        posInLine = tid % N;
    } else {
        // Column transform: tid = row * N + col, but we process column-major
        lineIdx = tid % N;
        posInLine = tid / N;
    }

    let halfLen = fftParams.stageLen;
    let fullLen = halfLen * 2u;

    // Which butterfly group and position within group
    let groupIdx = posInLine / fullLen;
    let posInGroup = posInLine % fullLen;

    // Determine even/odd partner
    var srcEvenPos: u32;
    var srcOddPos: u32;
    if (posInGroup < halfLen) {
        // This element reads from even position
        srcEvenPos = groupIdx * halfLen + posInGroup;
        srcOddPos = srcEvenPos + N / 2u;
    } else {
        // This element reads from odd position
        srcEvenPos = groupIdx * halfLen + (posInGroup - halfLen);
        srcOddPos = srcEvenPos + N / 2u;
    }

    // Convert to linear buffer indices (interleaved complex: index*2 for re, index*2+1 for im)
    var evenLinear: u32;
    var oddLinear: u32;
    if (fftParams.axis == 0u) {
        evenLinear = (lineIdx * N + srcEvenPos) * 2u;
        oddLinear = (lineIdx * N + srcOddPos) * 2u;
    } else {
        evenLinear = (srcEvenPos * N + lineIdx) * 2u;
        oddLinear = (srcOddPos * N + lineIdx) * 2u;
    }

    let eRe = bufA[evenLinear];
    let eIm = bufA[evenLinear + 1u];
    let oRe = bufA[oddLinear];
    let oIm = bufA[oddLinear + 1u];

    // Twiddle factor: W = exp(direction * 2πi * k / fullLen)
    let k = posInGroup % halfLen;
    let angle = f32(fftParams.direction) * TWO_PI * f32(k) / f32(fullLen);
    let wRe = cos(angle);
    let wIm = sin(angle);

    // Butterfly: out = even + W * odd (top) or even - W * odd (bottom)
    let tRe = wRe * oRe - wIm * oIm;
    let tIm = wRe * oIm + wIm * oRe;

    var outRe: f32;
    var outIm: f32;
    if (posInGroup < halfLen) {
        outRe = eRe + tRe;
        outIm = eIm + tIm;
    } else {
        outRe = eRe - tRe;
        outIm = eIm - tIm;
    }

    // IFFT normalization on last stage
    if (fftParams.isLastStage != 0u) {
        outRe *= fftParams.invN;
        outIm *= fftParams.invN;
    }

    // Write to output buffer
    var outLinear: u32;
    if (fftParams.axis == 0u) {
        outLinear = (lineIdx * N + posInLine) * 2u;
    } else {
        outLinear = (posInLine * N + lineIdx) * 2u;
    }
    bufB[outLinear] = outRe;
    bufB[outLinear + 1u] = outIm;
}

// ─── Pointwise Complex Multiply ───
// Φ̂ = ρ̂ · Ĝ. Both in bufA (ρ̂) and greenHat (Ĝ). Result written to bufA.
@group(1) @binding(0) var<storage, read> greenHat: array<f32>;  // GRID*GRID*2 (precomputed)

@compute @workgroup_size(256)
fn complexMultiply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let N2 = fftParams.gridSize * fftParams.gridSize;
    if (idx >= N2) { return; }
    let i = idx * 2u;
    let aRe = bufA[i];
    let aIm = bufA[i + 1u];
    let bRe = greenHat[i];
    let bIm = greenHat[i + 1u];
    bufA[i]      = aRe * bRe - aIm * bIm;
    bufA[i + 1u] = aRe * bIm + aIm * bRe;
}
