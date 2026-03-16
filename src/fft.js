// ─── Compact Radix-2 FFT ───
// In-place Cooley-Tukey, split-radix for 2D convolution.
// No dependencies. Operates on separate real/imag Float64Arrays.

/**
 * In-place 1D FFT (Cooley-Tukey, decimation-in-time).
 * @param {Float64Array} re - Real part (length must be power of 2)
 * @param {Float64Array} im - Imaginary part
 * @param {boolean} inverse - true for IFFT
 */
export function fft1d(re, im, inverse) {
    const n = re.length;
    if (n <= 1) return;

    // Bit-reversal permutation
    for (let i = 1, j = 0; i < n; i++) {
        let bit = n >> 1;
        while (j & bit) { j ^= bit; bit >>= 1; }
        j ^= bit;
        if (i < j) {
            let tmp = re[i]; re[i] = re[j]; re[j] = tmp;
            tmp = im[i]; im[i] = im[j]; im[j] = tmp;
        }
    }

    // Butterfly passes
    const sign = inverse ? 1 : -1;
    for (let len = 2; len <= n; len <<= 1) {
        const halfLen = len >> 1;
        const angle = sign * Math.PI / halfLen;
        const wRe = Math.cos(angle);
        const wIm = Math.sin(angle);
        for (let i = 0; i < n; i += len) {
            let curRe = 1, curIm = 0;
            for (let j = 0; j < halfLen; j++) {
                const a = i + j;
                const b = a + halfLen;
                const tRe = curRe * re[b] - curIm * im[b];
                const tIm = curRe * im[b] + curIm * re[b];
                re[b] = re[a] - tRe;
                im[b] = im[a] - tIm;
                re[a] += tRe;
                im[a] += tIm;
                const nextRe = curRe * wRe - curIm * wIm;
                curIm = curRe * wIm + curIm * wRe;
                curRe = nextRe;
            }
        }
    }

    // Normalize inverse
    if (inverse) {
        const invN = 1 / n;
        for (let i = 0; i < n; i++) {
            re[i] *= invN;
            im[i] *= invN;
        }
    }
}

/**
 * In-place 2D FFT on N×N grid (row-major).
 * @param {Float64Array} re - Real part (N*N)
 * @param {Float64Array} im - Imaginary part (N*N)
 * @param {number} N - Grid size (power of 2)
 * @param {boolean} inverse
 */
export function fft2d(re, im, N, inverse) {
    const rowRe = new Float64Array(N);
    const rowIm = new Float64Array(N);

    // Transform rows
    for (let y = 0; y < N; y++) {
        const off = y * N;
        for (let x = 0; x < N; x++) {
            rowRe[x] = re[off + x];
            rowIm[x] = im[off + x];
        }
        fft1d(rowRe, rowIm, inverse);
        for (let x = 0; x < N; x++) {
            re[off + x] = rowRe[x];
            im[off + x] = rowIm[x];
        }
    }

    // Transform columns
    const colRe = new Float64Array(N);
    const colIm = new Float64Array(N);
    for (let x = 0; x < N; x++) {
        for (let y = 0; y < N; y++) {
            colRe[y] = re[y * N + x];
            colIm[y] = im[y * N + x];
        }
        fft1d(colRe, colIm, inverse);
        for (let y = 0; y < N; y++) {
            re[y * N + x] = colRe[y];
            im[y * N + x] = colIm[y];
        }
    }
}
