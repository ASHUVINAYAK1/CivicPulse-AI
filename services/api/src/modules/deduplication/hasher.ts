import sharp from 'sharp';

export interface ImageHashes {
  dHash: string;
  pHash: string;
}

/**
 * Computes 1D Discrete Cosine Transform (DCT-II).
 */
function dct1d(input: number[]): number[] {
  const N = input.length;
  const output: number[] = new Array(N).fill(0);
  const factor = Math.PI / N;

  for (let k = 0; k < N; k++) {
    let sum = 0;
    for (let n = 0; n < N; n++) {
      sum += input[n] * Math.cos((n + 0.5) * k * factor);
    }
    const c = k === 0 ? Math.sqrt(1 / N) : Math.sqrt(2 / N);
    output[k] = sum * c;
  }
  return output;
}

/**
 * Computes 2D Discrete Cosine Transform (DCT-II) on an N x N matrix.
 */
function dct2d(matrix: number[][], size: number): number[][] {
  const rowDct: number[][] = [];
  for (let r = 0; r < size; r++) {
    rowDct.push(dct1d(matrix[r]));
  }

  // Transpose, apply 1D DCT on columns, and transpose back
  const colDct: number[][] = Array.from({ length: size }, () => new Array(size).fill(0));
  for (let c = 0; c < size; c++) {
    const colVals = rowDct.map((row) => row[c]);
    const transformedCol = dct1d(colVals);
    for (let r = 0; r < size; r++) {
      colDct[r][c] = transformedCol[r];
    }
  }

  return colDct;
}

/**
 * Computes perceptual hashes (dHash and pHash) from an image buffer.
 * Automatically normalizes EXIF orientation and handles all formats.
 */
export async function computePerceptualHash(buffer: Buffer): Promise<ImageHashes> {
  // 1. Difference Hash (dHash): 9x8 grayscale
  const dHashPixels = await sharp(buffer)
    .rotate() // Normalize EXIF orientation
    .grayscale()
    .resize(9, 8, { fit: 'fill' })
    .raw()
    .toBuffer();

  let dHashBits = '';
  for (let row = 0; row < 8; row++) {
    for (let col = 0; col < 8; col++) {
      const left = dHashPixels[row * 9 + col];
      const right = dHashPixels[row * 9 + col + 1];
      dHashBits += left > right ? '1' : '0';
    }
  }

  // 2. DCT-based Perceptual Hash (pHash): 32x32 grayscale
  const pHashPixels = await sharp(buffer)
    .rotate() // Normalize EXIF orientation
    .grayscale()
    .resize(32, 32, { fit: 'fill' })
    .raw()
    .toBuffer();

  const matrix32: number[][] = [];
  for (let r = 0; r < 32; r++) {
    const row: number[] = [];
    for (let c = 0; c < 32; c++) {
      row.push(pHashPixels[r * 32 + c]);
    }
    matrix32.push(row);
  }

  const dctMatrix = dct2d(matrix32, 32);

  // Extract top-left 8x8 low-frequency DCT coefficients (exclude DC [0,0] for median)
  const lowFreqVals: number[] = [];
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      if (r === 0 && c === 0) continue;
      lowFreqVals.push(dctMatrix[r][c]);
    }
  }

  const sortedVals = [...lowFreqVals].sort((a, b) => a - b);
  const median = sortedVals[Math.floor(sortedVals.length / 2)];

  let pHashBits = '';
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      pHashBits += dctMatrix[r][c] > median ? '1' : '0';
    }
  }

  // Convert 64-bit binary strings to 16-character hexadecimal representations
  const dHashHex = BigInt('0b' + dHashBits).toString(16).padStart(16, '0');
  const pHashHex = BigInt('0b' + pHashBits).toString(16).padStart(16, '0');

  return {
    dHash: dHashHex,
    pHash: pHashHex,
  };
}

/**
 * Computes Hamming distance between two 16-character hexadecimal hash strings.
 * Returns integer in [0, 64].
 */
export function computeHammingDistance(h1: string, h2: string): number {
  if (!h1 || !h2 || h1.length !== 16 || h2.length !== 16) {
    return 64; // Maximum distance for invalid/missing hashes
  }

  try {
    const b1 = BigInt('0x' + h1);
    const b2 = BigInt('0x' + h2);
    let xor = b1 ^ b2;

    // Brian Kernighan's bit-counting algorithm for BigInt
    let distance = 0;
    while (xor > 0n) {
      xor &= xor - 1n;
      distance++;
    }
    return distance;
  } catch {
    return 64;
  }
}

/**
 * Computes perceptual similarity in range [0.0, 1.0].
 * 1.0 = identical perceptual hash; 0.0 = completely opposite.
 */
export function computeHashSimilarity(h1: string, h2: string): number {
  const distance = computeHammingDistance(h1, h2);
  return Math.max(0, 1.0 - distance / 64.0);
}

