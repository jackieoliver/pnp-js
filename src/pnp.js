/**
 * pnp-js: Perspective-n-Point camera pose from 2D-3D correspondences.
 *
 * Pipeline: Hartley-normalised DLT -> orthogonalise the 3x3 block to get an
 * initial rotation -> damped Gauss-Newton (Levenberg-Marquardt) refinement on
 * a 6-DoF (Rodrigues vector, translation) parameterisation using numerical
 * Jacobians. No runtime dependencies; every linear-algebra routine is here.
 *
 * Conventions: the camera looks down +Z, image x right, image y down.
 * `worldToCamera` maps world points to camera space (Rwc * X + twc).
 * `R`, `t` are the inverse (camera-to-world), i.e. `t` is the camera centre.
 */

const DEFAULT_IMAGE_WIDTH = 1920;
const DEFAULT_IMAGE_HEIGHT = 1440;
const DEFAULT_FOV_DEG = 155;

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function degToRad(deg) {
  return (Number(deg) || 0) * (Math.PI / 180);
}

/**
 * Build a pinhole intrinsics object from an image size and horizontal FOV.
 * Defaults describe a wide-angle 4:3 action camera (1920x1440, 155 deg).
 * @param {{width?: number, height?: number, fovDeg?: number}} [options]
 * @returns {{width:number,height:number,fovDeg:number,fx:number,fy:number,cx:number,cy:number}}
 */
export function buildDefaultIntrinsics(options = {}) {
  const width = Math.max(1, Number(options.width) || DEFAULT_IMAGE_WIDTH);
  const height = Math.max(1, Number(options.height) || DEFAULT_IMAGE_HEIGHT);
  const fovDeg = clamp(Number(options.fovDeg) || DEFAULT_FOV_DEG, 1, 179.9);
  const focal = width / (2 * Math.tan(degToRad(fovDeg) * 0.5));
  const cx = width * 0.5;
  const cy = height * 0.5;
  return {
    width,
    height,
    fovDeg,
    fx: focal,
    fy: focal,
    cx,
    cy
  };
}

export const DEFAULT_CAMERA_INTRINSICS = buildDefaultIntrinsics();

function isFiniteVec3(value) {
  return Array.isArray(value) && value.length >= 3
    && Number.isFinite(Number(value[0]))
    && Number.isFinite(Number(value[1]))
    && Number.isFinite(Number(value[2]));
}

function isFiniteVec2(value) {
  return Array.isArray(value) && value.length >= 2
    && Number.isFinite(Number(value[0]))
    && Number.isFinite(Number(value[1]));
}

function parseCorrespondence(item) {
  if (!item || typeof item !== "object") return null;
  if (!isFiniteVec2(item.point2d) || !isFiniteVec3(item.point3d)) return null;
  return {
    point2d: [Number(item.point2d[0]), Number(item.point2d[1])],
    point3d: [Number(item.point3d[0]), Number(item.point3d[1]), Number(item.point3d[2])]
  };
}

function normalizeCorrespondences(correspondences, intrinsics) {
  const parsed = [];
  let max2d = 0;
  for (const item of correspondences || []) {
    const c = parseCorrespondence(item);
    if (!c) continue;
    parsed.push(c);
    max2d = Math.max(max2d, Math.abs(c.point2d[0]), Math.abs(c.point2d[1]));
  }

  if (parsed.length < 6) {
    throw new Error("PnP requires at least 6 valid 2D-3D correspondences.");
  }

  const treat2dAsNormalized = max2d <= 2.0;
  if (treat2dAsNormalized) {
    for (const c of parsed) {
      c.point2d[0] *= intrinsics.width;
      c.point2d[1] *= intrinsics.height;
    }
  }

  return parsed;
}

function createMatrix(rows, cols, fill = 0) {
  const out = new Array(rows);
  for (let r = 0; r < rows; r += 1) {
    out[r] = new Float64Array(cols);
    if (fill !== 0) out[r].fill(fill);
  }
  return out;
}

function identityMatrix(size) {
  const out = createMatrix(size, size);
  for (let i = 0; i < size; i += 1) out[i][i] = 1;
  return out;
}

function transpose(A) {
  const rows = A.length;
  const cols = A[0].length;
  const out = createMatrix(cols, rows);
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      out[c][r] = A[r][c];
    }
  }
  return out;
}

function matMul(A, B) {
  const rows = A.length;
  const inner = A[0].length;
  const cols = B[0].length;
  const out = createMatrix(rows, cols);
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      let sum = 0;
      for (let k = 0; k < inner; k += 1) {
        sum += A[r][k] * B[k][c];
      }
      out[r][c] = sum;
    }
  }
  return out;
}

function inverse3x3(M) {
  const det = M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
    - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
    + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
  if (Math.abs(det) < 1e-12) throw new Error("Matrix inversion failed: singular 3x3 matrix.");
  const invDet = 1 / det;
  const out = createMatrix(3, 3);
  out[0][0] = (M[1][1] * M[2][2] - M[1][2] * M[2][1]) * invDet;
  out[0][1] = (M[0][2] * M[2][1] - M[0][1] * M[2][2]) * invDet;
  out[0][2] = (M[0][1] * M[1][2] - M[0][2] * M[1][1]) * invDet;
  out[1][0] = (M[1][2] * M[2][0] - M[1][0] * M[2][2]) * invDet;
  out[1][1] = (M[0][0] * M[2][2] - M[0][2] * M[2][0]) * invDet;
  out[1][2] = (M[0][2] * M[1][0] - M[0][0] * M[1][2]) * invDet;
  out[2][0] = (M[1][0] * M[2][1] - M[1][1] * M[2][0]) * invDet;
  out[2][1] = (M[0][1] * M[2][0] - M[0][0] * M[2][1]) * invDet;
  out[2][2] = (M[0][0] * M[1][1] - M[0][1] * M[1][0]) * invDet;
  return out;
}

function det3(M) {
  return M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
       - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
       + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
}

function transpose3(M) {
  return [
    [M[0][0], M[1][0], M[2][0]],
    [M[0][1], M[1][1], M[2][1]],
    [M[0][2], M[1][2], M[2][2]]
  ];
}

function mat3Vec3Mul(M, v) {
  return [
    M[0][0] * v[0] + M[0][1] * v[1] + M[0][2] * v[2],
    M[1][0] * v[0] + M[1][1] * v[1] + M[1][2] * v[2],
    M[2][0] * v[0] + M[2][1] * v[1] + M[2][2] * v[2]
  ];
}

function rodriguesFromVector(w) {
  const wx = w[0];
  const wy = w[1];
  const wz = w[2];
  const theta = Math.hypot(wx, wy, wz);
  if (theta < 1e-12) {
    return [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1]
    ];
  }
  const kx = wx / theta;
  const ky = wy / theta;
  const kz = wz / theta;
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  const v = 1 - c;
  return [
    [c + kx * kx * v, kx * ky * v - kz * s, kx * kz * v + ky * s],
    [ky * kx * v + kz * s, c + ky * ky * v, ky * kz * v - kx * s],
    [kz * kx * v - ky * s, kz * ky * v + kx * s, c + kz * kz * v]
  ];
}

function rotationVectorFromMatrix(R) {
  const trace = R[0][0] + R[1][1] + R[2][2];
  const cosTheta = clamp((trace - 1) * 0.5, -1, 1);
  const theta = Math.acos(cosTheta);
  if (theta < 1e-12) return [0, 0, 0];

  const sinTheta = Math.sin(theta);
  if (Math.abs(sinTheta) < 1e-6) {
    // theta ~ pi: R = 2 k k^T - I, so (R + I) / 2 = k k^T. Take the column
    // with the largest diagonal entry to keep the axis signs consistent.
    let col = 0;
    if (R[1][1] > R[col][col]) col = 1;
    if (R[2][2] > R[col][col]) col = 2;
    const k = [(R[0][col] + (col === 0 ? 1 : 0)) * 0.5,
               (R[1][col] + (col === 1 ? 1 : 0)) * 0.5,
               (R[2][col] + (col === 2 ? 1 : 0)) * 0.5];
    const kn = Math.hypot(k[0], k[1], k[2]);
    return [k[0] / kn * theta, k[1] / kn * theta, k[2] / kn * theta];
  }

  const kx = (R[2][1] - R[1][2]) / (2 * sinTheta);
  const ky = (R[0][2] - R[2][0]) / (2 * sinTheta);
  const kz = (R[1][0] - R[0][1]) / (2 * sinTheta);
  return [kx * theta, ky * theta, kz * theta];
}

function solveLinearSystem(AInput, bInput) {
  const n = AInput.length;
  const A = createMatrix(n, n);
  const b = new Float64Array(n);

  for (let r = 0; r < n; r += 1) {
    for (let c = 0; c < n; c += 1) A[r][c] = AInput[r][c];
    b[r] = bInput[r];
  }

  for (let col = 0; col < n; col += 1) {
    let pivotRow = col;
    let maxAbs = Math.abs(A[col][col]);
    for (let r = col + 1; r < n; r += 1) {
      const value = Math.abs(A[r][col]);
      if (value > maxAbs) {
        maxAbs = value;
        pivotRow = r;
      }
    }

    if (maxAbs < 1e-12) return null;

    if (pivotRow !== col) {
      const tmpRow = A[col];
      A[col] = A[pivotRow];
      A[pivotRow] = tmpRow;
      const tmpB = b[col];
      b[col] = b[pivotRow];
      b[pivotRow] = tmpB;
    }

    const pivot = A[col][col];
    for (let c = col; c < n; c += 1) A[col][c] /= pivot;
    b[col] /= pivot;

    for (let r = 0; r < n; r += 1) {
      if (r === col) continue;
      const factor = A[r][col];
      if (Math.abs(factor) < 1e-20) continue;
      for (let c = col; c < n; c += 1) {
        A[r][c] -= factor * A[col][c];
      }
      b[r] -= factor * b[col];
    }
  }

  return Array.from(b);
}

function solveLeastSquaresNormal(A, b) {
  const rows = A.length;
  const cols = A[0].length;
  const AtA = createMatrix(cols, cols);
  const Atb = new Float64Array(cols);

  for (let r = 0; r < rows; r += 1) {
    for (let i = 0; i < cols; i += 1) {
      Atb[i] += A[r][i] * b[r];
      for (let j = 0; j < cols; j += 1) {
        AtA[i][j] += A[r][i] * A[r][j];
      }
    }
  }

  return solveLinearSystem(AtA, Atb);
}

function jacobiEigenDecompositionSymmetric(matrix, maxSweeps = 64) {
  const n = matrix.length;
  const A = createMatrix(n, n);
  const V = createMatrix(n, n);

  for (let r = 0; r < n; r += 1) {
    for (let c = 0; c < n; c += 1) {
      A[r][c] = matrix[r][c];
      V[r][c] = r === c ? 1 : 0;
    }
  }

  for (let sweep = 0; sweep < maxSweeps; sweep += 1) {
    let p = 0;
    let q = 1;
    let maxAbs = 0;

    for (let i = 0; i < n - 1; i += 1) {
      for (let j = i + 1; j < n; j += 1) {
        const value = Math.abs(A[i][j]);
        if (value > maxAbs) {
          maxAbs = value;
          p = i;
          q = j;
        }
      }
    }

    if (maxAbs < 1e-12) break;

    const app = A[p][p];
    const aqq = A[q][q];
    const apq = A[p][q];

    const tau = (aqq - app) / (2 * apq);
    const t = tau >= 0
      ? 1 / (tau + Math.sqrt(1 + tau * tau))
      : -1 / (-tau + Math.sqrt(1 + tau * tau));
    const c = 1 / Math.sqrt(1 + t * t);
    const s = t * c;

    A[p][p] = app - t * apq;
    A[q][q] = aqq + t * apq;
    A[p][q] = 0;
    A[q][p] = 0;

    for (let k = 0; k < n; k += 1) {
      if (k === p || k === q) continue;
      const akp = A[k][p];
      const akq = A[k][q];
      A[k][p] = c * akp - s * akq;
      A[p][k] = A[k][p];
      A[k][q] = s * akp + c * akq;
      A[q][k] = A[k][q];
    }

    for (let k = 0; k < n; k += 1) {
      const vkp = V[k][p];
      const vkq = V[k][q];
      V[k][p] = c * vkp - s * vkq;
      V[k][q] = s * vkp + c * vkq;
    }
  }

  const eigenvalues = new Float64Array(n);
  for (let i = 0; i < n; i += 1) {
    eigenvalues[i] = A[i][i];
  }

  return { eigenvalues, eigenvectors: V };
}

function extractSmallestEigenvector(matrix) {
  const { eigenvalues, eigenvectors } = jacobiEigenDecompositionSymmetric(matrix);
  let minIndex = 0;
  for (let i = 1; i < eigenvalues.length; i += 1) {
    if (eigenvalues[i] < eigenvalues[minIndex]) minIndex = i;
  }

  const v = new Float64Array(eigenvalues.length);
  for (let r = 0; r < eigenvalues.length; r += 1) {
    v[r] = eigenvectors[r][minIndex];
  }

  let norm = 0;
  for (let i = 0; i < v.length; i += 1) norm += v[i] * v[i];
  norm = Math.sqrt(norm);
  if (norm < 1e-12) throw new Error("DLT failed: degenerate correspondence matrix.");
  for (let i = 0; i < v.length; i += 1) v[i] /= norm;
  return v;
}

function buildDltMatrix(correspondences) {
  const A = createMatrix(correspondences.length * 2, 12);
  for (let i = 0; i < correspondences.length; i += 1) {
    const { point2d, point3d } = correspondences[i];
    const [u, v] = point2d;
    const [X, Y, Z] = point3d;
    const r0 = i * 2;
    const r1 = r0 + 1;

    A[r0][0] = X;
    A[r0][1] = Y;
    A[r0][2] = Z;
    A[r0][3] = 1;
    A[r0][8] = -u * X;
    A[r0][9] = -u * Y;
    A[r0][10] = -u * Z;
    A[r0][11] = -u;

    A[r1][4] = X;
    A[r1][5] = Y;
    A[r1][6] = Z;
    A[r1][7] = 1;
    A[r1][8] = -v * X;
    A[r1][9] = -v * Y;
    A[r1][10] = -v * Z;
    A[r1][11] = -v;
  }
  return A;
}

function normalizePoints2D(correspondences) {
  const out = correspondences.map((item) => ({
    point2d: [item.point2d[0], item.point2d[1]],
    point3d: [item.point3d[0], item.point3d[1], item.point3d[2]]
  }));

  let cx = 0;
  let cy = 0;
  for (const item of out) {
    cx += item.point2d[0];
    cy += item.point2d[1];
  }
  cx /= out.length;
  cy /= out.length;

  let meanDist = 0;
  for (const item of out) {
    const dx = item.point2d[0] - cx;
    const dy = item.point2d[1] - cy;
    meanDist += Math.hypot(dx, dy);
  }
  meanDist /= out.length;

  const scale = meanDist > 1e-9 ? Math.sqrt(2) / meanDist : 1;
  const T = identityMatrix(3);
  T[0][0] = scale;
  T[1][1] = scale;
  T[0][2] = -scale * cx;
  T[1][2] = -scale * cy;

  for (const item of out) {
    item.point2d[0] = scale * (item.point2d[0] - cx);
    item.point2d[1] = scale * (item.point2d[1] - cy);
  }

  return { normalized: out, T };
}

function normalizePoints3D(correspondences) {
  const out = correspondences.map((item) => ({
    point2d: [item.point2d[0], item.point2d[1]],
    point3d: [item.point3d[0], item.point3d[1], item.point3d[2]]
  }));

  let cx = 0;
  let cy = 0;
  let cz = 0;
  for (const item of out) {
    cx += item.point3d[0];
    cy += item.point3d[1];
    cz += item.point3d[2];
  }
  cx /= out.length;
  cy /= out.length;
  cz /= out.length;

  let meanDist = 0;
  for (const item of out) {
    const dx = item.point3d[0] - cx;
    const dy = item.point3d[1] - cy;
    const dz = item.point3d[2] - cz;
    meanDist += Math.hypot(dx, dy, dz);
  }
  meanDist /= out.length;

  const scale = meanDist > 1e-9 ? Math.sqrt(3) / meanDist : 1;
  const U = identityMatrix(4);
  U[0][0] = scale;
  U[1][1] = scale;
  U[2][2] = scale;
  U[0][3] = -scale * cx;
  U[1][3] = -scale * cy;
  U[2][3] = -scale * cz;

  for (const item of out) {
    item.point3d[0] = scale * (item.point3d[0] - cx);
    item.point3d[1] = scale * (item.point3d[1] - cy);
    item.point3d[2] = scale * (item.point3d[2] - cz);
  }

  return { normalized: out, U };
}

function projectWorldToPixel(worldPoint, Rwc, twc, intrinsics) {
  const cam = mat3Vec3Mul(Rwc, worldPoint);
  cam[0] += twc[0];
  cam[1] += twc[1];
  cam[2] += twc[2];

  const z = cam[2];
  if (Math.abs(z) < 1e-9) return null;

  return [
    intrinsics.fx * (cam[0] / z) + intrinsics.cx,
    intrinsics.fy * (cam[1] / z) + intrinsics.cy
  ];
}

function projectWithParams(worldPoint, params, intrinsics) {
  const Rwc = rodriguesFromVector([params[0], params[1], params[2]]);
  const twc = [params[3], params[4], params[5]];
  return projectWorldToPixel(worldPoint, Rwc, twc, intrinsics);
}

function solveExtrinsicsFromProjection(P, intrinsics) {
  const Kinv = [
    [1 / intrinsics.fx, 0, -intrinsics.cx / intrinsics.fx],
    [0, 1 / intrinsics.fy, -intrinsics.cy / intrinsics.fy],
    [0, 0, 1]
  ];

  const E = createMatrix(3, 4);
  for (let c = 0; c < 4; c += 1) {
    const col = [P[0][c], P[1][c], P[2][c]];
    const transformed = mat3Vec3Mul(Kinv, col);
    E[0][c] = transformed[0];
    E[1][c] = transformed[1];
    E[2][c] = transformed[2];
  }

  const M = createMatrix(3, 3);
  for (let r = 0; r < 3; r += 1) {
    for (let c = 0; c < 3; c += 1) {
      M[r][c] = E[r][c];
    }
  }

  const MT = transpose(M);
  const MTM = matMul(MT, M);
  const eig = jacobiEigenDecompositionSymmetric(MTM, 64);

  const order = [0, 1, 2].sort((a, b) => eig.eigenvalues[b] - eig.eigenvalues[a]);
  const V = createMatrix(3, 3);
  const sigma = [0, 0, 0];
  for (let col = 0; col < 3; col += 1) {
    const idx = order[col];
    sigma[col] = Math.sqrt(Math.max(eig.eigenvalues[idx], 1e-16));
    for (let row = 0; row < 3; row += 1) {
      V[row][col] = eig.eigenvectors[row][idx];
    }
  }

  const invSqrtDiag = createMatrix(3, 3);
  for (let i = 0; i < 3; i += 1) {
    invSqrtDiag[i][i] = 1 / Math.max(sigma[i], 1e-8);
  }
  const invSqrtMTM = matMul(matMul(V, invSqrtDiag), transpose(V));
  const Rwc = matMul(M, invSqrtMTM);

  const scale = (sigma[0] + sigma[1] + sigma[2]) / 3;
  if (scale < 1e-12) throw new Error("PnP failed: invalid projection scale.");
  const twc = [E[0][3] / scale, E[1][3] / scale, E[2][3] / scale];

  if (det3(Rwc) < 0) {
    for (let r = 0; r < 3; r += 1) {
      for (let c = 0; c < 3; c += 1) Rwc[r][c] *= -1;
    }
    twc[0] *= -1;
    twc[1] *= -1;
    twc[2] *= -1;
  }

  return { Rwc, twc };
}

function projectionFromCorrespondences(correspondences) {
  const normalized2d = normalizePoints2D(correspondences);
  const normalized3d = normalizePoints3D(normalized2d.normalized);
  const rows = normalized3d.normalized.length * 2;
  const A = createMatrix(rows, 11);
  const b = new Float64Array(rows);
  for (let i = 0; i < normalized3d.normalized.length; i += 1) {
    const { point2d, point3d } = normalized3d.normalized[i];
    const [u, v] = point2d;
    const [X, Y, Z] = point3d;
    const r0 = i * 2;
    const r1 = r0 + 1;

    A[r0][0] = X;
    A[r0][1] = Y;
    A[r0][2] = Z;
    A[r0][3] = 1;
    A[r0][8] = -u * X;
    A[r0][9] = -u * Y;
    A[r0][10] = -u * Z;
    b[r0] = u;

    A[r1][4] = X;
    A[r1][5] = Y;
    A[r1][6] = Z;
    A[r1][7] = 1;
    A[r1][8] = -v * X;
    A[r1][9] = -v * Y;
    A[r1][10] = -v * Z;
    b[r1] = v;
  }

  let q = solveLeastSquaresNormal(A, b);
  if (!q) {
    const fallbackA = buildDltMatrix(normalized3d.normalized);
    const At = transpose(fallbackA);
    const AtA = matMul(At, fallbackA);
    const p = extractSmallestEigenvector(AtA);
    q = Array.from(p.slice(0, 11));
  }

  const Pn = createMatrix(3, 4);
  const p = [...q, 1];
  let idx = 0;
  for (let r = 0; r < 3; r += 1) {
    for (let c = 0; c < 4; c += 1) {
      Pn[r][c] = p[idx];
      idx += 1;
    }
  }

  const Tinv = inverse3x3(normalized2d.T);
  const P = matMul(matMul(Tinv, Pn), normalized3d.U);
  return P;
}

function computeReprojectionError(correspondences, Rwc, twc, intrinsics) {
  let sumSq = 0;
  let count = 0;
  for (const c of correspondences) {
    const projected = projectWorldToPixel(c.point3d, Rwc, twc, intrinsics);
    if (!projected) continue;
    const du = projected[0] - c.point2d[0];
    const dv = projected[1] - c.point2d[1];
    sumSq += du * du + dv * dv;
    count += 1;
  }
  if (count === 0) return Number.POSITIVE_INFINITY;
  return Math.sqrt(sumSq / count);
}

function computeReprojectionErrorFromParams(correspondences, params, intrinsics) {
  let sumSq = 0;
  let count = 0;
  for (const c of correspondences) {
    const projected = projectWithParams(c.point3d, params, intrinsics);
    if (!projected) continue;
    const du = projected[0] - c.point2d[0];
    const dv = projected[1] - c.point2d[1];
    sumSq += du * du + dv * dv;
    count += 1;
  }
  if (count === 0) return Number.POSITIVE_INFINITY;
  return Math.sqrt(sumSq / count);
}

function refineExtrinsics(correspondences, initialRwc, initialTwc, intrinsics, options = {}) {
  let params = [
    ...rotationVectorFromMatrix(initialRwc),
    initialTwc[0],
    initialTwc[1],
    initialTwc[2]
  ];

  const maxIters = Number.isFinite(options.maxIters) ? Math.max(1, Math.floor(options.maxIters)) : 18;
  let lambda = Number.isFinite(options.lambda) ? Math.max(1e-9, options.lambda) : 1e-3;
  let bestError = computeReprojectionErrorFromParams(correspondences, params, intrinsics);
  // Forward-difference steps: fixed for the rotation vector (radians), and
  // relative to the scene scale for translation so the solver is unit-agnostic.
  const sceneScale = Math.max(1e-6, Math.hypot(initialTwc[0], initialTwc[1], initialTwc[2]));
  const tEps = 1e-6 * sceneScale;
  const eps = [1e-6, 1e-6, 1e-6, tEps, tEps, tEps];

  for (let iter = 0; iter < maxIters; iter += 1) {
    const residuals = new Float64Array(correspondences.length * 2);
    const J = createMatrix(correspondences.length * 2, 6);

    for (let i = 0; i < correspondences.length; i += 1) {
      const row = i * 2;
      const c = correspondences[i];
      const baseProj = projectWithParams(c.point3d, params, intrinsics);
      if (!baseProj) {
        residuals[row] = 0;
        residuals[row + 1] = 0;
        continue;
      }

      residuals[row] = c.point2d[0] - baseProj[0];
      residuals[row + 1] = c.point2d[1] - baseProj[1];

      for (let k = 0; k < 6; k += 1) {
        const perturbed = params.slice();
        perturbed[k] += eps[k];
        const perturbedProj = projectWithParams(c.point3d, perturbed, intrinsics);
        if (!perturbedProj) {
          J[row][k] = 0;
          J[row + 1][k] = 0;
          continue;
        }
        // J = d(projection)/d(params). With residual r = observed - projected,
        // the Gauss-Newton step solves (J^T J + lambda I) delta = J^T r.
        J[row][k] = (perturbedProj[0] - baseProj[0]) / eps[k];
        J[row + 1][k] = (perturbedProj[1] - baseProj[1]) / eps[k];
      }
    }

    const JT = transpose(J);
    const JTJ = matMul(JT, J);
    const JTr = new Float64Array(6);
    for (let r = 0; r < 6; r += 1) {
      let sum = 0;
      for (let i = 0; i < residuals.length; i += 1) {
        sum += JT[r][i] * residuals[i];
      }
      JTr[r] = sum;
    }

    for (let i = 0; i < 6; i += 1) {
      JTJ[i][i] += lambda;
    }

    const delta = solveLinearSystem(JTJ, JTr);
    if (!delta) break;

    const candidate = params.slice();
    for (let i = 0; i < 6; i += 1) candidate[i] += delta[i];

    const candidateError = computeReprojectionErrorFromParams(correspondences, candidate, intrinsics);
    if (candidateError < bestError) {
      params = candidate;
      bestError = candidateError;
      lambda *= 0.5;
      if (Math.hypot(...delta) < 1e-7) break;
    } else {
      lambda *= 2;
    }
  }

  return {
    Rwc: rodriguesFromVector([params[0], params[1], params[2]]),
    twc: [params[3], params[4], params[5]],
    error: bestError
  };
}

/**
 * Solve camera pose from at least six 2D-3D correspondences.
 *
 * @param {Array<{point2d:[number,number], point3d:[number,number,number]}>} correspondences
 *   Pixel coordinates and world coordinates. If every |point2d| <= 2 the 2D
 *   points are treated as normalised [0,1] image coordinates and scaled by
 *   the intrinsics' width/height.
 * @param {object} [options]
 * @param {object} [options.intrinsics] Partial override of fx, fy, cx, cy, width, height.
 * @param {number} [options.width] Used by buildDefaultIntrinsics when intrinsics are not given.
 * @param {number} [options.height]
 * @param {number} [options.fovDeg]
 * @param {false|{maxIters?: number, lambda?: number}} [options.refine]
 *   `false` skips Levenberg-Marquardt and returns the DLT initialisation.
 * @returns {{
 *   R: number[][], t: number[], error: number, pointCount: number,
 *   intrinsics: object, worldToCamera: {R: number[][], t: number[]}
 * }} `error` is the RMS reprojection error in pixels.
 * @throws {Error} when fewer than six valid correspondences are supplied or
 *   the geometry is degenerate.
 */
export function solvePnP(correspondences, options = {}) {
  const intrinsics = {
    ...buildDefaultIntrinsics(options),
    ...options.intrinsics
  };

  const normalized = normalizeCorrespondences(correspondences, intrinsics);
  const P = projectionFromCorrespondences(normalized);
  const initial = solveExtrinsicsFromProjection(P, intrinsics);
  const shouldRefine = options.refine !== false;
  const refined = shouldRefine
    ? refineExtrinsics(normalized, initial.Rwc, initial.twc, intrinsics, options.refine || {})
    : null;
  const Rwc = refined?.Rwc || initial.Rwc;
  const twc = refined?.twc || initial.twc;
  const reprojectionErrorPx = Number.isFinite(refined?.error)
    ? refined.error
    : computeReprojectionError(normalized, Rwc, twc, intrinsics);

  // Cheirality: a mirrored solution reprojects perfectly but places the
  // scene behind the camera. This mostly happens with coplanar inputs, where
  // the 12-parameter DLT is rank deficient. Refuse rather than return it.
  let behind = 0;
  for (const c of normalized) {
    const cam = mat3Vec3Mul(Rwc, c.point3d);
    if (cam[2] + twc[2] <= 0) behind += 1;
  }
  if (behind * 2 > normalized.length) {
    throw new Error(
      `PnP failed cheirality check: ${behind}/${normalized.length} points lie behind the camera `
      + "(coplanar or otherwise degenerate correspondences?)."
    );
  }

  const R = transpose3(Rwc);
  const negTwc = [-twc[0], -twc[1], -twc[2]];
  const t = mat3Vec3Mul(R, negTwc);

  return {
    R,
    t,
    error: reprojectionErrorPx,
    pointCount: normalized.length,
    intrinsics,
    worldToCamera: {
      R: Rwc,
      t: twc
    }
  };
}

/**
 * Project a world point through a solved pose. Returns null when the point
 * lies on the camera plane (z ~ 0).
 * @param {[number,number,number]} point3d
 * @param {{worldToCamera:{R:number[][], t:number[]}}} pose Result of solvePnP.
 * @param {object} [intrinsics]
 * @returns {[number,number]|null} Pixel coordinates.
 */
export function projectPoint(point3d, pose, intrinsics = DEFAULT_CAMERA_INTRINSICS) {
  const Rwc = pose?.worldToCamera?.R;
  const twc = pose?.worldToCamera?.t;
  if (!Array.isArray(Rwc) || !Array.isArray(twc)) return null;
  return projectWorldToPixel(point3d, Rwc, twc, intrinsics);
}

export { rodriguesFromVector, rotationVectorFromMatrix };
