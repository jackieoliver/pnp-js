import { rodriguesFromVector } from "../src/pnp.js";

/** Deterministic PRNG (mulberry32) so failures are reproducible. */
export function makeRng(seed) {
  let a = seed >>> 0;
  const uniform = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const gaussian = () => {
    const u = Math.max(uniform(), 1e-12);
    const v = uniform();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  return { uniform, gaussian };
}

export function matVec(M, v) {
  return [
    M[0][0] * v[0] + M[0][1] * v[1] + M[0][2] * v[2],
    M[1][0] * v[0] + M[1][1] * v[1] + M[1][2] * v[2],
    M[2][0] * v[0] + M[2][1] * v[1] + M[2][2] * v[2]
  ];
}

export function transpose(M) {
  return [
    [M[0][0], M[1][0], M[2][0]],
    [M[0][1], M[1][1], M[2][1]],
    [M[0][2], M[1][2], M[2][2]]
  ];
}

export function matMul(A, B) {
  const out = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let r = 0; r < 3; r += 1) {
    for (let c = 0; c < 3; c += 1) {
      out[r][c] = A[r][0] * B[0][c] + A[r][1] * B[1][c] + A[r][2] * B[2][c];
    }
  }
  return out;
}

/** Angle in degrees between two rotation matrices. */
export function rotationErrorDeg(Ra, Rb) {
  const D = matMul(transpose(Ra), Rb);
  const cos = Math.max(-1, Math.min(1, (D[0][0] + D[1][1] + D[2][2] - 1) / 2));
  return (Math.acos(cos) * 180) / Math.PI;
}

export function norm(v) {
  return Math.hypot(...v);
}

export function sub(a, b) {
  return a.map((x, i) => x - b[i]);
}

export function projectPinhole(X, Rwc, twc, K) {
  const cam = matVec(Rwc, X);
  const x = cam[0] + twc[0];
  const y = cam[1] + twc[1];
  const z = cam[2] + twc[2];
  return [K.fx * (x / z) + K.cx, K.fy * (y / z) + K.cy];
}

/**
 * Build a synthetic scene: a random camera pose looking roughly at a cloud of
 * world points, projected through `intrinsics` with optional pixel noise.
 * Points are generated in camera space (guaranteed in front of the camera and
 * inside the image) and then mapped back to world space with the inverse pose.
 */
export function makeScene({
  seed = 1,
  count = 30,
  noisePx = 0,
  intrinsics,
  maxAngleDeg = 120,
  depthRange = [0.4, 3.0],
  planar = false
}) {
  const rng = makeRng(seed);
  const K = intrinsics;

  // Random world-to-camera rotation via a random axis-angle.
  const axis = [rng.gaussian(), rng.gaussian(), rng.gaussian()];
  const an = norm(axis);
  const angle = ((rng.uniform() * maxAngleDeg) * Math.PI) / 180;
  const Rwc = rodriguesFromVector(axis.map((a) => (a / an) * angle));
  const twc = [rng.uniform() * 2 - 1, rng.uniform() * 2 - 1, rng.uniform() * 2 - 1];
  const Rcw = transpose(Rwc);

  const correspondences = [];
  const planeDepth = depthRange[0] + rng.uniform() * (depthRange[1] - depthRange[0]);
  for (let i = 0; i < count; i += 1) {
    // Sample a pixel inside the central 80% of the image and a depth.
    const u = K.width * (0.1 + 0.8 * rng.uniform());
    const v = K.height * (0.1 + 0.8 * rng.uniform());
    const z = planar ? planeDepth : depthRange[0] + rng.uniform() * (depthRange[1] - depthRange[0]);
    const cam = [((u - K.cx) / K.fx) * z, ((v - K.cy) / K.fy) * z, z];
    const world = matVec(Rcw, sub(cam, twc));
    const clean = projectPinhole(world, Rwc, twc, K);
    correspondences.push({
      point3d: world,
      point2d: [clean[0] + noisePx * rng.gaussian(), clean[1] + noisePx * rng.gaussian()]
    });
  }

  return { correspondences, Rwc, twc, Rcw, centre: matVec(Rcw, twc.map((x) => -x)) };
}
