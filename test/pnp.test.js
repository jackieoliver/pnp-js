import { test } from "node:test";
import assert from "node:assert/strict";
import {
  solvePnP,
  buildDefaultIntrinsics,
  projectPoint,
  rodriguesFromVector,
  rotationVectorFromMatrix
} from "../src/pnp.js";
import { makeScene, rotationErrorDeg, norm, sub, makeRng } from "./helpers.js";

const K = buildDefaultIntrinsics({ width: 1280, height: 960, fovDeg: 90 });

function check(scene, result, { rotDeg, transAbs }) {
  const rotErr = rotationErrorDeg(scene.Rwc, result.worldToCamera.R);
  const tErr = norm(sub(scene.twc, result.worldToCamera.t));
  assert.ok(rotErr < rotDeg, `rotation error ${rotErr.toFixed(4)} deg >= ${rotDeg}`);
  assert.ok(tErr < transAbs, `translation error ${tErr.toFixed(5)} >= ${transAbs}`);
  return { rotErr, tErr };
}

test("recovers an exact pose from noise-free correspondences", () => {
  const scene = makeScene({ seed: 7, count: 30, intrinsics: K });
  const result = solvePnP(scene.correspondences, { intrinsics: K });
  check(scene, result, { rotDeg: 1e-4, transAbs: 1e-5 });
  assert.ok(result.error < 1e-6, `reprojection ${result.error}`);
  assert.equal(result.pointCount, 30);
});

test("recovers pose from the minimal six correspondences", () => {
  const scene = makeScene({ seed: 11, count: 6, intrinsics: K });
  const result = solvePnP(scene.correspondences, { intrinsics: K });
  check(scene, result, { rotDeg: 1e-3, transAbs: 1e-4 });
});

test("recovers pose under 0.5 px Gaussian pixel noise (30 points)", () => {
  const scene = makeScene({ seed: 3, count: 30, noisePx: 0.5, intrinsics: K });
  const result = solvePnP(scene.correspondences, { intrinsics: K });
  check(scene, result, { rotDeg: 0.25, transAbs: 0.01 });
  assert.ok(result.error < 1.0, `reprojection RMS ${result.error} px`);
});

test("stays within tolerance across 40 random scenes with 1 px noise", () => {
  let worstRot = 0;
  let worstT = 0;
  let worstReproj = 0;
  for (let seed = 100; seed < 140; seed += 1) {
    const scene = makeScene({ seed, count: 40, noisePx: 1.0, intrinsics: K });
    const result = solvePnP(scene.correspondences, { intrinsics: K });
    const { rotErr, tErr } = check(scene, result, { rotDeg: 0.5, transAbs: 0.02 });
    worstRot = Math.max(worstRot, rotErr);
    worstT = Math.max(worstT, tErr);
    worstReproj = Math.max(worstReproj, result.error);
  }
  // `error` is the per-point Euclidean RMS, so isotropic 1 px noise gives
  // about sqrt(2) = 1.41 px. Allow 20% headroom.
  assert.ok(worstReproj < 1.7, `worst reprojection RMS ${worstReproj} px`);
  console.log(`  worst over 40 scenes: rot ${worstRot.toFixed(4)} deg, t ${worstT.toFixed(5)}, reproj ${worstReproj.toFixed(3)} px`);
});

test("works with the default wide-angle intrinsics (155 deg FOV)", () => {
  const wide = buildDefaultIntrinsics();
  const scene = makeScene({ seed: 21, count: 40, noisePx: 0.5, intrinsics: wide });
  const result = solvePnP(scene.correspondences);
  check(scene, result, { rotDeg: 0.3, transAbs: 0.01 });
  assert.equal(result.intrinsics.width, 1920);
});

test("Levenberg-Marquardt refinement improves on the DLT initialisation", () => {
  const scene = makeScene({ seed: 5, count: 20, noisePx: 2.0, intrinsics: K });
  const init = solvePnP(scene.correspondences, { intrinsics: K, refine: false });
  const refined = solvePnP(scene.correspondences, { intrinsics: K });
  assert.ok(Number.isFinite(init.error));
  assert.ok(refined.error < init.error, `refined ${refined.error} >= init ${init.error}`);
  assert.ok(
    rotationErrorDeg(scene.Rwc, refined.worldToCamera.R) < rotationErrorDeg(scene.Rwc, init.worldToCamera.R)
  );
  console.log(`  DLT init ${init.error.toFixed(3)} px -> LM ${refined.error.toFixed(3)} px`);
});

test("accepts normalised [0,1] image coordinates", () => {
  const scene = makeScene({ seed: 9, count: 25, intrinsics: K });
  const normalised = scene.correspondences.map((c) => ({
    point3d: c.point3d,
    point2d: [c.point2d[0] / K.width, c.point2d[1] / K.height]
  }));
  const a = solvePnP(scene.correspondences, { intrinsics: K });
  const b = solvePnP(normalised, { intrinsics: K });
  assert.ok(rotationErrorDeg(a.worldToCamera.R, b.worldToCamera.R) < 1e-6);
  assert.ok(norm(sub(a.worldToCamera.t, b.worldToCamera.t)) < 1e-6);
});

test("camera-to-world output is the inverse of world-to-camera", () => {
  const scene = makeScene({ seed: 13, count: 20, intrinsics: K });
  const result = solvePnP(scene.correspondences, { intrinsics: K });
  assert.ok(norm(sub(result.t, scene.centre)) < 1e-5, "t should be the camera centre");
  assert.ok(rotationErrorDeg(result.R, scene.Rcw) < 1e-4);
});

test("projectPoint reproduces the input pixels", () => {
  const scene = makeScene({ seed: 17, count: 12, intrinsics: K });
  const result = solvePnP(scene.correspondences, { intrinsics: K });
  for (const c of scene.correspondences) {
    const p = projectPoint(c.point3d, result, K);
    assert.ok(norm(sub(p, c.point2d)) < 1e-4);
  }
  assert.equal(projectPoint([0, 0, 1], {}), null);
});

test("skips malformed entries and throws below six valid correspondences", () => {
  const scene = makeScene({ seed: 2, count: 8, intrinsics: K });
  const dirty = [
    null,
    { point2d: [1, 2] },
    { point2d: ["x", 1], point3d: [0, 0, 1] },
    ...scene.correspondences
  ];
  assert.equal(solvePnP(dirty, { intrinsics: K }).pointCount, 8);
  assert.throws(
    () => solvePnP(scene.correspondences.slice(0, 5), { intrinsics: K }),
    /at least 6/
  );
  assert.throws(() => solvePnP([], { intrinsics: K }), /at least 6/);
});

test("degenerate: collinear world points cannot determine a pose", () => {
  // All 3D points on one line -> DLT system is rank deficient.
  const rng = makeRng(42);
  const correspondences = [];
  for (let i = 0; i < 10; i += 1) {
    const s = i * 0.1;
    correspondences.push({ point3d: [s, 2 * s, 1 + s], point2d: [rng.uniform() * 1000, rng.uniform() * 800] });
  }
  let threw = false;
  let result = null;
  try {
    result = solvePnP(correspondences, { intrinsics: K });
  } catch (err) {
    threw = true;
    assert.match(err.message, /degenerate|singular|failed/i);
  }
  // Either an explicit failure or a result whose reprojection error exposes it.
  assert.ok(threw || !Number.isFinite(result.error) || result.error > 10,
    `collinear input produced a suspiciously confident pose (error ${result?.error})`);
});

test("degenerate: coplanar world points never produce a silently wrong pose", () => {
  // A planar target (e.g. a checkerboard) makes the 12-parameter DLT rank
  // deficient, so this solver does not support it (see README). What it must
  // do is fail loudly: throw, report a large reprojection error, or -- if it
  // happens to land on the right basin -- be accurate.
  let outcomes = { accurate: 0, threw: 0, highError: 0 };
  for (let seed = 30; seed < 40; seed += 1) {
    const scene = makeScene({ seed, count: 30, noisePx: 0.3, intrinsics: K, planar: true, maxAngleDeg: 40 });
    let result;
    try {
      result = solvePnP(scene.correspondences, { intrinsics: K });
    } catch (err) {
      assert.match(err.message, /cheirality|degenerate|singular|failed/i);
      outcomes.threw += 1;
      continue;
    }
    const rotErr = rotationErrorDeg(scene.Rwc, result.worldToCamera.R);
    if (rotErr < 0.5 && norm(sub(scene.twc, result.worldToCamera.t)) < 0.02) {
      outcomes.accurate += 1;
    } else {
      assert.ok(result.error > 10, `seed ${seed}: wrong pose (rot ${rotErr.toFixed(1)} deg) with low error ${result.error}`);
      outcomes.highError += 1;
    }
  }
  console.log(`  planar outcomes: ${JSON.stringify(outcomes)}`);
});

test("Rodrigues round-trips, including near 180 degrees", () => {
  const cases = [
    [0, 0, 0],
    [0.1, -0.2, 0.3],
    [1.2, 0.4, -0.9],
    [Math.PI - 1e-4, 0, 0],
    [0, (Math.PI - 1e-4) / Math.SQRT2, (Math.PI - 1e-4) / Math.SQRT2],
    // Exactly pi with a mixed-sign axis exercises the sin(theta) ~ 0 branch.
    [0.6 * Math.PI, -0.8 * Math.PI, 0],
    [-Math.PI / Math.sqrt(3), Math.PI / Math.sqrt(3), -Math.PI / Math.sqrt(3)]
  ];
  for (const w of cases) {
    const R = rodriguesFromVector(w);
    const back = rodriguesFromVector(rotationVectorFromMatrix(R));
    assert.ok(rotationErrorDeg(R, back) < 1e-3, `round trip failed for ${w}`);
  }
});
