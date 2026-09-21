# pnp-js

A Perspective-n-Point camera pose solver in plain JavaScript, with no runtime
dependencies. Give it six or more 2D-3D correspondences and pinhole intrinsics
and it returns the camera rotation and translation.

It was extracted from a multi-camera capture rig, where a browser-based
calibration tool needed to register one of the cameras against a 3D scan. The
tool ran entirely client-side and pulling in OpenCV.js (about 8 MB of WASM) for
one function was not worth it, so the solver was written by hand.

## Usage

```js
import { solvePnP, buildDefaultIntrinsics, projectPoint } from "pnp-js";

const intrinsics = buildDefaultIntrinsics({ width: 1280, height: 960, fovDeg: 90 });

const pose = solvePnP(
  [
    { point2d: [612.4, 388.1], point3d: [0.12, -0.05, 1.40] },
    // ... at least six, not all on one plane
  ],
  { intrinsics }
);

pose.worldToCamera.R; // 3x3, maps world -> camera
pose.worldToCamera.t; // [x, y, z]
pose.R;               // camera -> world rotation
pose.t;               // camera centre in world coordinates
pose.error;           // RMS reprojection error in pixels

projectPoint([0.12, -0.05, 1.40], pose, intrinsics); // -> [612.4, 388.1]
```

`intrinsics` can be a full `{ fx, fy, cx, cy, width, height }` object or a
partial override of the defaults. If every 2D coordinate has magnitude <= 2 the
points are treated as normalised `[0, 1]` image coordinates and scaled by
`width`/`height`. Pass `refine: false` to get the closed-form initialisation
only, or `refine: { maxIters, lambda }` to tune the refinement.

Conventions: camera looks down +Z, image x right, image y down. Rotation
matrices are plain nested arrays.

## Algorithm

1. **Normalisation.** 2D points are translated to their centroid and scaled to
   mean distance sqrt(2); 3D points likewise to sqrt(3) (Hartley normalisation),
   so the linear system below is well conditioned regardless of units.
2. **DLT initialisation.** The 3x4 projection matrix is solved linearly by
   fixing P34 = 1 and solving the 11-unknown normal equations with partial
   pivoting. If that system is singular, it falls back to the homogeneous
   form and takes the smallest eigenvector of AᵀA using a hand-rolled Jacobi
   eigendecomposition.
3. **Extrinsics from P.** K⁻¹P gives [M | m]. M is orthogonalised into a
   proper rotation via M (MᵀM)^(-1/2) (again via Jacobi), the scale is taken
   from the singular values, and the determinant sign is fixed.
4. **Levenberg-Marquardt refinement.** The pose is reparameterised as a
   Rodrigues rotation vector plus translation (6 DoF). Each iteration builds a
   forward-difference Jacobian of the pixel residuals, solves the damped normal
   equations (JᵀJ + λI)δ = Jᵀr, and accepts the step only if the RMS
   reprojection error drops, halving λ on success and doubling it on failure.
5. **Cheirality check.** If most points end up behind the camera the solver
   throws rather than return a mirrored pose.

Rodrigues conversions in both directions are included, with the near-180°
case handled via the (R + I)/2 column trick.

## Accuracy on the synthetic tests

`npm test` builds random scenes (random rotation up to 120°, translation in a
±1 cube, 6-40 points at depths 0.4-3.0), projects them through a 90° FOV
1280x960 camera, adds Gaussian pixel noise, and recovers the pose.

| Scenario                          | Rotation error | Translation error | Reprojection RMS |
| --------------------------------- | -------------- | ----------------- | ---------------- |
| Noise-free, 30 points             | < 1e-4°        | < 1e-5 units      | < 1e-6 px        |
| Noise-free, minimal 6 points      | < 1e-3°        | < 1e-4 units      | -                |
| 0.5 px noise, 30 points           | < 0.25°        | < 0.01 units      | < 1 px           |
| 1 px noise, 40 points, 40 scenes  | worst 0.10°    | worst 0.003 units | worst 1.58 px    |

For reference, isotropic 1 px noise gives an irreducible per-point RMS of about
1.41 px, so the refinement is converging to the noise floor. The LM step
typically takes the DLT initialisation from several pixels of error down to
that floor (6.3 px to 2.2 px in the 2 px-noise test).

Translation units are whatever the 3D points are in; the solver is
unit-agnostic.

## Limitations

- **Coplanar points are not supported.** The 12-parameter DLT is rank
  deficient for a planar target (a checkerboard, say). The solver will throw a
  cheirality error or return a pose with a large reprojection error, but it
  will not recover the right answer. A homography-based initialisation would
  fix this and is the obvious next step.
- Needs at least 6 correspondences; there is no P3P/EPnP minimal solver and
  no RANSAC, so outliers must be removed beforehand.
- No lens distortion model. Undistort pixels first.
- Jacobians are numerical (forward differences), which is fine at this size
  (6 parameters, tens of points) but not the fastest option.
- The Jacobi eigensolver is O(n³) per sweep and only used on 3x3 and 12x12
  matrices; it is not a general-purpose linear-algebra library.
- The `<= 2` heuristic for detecting normalised coordinates will misfire on a
  genuinely tiny image.

## Development

```
npm test
```

Node 20 or later; uses the built-in `node --test` runner. No build step.

## License

MIT
