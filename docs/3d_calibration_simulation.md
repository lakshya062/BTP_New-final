# Internal 3D calibration simulation

The repository now has a reproducible camera-calibration and 3D-reconstruction
simulation. It runs without a physical camera, database, network, Blender, or a
human video. OpenCV and NumPy provide the renderer and solvers. The existing live
exercise analyzer still uses 2D joint angles and MediaPipe relative depth; running
this simulation does **not** enable calibrated 3D tracking in the live app.

## Run it

With the Python runtime from [runtime_setup.md](runtime_setup.md), run from the
repository root:

```bash
python scripts/simulate_3d_calibration.py
```

The program writes these generated files under `artifacts/calibration_simulation/`
and exits nonzero if a required quality check fails:

- `report.json`: fitted cameras, stereo transforms, held-out errors, noise
  benchmarks, and each quality gate's outcome.
- `detected_checkerboards.png`: actual raster checkerboard images with the corner
  detector's measured points overlaid.
- `calibration_results.png`: camera-angle bias, metric stereo error, and comparison
  of 35 cm and 70 cm camera spacing.

Use `--output-dir PATH`, `--seed INTEGER`, `--views INTEGER`, or `--no-plots` to
change the output location, repeat a different generated capture, adjust the
number of mono calibration views, or run without Matplotlib. The default is 48
mono board views; six are held out entirely from calibration. Generated artifacts
are ignored by Git. The core simulation needs only the existing OpenCV/NumPy
dependencies; plotting uses Matplotlib, already present with the mirror runtime.

## What is tested

1. A known 960×720 virtual camera sees a physical 9×6-inner-corner checkerboard
   with 3 cm squares, varied 3D positions/tilts, radial/tangential lens distortion,
   antialiasing, blur, and image noise.
2. OpenCV detects corners from the rendered image pixels. Calibration receives
   those measured pixels, not the renderer's exact projected coordinates.
3. Focal lengths, principal point, and five distortion coefficients are fitted
   from 42 images. Six unseen board poses measure recovered camera-ray accuracy
   against known 3D geometry, without refitting the held-out board poses.
4. Eighteen simultaneous rendered board pairs estimate the stereo cameras'
   relative rotation and translation. A second rig repeats the experiment with
   narrower camera spacing. Both identical virtual cameras share recovered
   intrinsics; their relative transform is estimated from measured raster corners.
5. Known 3D arm movements spanning 35–165° are projected into both cameras at a
   3 m subject distance. Four viewing yaws and four landmark-noise levels test
   triangulated positions and 3D joint angles against ground truth.
6. Invalid inputs, repeated/front-facing captures, folded distortion fits,
   excessive per-view errors, zero baselines, almost parallel rays, mismatched
   point pairs, and points behind cameras are rejected by regression tests.

The renderer uses known corner IDs to resolve the symmetry of an unmarked
checkerboard. This only reorders measured pixels. Real stereo capture needs
consistently matched IDs, for example a marked target, and synchronized frames.

## Measured results

Default seed 42, 48 mono views, existing OpenCV 4.13 runtime:

| Measurement | Result |
| --- | --- |
| Raster boards detected | 48 / 48 |
| Calibration / held-out images | 42 / 6 |
| Training reprojection RMS | 0.0584 px |
| Focal length error, x / y | 0.195% / 0.198% |
| Held-out ray error, p95 | 0.975 px |
| Stereo translation error | 1.263 mm |
| Stereo rotation error | 0.0470° |
| Worst mean 3D position error across tested yaws, 0.5 px landmark noise, 70 cm rig | 13.6 mm |
| Worst mean 3D joint-angle error at the same conditions | 2.55° |
| Worst mean 3D joint-angle error with 35 cm spacing | 4.89° |

The fixed benchmark requires mean joint-angle error below 3° and mean position
error below 20 mm at 0.5 px landmark noise, separately for each tested yaw. The
70 cm rig passes; the 35 cm comparison fails the angle target. Both outcomes are
retained. At 1 px noise, some 70 cm cases also exceed 3°: the lower-noise result
must not be extrapolated to worse landmarks. Full mean/p95/max errors are in JSON.

Camera calibration cannot resolve a single camera's unobserved depth. The test
suite includes two different 3D arms that produce the same image pixels. At 80°
motion-plane yaw, the default clean single-camera angle has roughly 25° mean
error even after lens correction, while stereo's clean angle error is below 0.2°.

## Stress checks and failure evidence

Forty-eight-view calibration passed the unchanged 2 px held-out ray gate for six
seeds (0, 1, 2, 7, 42, 99). The p95 errors were 1.166, 0.621, 0.802, 1.436,
0.975, and 0.691 px. A separately generated fixed holdout also passed for all six.
Full default and seed 7 simulations passed all 11 quality gates.

A smaller capture remains a useful failure example:

```bash
python scripts/simulate_3d_calibration.py --seed 7 --views 24 --no-plots \
  --output-dir artifacts/calibration_sparse_failure
```

This run is expected to return nonzero with the tested OpenCV versions. Despite
0.054 px training RMS, its held-out ray error reaches 2.324 px at p95 and fails the
2 px gate. Increasing view count improved this case; blindly adding images or
using more extreme tilts did not improve every dataset. Keep held-out validation.
The original rejected report and the passing 48-view seed 7 report were preserved
as `rejected_sparse_capture.json` and `alternate_seed_report.json` alongside the
default report during this verification.

## Tests and implementation

```bash
PYTHONPATH=Mirror_Code python -m pytest \
  Mirror_Code/tests/test_camera_calibration.py \
  Mirror_Code/tests/test_synthetic_camera.py \
  Mirror_Code/tests/test_synthetic_pose.py \
  Mirror_Code/tests/test_calibration_simulation.py -q
```

All **43 new tests and 90 subtests** passed in both the existing Python 3.10 /
OpenCV 4.13 / NumPy 2.2 environment and a separate Python 3.11 environment with
the project's pinned **OpenCV 4.10.0.84 / NumPy 1.26.4**. The temporary pinned
environment also passed `pip check`; the existing environment was unchanged.

- `Mirror_Code/core/camera_calibration.py`: validated intrinsic/stereo calibration
  and triangulation, independent of the UI and pose model.
- `Mirror_Code/core/synthetic_camera.py`: deterministic distorted raster boards.
- `Mirror_Code/core/synthetic_pose.py`: known metric arm motion, projections,
  normalized camera rays, and 3D joint angles.
- `scripts/simulate_3d_calibration.py`: end-to-end report, plots, and quality gates.

The conventions and camera equations follow the official
[OpenCV calibration/reconstruction reference](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
and [camera-calibration tutorial](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html).

## Scope before a real deployment

These tests validate geometry, image-corner processing, and failure handling.
They do not measure MediaPipe's ability to locate real human joints. Synthetic
landmarks use known correspondences plus controlled Gaussian pixel noise; actual
pose-model errors, occlusion, rolling shutter, changing focus, exposure, and
camera synchronization can produce different and larger errors.

A physical installation still needs actual intrinsics at its intended resolution
and focus, matched stereo extrinsics if using two cameras, and accuracy checks
against measured distances and labeled human recordings. Existing per-camera
exercise workers are independent. An offline synchronized multi-camera matching,
occlusion, and local-ingestion pipeline is now available; see
[multicamera_simulation.md](multicamera_simulation.md). Connecting it to those live
workers and integrating metric 3D angles into exercise gates remain separate work.
