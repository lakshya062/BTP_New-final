# Codebase audit and verification — 18 October 2026

The mirror, Django backend, mobile services, dependency manifests, and deployment
scripts were reviewed and concrete defects were fixed. Existing local edits were
preserved. The root repository, `gym_backend/`, and `Gym_App/` are **three separate
Git repositories**: backend and mobile changes must be included separately when
saving or deploying this work. No changes were committed or deployed.

## AI and identity fixes

- Corrected aspect-ratio distortion in pose angles/body ratios, and crop-to-frame
  depth scaling. Overlay/tracking coordinates remain in full-frame normalized units.
- Rejected occluded, nonfinite, out-of-frame, and degenerate geometry instead of
  manufacturing usable zero-degree angles or border landmarks.
- Discarded incomplete reps after tracking/identity interruption and prevented reps
  assembled across different arms or body sides. Completed activity is retained.
- Required consecutive initial readiness, corrected lateral-raise angle ranges,
  and fixed partial-range feedback for exercises whose angles increase on lifting.
- Used monotonic rep timing and handled malformed numeric environment settings.
- Added an offline rendered-checkerboard and known-3D-motion calibration simulator,
  with intrinsic/stereo calibration, metric triangulation, held-out checks, and
  repeatable failure cases. This does not switch the live analyzer to 3D; see
  [3d_calibration_simulation.md](3d_calibration_simulation.md).
- Added an offline four-camera synchronization, multi-person association, robust
  per-joint 3D fusion, and separate durable local-ingestion pipeline. Tests cover
  complementary occlusion, ambiguous crossings, incorrect labels, camera and clock
  faults, three-person entry/exit, and combined stream failures. See
  [multicamera_simulation.md](multicamera_simulation.md); live integration remains
  explicit future work.
- Prevented a smoothed old face label or another person's accumulated observation
  time from starting the wrong member's session.
- Validated biometric samples and labels, supported Unicode names, bounded stored
  samples, skipped malformed HDF5 groups, and made model updates/deletion/legacy
  migration atomic. File operations coordinate application threads and recognizer
  instances; use a separate state directory for each mirror process.

## Persistence, API, and runtime fixes

- Added expiring signed member credentials, password/account revocation, history
  ownership checks, login throttling, and request validation. The mobile client
  persists the token, discards expired/corrupt/legacy sessions, and prevents token
  forwarding to another origin, including per-request Axios base URL overrides.
- Removed controller auto-registration with a test key and cross-gym reassignment.
  Ingest now verifies controller/mirror boundaries, record identity, and payload
  integrity; applies batches atomically; and permits safe retry after failure.
- Sync now requires an explicit `APPLIED` acknowledgement for the submitted batch
  before removing records from its retry queue. Redirects and empty 2xx responses
  cannot silently discard pending activity.
- Preserved trailing partial-set reps. A payload with one completed 12-rep set
  and 5 more reps stores all 17, with the trailing set marked in rep telemetry.
- Fixed database initialization recovery/concurrency and failed-sync timestamps.
- Fixed a synchronous Qt tab callback during initialization and joined background
  sync before closing its database/network resources. Shutdown can wait for the
  outstanding configured request timeout.
- Replaced conflicting dependency freezes with focused Python 3.10/3.11 manifests;
  fixed Docker native libraries, HDF5, OpenCV variants, and setuptools compatibility.
- Added explicit backend deployment settings, safer sync defaults, persistent
  Docker mirror state, logging recovery, and setup-script error handling. Django
  migration commands no longer silently provision PostgreSQL roles/databases.

## Verification completed

| Check | Result |
| --- | --- |
| Full native mirror regression suite | **319 passed**, plus **264 subtests**, no skips |
| Calibration and multi-camera tests in isolated pinned OpenCV 4.10 / NumPy 1.26 environment | **214 passed**, plus **200 subtests** |
| Full rendered calibration and metric 3D simulation | All **11** quality gates passed; narrow-rig and sparse-capture failures retained |
| Four-camera replay with estimated calibration | **45 scenario runs**, **810 bundles**, zero mixed-person groups or wrong member assignments |
| Real PostgreSQL fused-packet ingest | **15 scenarios**, **90 packets**; replay, rollback, reopen and concurrent-writer checks passed |
| Django suite on isolated real PostgreSQL 17 | **27 passed** |
| Mobile service regressions using actual TypeScript/Axios code | **24 passed** |
| TypeScript type checking | Passed |
| Production Expo web export | Passed, 689 modules; placeholder HTTPS API URL |
| Native ArUco, dlib face, and MediaPipe inference | Passed on synthetic frames |
| Real HTTP save → ingest → authenticated history pipeline | Passed; 17/17 reps retained |
| Anonymous/cross-member history access | Rejected with HTTP 401/403 |
| Repeat upload | No duplicate reps |
| Real Qt offscreen startup, populated dashboard, shutdown | Passed |
| Backend deployment checks with explicit HTTPS settings | Zero issues |
| Migration consistency | No changes detected |
| Fresh Python 3.11 dependency resolution | Passed; dry run in isolated environment |
| Python compilation, Compose config, shell syntax | Passed |
| Changed-file whitespace checks | Passed |

The 370 main automated tests (319 mirror, 27 backend, 24 mobile) are separate from the native and full-pipeline
smoke checks. Mirror tests run with the existing Python 3.10 native environment;
the 214 calibration/multi-camera tests also pass in a separate Python 3.11 environment with
pinned OpenCV/NumPy. The full proposed Python 3.11 dependency stack was resolved
without installing it over the existing environment. The isolated PostgreSQL cluster used for integration
tests was stopped cleanly; configured application databases and external APIs were
not modified. A pre-existing backend `.gitignore` EOF whitespace change was left intact.

## Run the checks

Follow [runtime_setup.md](runtime_setup.md) to install a clean runtime. Then, from
the root:

```bash
PYTHONPATH=Mirror_Code QT_QPA_PLATFORM=offscreen python -m pytest Mirror_Code/tests -q
python scripts/smoke_ai.py
python scripts/simulate_3d_calibration.py
python scripts/simulate_multicamera.py
python -m compileall -q Mirror_Code gym_backend scripts/smoke_ai.py
docker compose -f docker-compose.centralized.yml --profile mirror config --quiet
```

For backend tests, configure a **disposable** PostgreSQL database and a test role
with database-creation permission, then run `python manage.py test gym` from
`gym_backend/`. Run `npm ci`, `npm test`, and `npm run typecheck` from `Gym_App/`.
See each nested repository's README for deployment and authentication setup.

## Remaining validation and integration work

- Physical cameras, real-person recognition, varied lifting recordings, intended
  displays, native Android/iOS devices, Windows, and audio hardware were not tested.
  Synthetic regressions establish correctness of the implemented rules; they do
  not establish recognition accuracy or the safety of exercise-form advice. The
  heuristic form thresholds still need evaluation against representative labeled
  recordings. No model retraining or new accuracy benchmark is claimed.
- The new multi-camera pipeline is offline and uses simulated landmark visibility
  and identity evidence. The live UI still has independent camera workers, and
  fused packets are separate from exercise-history ingestion. Incorrect physical
  clock calibration can degrade moving joints even if reported timestamps agree.
- Docker configuration validates, but the local daemon is stopped/unavailable.
  Image builds and container startup remain unverified.
- The included backend implements member login/history and controller ingest. The
  mobile owner/signup/video flows reference endpoints that are not implemented in
  this backend. Implement and authenticate those workflows before enabling them
  in a deployment that needs them.
- Deploy the updated backend/client together. Existing tokenless sessions require
  login again. Configure a stable `GYM_SECRET_KEY`, database credentials, allowed
  hosts/origins, HTTPS, and a provisioned controller key before enabling server sync.
- The existing `gymstartup` environment still has unrelated torch/torchaudio package
  metadata conflicts. The repaired manifests exclude those unused packages; use a
  fresh environment rather than treating that legacy environment as a deployment
  lockfile.
