# Native runtime setup and verification

Use CPython **3.10 or 3.11** for the mirror. The pinned MediaPipe 0.10.14
provides these wheels and the application uses its `mp.solutions` API; installing
the latest MediaPipe or using Python 3.13 is not a compatible substitute.
The container uses Python 3.11. On macOS, PySide6 6.9.1 requires macOS 12 or later.
See the [MediaPipe release files](https://pypi.org/project/mediapipe/0.10.14/#files)
and [PySide6 release files](https://pypi.org/project/PySide6/6.9.1/#files).

## Install in a fresh environment

Run these commands from the repository root (use `python3.10` instead if needed):

```bash
python3.11 -m venv .venv-mirror
source .venv-mirror/bin/activate
python -m pip install --upgrade pip wheel 'setuptools<82'
python -m pip install -r requirements-dev.txt
python -m pip check
```

On Windows, activate with `.venv-mirror\Scripts\Activate.ps1` in PowerShell.
For mirror runtime only, install `Mirror_Code/requirements.txt` instead.
For the optional API only, install `gym_backend/requirements.txt`.
The root requirements include both runtimes; `requirements-dev.txt` adds pytest.
Use a fresh environment so old `opencv-python` or `opencv-python-headless` wheels
do not share the same `cv2` import with `opencv-contrib-python`.

Native `dlib` installation needs CMake and a C++ compiler: Xcode command line
tools plus CMake on macOS, `build-essential` and `cmake` on Debian/Ubuntu, or
Visual Studio C++ build tools plus CMake on Windows. Linux also needs the Qt/X11,
OpenGL and PortAudio libraries listed in `docker/mirror/Dockerfile`.
The face model package still imports `pkg_resources`, which
[setuptools removed in version 82](https://setuptools.pypa.io/en/latest/deprecated/pkg_resources.html),
so the runtime constrains setuptools below that release.

Audio feedback is optional. Install `pyttsx3` separately and provide its native
speech engine if needed; the mirror continues without it. Audio remains disabled
by default on macOS and in the Docker bundle.

## Configure and run the mirror

The mirror requires an initialized PostgreSQL database and a working camera.
Its normal runtime does not fall back to SQLite. Set credentials explicitly:

```bash
export SMART_MIRROR_PG_DB=gym_main_db
export SMART_MIRROR_PG_USER=smartgym
export SMART_MIRROR_PG_PASSWORD='your-database-password'
export SMART_MIRROR_PG_HOST=127.0.0.1
export SMART_MIRROR_PG_PORT=5432
export SMART_GYM_SERVER_SYNC_ENABLED=0
python Mirror_Code/app.py
```

`scripts/setup_local_postgres.sh` provisions a local main database and optional
server database. Set `APP_DB_USER`, `APP_DB_PASSWORD`, and the PostgreSQL bootstrap
connection before running it. It changes application role credentials and applies
`infra/postgres/main_schema.sql`; run it only against the intended development
database. The Docker bundle initializes its own PostgreSQL volume automatically.

The optional Django API has separate credentials and provisioning requirements;
follow the backend setup documentation before enabling mirror sync. An API server
is not required for mirror-only operation.

Application state (face data, settings, logs) lives under the platform user data
directory. Set `SMART_MIRROR_STATE_DIR` to choose a writable location. Back up that
directory together with PostgreSQL. Docker stores state in the named
`smart-gym-mirror-state` volume and database data in `smart-gym-postgres`.

## Automated checks

```bash
PYTHONPATH=Mirror_Code QT_QPA_PLATFORM=offscreen python -m pytest Mirror_Code/tests -q
python -m compileall -q Mirror_Code gym_backend
python scripts/smoke_ai.py
docker compose -f docker-compose.centralized.yml --profile mirror config --quiet
```

The native smoke test runs the actual ArUco, face, and pose libraries on synthetic
frames without opening a camera or connecting to a database. On macOS, MediaPipe
requires a working graphics context even for its CPU pose graph; run outside a
restricted graphics sandbox. It verifies library integration, not model accuracy.

For rendered checkerboard calibration, metric stereo reconstruction, and known
3D movement benchmarks, run `python scripts/simulate_3d_calibration.py`. See
[3d_calibration_simulation.md](3d_calibration_simulation.md) for measured results,
quality gates, repeatable failure cases, and the distinction from live pose accuracy.

The database tests may use isolated test substitutes; passing them does not prove
a live PostgreSQL deployment, physical camera, display passthrough, or real-person
recognition accuracy. Validate those with the intended hardware and representative
recorded movements before operating unattended. Synthetic pose tests establish
geometry and counting behavior, not a clinical assessment of lifting safety.

The optional backend Docker entrypoint uses Gunicorn and migrations. Its source
is excluded from the current mirror-only `.dockerignore` bundle; to package the
API separately, use a build context that includes the backend and its Docker files.
