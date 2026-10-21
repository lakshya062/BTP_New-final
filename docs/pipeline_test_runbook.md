# Smart Gym Centralized Runtime Runbook

## Quick Start
Install the Python 3.10/3.11 runtime and configure PostgreSQL using
[runtime_setup.md](runtime_setup.md). The API is optional for mirror-only use.
For the full pipeline, use two terminals with the same activated environment.

Terminal A:
```bash
cd gym_backend
export GYM_DEBUG=1
# Set GYM_DB_* to your provisioned development database first.
python3 manage.py migrate
python3 manage.py runserver 0.0.0.0:8000
```

Terminal B:
```bash
cd Mirror_Code
python3 app.py
```

## What happens automatically
- Django uses PostgreSQL (`gym_server_db`); apply migrations explicitly before starting it.
- The mirror app uses PostgreSQL (`gym_main_db`) in centralized `MAIN` mode only.
- Exercise sessions are saved into the main database. Server sync is disabled by
  default; provision a controller and its key using the backend README, then set
  the server URL, matching gym/controller IDs, and `SMART_GYM_SERVER_SYNC_ENABLED=1`.
  Sync uses `POST /api/ingest/main-batch/` and requires an `APPLIED` acknowledgement.
- Unknown-user flow collects full registration details and stores them in `members`.

## Portrait display behavior
- The smart-display output uses a fixed portrait layout:
  - top `20%`: live status and exercise text
  - middle `70%`: analyzed camera feed
  - bottom `10%`: feedback, angle summary, and timestamp
- Landscape cameras are fitted with aspect preservation instead of vertical stretching.
- Default fit mode is `cover`, which fills the middle region and crops the sides if needed.

## Useful environment variables
- `SMART_MIRROR_DISPLAY_FIT_MODE=cover|contain`
- `SMART_MIRROR_CAMERA_ROTATION=0|90|180|270`
- `SMART_MIRROR_CAMERA_ROTATION_CAM_0=90`
- `SMART_MIRROR_CAMERA_MIRROR=1`
- `SMART_MIRROR_SMART_WINDOW_FPS=60`
- `SMART_MIRROR_TRACKED_USER_HOLD_SECONDS=2.0`
- `SMART_MIRROR_TARGET_LOCK_ROI_SCALE=1.65`
- `SMART_MIRROR_TARGET_LOCK_FACE_RECHECK_FRAMES=12`
- `SMART_MIRROR_TARGET_LOCK_MAX_CENTER_DRIFT=0.42`
- `SMART_MIRROR_TARGET_LOCK_MAX_SCALE_DRIFT=1.6`

## PostgreSQL defaults
- `SMART_MIRROR_PG_DB=gym_main_db`
- Set `SMART_MIRROR_PG_USER` and `SMART_MIRROR_PG_PASSWORD` explicitly.
- Set the optional server's `GYM_DB_USER` and `GYM_DB_PASSWORD` explicitly.

The included backend implements member login/history and controller ingest. Mobile
owner and signup flows need their separate API endpoints. Member history requires
the Bearer token returned by login. See `gym_backend/README.md` and
`Gym_App/README.md` for deployment and client compatibility requirements.
