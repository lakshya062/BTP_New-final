# Smart Gym Centralized Runtime Runbook

## Quick Start
Use two terminals.

Terminal A:
```bash
cd gym_backend
python3 manage.py runserver 0.0.0.0:8000
```

Terminal B:
```bash
cd Mirror_Code
python3 app.py
```

## What happens automatically
- Django uses PostgreSQL (`gym_server_db`) and auto-runs migrations during `runserver`.
- The mirror app uses PostgreSQL (`gym_main_db`) in centralized `MAIN` mode only.
- Exercise sessions are saved locally into the main database and then synced to the server API (`POST /api/ingest/main-batch/`).
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
- `SMART_MIRROR_PG_PASSWORD=admin`
- `GYM_DB_PASSWORD=admin`
