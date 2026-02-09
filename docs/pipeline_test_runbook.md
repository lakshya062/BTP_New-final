# Smart Gym Pipeline Test Runbook

## Quick Start (Default)
Use two terminals only.

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
- Django server defaults to PostgreSQL (`gym_server_db`) and auto-runs migrations on `runserver`.
- Main system app defaults to PostgreSQL (`gym_main_db`) and main-node mode.
- On macOS, TTS audio feedback is disabled by default for stability. To force-enable it, set `SMART_MIRROR_ENABLE_AUDIO_FEEDBACK=1` before `python3 app.py`.
- Unknown-user flow now asks full registration details (username, email, plus profile fields), stores them in `members`, and uses DB data when recognized later.
- Local SQLite edge buffer is disabled; all persistence is PostgreSQL-only.
- When a user exits frame and session data is saved, transfer sync is triggered immediately (no wait for 1-minute timer).
- Main system sync path remains:
  - edge data pull (SSH) -> main DB -> server API (`POST /api/ingest/main-batch/`) -> server DB.

## Optional: Local edge-mode simulation
If you want to simulate an edge runtime on the same machine, use a third terminal:
```bash
cd Mirror_Code
export SMART_MIRROR_EDGE_MODE=1
python3 app.py
```

## Edge deploy notes
- Add Device now supports:
  - Manual IP entry (if network scan misses an online edge device).
  - Optional PostgreSQL provisioning on edge during deploy.
- If PostgreSQL setup is enabled for that device, deploy runs:
  - `Mirror_Code/scripts/edge_setup_postgres.sh`
  - Then applies edge schema from `Mirror_Code/resources/sql/edge_schema.sql`.

## PostgreSQL credentials used by default
- `GYM_DB_ENGINE=postgres`
- `GYM_DB_PASSWORD=admin`
- `SMART_MIRROR_PG_PASSWORD=admin`
