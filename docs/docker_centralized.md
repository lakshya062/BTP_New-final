# Centralized Docker Bundle

## What this gives you
- `postgres`: shared database server for the centralized runtime, with the gym databases created inside the image startup flow
- `backend`: Django API server on port `8000`
- `mirror`: centralized PySide display runtime in a Linux container

## Export from the Mac
```bash
./scripts/export_docker_bundle.sh
```

By default this builds an `amd64` bundle for Windows/Linux x86_64 machines. For Apple Silicon Macs or ARM Linux, export an `arm64` bundle:
```bash
TARGET_PLATFORM=linux/arm64 ./scripts/export_docker_bundle.sh
```

This creates:
- `dist/centralized_docker_bundle_<arch>/smart-gym-centralized-images-<arch>.tar`
- `dist/smart-gym-centralized-bundle-<arch>.tgz`
- `dist/centralized_docker_bundle_<arch>/.env`

## One-command rebuild pipeline
If you update the code and want fresh images plus fresh transferable bundles, run:
```bash
./scripts/rebuild_portable_bundles.sh
```

That rebuilds the portable Docker images and bundle archives for both:
- `amd64` for Windows and standard Linux PCs
- `arm64` for Apple Silicon Macs and ARM Linux

Useful shortcuts:
```bash
./scripts/rebuild_portable_bundles.sh --host-only
./scripts/rebuild_portable_bundles.sh --host-only --start
./scripts/rebuild_portable_bundles.sh --host-only --start-mirror
./scripts/rebuild_portable_bundles.sh --amd64
./scripts/rebuild_portable_bundles.sh --arm64
```

What this gives you after every rebuild:
- updated local Docker image tags for the selected architectures
- updated `.tar` image archives inside `dist/centralized_docker_bundle_<arch>/`
- updated `.tgz` files ready to transfer to another machine

## Run on Linux
1. Copy the exported bundle to the target machine.
2. Extract it.
3. Run:
```bash
./run_centralized_linux.sh
```
4. When GUI forwarding and camera passthrough are ready:
```bash
./run_centralized_linux.sh --start-mirror
```

## Run on Windows
1. Copy the exported bundle to the target machine.
2. Extract it.
3. Run in PowerShell:
```powershell
.\run_centralized_windows.ps1
```
4. If the Windows machine has WSLg or an X server for GUI forwarding, start the mirror container too:
```powershell
.\run_centralized_windows.ps1 -StartMirror
```

## Run on macOS
1. Copy the exported bundle to the target machine.
2. Extract it.
3. Run:
```bash
./run_centralized_macos.sh
```
4. If XQuartz and camera/device passthrough are ready:
```bash
./run_centralized_macos.sh --start-mirror
```

## Why the Docker requirements files are short
- `docker/requirements/backend.txt` only lists Python packages that the Django backend needs at runtime.
- `docker/requirements/mirror.txt` only lists Python packages that the mirror container needs at runtime.
- Standard-library modules are not listed because Python already provides them.
- System packages such as `cmake` are installed in the Dockerfiles, not in `requirements.txt`.
- `dlib` and `face-recognition` are installed directly in `docker/mirror/Dockerfile` because they need a custom source build for cross-platform container compatibility.

## Why `docker run smart-gym-backend...` failed
- The backend container is not supposed to contain PostgreSQL inside the same image.
- In the centralized stack, PostgreSQL is its own Docker image/container and the backend connects to it over the Docker network.
- Running only the backend image makes it look for a database that is not there, which is why you saw `connection refused`.
- In the command you pasted, you also ran `smart-gym-backend:portable-arm64`. On Windows/Linux x86_64 you should use the `amd64` bundle instead.

## If you want raw `docker run` instead of `docker compose`
Start both containers on the same Docker network:
```bash
docker network create smart-gym-centralized
docker run -d --name smart-gym-postgres --network smart-gym-centralized -e POSTGRES_USER=smartgym -e POSTGRES_PASSWORD=smartgym smart-gym-postgres:portable-amd64
docker run -it --rm -p 8000:8000 --network smart-gym-centralized -e GYM_DB_HOST=smart-gym-postgres smart-gym-backend:portable-amd64
```

## Important host notes
- The Docker images are Linux containers. They run on Linux directly and on Windows through Docker Desktop running Linux containers.
- They also run on macOS through Docker Desktop or OrbStack running Linux containers.
- Camera, GPU, audio, and GUI passthrough still depend on the target host.
- For Linux mirror output, the bundle expects X11 access through `/tmp/.X11-unix`.
- For Windows mirror output, use WSLg or an X server such as VcXsrv or X410.
- For macOS mirror output, use XQuartz and allow Docker/XQuartz display access.
- If the camera is mounted in a rotated orientation, set `SMART_MIRROR_CAMERA_ROTATION` or `SMART_MIRROR_CAMERA_ROTATION_CAM_<index>`.
- If you want black bars instead of center-cropping, set `SMART_MIRROR_DISPLAY_FIT_MODE=contain`.

## Why this is not a single perfect universal binary
Docker can package the application stack and make deployment reproducible, but native screen and camera passthrough are controlled by the target OS and host drivers. The bundle here makes the software portable; the final GUI/camera wiring still needs the host to expose those devices to Docker.
