import logging
import os
import platform
import threading
import time

import cv2

_camera_cache_lock = threading.Lock()
_camera_cache = {}

_ROTATION_ALIASES = {
    "0": 0,
    "none": 0,
    "90": 90,
    "-270": 90,
    "cw": 90,
    "right": 90,
    "180": 180,
    "-180": 180,
    "flip": 180,
    "270": 270,
    "-90": 270,
    "ccw": 270,
    "left": 270,
}


def _camera_env_names(base_name, camera_source):
    names = []
    camera_id = None
    try:
        camera_id = int(camera_source)
    except (TypeError, ValueError):
        camera_id = None
    if camera_id is not None:
        names.append(f"{base_name}_CAM_{camera_id}")
        names.append(f"{base_name}_{camera_id}")
    names.append(base_name)
    return names


def _read_env_bool(var_name, default=False):
    raw = os.getenv(var_name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _read_env_int(var_name, default, minimum=0):
    raw = os.getenv(var_name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return max(minimum, value)


def _read_env_float(var_name, default, minimum=0.0):
    raw = os.getenv(var_name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return max(minimum, value)


def resolve_camera_rotation(camera_source=None, default=0):
    for env_name in _camera_env_names("SMART_MIRROR_CAMERA_ROTATION", camera_source):
        raw_value = os.getenv(env_name)
        if raw_value is None:
            continue
        normalized = raw_value.strip().lower()
        if normalized not in _ROTATION_ALIASES:
            logging.warning("Ignoring unsupported %s value: %s", env_name, raw_value)
            return default
        return _ROTATION_ALIASES[normalized]
    return default


def resolve_camera_mirror(camera_source=None, default=False):
    for env_name in _camera_env_names("SMART_MIRROR_CAMERA_MIRROR", camera_source):
        raw_value = os.getenv(env_name)
        if raw_value is None:
            continue
        return raw_value.strip().lower() not in {"0", "false", "no", "off"}
    return default


def transform_camera_frame(frame, camera_source=None, rotation=None, mirror=None):
    if frame is None or getattr(frame, "size", 0) == 0:
        return frame

    resolved_rotation = resolve_camera_rotation(camera_source) if rotation is None else rotation
    if resolved_rotation == 90:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif resolved_rotation == 180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif resolved_rotation == 270:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    resolved_mirror = resolve_camera_mirror(camera_source) if mirror is None else bool(mirror)
    if resolved_mirror:
        frame = cv2.flip(frame, 1)

    return frame


def _camera_backend_candidates():
    system_name = platform.system().lower()
    raw_candidates = []

    if system_name == "darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
        raw_candidates.append(("AVFoundation", cv2.CAP_AVFOUNDATION))
    elif system_name == "linux" and hasattr(cv2, "CAP_V4L2"):
        raw_candidates.append(("V4L2", cv2.CAP_V4L2))
    elif system_name == "windows":
        if hasattr(cv2, "CAP_MSMF"):
            raw_candidates.append(("MSMF", cv2.CAP_MSMF))
        if hasattr(cv2, "CAP_DSHOW"):
            raw_candidates.append(("DSHOW", cv2.CAP_DSHOW))

    raw_candidates.append(("Default", None))

    candidates = []
    seen_codes = set()
    for backend_name, backend_code in raw_candidates:
        if backend_code in seen_codes:
            continue
        seen_codes.add(backend_code)
        candidates.append((backend_name, backend_code))
    return candidates


def _decode_fourcc(raw_value):
    try:
        fourcc_value = int(raw_value)
    except (TypeError, ValueError):
        return ""

    chars = []
    for shift in range(4):
        code = (fourcc_value >> (8 * shift)) & 0xFF
        if 32 <= code <= 126:
            chars.append(chr(code))
    return "".join(chars).strip()


def _open_capture_with_backend(camera_source, backend_code):
    try:
        if backend_code is None:
            return cv2.VideoCapture(camera_source)
        return cv2.VideoCapture(camera_source, backend_code)
    except Exception:
        logging.exception(
            "OpenCV raised while opening camera %s with backend %s.",
            camera_source,
            backend_code,
        )
        return None


def _configure_camera_capture(
    cap,
    preferred_width=None,
    preferred_height=None,
    preferred_fourcc=None,
    preferred_fps=None,
):
    prefer_mjpeg = _read_env_bool("SMART_MIRROR_CAMERA_PREFER_MJPEG", default=True)
    requested_width = (
        _read_env_int("SMART_MIRROR_CAMERA_WIDTH", 1920, minimum=1)
        if preferred_width is None
        else max(1, int(preferred_width))
    )
    requested_height = (
        _read_env_int("SMART_MIRROR_CAMERA_HEIGHT", 1080, minimum=1)
        if preferred_height is None
        else max(1, int(preferred_height))
    )
    requested_fourcc = (
        preferred_fourcc
        if preferred_fourcc is not None
        else os.getenv("SMART_MIRROR_CAMERA_FOURCC", "MJPG")
    )
    requested_fourcc = (requested_fourcc or "").strip().upper()
    requested_fps = (
        _read_env_float("SMART_MIRROR_CAMERA_REQUEST_FPS", 0.0, minimum=0.0)
        if preferred_fps is None
        else max(0.0, float(preferred_fps))
    )

    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if prefer_mjpeg and len(requested_fourcc) == 4:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*requested_fourcc))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, requested_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, requested_height)

    if requested_fps > 0:
        cap.set(cv2.CAP_PROP_FPS, requested_fps)

    # Best effort: many webcams treat minimum zoom as the widest available view.
    # Unsupported cameras simply ignore this property.
    if _read_env_bool("SMART_MIRROR_CAMERA_WIDE_ANGLE", default=True):
        requested_zoom = _read_env_float(
            "SMART_MIRROR_CAMERA_WIDE_ANGLE_ZOOM",
            default=0.0,
            minimum=0.0,
        )
        zoom_prop = getattr(cv2, "CAP_PROP_ZOOM", None)
        if zoom_prop is not None:
            try:
                zoom_set = cap.set(zoom_prop, requested_zoom)
            except Exception:
                logging.exception(
                    "OpenCV raised while enabling wide-angle mode for the camera."
                )
            else:
                if zoom_set:
                    actual_zoom = float(cap.get(zoom_prop) or requested_zoom)
                    logging.info(
                        "Requested wide-angle camera mode by setting zoom to %.2f "
                        "(camera reported %.2f).",
                        requested_zoom,
                        actual_zoom,
                    )


def open_camera_capture(
    camera_source,
    *,
    apply_preferences=True,
    preferred_width=None,
    preferred_height=None,
    preferred_fourcc=None,
    preferred_fps=None,
):
    for backend_name, backend_code in _camera_backend_candidates():
        cap = _open_capture_with_backend(camera_source, backend_code)
        if cap is None:
            continue
        if not cap.isOpened():
            cap.release()
            continue

        if apply_preferences:
            _configure_camera_capture(
                cap,
                preferred_width=preferred_width,
                preferred_height=preferred_height,
                preferred_fourcc=preferred_fourcc,
                preferred_fps=preferred_fps,
            )

        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        actual_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        actual_fourcc = _decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC)) or "unknown"
        logging.info(
            "Opened camera %s via %s backend at %sx%s (fourcc=%s, fps=%.2f).",
            camera_source,
            backend_name,
            actual_width,
            actual_height,
            actual_fourcc,
            actual_fps,
        )
        return cap

    logging.error("Could not open camera %s with any supported backend.", camera_source)
    return None


def detect_available_cameras(max_cameras=10, cache_ttl_sec=None, force_refresh=False):
    ttl = cache_ttl_sec
    if ttl is None:
        ttl = float(os.getenv("SMART_MIRROR_CAMERA_SCAN_CACHE_SEC", "3.0"))
    ttl = max(0.0, float(ttl))
    cache_key = max(1, int(max_cameras))
    now = time.monotonic()

    with _camera_cache_lock:
        cached = _camera_cache.get(cache_key)
        if (
            not force_refresh
            and cached is not None
            and (now - cached["ts"]) <= ttl
        ):
            return list(cached["values"])

    available_cams = []
    for index in range(cache_key):
        cap = open_camera_capture(index, apply_preferences=False)
        if cap is None:
            continue
        if cap.isOpened():
            available_cams.append(index)
        cap.release()

    with _camera_cache_lock:
        _camera_cache[cache_key] = {"ts": now, "values": tuple(available_cams)}
    return available_cams
