import os

import cv2
import numpy as np

TOP_PANEL_RATIO = 0.20
FEED_PANEL_RATIO = 0.70
BOTTOM_PANEL_RATIO = 0.10


def resolve_display_fit_mode(default="contain"):
    raw_mode = os.getenv("SMART_MIRROR_DISPLAY_FIT_MODE", default).strip().lower()
    if raw_mode not in {"cover", "contain"}:
        return default
    return raw_mode


def _build_gradient_canvas(width, height):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    top_color = np.array([15, 22, 32], dtype=np.float32)
    bottom_color = np.array([6, 10, 17], dtype=np.float32)
    blend = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    gradient_row = (top_color * (1.0 - blend) + bottom_color * blend).astype(np.uint8)
    canvas[:, :, 0] = gradient_row[:, 0:1]
    canvas[:, :, 1] = gradient_row[:, 1:2]
    canvas[:, :, 2] = gradient_row[:, 2:3]
    return canvas


def fit_frame_to_region(frame, target_width, target_height, fit_mode="cover"):
    target_width = max(1, int(target_width))
    target_height = max(1, int(target_height))
    if frame is None or getattr(frame, "size", 0) == 0:
        return np.zeros((target_height, target_width, 3), dtype=np.uint8)

    src_h, src_w = frame.shape[:2]
    if src_h <= 0 or src_w <= 0:
        return np.zeros((target_height, target_width, 3), dtype=np.uint8)

    scale_x = target_width / float(src_w)
    scale_y = target_height / float(src_h)
    if fit_mode == "contain":
        scale = min(scale_x, scale_y)
    else:
        scale = max(scale_x, scale_y)

    resized_width = max(1, int(round(src_w * scale)))
    resized_height = max(1, int(round(src_h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=interpolation)

    if fit_mode == "contain":
        # Fill the full portrait panel with a softened version of the camera
        # feed so the live image does not look like a small strip in the middle.
        output = fit_frame_to_region(
            frame,
            target_width=target_width,
            target_height=target_height,
            fit_mode="cover",
        )
        output = cv2.GaussianBlur(output, (0, 0), sigmaX=18.0, sigmaY=18.0)
        output = cv2.addWeighted(output, 0.72, np.zeros_like(output), 0.28, 0.0)
        offset_x = max(0, (target_width - resized_width) // 2)
        offset_y = max(0, (target_height - resized_height) // 2)
        output[offset_y:offset_y + resized_height, offset_x:offset_x + resized_width] = resized
        return output

    start_x = max(0, (resized_width - target_width) // 2)
    start_y = max(0, (resized_height - target_height) // 2)
    end_x = start_x + target_width
    end_y = start_y + target_height
    return resized[start_y:end_y, start_x:end_x]


def build_display_lines(overlay_data=None, feedback_texts=None):
    overlay_data = dict(overlay_data or {})
    feedback_texts = [str(text).strip() for text in (feedback_texts or []) if str(text).strip()]

    username = overlay_data.get("username") or "Athlete"
    exercise = overlay_data.get("exercise") or "Exercise"
    reps = int(overlay_data.get("rep_count") or 0)
    sets_done = int(overlay_data.get("set_count") or 0)
    reps_per_set = int(overlay_data.get("reps_per_set") or 0)
    current_weight = int(overlay_data.get("current_weight") or 0)
    weight_text = f"{current_weight} lbs" if current_weight > 0 else "--"
    rep_progress = f"{reps}/{reps_per_set}" if reps_per_set > 0 else str(reps)

    top_lines = [
        f"{username} | {exercise}",
        f"Reps {rep_progress}   Sets {sets_done}   Weight {weight_text}",
    ]

    motivation = str(overlay_data.get("motivation") or "").strip()
    if motivation:
        top_lines.append(motivation)

    bottom_lines = []
    if feedback_texts:
        bottom_lines.append(feedback_texts[0])

    metadata_parts = []
    angles_line = str(overlay_data.get("angles_line") or "").strip()
    if angles_line:
        metadata_parts.append(angles_line)
    timestamp = str(overlay_data.get("timestamp") or "").strip()
    if timestamp:
        metadata_parts.append(timestamp)
    if metadata_parts:
        bottom_lines.append("   |   ".join(metadata_parts))

    if not bottom_lines and motivation:
        bottom_lines.append(motivation)

    return top_lines[:3], bottom_lines[:2]


def _wrap_text(text, max_width, font, font_scale, thickness):
    words = str(text or "").split()
    if not words:
        return []

    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        width = cv2.getTextSize(candidate, font, font_scale, thickness)[0][0]
        if width <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _draw_text_block(
    canvas,
    lines,
    x,
    y,
    width,
    height,
    font_scale,
    color,
    thickness,
    line_gap,
):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cursor_y = int(y)
    max_y = int(y + height)

    for raw_line in lines:
        wrapped_lines = _wrap_text(raw_line, width, font, font_scale, thickness)
        for line in wrapped_lines:
            text_height = cv2.getTextSize(line, font, font_scale, thickness)[0][1]
            baseline = text_height + 8
            if (cursor_y + baseline) > max_y:
                return
            cv2.putText(
                canvas,
                line,
                (int(x), cursor_y + text_height),
                font,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )
            cursor_y += baseline + line_gap


def compose_mirror_frame(
    frame,
    canvas_width,
    canvas_height,
    overlay_data=None,
    feedback_texts=None,
    fit_mode="cover",
):
    canvas_width = max(1, int(canvas_width))
    canvas_height = max(1, int(canvas_height))
    canvas = _build_gradient_canvas(canvas_width, canvas_height)

    top_height = max(1, int(round(canvas_height * TOP_PANEL_RATIO)))
    bottom_height = max(1, int(round(canvas_height * BOTTOM_PANEL_RATIO)))
    feed_height = max(1, canvas_height - top_height - bottom_height)

    top_panel = canvas[0:top_height, :]
    feed_panel = canvas[top_height:top_height + feed_height, :]
    bottom_panel = canvas[top_height + feed_height:, :]

    top_panel[:] = (12, 18, 28)
    bottom_panel[:] = (10, 14, 22)
    feed_panel[:] = (3, 6, 10)

    fitted_frame = fit_frame_to_region(
        frame,
        target_width=canvas_width,
        target_height=feed_height,
        fit_mode=fit_mode,
    )
    feed_panel[:, :] = fitted_frame
    cv2.rectangle(
        canvas,
        (0, top_height),
        (canvas_width - 1, top_height + feed_height - 1),
        (88, 162, 224),
        max(1, canvas_height // 540),
        cv2.LINE_AA,
    )

    top_lines, bottom_lines = build_display_lines(
        overlay_data=overlay_data,
        feedback_texts=feedback_texts,
    )

    padding_x = max(18, canvas_width // 32)
    top_font_scale = max(0.7, canvas_height / 1050.0)
    top_line_gap = max(4, canvas_height // 240)
    _draw_text_block(
        canvas,
        top_lines,
        x=padding_x,
        y=max(28, top_height // 4),
        width=canvas_width - (padding_x * 2),
        height=top_height - max(24, top_height // 5),
        font_scale=top_font_scale,
        color=(236, 244, 255),
        thickness=max(1, canvas_height // 720),
        line_gap=top_line_gap,
    )

    bottom_font_scale = max(0.48, canvas_height / 1500.0)
    bottom_line_gap = max(2, canvas_height // 360)
    _draw_text_block(
        canvas,
        bottom_lines,
        x=padding_x,
        y=top_height + feed_height + max(20, bottom_height // 3),
        width=canvas_width - (padding_x * 2),
        height=bottom_height - max(16, bottom_height // 5),
        font_scale=bottom_font_scale,
        color=(210, 224, 242),
        thickness=max(1, canvas_height // 900),
        line_gap=bottom_line_gap,
    )

    return canvas
