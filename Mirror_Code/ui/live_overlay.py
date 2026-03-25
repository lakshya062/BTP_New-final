from datetime import datetime

import cv2


def draw_translucent_panel(
    frame,
    x,
    y,
    width,
    height,
    fill_color=(20, 28, 38),
    border_color=(103, 188, 255),
    alpha=0.62,
    border_thickness=2,
):
    """Draw a translucent rectangular panel with a border."""
    x = max(0, int(x))
    y = max(0, int(y))
    width = max(1, int(width))
    height = max(1, int(height))

    x2 = min(frame.shape[1] - 1, x + width)
    y2 = min(frame.shape[0] - 1, y + height)
    if x >= x2 or y >= y2:
        return

    roi = frame[y:y2, x:x2]
    if roi.size == 0:
        return
    overlay = roi.copy()
    cv2.rectangle(
        overlay,
        (0, 0),
        (overlay.shape[1] - 1, overlay.shape[0] - 1),
        fill_color,
        -1,
    )
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)
    cv2.rectangle(frame, (x, y), (x2, y2), border_color, border_thickness)


def draw_live_overlay(frame, overlay_data, feedback_texts=None):
    feedback_texts = feedback_texts or []
    h, w = frame.shape[:2]
    scale = max(0.72, min(w / 1280.0, h / 720.0))

    margin = int(20 * scale)
    panel_w = int(470 * scale)
    panel_h = int(250 * scale)

    panel_x = margin
    panel_y = margin

    draw_translucent_panel(
        frame,
        panel_x,
        panel_y,
        panel_w,
        panel_h,
        fill_color=(22, 30, 43),
        border_color=(108, 201, 255),
        alpha=0.66,
        border_thickness=max(1, int(2 * scale)),
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    title_scale = 0.88 * scale
    label_scale = 0.53 * scale
    value_scale = 0.82 * scale

    cv2.putText(
        frame,
        "LIVE PERFORMANCE",
        (panel_x + int(18 * scale), panel_y + int(34 * scale)),
        font,
        title_scale,
        (240, 248, 255),
        max(1, int(2 * scale)),
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"User: {overlay_data['username']}",
        (panel_x + int(18 * scale), panel_y + int(64 * scale)),
        font,
        0.64 * scale,
        (186, 227, 255),
        max(1, int(2 * scale)),
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Exercise: {overlay_data['exercise']}",
        (panel_x + int(18 * scale), panel_y + int(88 * scale)),
        font,
        label_scale,
        (212, 223, 240),
        max(1, int(1 * scale)),
        cv2.LINE_AA,
    )

    chip_w = int((panel_w - int(52 * scale)) / 3)
    chip_h = int(90 * scale)
    chip_y = panel_y + int(102 * scale)
    chip_gap = int(14 * scale)

    stats = [
        ("REPS", str(overlay_data["rep_count"])),
        ("SETS", str(overlay_data["set_count"])),
        (
            "WEIGHT",
            f"{overlay_data['current_weight']} lbs" if overlay_data["current_weight"] > 0 else "--",
        ),
    ]

    for idx, (label, value) in enumerate(stats):
        chip_x = panel_x + int(16 * scale) + idx * (chip_w + chip_gap)
        draw_translucent_panel(
            frame,
            chip_x,
            chip_y,
            chip_w,
            chip_h,
            fill_color=(30, 41, 57),
            border_color=(92, 166, 236),
            alpha=0.7,
            border_thickness=max(1, int(1 * scale)),
        )
        cv2.putText(
            frame,
            label,
            (chip_x + int(12 * scale), chip_y + int(30 * scale)),
            font,
            0.48 * scale,
            (176, 210, 246),
            max(1, int(1 * scale)),
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            value,
            (chip_x + int(12 * scale), chip_y + int(68 * scale)),
            font,
            value_scale,
            (245, 250, 255),
            max(1, int(2 * scale)),
            cv2.LINE_AA,
        )

    cv2.putText(
        frame,
        overlay_data["angles_line"],
        (panel_x + int(18 * scale), panel_y + panel_h - int(20 * scale)),
        font,
        0.52 * scale,
        (178, 197, 218),
        max(1, int(1 * scale)),
        cv2.LINE_AA,
    )

    time_panel_w = int(390 * scale)
    time_panel_h = int(62 * scale)
    time_x = max(margin, w - time_panel_w - margin)
    time_y = margin
    draw_translucent_panel(
        frame,
        time_x,
        time_y,
        time_panel_w,
        time_panel_h,
        fill_color=(22, 30, 43),
        border_color=(108, 201, 255),
        alpha=0.64,
        border_thickness=max(1, int(2 * scale)),
    )
    cv2.putText(
        frame,
        overlay_data["timestamp"],
        (time_x + int(16 * scale), time_y + int(39 * scale)),
        font,
        0.60 * scale,
        (238, 246, 255),
        max(1, int(2 * scale)),
        cv2.LINE_AA,
    )

    if overlay_data.get("show_range_bars"):
        rep_angle = float(overlay_data.get("rep_angle") or 0.0)
        up_range = overlay_data.get("up_range") or (0.0, 0.0)
        down_range = overlay_data.get("down_range") or (0.0, 0.0)
        up_min, up_max = float(up_range[0]), float(up_range[1])
        down_min, down_max = float(down_range[0]), float(down_range[1])
        rep_state = int(overlay_data.get("rep_state", 0))
        blink_on = ((datetime.now().microsecond // 180000) % 2) == 0

        range_panel_w = int(390 * scale)
        range_panel_h = int(154 * scale)
        range_x = max(margin, w - range_panel_w - margin)
        range_y = time_y + time_panel_h + int(12 * scale)
        draw_translucent_panel(
            frame,
            range_x,
            range_y,
            range_panel_w,
            range_panel_h,
            fill_color=(22, 30, 43),
            border_color=(108, 201, 255),
            alpha=0.64,
            border_thickness=max(1, int(2 * scale)),
        )
        cv2.putText(
            frame,
            "RANGE CHECK",
            (range_x + int(16 * scale), range_y + int(28 * scale)),
            font,
            0.58 * scale,
            (232, 244, 255),
            max(1, int(2 * scale)),
            cv2.LINE_AA,
        )

        bar_left = range_x + int(16 * scale)
        bar_w = range_panel_w - int(32 * scale)
        bar_h = int(18 * scale)
        up_bar_y = range_y + int(48 * scale)
        down_bar_y = range_y + int(100 * scale)

        angle_span = max(1.0, down_max - up_min)
        up_progress = max(0.0, min(1.0, (down_max - rep_angle) / angle_span))
        down_progress = max(0.0, min(1.0, (rep_angle - up_min) / angle_span))
        up_hit = up_min <= rep_angle <= up_max
        down_hit = down_min <= rep_angle <= down_max
        up_miss = bool(overlay_data.get("up_range_miss", False))
        down_miss = bool(overlay_data.get("down_range_miss", False))
        up_range_label = str(overlay_data.get("up_range_label", "UP"))
        down_range_label = str(overlay_data.get("down_range_label", "DOWN"))

        def draw_range_bar(label, y, progress, target_min, target_max, in_target, expected_now, miss_now):
            draw_translucent_panel(
                frame,
                bar_left,
                y,
                bar_w,
                bar_h,
                fill_color=(26, 38, 54),
                border_color=(76, 132, 188),
                alpha=0.74,
                border_thickness=max(1, int(1 * scale)),
            )

            fill_w = max(1, int(bar_w * progress))
            if in_target:
                fill_color = (64, 188, 120)
                border_color = (118, 242, 130)
                state_text = "ON TARGET"
            elif miss_now and blink_on:
                fill_color = (52, 46, 188)
                border_color = (92, 92, 255)
                state_text = "MISSED RANGE"
            elif expected_now:
                fill_color = (82, 132, 214)
                border_color = (116, 168, 248)
                state_text = "IN PROGRESS"
            else:
                fill_color = (66, 108, 164)
                border_color = (100, 144, 206)
                state_text = "READY"

            cv2.rectangle(
                frame,
                (bar_left, y),
                (bar_left + fill_w, y + bar_h),
                fill_color,
                -1,
            )
            cv2.rectangle(
                frame,
                (bar_left, y),
                (bar_left + bar_w, y + bar_h),
                border_color,
                max(1, int(2 * scale)),
            )

            cv2.putText(
                frame,
                f"{label} {int(target_min)}-{int(target_max)}deg",
                (bar_left, y - int(7 * scale)),
                font,
                0.46 * scale,
                (196, 220, 246),
                max(1, int(1 * scale)),
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                state_text,
                (bar_left + int(6 * scale), y + bar_h + int(16 * scale)),
                font,
                0.44 * scale,
                (214, 232, 250),
                max(1, int(1 * scale)),
                cv2.LINE_AA,
            )

        draw_range_bar(
            up_range_label,
            up_bar_y,
            up_progress,
            up_min,
            up_max,
            up_hit,
            rep_state == 1,
            up_miss,
        )
        draw_range_bar(
            down_range_label,
            down_bar_y,
            down_progress,
            down_min,
            down_max,
            down_hit,
            rep_state == 2,
            down_miss,
        )

    motivation_h = int(72 * scale)
    motivation_x = margin
    motivation_y = h - motivation_h - margin
    motivation_w = max(100, w - (2 * margin))

    draw_translucent_panel(
        frame,
        motivation_x,
        motivation_y,
        motivation_w,
        motivation_h,
        fill_color=(17, 77, 88),
        border_color=(88, 223, 230),
        alpha=0.68,
        border_thickness=max(1, int(2 * scale)),
    )
    cv2.putText(
        frame,
        overlay_data["motivation"],
        (motivation_x + int(18 * scale), motivation_y + int(45 * scale)),
        font,
        0.68 * scale,
        (235, 255, 255),
        max(1, int(2 * scale)),
        cv2.LINE_AA,
    )

    warning_texts = [text for text in feedback_texts if "Warning" in text]
    if warning_texts:
        warning_text = warning_texts[0]
        warning_w = min(int(640 * scale), w - 2 * margin)
        warning_h = int(58 * scale)
        warning_x = max(margin, (w - warning_w) // 2)
        warning_y = margin + int(80 * scale)
        draw_translucent_panel(
            frame,
            warning_x,
            warning_y,
            warning_w,
            warning_h,
            fill_color=(61, 25, 32),
            border_color=(255, 124, 124),
            alpha=0.74,
            border_thickness=max(1, int(2 * scale)),
        )
        cv2.putText(
            frame,
            warning_text,
            (warning_x + int(16 * scale), warning_y + int(36 * scale)),
            font,
            0.62 * scale,
            (255, 230, 230),
            max(1, int(2 * scale)),
            cv2.LINE_AA,
        )
