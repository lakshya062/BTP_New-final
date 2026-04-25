from core.config import exercise_config


def _coaching_text(exercise_choice, key, default):
    return exercise_config.get(exercise_choice, {}).get("coaching", {}).get(key, default)


def range_miss_message(exercise_choice, up_range_miss, down_range_miss):
    if up_range_miss:
        if exercise_choice == "deadlift":
            return "Go deeper into the hinge before standing up.", "up"
        return _coaching_text(
            exercise_choice,
            "up_miss",
            "Go higher. Reach the top range before lowering.",
        ), "up"
    if down_range_miss:
        if exercise_choice == "deadlift":
            return "Stand tall at lockout before the next rep.", "down"
        return _coaching_text(
            exercise_choice,
            "down_miss",
            "Lower fully. Reach the down range before next rep.",
        ), "down"
    return None, None


def motivation_line(
    exercise_choice,
    reps,
    sets_done,
    reps_per_set,
    feedback_texts,
    status_hint="",
    rep_state=0,
    stable_start_detected=False,
    up_range_miss=False,
    down_range_miss=False,
    stagnant_seconds=0.0,
):
    is_deadlift = exercise_choice == "deadlift"

    if any("Warning" in text for text in feedback_texts):
        return "Reset your posture first, then we go again with control."
    if up_range_miss:
        if is_deadlift:
            return "You are close. Hinge deeper with control before driving up."
        return _coaching_text(
            exercise_choice,
            "up_miss",
            "You are close. Lift a little higher and squeeze at the top.",
        )
    if down_range_miss:
        if is_deadlift:
            return "Strong pull. Finish taller at lockout before the next rep."
        return _coaching_text(
            exercise_choice,
            "down_miss",
            "Nice effort. Lower all the way down, then start the next curl.",
        )
    if "Set complete!" in feedback_texts:
        return "Great set. Take two deep breaths and get ready for the next round."
    if "Ready to start!" in feedback_texts:
        return "Setup looks strong. Start smooth and own the first rep."

    if stable_start_detected and reps > 0:
        if stagnant_seconds >= 14:
            lines = [
                "You are not done yet. One clean rep breaks this plateau.",
                (
                    "Stay with me. Brace your core, drive through your heels, and finish this rep."
                    if is_deadlift
                    else "Stay with me. Elbows tight, breathe out, and finish this rep."
                ),
                "Tough moment right now. Slow down, reset, and drive through.",
            ]
            return lines[int(stagnant_seconds // 4) % len(lines)]
        if stagnant_seconds >= 8:
            lines = [
                "You are stuck on this rep. Small reset, then go again.",
                "Keep calm and stay tight. You can get this next rep.",
                "One solid rep right now. Controlled up, controlled down.",
            ]
            return lines[int(stagnant_seconds // 3) % len(lines)]

    if stable_start_detected and sets_done == 0 and reps == 0:
        if is_deadlift:
            return "Start position locked. Hinge back smoothly, then stand tall at lockout."
        return _coaching_text(
            exercise_choice,
            "start",
            "Start position locked. Curl up with intent and return with control.",
        )
    if sets_done > 0 and reps == 0:
        return "Strong previous set. Stay composed and attack this next one."
    if reps_per_set > 0 and reps >= max(1, int(reps_per_set * 0.9)):
        return "Last push. Finish every inch of these final reps."
    if reps_per_set > 0 and reps >= max(1, int(reps_per_set * 0.75)):
        return "You are in the hard zone now. Keep your form crisp."
    if reps_per_set > 0 and reps >= max(1, reps_per_set // 2):
        return "Halfway there. Keep breathing and keep moving clean."
    if reps_per_set > 0 and reps >= max(1, int(reps_per_set * 0.25)):
        return "Rhythm looks good. Stack clean reps one by one."
    if reps > 0:
        if rep_state == 1:
            if is_deadlift:
                return "Push the floor away and drive hips forward to stand tall."
            return _coaching_text(
                exercise_choice,
                "phase_up",
                "Drive up through the curl. Elbows stay pinned.",
            )
        if rep_state == 2:
            if is_deadlift:
                return "Control the hinge down. Keep the spine neutral and lats tight."
            return _coaching_text(
                exercise_choice,
                "phase_down",
                "Control the way down. Earn the full range.",
            )
        return "Good tempo. Stay smooth and keep tension."
    if status_hint:
        return status_hint
    return _coaching_text(
        exercise_choice,
        "start",
        "Stand centered and start when you are ready.",
    )
