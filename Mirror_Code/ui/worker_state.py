from dataclasses import dataclass, field
from math import isfinite


def _clamp_seconds(value, minimum=0.0):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(minimum)
    return max(float(minimum), numeric) if isfinite(numeric) else float(minimum)


@dataclass
class StablePrimaryLabelTracker:
    buildup_units_per_second: float
    decay_units_per_second: float
    min_score: float
    switch_margin: float
    forget_after_seconds: float
    max_score: float = 100.0
    _scores: dict = field(default_factory=dict)
    _stable_primary_label: str | None = None
    _seconds_without_primary_face: float = 0.0

    def reset(self):
        self._scores = {}
        self._stable_primary_label = None
        self._seconds_without_primary_face = 0.0

    def observe(self, observed_label, elapsed_seconds):
        dt = _clamp_seconds(elapsed_seconds)
        if observed_label is None:
            self._seconds_without_primary_face += dt
            decay_amount = self.decay_units_per_second * dt
            for label in list(self._scores.keys()):
                updated_score = self._scores[label] - decay_amount
                if updated_score <= 0.0:
                    del self._scores[label]
                else:
                    self._scores[label] = updated_score
            if self._seconds_without_primary_face >= self.forget_after_seconds:
                self._stable_primary_label = None
            return self._stable_primary_label

        self._seconds_without_primary_face = 0.0
        buildup_amount = self.buildup_units_per_second * dt
        decay_amount = self.decay_units_per_second * dt

        for label in list(self._scores.keys()):
            if label == observed_label:
                self._scores[label] = min(
                    self.max_score,
                    self._scores[label] + buildup_amount,
                )
            else:
                updated_score = self._scores[label] - decay_amount
                if updated_score <= 0.0:
                    del self._scores[label]
                else:
                    self._scores[label] = updated_score

        if observed_label not in self._scores:
            self._scores[observed_label] = min(self.max_score, max(buildup_amount, 0.01))

        top_label, top_score = max(self._scores.items(), key=lambda item: item[1])
        if self._stable_primary_label is None:
            if top_score >= self.min_score:
                self._stable_primary_label = top_label
        elif top_label != self._stable_primary_label:
            stable_score = self._scores.get(self._stable_primary_label, 0.0)
            if (
                top_score >= self.min_score
                and top_score >= stable_score + self.switch_margin
            ):
                self._stable_primary_label = top_label

        return self._stable_primary_label or top_label


@dataclass
class FaceRecognitionState:
    unknown_threshold_seconds: float
    known_threshold_seconds: float
    switch_penalty_seconds: float
    min_observation_seconds: float
    label_tracker: StablePrimaryLabelTracker
    unknown_elapsed_seconds: float = 0.0
    known_elapsed_seconds: float = 0.0
    last_recognized_user: str | None = None
    unknown_prompted: bool = False

    def reset(self):
        self.unknown_elapsed_seconds = 0.0
        self.known_elapsed_seconds = 0.0
        self.last_recognized_user = None
        self.unknown_prompted = False
        self.label_tracker.reset()

    def update_primary_label(self, observed_label, elapsed_seconds):
        return self.label_tracker.observe(observed_label, elapsed_seconds)

    def observe_unknown(self, elapsed_seconds):
        dt = _clamp_seconds(elapsed_seconds)
        self.unknown_elapsed_seconds = min(
            self.unknown_threshold_seconds,
            self.unknown_elapsed_seconds + dt,
        )
        self.clear_known_progress()
        if (
            self.unknown_elapsed_seconds >= self.unknown_threshold_seconds
            and not self.unknown_prompted
        ):
            self.unknown_prompted = True
            return True
        return False

    def observe_recognized(self, recognized_user, elapsed_seconds):
        dt = _clamp_seconds(elapsed_seconds)
        self.unknown_elapsed_seconds = max(0.0, self.unknown_elapsed_seconds - dt)
        if self.unknown_elapsed_seconds <= 0.0:
            self.unknown_prompted = False

        if not recognized_user:
            return False

        if self.last_recognized_user != recognized_user:
            self.last_recognized_user = recognized_user
            # Evidence belongs to one identity and must never transfer to a
            # different person when the primary face changes.
            self.known_elapsed_seconds = min(self.known_threshold_seconds, dt)
        else:
            self.known_elapsed_seconds = min(
                self.known_threshold_seconds,
                self.known_elapsed_seconds + dt,
            )

        return self.known_elapsed_seconds >= self.known_threshold_seconds

    def observe_no_face(self, elapsed_seconds):
        dt = _clamp_seconds(elapsed_seconds)
        self.unknown_elapsed_seconds = max(0.0, self.unknown_elapsed_seconds - dt)
        self.clear_known_progress()
        if self.unknown_elapsed_seconds <= 0.0:
            self.unknown_prompted = False

    def clear_known_progress(self):
        self.known_elapsed_seconds = 0.0
        self.last_recognized_user = None


@dataclass
class RegistrationCaptureState:
    target_capture_seconds: float
    minimum_samples: int
    maximum_samples: int
    active: bool = False
    user_name: str | None = None
    encodings: list = field(default_factory=list)
    valid_capture_seconds: float = 0.0

    def reset(self):
        self.active = False
        self.user_name = None
        self.encodings = []
        self.valid_capture_seconds = 0.0

    def start(self, user_name):
        self.reset()
        self.active = True
        self.user_name = str(user_name or "").strip() or None

    def add_valid_encoding(self, encoding, elapsed_seconds):
        if not self.active or encoding is None:
            return
        dt = _clamp_seconds(elapsed_seconds)
        self.valid_capture_seconds = min(
            self.target_capture_seconds,
            self.valid_capture_seconds + dt,
        )
        if encoding is not None and len(self.encodings) < self.maximum_samples:
            self.encodings.append(encoding)

    @property
    def is_complete(self):
        return (
            self.active
            and self.valid_capture_seconds >= self.target_capture_seconds
            and len(self.encodings) >= self.minimum_samples
        )
