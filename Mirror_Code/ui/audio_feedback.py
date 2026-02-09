import logging
import os
import queue
import threading
import time

class AudioFeedbackPlayer:
    def __init__(self):
        # pyttsx3 on macOS can crash when driven from worker threads via AppKit.
        # Keep it disabled by default on macOS unless explicitly enabled.
        self._force_enable = os.environ.get("SMART_MIRROR_ENABLE_AUDIO_FEEDBACK", "0") == "1"
        self._force_disable = os.environ.get("SMART_MIRROR_DISABLE_AUDIO_FEEDBACK", "0") == "1"
        self._is_macos = os.uname().sysname == "Darwin" if hasattr(os, "uname") else False
        self._tts_module = None
        self.available = False
        self._queue = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = None
        self._last_enqueued_text = ""
        self._last_enqueued_ts = 0.0
        self._repeat_suppression_sec = 2.0

        if self._force_disable:
            logging.info("Audio feedback disabled by SMART_MIRROR_DISABLE_AUDIO_FEEDBACK=1.")
            return

        if self._is_macos and not self._force_enable:
            logging.info(
                "Audio feedback disabled on macOS by default for stability. "
                "Set SMART_MIRROR_ENABLE_AUDIO_FEEDBACK=1 to force-enable."
            )
            return

        try:
            import pyttsx3  # type: ignore

            self._tts_module = pyttsx3
            self.available = True
        except Exception:
            self._tts_module = None
            self.available = False

        if not self.available:
            logging.warning("pyttsx3 is not available. Audio feedback is disabled.")
            return

        self._thread = threading.Thread(
            target=self._speak_loop,
            name="audio-feedback-player",
            daemon=True,
        )
        self._thread.start()

    def enqueue(self, text, high_priority=False):
        if not self.available:
            return

        normalized = " ".join(str(text or "").split())
        if not normalized:
            return

        now = time.monotonic()
        if (
            normalized == self._last_enqueued_text
            and (now - self._last_enqueued_ts) < self._repeat_suppression_sec
        ):
            return

        self._last_enqueued_text = normalized
        self._last_enqueued_ts = now

        if high_priority:
            self.clear_pending()
        self._queue.put(normalized)

    def clear_pending(self):
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def _speak_loop(self):
        try:
            if self._tts_module is None:
                raise RuntimeError("pyttsx3 is unavailable.")
            engine = self._tts_module.init()
            engine.setProperty("rate", 178)
            engine.setProperty("volume", 1.0)
        except Exception as exc:
            logging.error("Failed to initialize audio feedback engine: %s", exc)
            self.available = False
            return

        while not self._stop_event.is_set():
            try:
                text = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue

            if text is None:
                break

            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as exc:
                logging.debug("Audio feedback playback failed: %s", exc)

        try:
            engine.stop()
        except Exception:
            pass

    def stop(self):
        if not self.available:
            return

        self._stop_event.set()
        self._queue.put(None)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
