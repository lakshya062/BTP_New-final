from PySide6.QtCore import QThread, Signal


class ServerSyncWorker(QThread):
    progress_signal = Signal(str)
    finished_signal = Signal(object)

    def __init__(self, push_callable, parent=None):
        super().__init__(parent)
        self._push_callable = push_callable

    def run(self):
        result = {
            "synced": 0,
            "error": "",
        }
        try:
            if self.isInterruptionRequested():
                self.finished_signal.emit(result)
                return
            result["synced"] = int(self._push_callable(self.progress_signal.emit) or 0)
        except Exception as exc:
            result["error"] = str(exc)
        self.finished_signal.emit(result)
