from PySide6.QtCore import QThread, Signal

from ui.edge_utils import (
    resolve_edge_camera_index,
    resolve_edge_pg_password,
    resolve_edge_pg_user,
)


class EdgeDeployWorker(QThread):
    progress_signal = Signal(str)
    finished_signal = Signal(object)

    def __init__(self, edge_device_manager, edge_devices, parent=None):
        super().__init__(parent)
        self.edge_device_manager = edge_device_manager
        self.edge_devices = edge_devices

    def run(self):
        success_ips = []
        failures = []
        for edge in self.edge_devices:
            if self.isInterruptionRequested():
                break
            ip = (edge.get("ip") or "").strip()
            username = (edge.get("username") or "").strip()
            password = (edge.get("password") or "").strip()
            remote_dir = (edge.get("remote_dir") or "~/smart_mirror_edge").strip()
            display = (edge.get("display") or ":0").strip()
            install_deps = bool(edge.get("install_deps", False))

            if not ip:
                continue

            self.progress_signal.emit(f"Deploying edge device {ip}...")
            result = self.edge_device_manager.deploy_and_launch(
                ip=ip,
                username=username,
                password=password,
                remote_dir=remote_dir,
                display=display,
                camera_index=resolve_edge_camera_index(edge),
                install_deps=install_deps,
                setup_postgres=bool(edge.get("setup_postgres", False)),
                db_backend=edge.get("db_backend", "postgres"),
                pg_db=edge.get("pg_db", "gym_edge_db"),
                pg_user=resolve_edge_pg_user(edge),
                pg_password=resolve_edge_pg_password(edge),
                pg_host=edge.get("pg_host", "localhost"),
                pg_port=edge.get("pg_port", "5432"),
                mirror_id=edge.get("mirror_id", ""),
                gym_id=edge.get("gym_id", ""),
                main_system_id=edge.get("main_system_id", ""),
                progress_callback=self.progress_signal.emit,
            )
            if result.success:
                success_ips.append(ip)
                self.progress_signal.emit(f"Edge deployment successful for {ip}.")
            else:
                details = f" ({result.details})" if result.details else ""
                failures.append(f"{ip}: {result.message}{details}")
                self.progress_signal.emit(f"Edge deployment failed for {ip}.")

        self.finished_signal.emit({"success_ips": success_ips, "failures": failures})


class EdgeStopWorker(QThread):
    progress_signal = Signal(str)
    finished_signal = Signal(object)

    def __init__(self, edge_device_manager, edge_devices, parent=None):
        super().__init__(parent)
        self.edge_device_manager = edge_device_manager
        self.edge_devices = edge_devices

    def run(self):
        success_ips = []
        failures = []
        for edge in self.edge_devices:
            if self.isInterruptionRequested():
                break
            ip = (edge.get("ip") or "").strip()
            username = (edge.get("username") or "").strip()
            password = (edge.get("password") or "").strip()
            remote_dir = (edge.get("remote_dir") or "~/smart_mirror_edge").strip()
            if not ip:
                continue

            self.progress_signal.emit(f"Stopping edge runtime on {ip}...")
            result = self.edge_device_manager.stop_runtime(
                ip=ip,
                username=username,
                password=password,
                remote_dir=remote_dir,
                progress_callback=self.progress_signal.emit,
            )
            if result.success:
                success_ips.append(ip)
                self.progress_signal.emit(f"Edge runtime stop successful for {ip}.")
            else:
                details = f" ({result.details})" if result.details else ""
                failures.append(f"{ip}: {result.message}{details}")
                self.progress_signal.emit(f"Edge runtime stop failed for {ip}.")

        self.finished_signal.emit({"success_ips": success_ips, "failures": failures})


class SyncPipelineWorker(QThread):
    progress_signal = Signal(str)
    finished_signal = Signal(object)

    def __init__(self, pull_callable, push_callable, parent=None):
        super().__init__(parent)
        self._pull_callable = pull_callable
        self._push_callable = push_callable

    def run(self):
        result = {
            "pulled": 0,
            "synced": 0,
            "error": "",
        }
        try:
            if self.isInterruptionRequested():
                self.finished_signal.emit(result)
                return
            result["pulled"] = int(self._pull_callable(self.progress_signal.emit) or 0)
            if self.isInterruptionRequested():
                self.finished_signal.emit(result)
                return
            result["synced"] = int(self._push_callable(self.progress_signal.emit) or 0)
        except Exception as exc:
            result["error"] = str(exc)
        self.finished_signal.emit(result)
