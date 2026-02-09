import getpass
import ipaddress
import json
import logging
import os
import shlex
import socket
import tarfile
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import psutil

logger = logging.getLogger(__name__)


@dataclass
class DiscoveredEdgeDevice:
    ip: str
    interface: str
    hostname: str = ""
    linux_verified: bool = False

    @property
    def display_name(self):
        host = self.hostname or "unknown-host"
        verification = "linux" if self.linux_verified else "ssh-open"
        return f"{host} ({self.ip}) [{self.interface}, {verification}]"


@dataclass
class EdgeDeployResult:
    success: bool
    message: str
    details: str = ""


class EdgeDeviceManager:
    def __init__(self, project_root):
        self.project_root = Path(project_root).resolve()

    def discover_devices(self, username="", password=""):
        username = (username or getpass.getuser()).strip()
        password = (password or "").strip()

        interface_networks = self._collect_candidate_networks()
        if not interface_networks:
            return []

        ip_to_interface = {}
        for interface, ip_addr, network in interface_networks:
            for host in network.hosts():
                host_ip = str(host)
                if host_ip == ip_addr:
                    continue
                ip_to_interface.setdefault(host_ip, interface)

        open_hosts = self._scan_ssh_hosts(list(ip_to_interface.keys()))
        if not open_hosts:
            return []

        discovered = []
        if username and password:
            with ThreadPoolExecutor(max_workers=min(32, max(8, len(open_hosts)))) as executor:
                futures = {
                    executor.submit(self._probe_remote_identity, ip, username, password): ip
                    for ip in open_hosts
                }
                for future in as_completed(futures):
                    host_ip = futures[future]
                    interface = ip_to_interface.get(host_ip, "unknown")
                    hostname, linux_verified = future.result()
                    discovered.append(
                        DiscoveredEdgeDevice(
                            ip=host_ip,
                            interface=interface,
                            hostname=hostname,
                            linux_verified=linux_verified,
                        )
                    )
        else:
            for host_ip in open_hosts:
                discovered.append(
                    DiscoveredEdgeDevice(
                        ip=host_ip,
                        interface=ip_to_interface.get(host_ip, "unknown"),
                        hostname="",
                        linux_verified=False,
                    )
                )

        discovered.sort(
            key=lambda d: (
                0 if d.linux_verified else 1,
                d.interface,
                self._ip_sort_key(d.ip),
            )
        )
        return discovered

    def deploy_and_launch(
        self,
        ip,
        username,
        password,
        remote_dir="~/smart_mirror_edge",
        display=":0",
        install_deps=False,
        setup_postgres=True,
        db_backend="postgres",
        pg_db="gym_edge_db",
        pg_user="",
        pg_password="admin",
        pg_host="localhost",
        pg_port="5432",
        mirror_id="",
        gym_id="",
        main_system_id="",
        progress_callback=None,
    ):
        username = (username or "").strip()
        password = (password or "").strip()
        if not username:
            return EdgeDeployResult(False, "Username is required for edge deployment.")
        if not password:
            return EdgeDeployResult(False, "Password is required for edge deployment.")

        remote_dir = self._normalize_remote_dir((remote_dir or "~/smart_mirror_edge").strip())
        display = (display or ":0").strip()
        db_backend = "postgres"
        setup_postgres_flag = "1"
        escaped_pg_db = shlex.quote(str(pg_db or "gym_edge_db"))
        default_pg_user = (
            username
            or os.getenv("SMART_MIRROR_DEFAULT_PG_USER", os.getenv("PGUSER", "")).strip()
        )
        escaped_pg_user = shlex.quote(str(pg_user or default_pg_user))
        escaped_pg_password = shlex.quote(
            str(pg_password or os.getenv("SMART_MIRROR_DEFAULT_PG_PASSWORD", "admin") or "admin")
        )
        escaped_pg_host = shlex.quote(str(pg_host or "localhost"))
        escaped_pg_port = shlex.quote(str(pg_port or "5432"))
        escaped_db_backend = shlex.quote(db_backend)
        escaped_sudo_password = shlex.quote(str(password or ""))
        escaped_mirror_id = shlex.quote(str(mirror_id or ""))
        escaped_gym_id = shlex.quote(str(gym_id or ""))
        escaped_main_system_id = shlex.quote(str(main_system_id or ""))

        paramiko = self._get_paramiko()
        if paramiko is None:
            return EdgeDeployResult(
                False,
                "Missing dependency: paramiko. Install it with 'pip install paramiko'.",
            )

        local_bundle = None
        client = None
        sftp = None
        try:
            self._emit_progress(progress_callback, f"[edge:{ip}] Connecting over SSH...")
            client = self._connect_client(paramiko, ip, username, password, timeout=8)
            sftp = client.open_sftp()
            self._emit_progress(progress_callback, f"[edge:{ip}] SSH connected.")

            local_bundle = self._create_project_bundle()
            remote_bundle = f"/tmp/mirror_code_bundle_{int(time.time())}.tar.gz"
            self._emit_progress(progress_callback, f"[edge:{ip}] Uploading code bundle...")
            sftp.put(local_bundle, remote_bundle)
            self._emit_progress(progress_callback, f"[edge:{ip}] Upload complete.")

            escaped_remote_dir = self._remote_path_expr(remote_dir)
            escaped_remote_bundle = shlex.quote(remote_bundle)
            escaped_display = shlex.quote(display)
            force_reinstall_deps = "1" if install_deps else "0"
            remote_script = (
                f"set -e; "
                f"EDGE_LOG_DIR=\"$HOME/Desktop\"; "
                f"if [ ! -d \"$EDGE_LOG_DIR\" ]; then EDGE_LOG_DIR=\"$HOME\"; fi; "
                f"DEPLOY_LOG=\"$EDGE_LOG_DIR/mirror_edge_deploy.log\"; "
                f"RUNTIME_LOG=\"$EDGE_LOG_DIR/mirror_edge_runtime.log\"; "
                f"exec >> \"$DEPLOY_LOG\" 2>&1; "
                f"echo \"==== $(date) :: deploy start ====\"; "
                f"mkdir -p {escaped_remote_dir}; "
                f"cd {escaped_remote_dir}; "
                # Always refresh code to latest: remove stale files that may survive extraction.
                # Keep local runtime state files so face model/member DB and venv are not lost.
                f"find . -mindepth 1 -maxdepth 1 "
                f"! -name '.venv' "
                f"! -name 'trained_model.hdf5' "
                f"! -name '.edge_deps.hash' "
                f"! -name 'edge_app.pid' "
                f"-exec rm -rf {{}} +; "
                f"tar -xzf {escaped_remote_bundle} -C {escaped_remote_dir}; "
                f"rm -f {escaped_remote_bundle}; "
                f"cd {escaped_remote_dir}; "
                f"python3 -m venv .venv; "
                f". .venv/bin/activate; "
                f"REQ_FILE='requirements.edge.txt'; "
                f"if [ ! -f \"$REQ_FILE\" ]; then REQ_FILE='requirements.txt'; fi; "
                f"if [ ! -f \"$REQ_FILE\" ]; then "
                f"echo 'No requirements file found for edge runtime.' >&2; exit 2; "
                f"fi; "
                f"if command -v sha256sum >/dev/null 2>&1; then "
                f"REQ_HASH=$(sha256sum \"$REQ_FILE\" | awk '{{print $1}}'); "
                f"else "
                f"REQ_HASH=$(shasum -a 256 \"$REQ_FILE\" | awk '{{print $1}}'); "
                f"fi; "
                f"INSTALLED_HASH=''; "
                f"if [ -f '.edge_deps.hash' ]; then INSTALLED_HASH=$(cat .edge_deps.hash); fi; "
                f"if [ '{force_reinstall_deps}' = '1' ] || [ \"$REQ_HASH\" != \"$INSTALLED_HASH\" ]; then "
                f"export CMAKE_BUILD_PARALLEL_LEVEL=1 MAX_JOBS=1 MAKEFLAGS='-j1'; "
                f"pip install --upgrade pip; "
                f"pip install -r \"$REQ_FILE\"; "
                f"echo \"$REQ_HASH\" > .edge_deps.hash; "
                f"fi; "
                f"if [ '{setup_postgres_flag}' = '1' ] && [ -f 'scripts/edge_setup_postgres.sh' ]; then "
                f"SMART_MIRROR_PG_DB={escaped_pg_db} "
                f"SMART_MIRROR_PG_USER={escaped_pg_user} "
                f"SMART_MIRROR_PG_PASSWORD={escaped_pg_password} "
                f"SMART_MIRROR_PG_HOST={escaped_pg_host} "
                f"SMART_MIRROR_PG_PORT={escaped_pg_port} "
                f"PGDATABASE={escaped_pg_db} "
                f"PGUSER={escaped_pg_user} "
                f"PGPASSWORD={escaped_pg_password} "
                f"PGHOST={escaped_pg_host} "
                f"PGPORT={escaped_pg_port} "
                f"SMART_MIRROR_SUDO_PASSWORD={escaped_sudo_password} "
                f"bash scripts/edge_setup_postgres.sh; "
                f"fi; "
                f"echo \"==== $(date) :: launching runtime ====\"; "
                f"cd {escaped_remote_dir}; "
                f"PY_BIN='python3'; "
                f"if [ -x '.venv/bin/python' ]; then PY_BIN='.venv/bin/python'; fi; "
                f"if [ -f edge_app.pid ]; then "
                f"OLD_PID=$(cat edge_app.pid 2>/dev/null || true); "
                f"if [ -n \"$OLD_PID\" ] && kill -0 \"$OLD_PID\" 2>/dev/null; then "
                f"kill \"$OLD_PID\" 2>/dev/null || true; "
                f"sleep 1; "
                f"if kill -0 \"$OLD_PID\" 2>/dev/null; then kill -9 \"$OLD_PID\" 2>/dev/null || true; fi; "
                f"echo \"Stopped previous runtime PID: $OLD_PID\"; "
                f"fi; "
                f"fi; "
                f"set +e; "
                f"nohup env DISPLAY={escaped_display} QT_QPA_PLATFORM=xcb "
                f"SMART_MIRROR_EDGE_MODE=1 SMART_MIRROR_EDGE_AUTOSTART=1 SMART_MIRROR_DISABLE_ARUCO=1 "
                f"SMART_MIRROR_DB_BACKEND={escaped_db_backend} "
                f"SMART_MIRROR_PG_DB={escaped_pg_db} "
                f"SMART_MIRROR_PG_USER={escaped_pg_user} "
                f"SMART_MIRROR_PG_PASSWORD={escaped_pg_password} "
                f"SMART_MIRROR_PG_HOST={escaped_pg_host} "
                f"SMART_MIRROR_PG_PORT={escaped_pg_port} "
                f"PGDATABASE={escaped_pg_db} "
                f"PGUSER={escaped_pg_user} "
                f"PGPASSWORD={escaped_pg_password} "
                f"PGHOST={escaped_pg_host} "
                f"PGPORT={escaped_pg_port} "
                f"SMART_MIRROR_MIRROR_ID={escaped_mirror_id} "
                f"SMART_MIRROR_GYM_ID={escaped_gym_id} "
                f"SMART_MIRROR_MAIN_SYSTEM_ID={escaped_main_system_id} "
                f"SMART_MIRROR_NODE_ROLE=EDGE "
                f"PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 "
                f"$PY_BIN app.py >> \"$RUNTIME_LOG\" 2>&1 < /dev/null & "
                f"LAUNCH_STATUS=$?; "
                f"set -e; "
                f"if [ \"$LAUNCH_STATUS\" -ne 0 ]; then "
                f"echo \"Runtime launch command failed with status $LAUNCH_STATUS\"; "
                f"exit \"$LAUNCH_STATUS\"; "
                f"fi; "
                f"echo $! > edge_app.pid || true; "
                f"RUNTIME_PID=$(cat edge_app.pid 2>/dev/null || true); "
                f"if [ -n \"$RUNTIME_PID\" ]; then echo \"Runtime PID: $RUNTIME_PID\"; "
                f"else echo \"Runtime PID unavailable\"; fi; "
                f"echo \"Deploy log: $DEPLOY_LOG\"; "
                f"echo \"Runtime log: $RUNTIME_LOG\"; "
                f"echo \"==== $(date) :: deploy end ====\"; "
            )

            self._emit_progress(progress_callback, f"[edge:{ip}] Running remote setup/install/start...")
            exit_code, _, stderr = self._exec_remote(
                client,
                f"bash -lc {shlex.quote(remote_script)}",
                timeout=1200,
            )
            if exit_code != 0:
                self._emit_progress(progress_callback, f"[edge:{ip}] Remote setup failed.")
                remote_log_tail = self._collect_remote_log_tail(client)
                if remote_log_tail:
                    self._emit_progress(progress_callback, f"[edge:{ip}] Remote log tail:\n{remote_log_tail}")
                combined_details = stderr.strip()
                if remote_log_tail:
                    combined_details = (
                        f"{combined_details}\n\n--- edge log tail ---\n{remote_log_tail}"
                        if combined_details
                        else remote_log_tail
                    )
                return EdgeDeployResult(
                    False,
                    "Code transferred but failed to start on edge device.",
                    combined_details,
                )

            self._emit_progress(
                progress_callback,
                f"[edge:{ip}] Runtime started. Edge logs: ~/Desktop/mirror_edge_deploy.log and ~/Desktop/mirror_edge_runtime.log",
            )
            health_script = (
                f"cd {escaped_remote_dir}; "
                f"EDGE_LOG_DIR=\"$HOME/Desktop\"; "
                f"if [ ! -d \"$EDGE_LOG_DIR\" ]; then EDGE_LOG_DIR=\"$HOME\"; fi; "
                f"RUNTIME_LOG=\"$EDGE_LOG_DIR/mirror_edge_runtime.log\"; "
                f"if [ -f edge_app.pid ] && kill -0 \"$(cat edge_app.pid)\" 2>/dev/null; then "
                f"echo \"Runtime process is alive (PID $(cat edge_app.pid)).\"; "
                f"else "
                f"echo \"Runtime process is NOT running.\"; "
                f"fi; "
                f"if [ -f \"$RUNTIME_LOG\" ]; then "
                f"echo \"--- runtime tail ---\"; tail -n 20 \"$RUNTIME_LOG\"; "
                f"fi"
            )
            _, health_stdout, _ = self._exec_remote(
                client,
                f"bash -lc {shlex.quote(health_script)}",
                timeout=20,
            )
            if health_stdout.strip():
                for line in health_stdout.strip().splitlines():
                    self._emit_progress(progress_callback, f"[edge:{ip}] {line}")

            return EdgeDeployResult(
                True,
                f"Edge device {ip} deployed and started successfully.",
            )
        except Exception as exc:
            self._emit_progress(progress_callback, f"[edge:{ip}] Deployment error: {exc}")
            return EdgeDeployResult(
                False,
                f"Edge deployment failed for {ip}.",
                str(exc),
            )
        finally:
            if sftp is not None:
                try:
                    sftp.close()
                except Exception:
                    pass
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass
            if local_bundle and os.path.exists(local_bundle):
                try:
                    os.remove(local_bundle)
                except OSError:
                    pass

    def stop_runtime(
        self,
        ip,
        username,
        password,
        remote_dir="~/smart_mirror_edge",
        progress_callback=None,
    ):
        username = (username or "").strip()
        password = (password or "").strip()
        if not username:
            return EdgeDeployResult(False, "Username is required to stop edge runtime.")
        if not password:
            return EdgeDeployResult(False, "Password is required to stop edge runtime.")

        remote_dir = self._normalize_remote_dir((remote_dir or "~/smart_mirror_edge").strip())
        paramiko = self._get_paramiko()
        if paramiko is None:
            return EdgeDeployResult(
                False,
                "Missing dependency: paramiko. Install it with 'pip install paramiko'.",
            )

        client = None
        try:
            self._emit_progress(progress_callback, f"[edge:{ip}] Connecting over SSH for stop...")
            client = self._connect_client(paramiko, ip, username, password, timeout=8)
            escaped_remote_dir = self._remote_path_expr(remote_dir)

            stop_script = (
                "set +e; "
                f"cd {escaped_remote_dir} 2>/dev/null || true; "
                "RUNTIME_PID=''; "
                "if [ -f edge_app.pid ]; then "
                "RUNTIME_PID=$(cat edge_app.pid 2>/dev/null || true); "
                "fi; "
                "if [ -n \"$RUNTIME_PID\" ] && kill -0 \"$RUNTIME_PID\" 2>/dev/null; then "
                "kill \"$RUNTIME_PID\" 2>/dev/null || true; "
                "sleep 1; "
                "if kill -0 \"$RUNTIME_PID\" 2>/dev/null; then "
                "kill -9 \"$RUNTIME_PID\" 2>/dev/null || true; "
                "fi; "
                "fi; "
                "if [ -n \"$RUNTIME_PID\" ] && kill -0 \"$RUNTIME_PID\" 2>/dev/null; then "
                "echo \"Failed to stop runtime PID $RUNTIME_PID\"; "
                "exit 1; "
                "fi; "
                "rm -f edge_app.pid 2>/dev/null || true; "
                "if [ -n \"$RUNTIME_PID\" ]; then "
                "echo \"Stopped runtime PID $RUNTIME_PID\"; "
                "else "
                "echo \"No active runtime PID found.\"; "
                "fi; "
                "exit 0"
            )

            exit_code, stdout, stderr = self._exec_remote(
                client,
                f"bash -lc {shlex.quote(stop_script)}",
                timeout=90,
            )
            if exit_code != 0:
                details = (stderr or stdout).strip()
                self._emit_progress(progress_callback, f"[edge:{ip}] Runtime stop failed.")
                return EdgeDeployResult(
                    False,
                    f"Failed to stop edge runtime on {ip}.",
                    details,
                )

            details = stdout.strip()
            self._emit_progress(progress_callback, f"[edge:{ip}] Runtime stop completed.")
            if details:
                for line in details.splitlines():
                    self._emit_progress(progress_callback, f"[edge:{ip}] {line}")
            return EdgeDeployResult(
                True,
                f"Edge runtime stop command completed for {ip}.",
                details,
            )
        except Exception as exc:
            self._emit_progress(progress_callback, f"[edge:{ip}] Stop error: {exc}")
            return EdgeDeployResult(
                False,
                f"Edge stop failed for {ip}.",
                str(exc),
            )
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass

    def fetch_pending_exercise_data(
        self,
        ip,
        username,
        password,
        remote_dir="~/smart_mirror_edge",
        limit=500,
        db_backend="postgres",
        pg_db="gym_edge_db",
        pg_user="",
        pg_password="admin",
        pg_host="localhost",
        pg_port="5432",
        progress_callback=None,
    ):
        username = (username or "").strip()
        password = (password or "").strip()
        if not username:
            return False, [], "Username is required to pull edge data."
        if not password:
            return False, [], "Password is required to pull edge data."

        paramiko = self._get_paramiko()
        if paramiko is None:
            return False, [], "Missing dependency: paramiko."

        remote_dir = self._normalize_remote_dir((remote_dir or "~/smart_mirror_edge").strip())
        escaped_remote_dir = self._remote_path_expr(remote_dir)
        limit = max(1, int(limit))
        escaped_db_backend = shlex.quote("postgres")
        escaped_pg_db = shlex.quote(str(pg_db or "gym_edge_db"))
        default_pg_user = (
            username
            or os.getenv("SMART_MIRROR_DEFAULT_PG_USER", os.getenv("PGUSER", "")).strip()
        )
        escaped_pg_user = shlex.quote(str(pg_user or default_pg_user))
        escaped_pg_password = shlex.quote(
            str(pg_password or os.getenv("SMART_MIRROR_DEFAULT_PG_PASSWORD", "admin") or "admin")
        )
        escaped_pg_host = shlex.quote(str(pg_host or "localhost"))
        escaped_pg_port = shlex.quote(str(pg_port or "5432"))

        client = None
        try:
            self._emit_progress(progress_callback, f"[edge:{ip}] Pulling pending exercise data...")
            client = self._connect_client(paramiko, ip, username, password, timeout=8)
            command = (
                f"cd {escaped_remote_dir}; "
                f"PY_BIN='python3'; "
                f"if [ -x '.venv/bin/python' ]; then PY_BIN='.venv/bin/python'; fi; "
                f"SMART_MIRROR_DB_BACKEND={escaped_db_backend} "
                f"SMART_MIRROR_PG_DB={escaped_pg_db} "
                f"SMART_MIRROR_PG_USER={escaped_pg_user} "
                f"SMART_MIRROR_PG_PASSWORD={escaped_pg_password} "
                f"SMART_MIRROR_PG_HOST={escaped_pg_host} "
                f"SMART_MIRROR_PG_PORT={escaped_pg_port} "
                f"PGDATABASE={escaped_pg_db} "
                f"PGUSER={escaped_pg_user} "
                f"PGPASSWORD={escaped_pg_password} "
                f"PGHOST={escaped_pg_host} "
                f"PGPORT={escaped_pg_port} "
                f"PYTHONPATH=\"$PWD:${{PYTHONPATH:-}}\" "
                f"$PY_BIN scripts/edge_export_pending.py --limit {limit} --mark-queued"
            )
            exit_code, stdout, stderr = self._exec_remote(
                client,
                f"bash -lc {shlex.quote(command)}",
                timeout=90,
            )
            if exit_code != 0:
                return False, [], (stderr or stdout).strip()

            try:
                payload = json.loads(stdout or "{}")
            except json.JSONDecodeError as exc:
                return False, [], f"Invalid edge export payload: {exc}"

            records = payload.get("records") or []
            self._emit_progress(
                progress_callback,
                f"[edge:{ip}] Pulled {len(records)} pending record(s).",
            )
            return True, records, ""
        except Exception as exc:
            return False, [], str(exc)
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass

    def mark_remote_records_synced(
        self,
        ip,
        username,
        password,
        record_ids,
        remote_dir="~/smart_mirror_edge",
        status="SYNCED",
        db_backend="postgres",
        pg_db="gym_edge_db",
        pg_user="",
        pg_password="admin",
        pg_host="localhost",
        pg_port="5432",
        progress_callback=None,
    ):
        clean_ids = [str(item).strip() for item in (record_ids or []) if str(item).strip()]
        if not clean_ids:
            return True, ""

        username = (username or "").strip()
        password = (password or "").strip()
        if not username:
            return False, "Username is required to mark edge records."
        if not password:
            return False, "Password is required to mark edge records."

        paramiko = self._get_paramiko()
        if paramiko is None:
            return False, "Missing dependency: paramiko."

        remote_dir = self._normalize_remote_dir((remote_dir or "~/smart_mirror_edge").strip())
        escaped_remote_dir = self._remote_path_expr(remote_dir)
        escaped_ids = shlex.quote(json.dumps(clean_ids))
        escaped_status = shlex.quote((status or "SYNCED").upper())
        escaped_db_backend = shlex.quote("postgres")
        escaped_pg_db = shlex.quote(str(pg_db or "gym_edge_db"))
        default_pg_user = (
            username
            or os.getenv("SMART_MIRROR_DEFAULT_PG_USER", os.getenv("PGUSER", "")).strip()
        )
        escaped_pg_user = shlex.quote(str(pg_user or default_pg_user))
        escaped_pg_password = shlex.quote(
            str(pg_password or os.getenv("SMART_MIRROR_DEFAULT_PG_PASSWORD", "admin") or "admin")
        )
        escaped_pg_host = shlex.quote(str(pg_host or "localhost"))
        escaped_pg_port = shlex.quote(str(pg_port or "5432"))

        client = None
        try:
            self._emit_progress(
                progress_callback,
                f"[edge:{ip}] Marking {len(clean_ids)} record(s) as {status}...",
            )
            client = self._connect_client(paramiko, ip, username, password, timeout=8)
            command = (
                f"cd {escaped_remote_dir}; "
                f"PY_BIN='python3'; "
                f"if [ -x '.venv/bin/python' ]; then PY_BIN='.venv/bin/python'; fi; "
                f"SMART_MIRROR_DB_BACKEND={escaped_db_backend} "
                f"SMART_MIRROR_PG_DB={escaped_pg_db} "
                f"SMART_MIRROR_PG_USER={escaped_pg_user} "
                f"SMART_MIRROR_PG_PASSWORD={escaped_pg_password} "
                f"SMART_MIRROR_PG_HOST={escaped_pg_host} "
                f"SMART_MIRROR_PG_PORT={escaped_pg_port} "
                f"PGDATABASE={escaped_pg_db} "
                f"PGUSER={escaped_pg_user} "
                f"PGPASSWORD={escaped_pg_password} "
                f"PGHOST={escaped_pg_host} "
                f"PGPORT={escaped_pg_port} "
                f"PYTHONPATH=\"$PWD:${{PYTHONPATH:-}}\" "
                f"$PY_BIN scripts/edge_mark_synced.py --ids {escaped_ids} --status {escaped_status}"
            )
            exit_code, stdout, stderr = self._exec_remote(
                client,
                f"bash -lc {shlex.quote(command)}",
                timeout=60,
            )
            if exit_code != 0:
                return False, (stderr or stdout).strip()

            self._emit_progress(progress_callback, f"[edge:{ip}] Sync status update complete.")
            return True, stdout.strip()
        except Exception as exc:
            return False, str(exc)
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass

    def _collect_candidate_networks(self):
        candidates = []
        interfaces = psutil.net_if_addrs()
        stats = psutil.net_if_stats()

        for interface, addrs in interfaces.items():
            if interface.lower().startswith("lo"):
                continue
            if interface in stats and not stats[interface].isup:
                continue

            for addr in addrs:
                if addr.family != socket.AF_INET:
                    continue
                ip_addr = (addr.address or "").strip()
                netmask = (addr.netmask or "").strip()
                if not ip_addr or not netmask or ip_addr.startswith("127."):
                    continue

                try:
                    network = ipaddress.IPv4Network(f"{ip_addr}/{netmask}", strict=False)
                except ValueError:
                    continue

                if network.num_addresses > 256:
                    network = ipaddress.IPv4Network(f"{ip_addr}/24", strict=False)
                candidates.append((interface, ip_addr, network))
        return candidates

    def _scan_ssh_hosts(self, host_ips):
        if not host_ips:
            return []
        open_hosts = []
        scan_timeout = float(os.getenv("SMART_MIRROR_EDGE_SCAN_TIMEOUT_SEC", "0.8"))
        scan_retries = max(1, int(os.getenv("SMART_MIRROR_EDGE_SCAN_RETRIES", "2")))

        def _has_ssh(ip_addr):
            for attempt in range(scan_retries):
                sock = None
                try:
                    sock = socket.create_connection((ip_addr, 22), timeout=scan_timeout)
                    return ip_addr
                except OSError:
                    if attempt < (scan_retries - 1):
                        time.sleep(0.05)
                finally:
                    if sock:
                        sock.close()
            return None

        with ThreadPoolExecutor(max_workers=min(128, max(8, len(host_ips)))) as executor:
            futures = [executor.submit(_has_ssh, ip_addr) for ip_addr in host_ips]
            for future in as_completed(futures):
                value = future.result()
                if value:
                    open_hosts.append(value)
        return open_hosts

    def _probe_remote_identity(self, ip, username, password):
        paramiko = self._get_paramiko()
        if paramiko is None:
            return "", False
        client = None
        try:
            client = self._connect_client(paramiko, ip, username, password, timeout=4)
            exit_code, stdout, _ = self._exec_remote(client, "uname -s; hostname", timeout=8)
            if exit_code != 0:
                return "", False
            lines = [line.strip() for line in stdout.splitlines() if line.strip()]
            if not lines:
                return "", False
            is_linux = lines[0] == "Linux"
            hostname = lines[1] if len(lines) > 1 else ""
            return hostname, is_linux
        except Exception:
            return "", False
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass

    def _create_project_bundle(self):
        fd, bundle_path = tempfile.mkstemp(prefix="mirror_code_bundle_", suffix=".tar.gz")
        os.close(fd)
        bundle_path = Path(bundle_path)
        excluded_dirs = {
            ".git",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            ".idea",
            ".vscode",
            ".venv",
            "venv",
        }
        excluded_files = {
            "app.log",
            "worker.log",
            "network_utils.log",
            "local_edge_buffer.db",
        }

        with tarfile.open(bundle_path, "w:gz") as tar:
            for path in self.project_root.rglob("*"):
                rel_path = path.relative_to(self.project_root)
                parts = set(rel_path.parts)
                if parts & excluded_dirs:
                    continue
                if path.is_file() and (
                    path.name in excluded_files
                    or path.suffix in {".db", ".sqlite", ".sqlite3"}
                ):
                    continue
                tar.add(path, arcname=str(rel_path))
        return str(bundle_path)

    @staticmethod
    def _normalize_remote_dir(remote_dir):
        if remote_dir == "~":
            return "$HOME"
        if remote_dir.startswith("~/"):
            return f"$HOME/{remote_dir[2:]}"
        return remote_dir

    @staticmethod
    def _remote_path_expr(remote_path):
        if remote_path == "$HOME":
            return '"${HOME}"'
        if remote_path.startswith("$HOME/"):
            suffix = remote_path[len("$HOME/") :]
            suffix = (
                suffix.replace("\\", "\\\\")
                .replace('"', '\\"')
                .replace("$", "\\$")
                .replace("`", "\\`")
            )
            return f'"${{HOME}}/{suffix}"'
        return shlex.quote(remote_path)

    @staticmethod
    def _exec_remote(client, command, timeout=30):
        stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
        _ = stdin
        exit_code = stdout.channel.recv_exit_status()
        return exit_code, stdout.read().decode("utf-8", errors="replace"), stderr.read().decode(
            "utf-8", errors="replace"
        )

    @staticmethod
    def _connect_client(paramiko, ip, username, password, timeout=8):
        max_attempts = 2
        last_exc = None
        for attempt in range(max_attempts):
            # Fast pre-check avoids expensive SSH handshakes when target port
            # is unreachable and improves reliability on flaky Wi-Fi.
            probe = None
            try:
                probe = socket.create_connection((ip, 22), timeout=timeout)
            finally:
                if probe is not None:
                    try:
                        probe.close()
                    except Exception:
                        pass

            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            try:
                client.connect(
                    ip,
                    username=username,
                    password=password,
                    timeout=timeout,
                    auth_timeout=timeout,
                    banner_timeout=timeout,
                    look_for_keys=False,
                    allow_agent=False,
                )
                return client
            except Exception as exc:
                last_exc = exc
                try:
                    client.close()
                except Exception:
                    pass
                if attempt < (max_attempts - 1):
                    time.sleep(0.2)
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"Unable to connect to {ip} over SSH.")

    @staticmethod
    def _get_paramiko():
        try:
            import paramiko  # type: ignore

            return paramiko
        except Exception:
            return None

    @staticmethod
    def _ip_sort_key(ip_addr):
        try:
            return tuple(int(x) for x in ip_addr.split("."))
        except ValueError:
            return (999, 999, 999, 999)

    def _collect_remote_log_tail(self, client):
        try:
            script = (
                "EDGE_LOG_DIR=\"$HOME/Desktop\"; "
                "if [ ! -d \"$EDGE_LOG_DIR\" ]; then EDGE_LOG_DIR=\"$HOME\"; fi; "
                "DEPLOY_LOG=\"$EDGE_LOG_DIR/mirror_edge_deploy.log\"; "
                "RUNTIME_LOG=\"$EDGE_LOG_DIR/mirror_edge_runtime.log\"; "
                "if [ -f \"$DEPLOY_LOG\" ]; then "
                "echo \"[deploy]\"; tail -n 50 \"$DEPLOY_LOG\"; "
                "fi; "
                "if [ -f \"$RUNTIME_LOG\" ]; then "
                "echo \"[runtime]\"; tail -n 50 \"$RUNTIME_LOG\"; "
                "fi"
            )
            _, stdout, stderr = self._exec_remote(
                client,
                f"bash -lc {shlex.quote(script)}",
                timeout=25,
            )
            output = stdout.strip()
            if output:
                return output
            return stderr.strip()
        except Exception as exc:
            return f"Failed to fetch remote logs: {exc}"

    @staticmethod
    def _emit_progress(progress_callback, message):
        logger.info(message)
        if progress_callback:
            try:
                progress_callback(message)
            except Exception:
                pass
