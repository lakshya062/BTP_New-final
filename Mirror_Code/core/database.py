import json
import logging
import os
import sqlite3
import threading
import time
import uuid
import getpass
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import psycopg2
    from psycopg2.extras import Json
except Exception:  # pragma: no cover - optional runtime dependency
    psycopg2 = None
    Json = None


class DatabaseHandler:
    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super(DatabaseHandler, cls).__new__(cls)
        return cls._instance

    def __init__(self, local_db_file=None):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._initialized = True

        self.local_db_file = local_db_file or ""
        self.lock = threading.Lock()
        self.backend = os.getenv("SMART_MIRROR_DB_BACKEND", "postgres").strip().lower()
        self.connection = None
        self.cursor = None
        self._cached_default_mirror_id = None

        if self.backend == "postgres":
            self._connect_postgres()
        else:
            raise RuntimeError(
                "SQLite backend is disabled. Set SMART_MIRROR_DB_BACKEND=postgres."
            )

        self.setup_local_db()

    def _connect_sqlite(self):
        self.connection = sqlite3.connect(
            self.local_db_file,
            check_same_thread=False,
            timeout=30,
            isolation_level=None,
        )
        self.connection.execute("PRAGMA foreign_keys = ON;")
        self.connection.execute("PRAGMA journal_mode = WAL;")
        self.connection.execute("PRAGMA synchronous = NORMAL;")
        self.cursor = self.connection.cursor()
        try:
            os.chmod(self.local_db_file, 0o600)
        except OSError:
            pass
        logger.info("Database backend: sqlite (%s)", self.local_db_file)

    def _connect_postgres(self):
        if psycopg2 is None:
            raise RuntimeError("psycopg2 is required for PostgreSQL backend.")

        db_name = os.getenv("SMART_MIRROR_PG_DB", "gym_main_db")
        db_user = os.getenv(
            "SMART_MIRROR_PG_USER",
            os.getenv(
                "SMART_MIRROR_DEFAULT_PG_USER",
                os.getenv("PGUSER", getpass.getuser()),
            ),
        )
        db_password = os.getenv(
            "SMART_MIRROR_PG_PASSWORD",
            os.getenv("SMART_MIRROR_DEFAULT_PG_PASSWORD", os.getenv("PGPASSWORD", "admin")),
        )
        db_host = os.getenv("SMART_MIRROR_PG_HOST", "localhost")
        db_port = int(os.getenv("SMART_MIRROR_PG_PORT", "5432"))

        if db_user:
            os.environ.setdefault("PGUSER", db_user)
        if db_password:
            os.environ.setdefault("PGPASSWORD", db_password)
        os.environ.setdefault("PGHOST", db_host)
        os.environ.setdefault("PGPORT", str(db_port))
        os.environ.setdefault("PGDATABASE", db_name)

        connect_kwargs = {
            "dbname": db_name,
            "user": db_user,
            "password": db_password,
            "host": db_host,
            "port": db_port,
            "connect_timeout": 8,
        }

        self.connection = psycopg2.connect(**connect_kwargs)
        self.connection.autocommit = True
        self.cursor = self.connection.cursor()
        logger.info("Database backend: postgres (%s@%s:%s/%s)", db_user, db_host, db_port, db_name)

    def setup_local_db(self):
        with self.lock:
            try:
                if self.backend == "postgres":
                    self._setup_postgres_schema()
                else:
                    self._setup_sqlite_schema()
                logger.info("Local database schema setup completed.")
            except Exception as exc:
                logger.error("Database setup error: %s", exc)
                raise

    def _setup_sqlite_schema(self):
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS members (
                user_id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT DEFAULT 'NA',
                membership TEXT DEFAULT 'NA',
                joined_on TEXT NOT NULL,
                first_name TEXT,
                last_name TEXT,
                phone TEXT,
                gender TEXT,
                date_of_birth TEXT,
                height_cm REAL,
                weight_kg REAL,
                password_hash TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            )
            """
        )

        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS exercise_data (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                username_snapshot TEXT,
                exercise TEXT NOT NULL,
                set_count INTEGER DEFAULT 0,
                sets_reps TEXT,
                rep_data TEXT,
                timestamp TEXT,
                date TEXT,
                gym_id TEXT,
                main_system_id TEXT,
                mirror_id TEXT,
                source_node TEXT DEFAULT 'EDGE',
                equipment_type TEXT DEFAULT 'FREE_WEIGHT',
                machine_id TEXT,
                sync_status TEXT DEFAULT 'PENDING',
                synced_at TEXT,
                source_payload_hash TEXT,
                source_batch_id TEXT,
                source_edge_ip TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (user_id) REFERENCES members (user_id) ON DELETE CASCADE
            )
            """
        )

        # Backward-compatible migrations for existing local DB files.
        self._ensure_sqlite_column("members", "first_name", "TEXT")
        self._ensure_sqlite_column("members", "last_name", "TEXT")
        self._ensure_sqlite_column("members", "phone", "TEXT")
        self._ensure_sqlite_column("members", "gender", "TEXT")
        self._ensure_sqlite_column("members", "date_of_birth", "TEXT")
        self._ensure_sqlite_column("members", "height_cm", "REAL")
        self._ensure_sqlite_column("members", "weight_kg", "REAL")
        self._ensure_sqlite_column("members", "password_hash", "TEXT")
        self._ensure_sqlite_column("members", "created_at", "TEXT DEFAULT (datetime('now'))")
        self._ensure_sqlite_column("members", "updated_at", "TEXT DEFAULT (datetime('now'))")

        sqlite_columns = {
            "username_snapshot": "TEXT",
            "gym_id": "TEXT",
            "main_system_id": "TEXT",
            "mirror_id": "TEXT",
            "source_node": "TEXT DEFAULT 'EDGE'",
            "equipment_type": "TEXT DEFAULT 'FREE_WEIGHT'",
            "machine_id": "TEXT",
            "sync_status": "TEXT DEFAULT 'PENDING'",
            "synced_at": "TEXT",
            "source_payload_hash": "TEXT",
            "source_batch_id": "TEXT",
            "source_edge_ip": "TEXT",
            "created_at": "TEXT DEFAULT (datetime('now'))",
            "updated_at": "TEXT DEFAULT (datetime('now'))",
        }
        for col_name, col_def in sqlite_columns.items():
            self._ensure_sqlite_column("exercise_data", col_name, col_def)

        self.cursor.execute("UPDATE members SET created_at = datetime('now') WHERE created_at IS NULL")
        self.cursor.execute("UPDATE members SET updated_at = datetime('now') WHERE updated_at IS NULL")
        self.cursor.execute("UPDATE exercise_data SET created_at = datetime('now') WHERE created_at IS NULL")
        self.cursor.execute("UPDATE exercise_data SET updated_at = datetime('now') WHERE updated_at IS NULL")

        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_members_username ON members(username)")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_exercise_user_id ON exercise_data(user_id)")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_exercise_timestamp ON exercise_data(timestamp)")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_exercise_name ON exercise_data(exercise)")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_exercise_sync_status ON exercise_data(sync_status)")
        self.connection.commit()

    def _setup_postgres_schema(self):
        self.cursor.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS members (
                user_id UUID PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT DEFAULT 'NA',
                membership TEXT DEFAULT 'NA',
                joined_on DATE NOT NULL,
                first_name TEXT,
                last_name TEXT,
                phone TEXT,
                gender TEXT,
                date_of_birth DATE,
                height_cm DOUBLE PRECISION,
                weight_kg DOUBLE PRECISION,
                password_hash TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        self.cursor.execute("ALTER TABLE members ADD COLUMN IF NOT EXISTS first_name TEXT")
        self.cursor.execute("ALTER TABLE members ADD COLUMN IF NOT EXISTS last_name TEXT")
        self.cursor.execute("ALTER TABLE members ADD COLUMN IF NOT EXISTS phone TEXT")
        self.cursor.execute("ALTER TABLE members ADD COLUMN IF NOT EXISTS gender TEXT")
        self.cursor.execute("ALTER TABLE members ADD COLUMN IF NOT EXISTS date_of_birth DATE")
        self.cursor.execute("ALTER TABLE members ADD COLUMN IF NOT EXISTS height_cm DOUBLE PRECISION")
        self.cursor.execute("ALTER TABLE members ADD COLUMN IF NOT EXISTS weight_kg DOUBLE PRECISION")
        self.cursor.execute("ALTER TABLE members ADD COLUMN IF NOT EXISTS password_hash TEXT")

        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS exercise_data (
                id UUID PRIMARY KEY,
                user_id UUID NOT NULL REFERENCES members(user_id) ON DELETE CASCADE,
                username_snapshot TEXT,
                exercise TEXT NOT NULL,
                set_count INTEGER NOT NULL DEFAULT 0,
                sets_reps JSONB NOT NULL DEFAULT '[]'::jsonb,
                rep_data JSONB NOT NULL DEFAULT '[]'::jsonb,
                timestamp TIMESTAMPTZ NOT NULL,
                date DATE NOT NULL,
                gym_id UUID,
                main_system_id UUID,
                mirror_id UUID,
                source_node TEXT NOT NULL DEFAULT 'EDGE',
                equipment_type TEXT NOT NULL DEFAULT 'FREE_WEIGHT',
                machine_id UUID,
                sync_status TEXT NOT NULL DEFAULT 'PENDING',
                synced_at TIMESTAMPTZ,
                source_payload_hash TEXT,
                source_batch_id UUID,
                source_edge_ip TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )

        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_members_username ON members(username)")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_exercise_user_id ON exercise_data(user_id)")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_exercise_timestamp ON exercise_data(timestamp DESC)")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_exercise_name ON exercise_data(exercise)")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_exercise_sync_status ON exercise_data(sync_status)")
        self._ensure_postgres_exercise_data_columns()
        self.cursor.execute("ALTER TABLE exercise_data ALTER COLUMN mirror_id SET DEFAULT gen_random_uuid()")
        self.cursor.execute("UPDATE exercise_data SET mirror_id = gen_random_uuid() WHERE mirror_id IS NULL")

    def _postgres_column_exists(self, table_name, column_name):
        self.cursor.execute(
            """
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = %s
              AND column_name = %s
            LIMIT 1
            """,
            (table_name, column_name),
        )
        return self.cursor.fetchone() is not None

    def _ensure_postgres_exercise_data_columns(self):
        canonical_col = "source_edge_ip"
        legacy_col = "source_Edge_ip"

        has_canonical = self._postgres_column_exists("exercise_data", canonical_col)
        has_legacy = self._postgres_column_exists("exercise_data", legacy_col)

        if has_legacy and not has_canonical:
            self.cursor.execute(
                'ALTER TABLE exercise_data RENAME COLUMN "source_Edge_ip" TO source_edge_ip'
            )
            has_canonical = True

        if not has_canonical:
            self.cursor.execute("ALTER TABLE exercise_data ADD COLUMN IF NOT EXISTS source_edge_ip TEXT")

        default_mirror_id = self._resolve_default_mirror_id()
        if default_mirror_id:
            self.cursor.execute(
                "UPDATE exercise_data SET mirror_id = %s WHERE mirror_id IS NULL",
                (default_mirror_id,),
            )

    def _ensure_sqlite_column(self, table_name, column_name, column_def):
        self.cursor.execute(f"PRAGMA table_info({table_name})")
        existing_cols = {row[1] for row in self.cursor.fetchall()}
        if column_name not in existing_cols:
            safe_col_def = self._sqlite_safe_column_def(column_def)
            self.cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {safe_col_def}")

    @staticmethod
    def _sqlite_safe_column_def(column_def):
        # SQLite ALTER TABLE ADD COLUMN does not allow non-constant defaults
        # like datetime('now'). Strip them when adding columns to existing tables.
        normalized = str(column_def).strip()
        lowered = normalized.lower()
        if "default (datetime(" in lowered:
            return normalized.split("DEFAULT", 1)[0].strip()
        return normalized

    @staticmethod
    def retry_db_operation(func):
        def wrapper(self, *args, **kwargs):
            max_retries = 5
            delay = 0.1
            for _ in range(max_retries):
                try:
                    return func(self, *args, **kwargs)
                except sqlite3.OperationalError as exc:
                    if "database is locked" in str(exc):
                        logger.warning("Database is locked. Retrying in %s seconds...", delay)
                        time.sleep(delay)
                        delay *= 2
                    else:
                        raise
                except Exception as exc:
                    logger.error("Database error in %s: %s", func.__name__, exc)
                    raise
            logger.error("Failed to execute %s after retries.", func.__name__)
            return False

        return wrapper

    def _normalize_uuid(self, value):
        if not value:
            return None
        try:
            return str(uuid.UUID(str(value)))
        except ValueError:
            return None

    def _resolve_default_mirror_id(self):
        if self._cached_default_mirror_id:
            return self._cached_default_mirror_id

        mirror_id = self._normalize_uuid(os.getenv("SMART_MIRROR_MIRROR_ID"))
        if mirror_id:
            self._cached_default_mirror_id = mirror_id
            return mirror_id

        seed_parts = [
            os.getenv("SMART_MIRROR_NODE_ROLE", "MAIN"),
            os.getenv("SMART_MIRROR_GYM_ID", ""),
            os.getenv("SMART_MIRROR_MAIN_SYSTEM_ID", ""),
            os.getenv("SMART_MIRROR_PG_DB", ""),
            os.getenv("HOSTNAME", ""),
        ]
        seed = "|".join(seed_parts)
        mirror_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"smart-mirror:{seed}"))
        os.environ["SMART_MIRROR_MIRROR_ID"] = mirror_id
        self._cached_default_mirror_id = mirror_id
        logger.warning(
            "SMART_MIRROR_MIRROR_ID was missing/invalid. Using generated mirror_id=%s",
            mirror_id,
        )
        return mirror_id

    def _parse_timestamp(self, value):
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value)

    def _to_date_str(self, value):
        if not value:
            return datetime.utcnow().strftime("%Y-%m-%d")
        if isinstance(value, datetime):
            return value.strftime("%Y-%m-%d")
        value = str(value)
        if "T" in value:
            return value.split("T", 1)[0]
        if " " in value:
            return value.split(" ", 1)[0]
        return value

    def _json_load(self, value, default):
        if value is None:
            return default
        if isinstance(value, (list, dict)):
            return value
        try:
            return json.loads(value)
        except Exception:
            return default

    def _edge_role_default(self):
        role = os.getenv("SMART_MIRROR_NODE_ROLE", "EDGE").strip().upper()
        if role not in {"EDGE", "MAIN"}:
            return "EDGE"
        return role

    @staticmethod
    def _safe_float(value):
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def get_member_info_local(self, username):
        query = """
            SELECT user_id, username, email, membership, joined_on,
                   first_name, last_name, phone, gender, date_of_birth,
                   height_cm, weight_kg, password_hash
            FROM members
            WHERE username = %s
        """
        params = (username,)
        if self.backend == "sqlite":
            query = query.replace("%s", "?")

        try:
            with self.lock:
                self.cursor.execute(query, params)
                row = self.cursor.fetchone()
                if not row:
                    return None
                return {
                    "user_id": str(row[0]),
                    "username": row[1],
                    "email": row[2],
                    "membership": row[3],
                    "joined_on": str(row[4]),
                    "first_name": row[5],
                    "last_name": row[6],
                    "phone": row[7],
                    "gender": row[8],
                    "date_of_birth": str(row[9]) if row[9] is not None else None,
                    "height_cm": row[10],
                    "weight_kg": row[11],
                    "password_hash": row[12],
                }
        except Exception as exc:
            logger.error("Database error in get_member_info_local: %s", exc)
            return None

    @retry_db_operation
    def insert_member_local(self, member_info):
        user_id = self._normalize_uuid(member_info.get("user_id"))
        if not user_id:
            user_id = str(uuid.uuid4())

        username = (member_info.get("username") or "").strip()
        if not username:
            logger.error("insert_member_local missing username")
            return False

        email = member_info.get("email", "NA")
        membership = member_info.get("membership", "NA")
        joined_on = self._to_date_str(member_info.get("joined_on"))
        first_name = (member_info.get("first_name") or "").strip() or None
        last_name = (member_info.get("last_name") or "").strip() or None
        phone = (member_info.get("phone") or "").strip() or None
        gender = (member_info.get("gender") or "").strip() or None
        date_of_birth = self._to_date_str(member_info.get("date_of_birth")) if member_info.get("date_of_birth") else None
        height_cm = self._safe_float(member_info.get("height_cm"))
        weight_kg = self._safe_float(member_info.get("weight_kg"))
        password_hash = member_info.get("password_hash")

        try:
            with self.lock:
                if self.backend == "postgres":
                    self.cursor.execute(
                        """
                        INSERT INTO members (
                            user_id, username, email, membership, joined_on,
                            first_name, last_name, phone, gender, date_of_birth,
                            height_cm, weight_kg, password_hash
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (username) DO UPDATE
                        SET email = EXCLUDED.email,
                            membership = EXCLUDED.membership,
                            joined_on = EXCLUDED.joined_on,
                            first_name = EXCLUDED.first_name,
                            last_name = EXCLUDED.last_name,
                            phone = EXCLUDED.phone,
                            gender = EXCLUDED.gender,
                            date_of_birth = EXCLUDED.date_of_birth,
                            height_cm = EXCLUDED.height_cm,
                            weight_kg = EXCLUDED.weight_kg,
                            password_hash = EXCLUDED.password_hash,
                            updated_at = NOW()
                        """,
                        (
                            user_id,
                            username,
                            email,
                            membership,
                            joined_on,
                            first_name,
                            last_name,
                            phone,
                            gender,
                            date_of_birth,
                            height_cm,
                            weight_kg,
                            password_hash,
                        ),
                    )
                else:
                    self.cursor.execute(
                        """
                        INSERT INTO members (
                            user_id, username, email, membership, joined_on,
                            first_name, last_name, phone, gender, date_of_birth,
                            height_cm, weight_kg, password_hash
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(username) DO UPDATE SET
                            email = excluded.email,
                            membership = excluded.membership,
                            joined_on = excluded.joined_on,
                            first_name = excluded.first_name,
                            last_name = excluded.last_name,
                            phone = excluded.phone,
                            gender = excluded.gender,
                            date_of_birth = excluded.date_of_birth,
                            height_cm = excluded.height_cm,
                            weight_kg = excluded.weight_kg,
                            password_hash = excluded.password_hash,
                            updated_at = datetime('now')
                        """,
                        (
                            user_id,
                            username,
                            email,
                            membership,
                            joined_on,
                            first_name,
                            last_name,
                            phone,
                            gender,
                            date_of_birth,
                            height_cm,
                            weight_kg,
                            password_hash,
                        ),
                    )
                    self.connection.commit()

                logger.info("Inserted/updated member %s.", username)
                return True
        except Exception as exc:
            logger.error("Error inserting member %s: %s", username, exc)
            return False

    @retry_db_operation
    def delete_member_local(self, username):
        try:
            with self.lock:
                if self.backend == "postgres":
                    self.cursor.execute("SELECT user_id FROM members WHERE username = %s", (username,))
                else:
                    self.cursor.execute("SELECT user_id FROM members WHERE username = ?", (username,))
                row = self.cursor.fetchone()
                if not row:
                    logger.warning("No member found with username: %s", username)
                    return False
                user_id = row[0]

                if self.backend == "postgres":
                    self.cursor.execute("DELETE FROM exercise_data WHERE user_id = %s", (user_id,))
                    self.cursor.execute("DELETE FROM members WHERE username = %s", (username,))
                else:
                    self.cursor.execute("DELETE FROM exercise_data WHERE user_id = ?", (user_id,))
                    self.cursor.execute("DELETE FROM members WHERE username = ?", (username,))
                    self.connection.commit()

                logger.info("Deleted member %s and associated exercise data.", username)
                return True
        except Exception as exc:
            logger.error("Error deleting member %s: %s", username, exc)
            return False

    def get_all_members_local(self):
        try:
            with self.lock:
                self.cursor.execute(
                    """
                    SELECT user_id, username, email, membership, joined_on,
                           first_name, last_name, phone, gender, date_of_birth,
                           height_cm, weight_kg, password_hash
                    FROM members
                    ORDER BY username
                    """
                )
                rows = self.cursor.fetchall()
                return [
                    {
                        "user_id": str(row[0]),
                        "username": row[1],
                        "email": row[2],
                        "membership": row[3],
                        "joined_on": str(row[4]),
                        "first_name": row[5],
                        "last_name": row[6],
                        "phone": row[7],
                        "gender": row[8],
                        "date_of_birth": str(row[9]) if row[9] is not None else None,
                        "height_cm": row[10],
                        "weight_kg": row[11],
                        "password_hash": row[12],
                    }
                    for row in rows
                ]
        except Exception as exc:
            logger.error("Database error in get_all_members_local: %s", exc)
            return []

    def get_member_info(self, username):
        return self.get_member_info_local(username)

    def get_all_members(self):
        return self.get_all_members_local()

    def _normalize_exercise_record(self, record):
        rec_id = self._normalize_uuid(record.get("id"))
        if not rec_id:
            rec_id = str(uuid.uuid4())

        user_id = self._normalize_uuid(record.get("user_id"))
        if not user_id:
            raise ValueError("record.user_id must be a valid UUID")

        timestamp = record.get("timestamp") or datetime.utcnow().isoformat()
        date_str = self._to_date_str(record.get("date") or timestamp)

        mirror_id = (
            self._normalize_uuid(record.get("mirror_id"))
            or self._normalize_uuid(os.getenv("SMART_MIRROR_MIRROR_ID"))
            or self._resolve_default_mirror_id()
        )
        if not mirror_id:
            raise ValueError("record.mirror_id could not be resolved")

        normalized = {
            "id": rec_id,
            "user_id": user_id,
            "username_snapshot": record.get("username_snapshot"),
            "exercise": record.get("exercise") or "unknown",
            "set_count": int(record.get("set_count") or 0),
            "sets_reps": record.get("sets_reps") or [],
            "rep_data": record.get("rep_data") or [],
            "timestamp": self._parse_timestamp(timestamp),
            "date": date_str,
            "gym_id": self._normalize_uuid(record.get("gym_id") or os.getenv("SMART_MIRROR_GYM_ID")),
            "main_system_id": self._normalize_uuid(
                record.get("main_system_id") or os.getenv("SMART_MIRROR_MAIN_SYSTEM_ID")
            ),
            "mirror_id": mirror_id,
            "source_node": (record.get("source_node") or self._edge_role_default()).upper(),
            "equipment_type": (record.get("equipment_type") or "FREE_WEIGHT").upper(),
            "machine_id": self._normalize_uuid(record.get("machine_id")),
            "sync_status": (record.get("sync_status") or "PENDING").upper(),
            "synced_at": record.get("synced_at"),
            "source_payload_hash": record.get("source_payload_hash"),
            "source_batch_id": self._normalize_uuid(record.get("source_batch_id")),
            "source_edge_ip": record.get("source_edge_ip") or record.get("source_Edge_ip"),
        }
        return normalized

    @retry_db_operation
    def insert_exercise_data_local(self, record):
        try:
            normalized = self._normalize_exercise_record(record)
        except Exception as exc:
            logger.error("Invalid exercise record: %s", exc)
            return False

        if not normalized.get("mirror_id"):
            normalized["mirror_id"] = self._resolve_default_mirror_id()
        if not normalized.get("mirror_id"):
            logger.error("Unable to persist exercise record id=%s because mirror_id is unresolved.", normalized.get("id"))
            return False

        try:
            with self.lock:
                if self.backend == "postgres":
                    self.cursor.execute(
                        """
                        INSERT INTO exercise_data (
                            id, user_id, username_snapshot, exercise, set_count,
                            sets_reps, rep_data, timestamp, date,
                            gym_id, main_system_id, mirror_id, source_node,
                            equipment_type, machine_id, sync_status, synced_at,
                            source_payload_hash, source_batch_id, source_edge_ip
                        )
                        VALUES (
                            %s, %s, %s, %s, %s,
                            %s, %s, %s, %s,
                            %s, %s, COALESCE(%s::uuid, gen_random_uuid()), %s,
                            %s, %s, %s, %s,
                            %s, %s, %s
                        )
                        ON CONFLICT (id) DO UPDATE
                        SET user_id = EXCLUDED.user_id,
                            username_snapshot = EXCLUDED.username_snapshot,
                            exercise = EXCLUDED.exercise,
                            set_count = EXCLUDED.set_count,
                            sets_reps = EXCLUDED.sets_reps,
                            rep_data = EXCLUDED.rep_data,
                            timestamp = EXCLUDED.timestamp,
                            date = EXCLUDED.date,
                            gym_id = EXCLUDED.gym_id,
                            main_system_id = EXCLUDED.main_system_id,
                            mirror_id = COALESCE(EXCLUDED.mirror_id, exercise_data.mirror_id, gen_random_uuid()),
                            source_node = EXCLUDED.source_node,
                            equipment_type = EXCLUDED.equipment_type,
                            machine_id = EXCLUDED.machine_id,
                            sync_status = EXCLUDED.sync_status,
                            synced_at = EXCLUDED.synced_at,
                            source_payload_hash = EXCLUDED.source_payload_hash,
                            source_batch_id = EXCLUDED.source_batch_id,
                            source_edge_ip = EXCLUDED.source_edge_ip,
                            updated_at = NOW()
                        """,
                        (
                            normalized["id"],
                            normalized["user_id"],
                            normalized["username_snapshot"],
                            normalized["exercise"],
                            normalized["set_count"],
                            Json(normalized["sets_reps"]),
                            Json(normalized["rep_data"]),
                            normalized["timestamp"],
                            normalized["date"],
                            normalized["gym_id"],
                            normalized["main_system_id"],
                            normalized["mirror_id"],
                            normalized["source_node"],
                            normalized["equipment_type"],
                            normalized["machine_id"],
                            normalized["sync_status"],
                            normalized["synced_at"],
                            normalized["source_payload_hash"],
                            normalized["source_batch_id"],
                            normalized["source_edge_ip"],
                        ),
                    )
                else:
                    self.cursor.execute(
                        """
                        INSERT INTO exercise_data (
                            id, user_id, username_snapshot, exercise, set_count, sets_reps, rep_data,
                            timestamp, date, gym_id, main_system_id, mirror_id, source_node,
                            equipment_type, machine_id, sync_status, synced_at,
                            source_payload_hash, source_batch_id, source_edge_ip
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(id) DO UPDATE SET
                            user_id = excluded.user_id,
                            username_snapshot = excluded.username_snapshot,
                            exercise = excluded.exercise,
                            set_count = excluded.set_count,
                            sets_reps = excluded.sets_reps,
                            rep_data = excluded.rep_data,
                            timestamp = excluded.timestamp,
                            date = excluded.date,
                            gym_id = excluded.gym_id,
                            main_system_id = excluded.main_system_id,
                            mirror_id = excluded.mirror_id,
                            source_node = excluded.source_node,
                            equipment_type = excluded.equipment_type,
                            machine_id = excluded.machine_id,
                            sync_status = excluded.sync_status,
                            synced_at = excluded.synced_at,
                            source_payload_hash = excluded.source_payload_hash,
                            source_batch_id = excluded.source_batch_id,
                            source_edge_ip = excluded.source_edge_ip,
                            updated_at = datetime('now')
                        """,
                        (
                            normalized["id"],
                            normalized["user_id"],
                            normalized["username_snapshot"],
                            normalized["exercise"],
                            normalized["set_count"],
                            json.dumps(normalized["sets_reps"]),
                            json.dumps(normalized["rep_data"]),
                            normalized["timestamp"],
                            normalized["date"],
                            normalized["gym_id"],
                            normalized["main_system_id"],
                            normalized["mirror_id"],
                            normalized["source_node"],
                            normalized["equipment_type"],
                            normalized["machine_id"],
                            normalized["sync_status"],
                            normalized["synced_at"],
                            normalized["source_payload_hash"],
                            normalized["source_batch_id"],
                            normalized["source_edge_ip"],
                        ),
                    )
                    self.connection.commit()

                logger.info("Inserted/updated exercise data id=%s for user=%s.", normalized["id"], normalized["user_id"])
                return True
        except Exception as exc:
            logger.error("Error inserting exercise data: %s", exc)
            return False

    def get_exercise_data_for_user_local(self, user_id):
        user_id_norm = self._normalize_uuid(user_id)
        if not user_id_norm:
            return []

        try:
            with self.lock:
                if self.backend == "postgres":
                    self.cursor.execute(
                        """
                        SELECT id, user_id, exercise, set_count, sets_reps, rep_data,
                               timestamp, date, gym_id, main_system_id, mirror_id,
                               source_node, equipment_type, machine_id, sync_status, synced_at
                        FROM exercise_data
                        WHERE user_id = %s
                        ORDER BY timestamp DESC
                        """,
                        (user_id_norm,),
                    )
                else:
                    self.cursor.execute(
                        """
                        SELECT id, user_id, exercise, set_count, sets_reps, rep_data,
                               timestamp, date, gym_id, main_system_id, mirror_id,
                               source_node, equipment_type, machine_id, sync_status, synced_at
                        FROM exercise_data
                        WHERE user_id = ?
                        ORDER BY timestamp DESC
                        """,
                        (user_id_norm,),
                    )

                rows = self.cursor.fetchall()
                data = []
                for row in rows:
                    sets_reps = row[4] if self.backend == "postgres" else self._json_load(row[4], [])
                    rep_data = row[5] if self.backend == "postgres" else self._json_load(row[5], [])
                    data.append(
                        {
                            "id": str(row[0]),
                            "user_id": str(row[1]),
                            "exercise": row[2],
                            "set_count": row[3] or 0,
                            "sets_reps": sets_reps or [],
                            "rep_data": rep_data or [],
                            "timestamp": self._parse_timestamp(row[6]),
                            "date": str(row[7]) if row[7] is not None else "",
                            "gym_id": str(row[8]) if row[8] else None,
                            "main_system_id": str(row[9]) if row[9] else None,
                            "mirror_id": str(row[10]) if row[10] else None,
                            "source_node": row[11],
                            "equipment_type": row[12],
                            "machine_id": str(row[13]) if row[13] else None,
                            "sync_status": row[14],
                            "synced_at": self._parse_timestamp(row[15]),
                        }
                    )
                return data
        except Exception as exc:
            logger.error("Database error in get_exercise_data_for_user_local: %s", exc)
            return []

    def get_exercise_data_for_user(self, user_id):
        return self.get_exercise_data_for_user_local(user_id)

    def get_pending_exercise_data(self, limit=500):
        limit = max(1, int(limit))
        try:
            with self.lock:
                if self.backend == "postgres":
                    self.cursor.execute(
                        """
                        SELECT id, user_id, username_snapshot, exercise, set_count, sets_reps,
                               rep_data, timestamp, date, gym_id, main_system_id, mirror_id,
                               source_node, equipment_type, machine_id, sync_status,
                               synced_at, source_payload_hash, source_batch_id, source_edge_ip
                        FROM exercise_data
                        WHERE sync_status IN ('PENDING', 'FAILED', 'QUEUED')
                        ORDER BY timestamp ASC
                        LIMIT %s
                        """,
                        (limit,),
                    )
                else:
                    self.cursor.execute(
                        """
                        SELECT id, user_id, username_snapshot, exercise, set_count, sets_reps,
                               rep_data, timestamp, date, gym_id, main_system_id, mirror_id,
                               source_node, equipment_type, machine_id, sync_status,
                               synced_at, source_payload_hash, source_batch_id, source_edge_ip
                        FROM exercise_data
                        WHERE sync_status IN ('PENDING', 'FAILED', 'QUEUED')
                        ORDER BY timestamp ASC
                        LIMIT ?
                        """,
                        (limit,),
                    )

                rows = self.cursor.fetchall()
                payload = []
                for row in rows:
                    sets_reps = row[5] if self.backend == "postgres" else self._json_load(row[5], [])
                    rep_data = row[6] if self.backend == "postgres" else self._json_load(row[6], [])
                    timestamp_val = self._parse_timestamp(row[7]) or datetime.utcnow().isoformat()
                    date_val = str(row[8]) if row[8] else self._to_date_str(timestamp_val)
                    payload.append(
                        {
                            "id": str(row[0]),
                            "user_id": str(row[1]),
                            "username_snapshot": row[2],
                            "exercise": row[3],
                            "set_count": int(row[4] or 0),
                            "sets_reps": sets_reps or [],
                            "rep_data": rep_data or [],
                            "timestamp": timestamp_val,
                            "date": date_val,
                            "gym_id": str(row[9]) if row[9] else None,
                            "main_system_id": str(row[10]) if row[10] else None,
                            "mirror_id": str(row[11]) if row[11] else None,
                            "source_node": row[12],
                            "equipment_type": row[13],
                            "machine_id": str(row[14]) if row[14] else None,
                            "sync_status": row[15],
                            "synced_at": self._parse_timestamp(row[16]),
                            "source_payload_hash": row[17],
                            "source_batch_id": str(row[18]) if row[18] else None,
                            "source_edge_ip": row[19],
                        }
                    )
                return payload
        except Exception as exc:
            logger.error("Database error in get_pending_exercise_data: %s", exc)
            return []

    @retry_db_operation
    def mark_exercise_data_synced(self, record_ids, status="SYNCED"):
        cleaned_ids = [self._normalize_uuid(item) for item in (record_ids or [])]
        cleaned_ids = [item for item in cleaned_ids if item]
        if not cleaned_ids:
            return True

        status = (status or "SYNCED").upper()
        now_iso = datetime.utcnow().isoformat()

        with self.lock:
            if self.backend == "postgres":
                self.cursor.execute(
                    """
                    UPDATE exercise_data
                    SET sync_status = %s,
                        synced_at = NOW(),
                        updated_at = NOW()
                    WHERE id = ANY(%s::uuid[])
                    """,
                    (status, cleaned_ids),
                )
            else:
                placeholders = ",".join(["?"] * len(cleaned_ids))
                self.cursor.execute(
                    f"""
                    UPDATE exercise_data
                    SET sync_status = ?, synced_at = ?, updated_at = datetime('now')
                    WHERE id IN ({placeholders})
                    """,
                    (status, now_iso, *cleaned_ids),
                )
                self.connection.commit()
        return True

    def upsert_edge_records(self, records, source_edge_ip=None, return_record_ids=False):
        applied = 0
        applied_record_ids = []
        for record in records or []:
            user_id = self._normalize_uuid(record.get("user_id"))
            username = (record.get("username_snapshot") or record.get("username") or "").strip()

            canonical_user_id = user_id
            existing_member = self.get_member_info_local(username) if username else None
            if existing_member:
                existing_user_id = self._normalize_uuid(existing_member.get("user_id"))
                if existing_user_id:
                    canonical_user_id = existing_user_id

            if not username and canonical_user_id:
                username = f"user_{canonical_user_id[:8]}"

            if username and canonical_user_id:
                if user_id and canonical_user_id != user_id:
                    logger.warning(
                        "Edge record user_id remapped for username=%s (incoming=%s, canonical=%s).",
                        username,
                        user_id,
                        canonical_user_id,
                    )
                self.insert_member_local(
                    {
                        "user_id": canonical_user_id,
                        "username": username,
                        "email": record.get("email", "NA"),
                        "membership": record.get("membership", "NA"),
                        "joined_on": self._to_date_str(record.get("date") or record.get("timestamp")),
                    }
                )
            elif not canonical_user_id:
                logger.warning(
                    "Skipping edge record id=%s because canonical user_id could not be resolved.",
                    record.get("id"),
                )
                continue

            merged = dict(record)
            merged["source_node"] = "EDGE"
            merged["user_id"] = canonical_user_id
            if source_edge_ip:
                merged["source_edge_ip"] = source_edge_ip
            if not merged.get("gym_id"):
                merged["gym_id"] = os.getenv("SMART_MIRROR_GYM_ID")
            if not merged.get("main_system_id"):
                merged["main_system_id"] = os.getenv("SMART_MIRROR_MAIN_SYSTEM_ID")
            if not merged.get("gym_id") or not merged.get("main_system_id"):
                logger.warning(
                    "Skipping edge record id=%s because gym_id/main_system_id could not be resolved.",
                    merged.get("id"),
                )
                continue
            merged.setdefault("sync_status", "PENDING")

            if self.insert_exercise_data_local(merged):
                applied += 1
                if merged.get("id"):
                    applied_record_ids.append(str(merged["id"]))
        if return_record_ids:
            return applied, applied_record_ids
        return applied

    def close_connections(self):
        try:
            with self.lock:
                if self.connection:
                    self.connection.close()
                    logger.info("Closed database connection.")
        except Exception as exc:
            logger.error("Database close error: %s", exc)

    def get_total_members(self):
        try:
            with self.lock:
                self.cursor.execute("SELECT COUNT(*) FROM members")
                count = int(self.cursor.fetchone()[0] or 0)
                logger.info("Total members: %s", count)
                return count
        except Exception as exc:
            logger.error("Database error in get_total_members: %s", exc)
            return 0

    def get_active_exercises(self):
        try:
            with self.lock:
                self.cursor.execute("SELECT COUNT(DISTINCT exercise) FROM exercise_data")
                count = int(self.cursor.fetchone()[0] or 0)
                logger.info("Active exercises: %s", count)
                return count
        except Exception as exc:
            logger.error("Database error in get_active_exercises: %s", exc)
            return 0

    def get_recent_activities(self, limit=5):
        limit = max(1, int(limit))
        try:
            with self.lock:
                if self.backend == "postgres":
                    self.cursor.execute(
                        """
                        SELECT members.username, exercise_data.exercise, exercise_data.set_count,
                               exercise_data.rep_data, exercise_data.timestamp
                        FROM exercise_data
                        JOIN members ON exercise_data.user_id = members.user_id
                        ORDER BY exercise_data.timestamp DESC
                        LIMIT %s
                        """,
                        (limit,),
                    )
                else:
                    self.cursor.execute(
                        """
                        SELECT members.username, exercise_data.exercise, exercise_data.set_count,
                               exercise_data.rep_data, exercise_data.timestamp
                        FROM exercise_data
                        JOIN members ON exercise_data.user_id = members.user_id
                        ORDER BY exercise_data.timestamp DESC
                        LIMIT ?
                        """,
                        (limit,),
                    )

                rows = self.cursor.fetchall()
                activities = []
                for row in rows:
                    rep_data = row[3] if self.backend == "postgres" else self._json_load(row[3], [])
                    activities.append(
                        {
                            "username": row[0],
                            "exercise": row[1],
                            "set_count": int(row[2] or 0),
                            "rep_count": len(rep_data or []),
                            "timestamp": self._parse_timestamp(row[4]),
                        }
                    )
                logger.info("Retrieved %s recent activities.", len(activities))
                return activities
        except Exception as exc:
            logger.error("Error in get_recent_activities: %s", exc)
            return []

    def get_total_sets(self):
        try:
            with self.lock:
                self.cursor.execute("SELECT COALESCE(SUM(set_count), 0) FROM exercise_data")
                count = int(self.cursor.fetchone()[0] or 0)
                logger.info("Total sets: %s", count)
                return count
        except Exception as exc:
            logger.error("Database error in get_total_sets: %s", exc)
            return 0

    def get_total_reps(self):
        try:
            with self.lock:
                self.cursor.execute("SELECT rep_data FROM exercise_data")
                rows = self.cursor.fetchall()
                total_reps = 0
                for row in rows:
                    rep_data = row[0] if self.backend == "postgres" else self._json_load(row[0], [])
                    total_reps += len(rep_data or [])
                logger.info("Total reps: %s", total_reps)
                return total_reps
        except Exception as exc:
            logger.error("Database error in get_total_reps: %s", exc)
            return 0
