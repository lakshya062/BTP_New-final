# core/database.py

import sqlite3
import logging
import json
import threading
import time
import os

logger = logging.getLogger(__name__)


class DatabaseHandler:
    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super(DatabaseHandler, cls).__new__(cls)
        return cls._instance

    def __init__(self, local_db_file="local_members.db"):
        if hasattr(self, '_initialized') and self._initialized:
            return
        self._initialized = True

        self.local_db_file = local_db_file
        self.lock = threading.Lock()
        self.connection = sqlite3.connect(
            self.local_db_file,
            check_same_thread=False,
            timeout=30,  # Wait up to 30 seconds for the lock
            isolation_level=None  # Autocommit mode
        )
        self.connection.execute("PRAGMA foreign_keys = ON;")
        self.connection.execute("PRAGMA journal_mode = WAL;")
        self.connection.execute("PRAGMA synchronous = NORMAL;")
        self.cursor = self.connection.cursor()
        try:
            os.chmod(self.local_db_file, 0o600)
        except OSError:
            pass
        self.setup_local_db()

    def setup_local_db(self):
        """Create members and exercise_data tables if they don't exist."""
        with self.lock:
            try:
                self.cursor.execute('''
                    CREATE TABLE IF NOT EXISTS members (
                        user_id TEXT PRIMARY KEY,
                        username TEXT UNIQUE NOT NULL,
                        email TEXT DEFAULT 'NA',
                        membership TEXT DEFAULT 'NA',
                        joined_on TEXT NOT NULL
                    )
                ''')
                self.cursor.execute('''
                    CREATE TABLE IF NOT EXISTS exercise_data (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        exercise TEXT NOT NULL,
                        set_count INTEGER,
                        sets_reps TEXT,
                        rep_data TEXT,
                        timestamp TEXT,
                        date TEXT,
                        FOREIGN KEY (user_id) REFERENCES members (user_id) ON DELETE CASCADE
                    )
                ''')
                self.cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_members_username ON members(username)"
                )
                self.cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_exercise_user_id ON exercise_data(user_id)"
                )
                self.cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_exercise_timestamp ON exercise_data(timestamp)"
                )
                self.cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_exercise_name ON exercise_data(exercise)"
                )
                self.connection.commit()
                logger.info("Local SQLite database setup completed.")
            except sqlite3.Error as e:
                logger.error("SQLite error during setup_local_db: %s", e)

    # Retry decorator for write operations
    def retry_db_operation(func):
        def wrapper(self, *args, **kwargs):
            max_retries = 5
            delay = 0.1  # Initial delay in seconds
            for attempt in range(max_retries):
                try:
                    return func(self, *args, **kwargs)
                except sqlite3.OperationalError as e:
                    if "database is locked" in str(e):
                        logger.warning("Database is locked. Retrying in %s seconds...", delay)
                        time.sleep(delay)
                        delay *= 2  # Exponential backoff
                    else:
                        logger.error("SQLite OperationalError: %s", e)
                        raise
                except sqlite3.Error as e:
                    logger.error("SQLite error in %s: %s", func.__name__, e)
                    raise
            logger.error("Failed to execute %s after %s retries.", func.__name__, max_retries)
            return False
        return wrapper

    # Local SQLite Methods for Members
    def get_member_info_local(self, username):
        """Retrieve member info from local SQLite DB."""
        try:
            with self.lock:
                self.cursor.execute('SELECT * FROM members WHERE username = ?', (username,))
                row = self.cursor.fetchone()
                if row:
                    return {
                        "user_id": row[0],
                        "username": row[1],
                        "email": row[2],
                        "membership": row[3],
                        "joined_on": row[4]
                    }
                return None
        except sqlite3.Error as e:
            logger.error("SQLite error in get_member_info_local: %s", e)
            return None

    @retry_db_operation
    def insert_member_local(self, member_info):
        """Insert a new member into local SQLite DB."""
        try:
            with self.lock:
                self.cursor.execute('''
                    INSERT INTO members (user_id, username, email, membership, joined_on)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    member_info["user_id"],
                    member_info["username"],
                    member_info.get("email", "NA"),
                    member_info.get("membership", "NA"),
                    member_info["joined_on"]
                ))
                self.connection.commit()
                logger.info("Inserted member %s into local DB.", member_info["username"])
                return True
        except sqlite3.IntegrityError as e:
            logger.error("IntegrityError while inserting member: %s", e)
            return False
        except sqlite3.Error as e:
            logger.error("SQLite error while inserting member: %s", e)
            return False

    @retry_db_operation
    def delete_member_local(self, username):
        """Delete a member from local SQLite DB."""
        try:
            with self.lock:
                # Fetch user_id for logging
                self.cursor.execute('SELECT user_id FROM members WHERE username = ?', (username,))
                user_row = self.cursor.fetchone()
                if not user_row:
                    logger.warning("No member found with username: %s", username)
                    return False
                user_id = user_row[0]

                # Delete the member
                self.cursor.execute('DELETE FROM exercise_data WHERE exercise_data.user_id = ?', (user_id,))
                self.cursor.execute('DELETE FROM members WHERE username = ?', (username,))
                if self.cursor.rowcount > 0:
                    self.connection.commit()
                    self.cursor.execute('SELECT COUNT(*) FROM exercise_data WHERE user_id = ?', (user_id,))
                    remaining_exercises = self.cursor.fetchone()[0]
                    if not remaining_exercises:
                        logger.info("Deleted member %s and associated exercise data from local DB.", username)
                        return True
                    else:
                        logger.error("Exercise data for user %s was not fully deleted.", username)
                        return False
                else:
                    logger.warning("No member found with username: %s", username)
                    return False
        except sqlite3.IntegrityError as e:
            logger.error("IntegrityError while deleting member: %s", e)
            return False
        except sqlite3.Error as e:
            logger.error("SQLite error while deleting member: %s", e)
            return False


    def get_all_members_local(self):
        """Retrieve all members from local SQLite DB."""
        try:
            with self.lock:
                self.cursor.execute('SELECT * FROM members')
                rows = self.cursor.fetchall()
                members = []
                for row in rows:
                    members.append({
                        "user_id": row[0],
                        "username": row[1],
                        "email": row[2],
                        "membership": row[3],
                        "joined_on": row[4]
                    })
                return members
        except sqlite3.Error as e:
            logger.error("SQLite error in get_all_members_local: %s", e)
            return []

    def get_member_info(self, username):
        """Retrieve a member's information by username."""
        return self.get_member_info_local(username)
    
    def get_all_members(self):
        """Retrieve all members from local SQLite DB."""
        return self.get_all_members_local()

    # Local SQLite Methods for Exercise Data
    @retry_db_operation
    def insert_exercise_data_local(self, record):
        """Insert exercise data into the local database."""
        try:
            with self.lock:
                self.cursor.execute('''
                    INSERT INTO exercise_data (id, user_id, exercise, set_count, sets_reps, rep_data, timestamp, date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    record['id'],
                    record['user_id'],
                    record['exercise'],
                    record['set_count'],
                    json.dumps(record['sets_reps']),
                    json.dumps(record['rep_data']),
                    record['timestamp'],
                    record['date']
                ))
                self.connection.commit()
                logger.info("Inserted exercise data for user %s into local DB.", record["user_id"])
                return True
        except sqlite3.IntegrityError as ie:
            logger.error("IntegrityError while inserting exercise data: %s", ie)
            return False
        except sqlite3.Error as e:
            logger.error("SQLite error while inserting exercise data: %s", e)
            return False

    def get_exercise_data_for_user_local(self, user_id):
        """Retrieve exercise data for a specific user from local SQLite DB."""
        try:
            with self.lock:
                self.cursor.execute('SELECT * FROM exercise_data WHERE user_id = ?', (user_id,))
                rows = self.cursor.fetchall()
                data = []
                for row in rows:
                    data.append({
                        "id": row[0],
                        "user_id": row[1],
                        "exercise": row[2],
                        "set_count": row[3],
                        "sets_reps": json.loads(row[4]) if row[4] else [],
                        "rep_data": json.loads(row[5]) if row[5] else [],
                        "timestamp": row[6],
                        "date": row[7]
                    })
                return data
        except sqlite3.Error as e:
            logger.error("SQLite error in get_exercise_data_for_user_local: %s", e)
            return []

    def get_exercise_data_for_user(self, user_id):
        """Retrieve exercise data for a specific user from local SQLite DB."""
        return self.get_exercise_data_for_user_local(user_id)

    def close_connections(self):
        """Close the SQLite connection."""
        try:
            with self.lock:
                self.connection.close()
                logger.info("Closed SQLite connection.")
        except sqlite3.Error as e:
            logger.error("SQLite error while closing connection: %s", e)
    
    # Additional Methods Added Below

    def get_total_members(self):
        """Retrieve the total number of members."""
        try:
            with self.lock:
                self.cursor.execute('SELECT COUNT(*) FROM members')
                count = self.cursor.fetchone()[0]
                logger.info("Total members: %s", count)
                return count
        except sqlite3.Error as e:
            logger.error("SQLite error in get_total_members: %s", e)
            return 0

    def get_active_exercises(self):
        """Retrieve the number of distinct active exercises."""
        try:
            with self.lock:
                # Assuming 'active exercises' refers to distinct exercises recorded
                self.cursor.execute('SELECT COUNT(DISTINCT exercise) FROM exercise_data')
                count = self.cursor.fetchone()[0]
                logger.info("Active exercises: %s", count)
                return count
        except sqlite3.Error as e:
            logger.error("SQLite error in get_active_exercises: %s", e)
            return 0
        
    def get_recent_activities(self, limit=5):
        """Retrieve recent activities from the exercise_data table."""
        try:
            with self.lock:
                self.cursor.execute('''
                    SELECT members.username, exercise_data.exercise, exercise_data.set_count,
                           exercise_data.rep_data, exercise_data.timestamp
                    FROM exercise_data
                    JOIN members ON exercise_data.user_id = members.user_id
                    ORDER BY exercise_data.timestamp DESC
                    LIMIT ?
                ''', (limit,))
                rows = self.cursor.fetchall()
                activities = []
                for row in rows:
                    rep_data = json.loads(row[3]) if row[3] else []
                    activities.append({
                        "username": row[0],
                        "exercise": row[1],
                        "set_count": row[2] if row[2] else 0,
                        "rep_count": len(rep_data),
                        "timestamp": row[4]
                    })
                logger.info("Retrieved %s recent activities.", len(activities))
                return activities
        except (sqlite3.Error, json.JSONDecodeError) as e:
            logger.error("Error in get_recent_activities: %s", e)
            return []
        
    def get_total_sets(self):
        """Retrieve the total number of sets across all exercises."""
        try:
            with self.lock:
                self.cursor.execute('SELECT SUM(set_count) FROM exercise_data')
                result = self.cursor.fetchone()[0]
                count = result if result else 0
                logger.info("Total sets: %s", count)
                return count
        except sqlite3.Error as e:
            logger.error("SQLite error in get_total_sets: %s", e)
            return 0

    def get_total_reps(self):
        """Retrieve the total number of reps across all exercises."""
        try:
            with self.lock:
                self.cursor.execute('SELECT rep_data FROM exercise_data')
                rows = self.cursor.fetchall()
                total_reps = 0
                for row in rows:
                    rep_data = json.loads(row[0]) if row[0] else []
                    total_reps += len(rep_data)
                logger.info("Total reps: %s", total_reps)
                return total_reps
        except sqlite3.Error as e:
            logger.error("SQLite error in get_total_reps: %s", e)
            return 0
        except json.JSONDecodeError as je:
            logger.error("JSON decode error in get_total_reps: %s", je)
            return 0
