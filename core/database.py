# core/database.py

import sqlite3
from datetime import datetime
import uuid
import logging
import json
import threading
import time

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
        self.connection.execute("PRAGMA foreign_keys = ON;")  # Enable foreign keys
        self.cursor = self.connection.cursor()
        self.setup_local_db()

        # Configure logging
        logging.basicConfig(level=logging.INFO, filename='database.log',
                            format='%(asctime)s - %(levelname)s - %(message)s')

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
                self.connection.commit()
                logging.info("Local SQLite database setup completed.")
            except sqlite3.Error as e:
                logging.error(f"SQLite error during setup_local_db: {e}")

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
                        logging.warning(f"Database is locked. Retrying in {delay} seconds...")
                        time.sleep(delay)
                        delay *= 2  # Exponential backoff
                    else:
                        logging.error(f"SQLite OperationalError: {e}")
                        raise
                except sqlite3.Error as e:
                    logging.error(f"SQLite error in {func.__name__}: {e}")
                    raise
            logging.error(f"Failed to execute {func.__name__} after {max_retries} retries.")
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
            logging.error(f"SQLite error in get_member_info_local: {e}")
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
                logging.info(f"Inserted member {member_info['username']} into local DB.")
                return True
        except sqlite3.IntegrityError as e:
            logging.error(f"IntegrityError while inserting member: {e}")
            return False
        except sqlite3.Error as e:
            logging.error(f"SQLite error while inserting member: {e}")
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
                    logging.warning(f"No member found with username: {username}")
                    return False
                user_id = user_row[0]

                # Delete the member
                self.cursor.execute('DELETE FROM exercise_data WHERE exercise_data.user_id = ?', (user_id,))
                self.cursor.execute('DELETE FROM members WHERE username = ?', (username,))
                if self.cursor.rowcount > 0:
                    # Ensure exercise data is deleted
                    remaining_exercises = self.cursor.fetchall()
                    if not remaining_exercises:
                        logging.info(f"Deleted member {username} and associated exercise data from local DB.")
                        return True
                    else:
                        logging.error(f"Exercise data for user {username} was not fully deleted.")
                        return False
                else:
                    logging.warning(f"No member found with username: {username}")
                    return False
        except sqlite3.IntegrityError as e:
            logging.error(f"IntegrityError while deleting member: {e}")
            return False
        except sqlite3.Error as e:
            logging.error(f"SQLite error while deleting member: {e}")
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
            logging.error(f"SQLite error in get_all_members_local: {e}")
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
                logging.info(f"Inserted exercise data for user {record['user_id']} into local DB.")
                return True
        except sqlite3.IntegrityError as ie:
            logging.error(f"IntegrityError while inserting exercise data: {ie}")
            return False
        except sqlite3.Error as e:
            logging.error(f"SQLite error while inserting exercise data: {e}")
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
            logging.error(f"SQLite error in get_exercise_data_for_user_local: {e}")
            return []

    def get_exercise_data_for_user(self, user_id):
        """Retrieve exercise data for a specific user from local SQLite DB."""
        return self.get_exercise_data_for_user_local(user_id)

    def close_connections(self):
        """Close the SQLite connection."""
        try:
            with self.lock:
                self.connection.close()
                logging.info("Closed SQLite connection.")
        except sqlite3.Error as e:
            logging.error(f"SQLite error while closing connection: {e}")
    
    # Additional Methods Added Below

    def get_total_members(self):
        """Retrieve the total number of members."""
        try:
            with self.lock:
                self.cursor.execute('SELECT COUNT(*) FROM members')
                count = self.cursor.fetchone()[0]
                logging.info(f"Total members: {count}")
                return count
        except sqlite3.Error as e:
            logging.error(f"SQLite error in get_total_members: {e}")
            return 0

    def get_active_exercises(self):
        """Retrieve the number of distinct active exercises."""
        try:
            with self.lock:
                # Assuming 'active exercises' refers to distinct exercises recorded
                self.cursor.execute('SELECT COUNT(DISTINCT exercise) FROM exercise_data')
                count = self.cursor.fetchone()[0]
                logging.info(f"Active exercises: {count}")
                return count
        except sqlite3.Error as e:
            logging.error(f"SQLite error in get_active_exercises: {e}")
            return 0
        
    def get_recent_activities(self, limit=5):
        """Retrieve recent activities from the exercise_data table."""
        try:
            with self.lock:
                self.cursor.execute('''
                    SELECT members.username, exercise_data.exercise, exercise_data.timestamp
                    FROM exercise_data
                    JOIN members ON exercise_data.user_id = members.user_id
                    ORDER BY exercise_data.timestamp DESC
                    LIMIT ?
                ''', (limit,))
                rows = self.cursor.fetchall()
                activities = []
                for row in rows:
                    activities.append({
                        "username": row[0],
                        "exercise": row[1],
                        "timestamp": row[2]
                    })
                logging.info(f"Retrieved {len(activities)} recent activities.")
                return activities
        except sqlite3.Error as e:
            logging.error(f"SQLite error in get_recent_activities: {e}")
            return []
        
    def get_total_sets(self):
        """Retrieve the total number of sets across all exercises."""
        try:
            with self.lock:
                self.cursor.execute('SELECT SUM(set_count) FROM exercise_data')
                result = self.cursor.fetchone()[0]
                count = result if result else 0
                logging.info(f"Total sets: {count}")
                return count
        except sqlite3.Error as e:
            logging.error(f"SQLite error in get_total_sets: {e}")
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
                logging.info(f"Total reps: {total_reps}")
                return total_reps
        except sqlite3.Error as e:
            logging.error(f"SQLite error in get_total_reps: {e}")
            return 0
        except json.JSONDecodeError as je:
            logging.error(f"JSON decode error in get_total_reps: {je}")
            return 0