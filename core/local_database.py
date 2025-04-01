# core/local_database.py

import sqlite3
from sqlite3 import Error
import uuid
from datetime import datetime

class LocalDatabaseHandler:
    def __init__(self, db_file="local_members.db"):
        self.db_file = db_file
        self.conn = self.create_connection()
        self.create_members_table()

    def create_connection(self):
        """Create a database connection to the SQLite database."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_file)
            print(f"Connected to SQLite database: {self.db_file}")
        except Error as e:
            print(f"SQLite connection error: {e}")
        return conn

    def create_members_table(self):
        """Create the members table if it doesn't exist."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS members (
            user_id TEXT PRIMARY KEY,
            username TEXT NOT NULL UNIQUE,
            email TEXT DEFAULT 'NA',
            membership TEXT DEFAULT 'NA',
            joined_on TEXT NOT NULL
        );
        """
        try:
            c = self.conn.cursor()
            c.execute(create_table_sql)
            self.conn.commit()
            print("Members table ensured in local database.")
        except Error as e:
            print(f"Error creating members table: {e}")

    def insert_member(self, member_info):
        """Insert a new member into the members table."""
        sql = '''INSERT INTO members(user_id, username, email, membership, joined_on)
                 VALUES(?,?,?,?,?)'''
        try:
            c = self.conn.cursor()
            c.execute(sql, (
                member_info['user_id'],
                member_info['username'],
                member_info.get('email', 'NA'),
                member_info.get('membership', 'NA'),
                member_info['joined_on']
            ))
            self.conn.commit()
            print(f"Inserted member: {member_info['username']}")
            return True
        except sqlite3.IntegrityError:
            print(f"Member {member_info['username']} already exists.")
            return False
        except Error as e:
            print(f"Error inserting member: {e}")
            return False

    def delete_member(self, username):
        """Delete a member from the members table by username."""
        sql = '''DELETE FROM members WHERE username = ?'''
        try:
            c = self.conn.cursor()
            c.execute(sql, (username,))
            self.conn.commit()
            if c.rowcount > 0:
                print(f"Deleted member: {username}")
                return True
            else:
                print(f"No member found with username: {username}")
                return False
        except Error as e:
            print(f"Error deleting member: {e}")
            return False

    def get_all_members(self):
        """Retrieve all members from the members table."""
        sql = '''SELECT user_id, username, email, membership, joined_on FROM members'''
        try:
            c = self.conn.cursor()
            c.execute(sql)
            rows = c.fetchall()
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
        except Error as e:
            print(f"Error retrieving members: {e}")
            return []

    def get_member_info(self, username):
        """Retrieve a member's information by username."""
        sql = '''SELECT user_id, username, email, membership, joined_on FROM members WHERE username = ?'''
        try:
            c = self.conn.cursor()
            c.execute(sql, (username,))
            row = c.fetchone()
            if row:
                return {
                    "user_id": row[0],
                    "username": row[1],
                    "email": row[2],
                    "membership": row[3],
                    "joined_on": row[4]
                }
            else:
                return None
        except Error as e:
            print(f"Error retrieving member info: {e}")
            return None
