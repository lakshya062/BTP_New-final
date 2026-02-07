# core/local_database.py
#
# Backward-compatible adapter that forwards to the canonical DatabaseHandler.

from .database import DatabaseHandler


class LocalDatabaseHandler:
    def __init__(self, db_file="local_members.db"):
        self._handler = DatabaseHandler(local_db_file=db_file)

    def insert_member(self, member_info):
        return self._handler.insert_member_local(member_info)

    def delete_member(self, username):
        return self._handler.delete_member_local(username)

    def get_all_members(self):
        return self._handler.get_all_members_local()

    def get_member_info(self, username):
        return self._handler.get_member_info_local(username)
