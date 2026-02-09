-- Run with: psql -d postgres -f infra/postgres/drop_databases.sql
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE datname IN ('gym_edge_db', 'gym_main_db', 'gym_server_db')
  AND pid <> pg_backend_pid();

DROP DATABASE IF EXISTS gym_edge_db;
DROP DATABASE IF EXISTS gym_main_db;
DROP DATABASE IF EXISTS gym_server_db;
