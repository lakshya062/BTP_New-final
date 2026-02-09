-- Run with: psql -d postgres -f infra/postgres/create_databases.sql
SELECT 'CREATE DATABASE gym_edge_db'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'gym_edge_db')\gexec

SELECT 'CREATE DATABASE gym_main_db'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'gym_main_db')\gexec

SELECT 'CREATE DATABASE gym_server_db'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'gym_server_db')\gexec
