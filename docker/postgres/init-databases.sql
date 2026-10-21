SELECT 'CREATE DATABASE gym_main_db'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'gym_main_db')\gexec
