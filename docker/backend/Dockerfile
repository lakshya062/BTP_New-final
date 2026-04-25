FROM python:3.11-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GYM_DB_ENGINE=postgres \
    GYM_DB_NAME=gym_server_db \
    GYM_DB_USER=smartgym \
    GYM_DB_HOST=postgres \
    GYM_DB_PORT=5432

WORKDIR /app

COPY docker/requirements/backend.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt

COPY docker/backend/entrypoint.sh /usr/local/bin/start-smart-gym-backend
RUN chmod +x /usr/local/bin/start-smart-gym-backend

COPY gym_backend /app/gym_backend

WORKDIR /app/gym_backend

EXPOSE 8000

CMD ["start-smart-gym-backend"]
