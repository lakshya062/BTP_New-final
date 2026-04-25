FROM python:3.11-bookworm

ARG TARGETARCH

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    QT_X11_NO_MITSHM=1 \
    CMAKE_BUILD_PARALLEL_LEVEL=2 \
    PIP_DEFAULT_TIMEOUT=120 \
    PYTHONPATH=/app/Mirror_Code

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends cmake \
    && rm -rf /var/lib/apt/lists/*

COPY docker/requirements/mirror.txt /tmp/requirements.txt
RUN if [ "${TARGETARCH:-}" = "arm64" ]; then \
        sed -i 's/^PySide6==6.9.1$/PySide6==6.8.0.2/' /tmp/requirements.txt; \
    fi
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && cmake --version \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && mkdir -p /tmp/dlib-src \
    && pip download --no-binary=:all: --no-deps dlib==20.0.0 -d /tmp/dlib-src \
    && tar -xf /tmp/dlib-src/dlib-20.0.0.tar.gz -C /tmp/dlib-src \
    && cd /tmp/dlib-src/dlib-20.0.0 \
    && python setup.py install --set CMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF \
    && pip install --no-cache-dir face-recognition==1.3.0

COPY Mirror_Code /app/Mirror_Code

WORKDIR /app/Mirror_Code

CMD ["python", "app.py"]
