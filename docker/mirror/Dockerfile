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
    && apt-get install -y --no-install-recommends \
        cmake libgl1 libegl1 libglib2.0-0 libportaudio2 \
        libdbus-1-3 libfontconfig1 libxkbcommon-x11-0 libx11-xcb1 \
        libxcb-cursor0 libxcb-icccm4 libxcb-keysyms1 libxcb-shape0 \
        libxcb-xinerama0 libxcb-randr0 libxcb-render-util0 libxcb-xfixes0 \
    && rm -rf /var/lib/apt/lists/*

COPY docker/requirements/mirror.txt /tmp/requirements.txt
RUN if [ "${TARGETARCH:-}" = "arm64" ]; then \
        sed -i 's/^PySide6==6.9.1$/PySide6==6.8.0.2/' /tmp/requirements.txt; \
    fi
RUN pip install --no-cache-dir --upgrade pip 'setuptools<82' wheel \
    && cmake --version \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && mkdir -p /tmp/dlib-src \
    && pip download --no-binary=:all: --no-deps dlib==20.0.0 -d /tmp/dlib-src \
    && tar -xf /tmp/dlib-src/dlib-20.0.0.tar.gz -C /tmp/dlib-src \
    && cd /tmp/dlib-src/dlib-20.0.0 \
    && python setup.py install --set CMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF \
    && pip install --no-cache-dir face-recognition==1.3.0 \
    && pip check \
    && python -c "import cv2, h5py, mediapipe, face_recognition; from PySide6 import QtWidgets; assert hasattr(cv2, 'aruco'); assert hasattr(mediapipe, 'solutions')"

COPY Mirror_Code /app/Mirror_Code

WORKDIR /app/Mirror_Code

CMD ["python", "app.py"]
