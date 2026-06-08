FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu128

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    ca-certificates \
    build-essential \
    git \
    xvfb \
    libgl1 \
    libgl1-mesa-dri \
    libglx-mesa0 \
    libglib2.0-0 \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    python -m venv /opt/venv

ENV PATH="/opt/venv/bin:${PATH}"

RUN python -m pip config set global.index-url ${PIP_INDEX_URL} && \
    python -m pip config set global.extra-index-url ${PIP_EXTRA_INDEX_URL} && \
    python -m pip config set global.timeout 120 && \
    python -m pip config set global.retries 5

COPY requirements.docker.txt /app/requirements.docker.txt

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r /app/requirements.docker.txt

COPY . /app
COPY docker-entrypoint.sh /app/docker-entrypoint.sh

RUN chmod +x /app/docker-entrypoint.sh

# Build Cython extension used by OpenCOOD.
RUN cd /app/opencood/utils && python setup.py build_ext --inplace
# RUN cd /app/opencood/pcdet_utils && python setup.py build_ext --inplace

ENV PYTHONPATH=/app \
    LIBGL_ALWAYS_SOFTWARE=1 \
    QT_X11_NO_MITSHM=1 \
    QT_GRAPHICSSYSTEM=native \
    MODEL_CHECKPOINT_PATH=/app/checkpoints/point_pillar_where2comm/

EXPOSE 9002

ENTRYPOINT ["/app/docker-entrypoint.sh"]
