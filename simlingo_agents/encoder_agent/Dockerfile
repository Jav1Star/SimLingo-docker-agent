ARG BASE_IMAGE=nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

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
    curl \
    build-essential \
    git \
    libgl1 \
    libglib2.0-0 \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && \
    python /tmp/get-pip.py && \
    rm -f /tmp/get-pip.py

RUN python -m pip config set global.index-url ${PIP_INDEX_URL} && \
    python -m pip config set global.extra-index-url ${PIP_EXTRA_INDEX_URL} && \
    python -m pip config set global.timeout 120 && \
    python -m pip config set global.retries 5

COPY simlingo_agents/encoder_agent/requirements.docker.txt /app/requirements.docker.txt

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r /app/requirements.docker.txt

COPY simlingo_adaption_training /app/simlingo_adaption_training
COPY simlingo_agents/encoder_agent /app
RUN mkdir -p /app/models /app/checkpoints/simlingo_encoder
COPY --from=simlingo_models InternVL2-1B /app/models/InternVL2-1B
COPY --from=simlingo_models 2026_01_29_21_59_03_adaption_train_seed_9876/checkpoints/epoch=002.ckpt/pytorch_model.bin /app/checkpoints/simlingo_encoder/pytorch_model.bin

ENV PYTHONPATH=/app \
    ENCODER_MODEL_VARIANT=/app/models/InternVL2-1B \
    ENCODER_CHECKPOINT_PATH=/app/checkpoints/simlingo_encoder/pytorch_model.bin \
    NATS_SERVER_URL=nats://host.docker.internal:4222 \
    NATS_IN_SUBJECT=workflow.previousagent.result \
    NATS_IN_DURABLE=workflow-previousagent-result \
    NATS_OUT_SUBJECT=workflow.simlingo.scheduler_budget_input

EXPOSE 9011

CMD ["python", "main.py"]
