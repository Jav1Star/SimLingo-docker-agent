# SimLingo Encoder Agent

This Docker agent owns only the SimLingo encoding stage:

- prompt tokenization with the InternVL/Qwen tokenizer
- InternViT visual feature extraction
- Pixel Shuffle based visual token reshaping inside InternVL
- waypoint placeholder encoding
- A2A HTTP trigger plus NATS-based input/output data flow

It intentionally does not run Qwen2 inference and does not decide budget or scheduler plans.

## Model Packaging

The Docker image is built with model artifacts inside the image. The compose
files provide a named build context called `simlingo_models`, which defaults to:

```bash
/home/yyj/simlingo-adaption/models
```

During build, the image copies:

- `InternVL2-1B` to `/app/models/InternVL2-1B`
- the SimLingo encoder-side checkpoint source to `/app/checkpoints/simlingo_encoder/pytorch_model.bin`

At runtime the container no longer mounts host model directories.

Override the model build context when rebuilding on a different machine:

```bash
SIMLINGO_MODELS_CONTEXT=/path/to/models docker compose -f simlingo_agents/docker-compose.yml build simlingo-encoder-agent
```

Override the CUDA base image when Docker Hub is slow or an internal mirror is available:

```bash
BASE_IMAGE=your-registry.example.com/nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04 \
docker compose -f simlingo_agents/docker-compose.yml build simlingo-encoder-agent
```

After the image is built and pushed/exported, the target device does not need
the original host model directory.

## Run

From the repository root:

```bash
docker compose -f simlingo_agents/docker-compose.yml up --build simlingo-encoder-agent
```

Or from this directory:

```bash
docker compose up --build
```

The service listens on `9011`.

## Environment

- `ENCODER_MODEL_VARIANT`: model path or Hugging Face id. Defaults to `/app/models/InternVL2-1B`.
- `ENCODER_CHECKPOINT_PATH`: optional SimLingo checkpoint. Defaults to the checkpoint baked into `/app/checkpoints/simlingo_encoder/pytorch_model.bin`; only encoder-side weights are loaded.
- `NATS_SERVER_URL`: NATS server URL.
- `NATS_IN_SUBJECT`: input subject consumed from the previous agent.
- `NATS_IN_DURABLE`: durable consumer name for the input subject.
- `NATS_OUT_SUBJECT`: output subject published to the next agent.

## HTTP API

Health:

```bash
curl http://127.0.0.1:9011/health
```

A2A trigger:

```bash
curl -X POST http://127.0.0.1:9011/a2a/execute \
  -H 'Content-Type: application/json' \
  -d '{
    "sender_id": "L2_Scheduler",
    "receiver_id": "SimLingoEncoderAgent",
    "message_type": "request",
    "payload": {
      "task_id": "encode-001",
      "task_type": "vision",
      "task_description": "Run SimLingo encoder",
      "metadata": {
        "nats_in_subject": "workflow.previousagent.result",
        "nats_out_subject": "workflow.simlingo.encoded_tokens"
      }
    }
  }'
```

## Data Flow

The HTTP request only triggers execution. The actual business payload is:

- read from `NATS_IN_SUBJECT`
- decoded from structured numpy JSON
- processed by `encoder_runtime.encode(...)`
- encoded back into structured numpy JSON
- published to `NATS_OUT_SUBJECT`
