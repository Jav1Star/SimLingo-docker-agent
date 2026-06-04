# SimLingo Encoder Agent

This Docker agent owns only the SimLingo encoding stage:

- prompt tokenization with the InternVL/Qwen tokenizer
- InternViT visual feature extraction
- Pixel Shuffle based visual token reshaping inside InternVL
- waypoint placeholder encoding
- HTTP and optional NATS publication of encoded payloads

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

At runtime the container no longer mounts host model directories. Only
`/app/artifacts` is backed by a Docker named volume for encoded outputs.

Override the model build context when rebuilding on a different machine:

```bash
SIMLINGO_MODELS_CONTEXT=/path/to/models docker compose -f simlingo_agents/docker-compose.yml build simlingo-encoder-agent
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
- `ENCODER_ARTIFACT_DIR`: directory used for `.npy` tensor artifacts. Defaults to `/app/artifacts`.
- `ENCODER_PUBLISH_BY_DEFAULT`: whether `/encoder/encode` publishes to NATS when `publish` is omitted.
- `NATS_SERVER_URL`: NATS server URL.
- `NATS_SUBJECT`: default output subject, `workflow.simlingo.encoded_tokens`.

## HTTP API

Health:

```bash
curl http://127.0.0.1:9011/health
```

Encode without NATS publication:

```bash
curl -X POST http://127.0.0.1:9011/encoder/encode \
  -H 'Content-Type: application/json' \
  -d '{"prompt_texts":["<image>\nWhat should the ego do next?"],"publish":false}'
```

Set `inline_payload=true` only for small debug requests. The default response stores arrays as `.npy` files and returns `file://` URIs plus shape and dtype metadata.
