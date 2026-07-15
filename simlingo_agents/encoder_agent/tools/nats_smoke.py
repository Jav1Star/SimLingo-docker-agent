from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

import numpy as np

from protocols import NatsComm
from utils.numpy_utils import decode_structured_numpy, encode_structured_numpy


def _build_minimal_encoder_payload(
    *,
    route_key: str,
    frame_id: str,
    num_patches: int,
    image_size: int,
    ego_x: float,
    ego_y: float,
    ego_yaw: float,
    timestamp: float,
    token_prune_ratio: float | None,
    token_prune_mode: str,
    token_prune_min_keep: int,
) -> dict[str, Any]:
    camera_images = np.zeros((1, 1, num_patches, 3, image_size, image_size), dtype=np.float32)
    payload = {
        "route_key": route_key,
        "frame_id": frame_id,
        "prompt_texts": ["<image>\nWhat should the ego do next?"],
        "camera_images": camera_images,
        "num_patches": num_patches,
        "runtime_context": {
            "ego_xy": [float(ego_x), float(ego_y)],
            "ego_yaw": float(ego_yaw),
            "timestamp": float(timestamp),
        },
    }
    if token_prune_ratio is not None:
        if not 0.0 <= token_prune_ratio <= 1.0:
            raise ValueError(f"--token-prune-ratio must be in [0, 1], got {token_prune_ratio}")
        # 关键调用点：模拟 remote agent 每帧传入的 token prune 覆盖配置。
        payload["token_prune"] = {
            "mode": token_prune_mode,
            "prune_ratio": float(token_prune_ratio),
            "min_keep": int(token_prune_min_keep),
        }
    return payload


def _summarize_arrays(payload: Any) -> Any:
    if isinstance(payload, dict):
        if {"shape", "dtype", "data"}.issubset(payload):
            return {
                "shape": payload["shape"],
                "dtype": payload["dtype"],
                "data": f"<base64:{len(payload['data'])} chars>",
            }
        return {key: _summarize_arrays(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_summarize_arrays(item) for item in payload]
    if isinstance(payload, np.ndarray):
        return {
            "shape": list(payload.shape),
            "dtype": str(payload.dtype),
        }
    return payload


async def _publish_minimal_input(args: argparse.Namespace) -> None:
    comm = NatsComm(servers=[args.nats_server])
    payload = _build_minimal_encoder_payload(
        route_key=args.route_key,
        frame_id=args.frame_id,
        num_patches=args.num_patches,
        image_size=args.image_size,
        ego_x=args.ego_x,
        ego_y=args.ego_y,
        ego_yaw=args.ego_yaw,
        timestamp=args.timestamp,
        token_prune_ratio=args.token_prune_ratio,
        token_prune_mode=args.token_prune_mode,
        token_prune_min_keep=args.token_prune_min_keep,
    )
    ack = await comm.send(args.subject, encode_structured_numpy(payload))
    print(
        json.dumps(
            {
                "status": "published",
                "subject": args.subject,
                "ack": ack,
                "payload_summary": _summarize_arrays(payload),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    await comm.close()


async def _fetch_once(args: argparse.Namespace) -> None:
    comm = NatsComm(servers=[args.nats_server])
    messages = await comm.receive(
        subject=args.subject,
        durable=args.durable,
        batch=1,
        timeout_sec=args.timeout_sec,
    )
    if not messages:
        print(json.dumps({"status": "empty", "subject": args.subject}, indent=2, ensure_ascii=False))
        await comm.close()
        return

    message = messages[0]
    payload = message.payload
    if args.decode:
        payload = decode_structured_numpy(payload)
    if args.summary:
        payload = _summarize_arrays(payload)

    print(
        json.dumps(
            {
                "status": "received",
                "subject": message.subject,
                "stream": message.stream,
                "consumer": message.consumer,
                "payload": payload,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    await message.ack()
    await comm.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="NATS smoke-test helper for SimLingo agents")
    parser.add_argument("--nats-server", default="nats://host.docker.internal:4222")
    subparsers = parser.add_subparsers(dest="command", required=True)

    publish_parser = subparsers.add_parser("publish-minimal-input")
    publish_parser.add_argument("--subject", default="workflow.previousagent.result")
    publish_parser.add_argument("--route-key", default="smoke_route")
    publish_parser.add_argument("--frame-id", default="smoke_frame_0001")
    publish_parser.add_argument("--num-patches", type=int, default=2)
    publish_parser.add_argument("--image-size", type=int, default=448)
    publish_parser.add_argument("--ego-x", type=float, default=0.0)
    publish_parser.add_argument("--ego-y", type=float, default=0.0)
    publish_parser.add_argument("--ego-yaw", type=float, default=0.0)
    publish_parser.add_argument("--timestamp", type=float, default=0.0)
    publish_parser.add_argument(
        "--token-prune-ratio",
        type=float,
        default=None,
        help="Optional visual token prune ratio. Omit to use encoder container defaults.",
    )
    publish_parser.add_argument("--token-prune-mode", default="prune2drive", choices=["origin", "random", "prune2drive", "off"])
    publish_parser.add_argument("--token-prune-min-keep", type=int, default=1)

    fetch_parser = subparsers.add_parser("fetch-once")
    fetch_parser.add_argument("--subject", required=True)
    fetch_parser.add_argument("--durable", required=True)
    fetch_parser.add_argument("--timeout-sec", type=float, default=5.0)
    fetch_parser.add_argument("--decode", action="store_true")
    fetch_parser.add_argument("--summary", action="store_true")

    args = parser.parse_args()
    if args.command == "publish-minimal-input":
        asyncio.run(_publish_minimal_input(args))
    elif args.command == "fetch-once":
        asyncio.run(_fetch_once(args))


if __name__ == "__main__":
    main()
