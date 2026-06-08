from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
from flask_socketio import SocketIO
from collections import deque
import asyncio
import os
import numpy as np
import cv2
import queue
import subprocess
import time
from threading import Lock
import base64
import torch
import json

from utils.numpy_utils import decode_array_from_dict
from utils.logger_utils import get_logger
from utils.open3d_utils import init_vis, render_vis
from protocols import A2AMessage, A2ATaskRequest, A2ATaskResponse, NatsComm

from .model_runtime import model_runtime

FIXED_MODEL_CHECKPOINT_PATH = os.getenv(
    "MODEL_CHECKPOINT_PATH",
    "checkpoints/point_pillar_where2comm/",
)

NATS_SERVER_URL = os.getenv("NATS_SERVER_URL", "nats://nats:4222")
NATS_SUBJECT = os.getenv("NATS_SUBJECT", "workflow.demo.perception2feature.result")
NATS_DURABLE = os.getenv("NATS_DURABLE", "workflow-demo-perception2feature-result")

logger = get_logger(__name__)

class CooperativeFeatureFusionWebApp:
    def __init__(self):
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, cors_allowed_origins='*', async_mode='threading')
        # self.background_thread = None
        # self.thread_lock = Lock()
        self.data_queue = deque(maxlen=5)  # 用于存储待发送的数据，避免内存占用过大

        logger.info("NATS communication initialized with server: %s", NATS_SERVER_URL)
        self.nats_comm = NatsComm(servers=[NATS_SERVER_URL])
        self.model_runtime = model_runtime
        # self.vis = init_vis()

        self._register_routes()
        self._register_socket_events()

        self.model_runtime.load_model(FIXED_MODEL_CHECKPOINT_PATH)

    def __del__(self):
        try:
            self.socketio.stop()
            asyncio.run(self.nats_comm.close())
            logger.info("SocketIO server stopped successfully.")
        except Exception as e:
            logger.warning(f"Error while stopping SocketIO server: {e}")

    @staticmethod
    def _decode_structured_numpy(payload):
        """将传输友好的结构化数组递归还原为 numpy.ndarray。"""
        if isinstance(payload, dict):
            # 数组字典格式: {'shape': ..., 'dtype': ..., 'data': ...}
            if {'shape', 'dtype', 'data'}.issubset(payload.keys()):
                return decode_array_from_dict(payload)
            return {k: CooperativeFeatureFusionWebApp._decode_structured_numpy(v) for k, v in payload.items()}

        if isinstance(payload, list):
            return [CooperativeFeatureFusionWebApp._decode_structured_numpy(v) for v in payload]

        return payload

    @staticmethod
    def _convert_numpy_to_tensor(payload, device='cpu'):
        """递归遍历容器，将 numpy.ndarray 转换为 torch.Tensor。"""
        if isinstance(payload, dict):
            return {
                k: CooperativeFeatureFusionWebApp._convert_numpy_to_tensor(v, device=device)
                for k, v in payload.items()
            }

        if isinstance(payload, list):
            return [CooperativeFeatureFusionWebApp._convert_numpy_to_tensor(v, device=device) for v in payload]

        if isinstance(payload, tuple):
            return tuple(CooperativeFeatureFusionWebApp._convert_numpy_to_tensor(v, device=device) for v in payload)

        if isinstance(payload, np.ndarray):
            return torch.from_numpy(payload.copy()).to(device)

        return payload

    @staticmethod
    def _restore_original_feature_from_dict(feature_dict, target_hw=(96, 352), device='cpu') -> torch.Tensor:
        """从 dict 形式的输入中恢复原始特征图，要求包含 'feature' 和 'mask'。"""
        if not isinstance(feature_dict, dict):
            raise TypeError("feature_dict must be a dict")

        if 'feature' not in feature_dict or 'mask' not in feature_dict:
            raise KeyError("feature_dict must contain 'feature' and 'mask'")
        
        feature_dict = CooperativeFeatureFusionWebApp._convert_numpy_to_tensor(feature_dict, device=device)

        masked_feature_tensor = feature_dict['feature']
        mask_tensor = feature_dict['mask']
        
        if not isinstance(masked_feature_tensor, torch.Tensor) or not isinstance(mask_tensor, torch.Tensor):
            raise TypeError("'feature' and 'mask' must be torch.Tensor")

        # 1. 插值掩码到目标尺寸
        target_h, target_w = target_hw
        if mask_tensor.shape[-2:] != (target_h, target_w):
            mask_tensor = torch.nn.functional.interpolate(
                mask_tensor.float(),
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False,
            )     # [1, 1, H, W]

        # 2. 获取掩码的非零位置索引
        spatial_mask = mask_tensor.squeeze(0).squeeze(0)       # [H, W]
        non_zero_indices = torch.nonzero(spatial_mask != 0, as_tuple=False) # [N, 2]

        # 3. 创建全零特征图
        c = masked_feature_tensor.shape[0]
        feature = torch.zeros(
            (1, c, target_h, target_w),
            dtype=masked_feature_tensor.dtype,
            device=masked_feature_tensor.device,
        )             # [1, C, H, W]

        if non_zero_indices.numel() == 0:
            return feature

        if masked_feature_tensor.shape[1] != non_zero_indices.shape[0]:
            raise ValueError(
                f"Feature count mismatch: {masked_feature_tensor.shape[1]} vs {non_zero_indices.shape[0]}"
            )
        # 4. 将提取的特征填充到非零位置
        # 对于每个通道
        for ch in range(c):
            # 在非零位置填充值
            feature[0, ch, non_zero_indices[:, 0], non_zero_indices[:, 1]] = masked_feature_tensor[ch]

        return feature

    def build(self):
        return self.socketio, self.app

    def _load_model(self):
        try:
            model_runtime.load_model(FIXED_MODEL_CHECKPOINT_PATH)
            logger.info("Model loaded from fixed path: %s", FIXED_MODEL_CHECKPOINT_PATH)
        except FileNotFoundError as exc:
            # Keep service alive in demo mode even when fixed path is not present.
            logger.error("Fixed checkpoint missing (%s), loading built-in demo weights", exc)
        except Exception as exc:
            logger.exception("Failed to load model during startup")
            raise RuntimeError(f"Startup model loading failed: {exc}") from exc
        
    def _receive_data(self, nats_subject=NATS_SUBJECT, nats_durable=NATS_DURABLE) -> dict:
        async def _receive_once() -> dict | None:
            try:
                messages = await self.nats_comm.receive(
                    subject=nats_subject,
                    durable=nats_durable,
                    batch=1,
                    timeout_sec=5,
                )
                for message in messages:
                    logger.info(f"Received message on subject '{nats_subject}'")
                    await message.ack()
                    return message.payload
                return None
            finally:
                await self.nats_comm.close()

        return asyncio.run(_receive_once())


    def _register_routes(self):
        @self.app.route('/')
        def home():
            return render_template('home_5.html')

        @self.app.route('/temp/forward', methods=['GET'])
        def forward(nats_subject=NATS_SUBJECT, nats_durable=NATS_DURABLE):
            # data = self.data_queue.popleft() if len(self.data_queue) > 0 else None

            data = self._receive_data(nats_subject=nats_subject, nats_durable=nats_durable)

            if data is None:
                return jsonify(status='error', message='No data available'), 404

            try:
                decoded_data = self._decode_structured_numpy(data)
            except Exception as e:
                return jsonify(status='error', message=f'Failed to decode numpy payload: {e}'), 400

            # 这里得到的结构中，intermediate_feature/pcd 已被还原为 numpy.ndarray。
            # 例如:
            # decoded_data['intermediate_feature']['feature'] -> np.ndarray
            # decoded_data['intermediate_feature']['mask'] -> np.ndarray
            # decoded_data['pcd'] -> np.ndarray

            masked_intermediate_feature = decoded_data.get('intermediate_feature')
            if masked_intermediate_feature is None:
                return jsonify(status='error', message='intermediate_feature is missing in the payload'), 400

            masked_intermediate_feature_tensor = self._convert_numpy_to_tensor(masked_intermediate_feature, device=self.model_runtime.device)
            try:
                intermediate_feature_tensor = self._restore_original_feature_from_dict(
                    masked_intermediate_feature_tensor,
                    target_hw=(96, 352),
                    device=self.model_runtime.device
                )
            except (TypeError, KeyError, ValueError) as e:
                return jsonify(status='error', message=f'Failed to restore feature map: {e}'), 400

            intermediate_features = [intermediate_feature_tensor]

            ego_pred_box, fused_pred_box, ego_feature, fused_feature = self.model_runtime.process_intermediate_features(intermediate_features)

            pcd = decoded_data.get('pcd')
            # pcd_img = render_vis(self.vis, pcd, ego_pred_box, fused_pred_box)
            pcd_img = render_vis(None, pcd, ego_pred_box, fused_pred_box)

            self._update_images(pcd_img=pcd_img, ego_feature=ego_feature, fused_feature=fused_feature)

            data = {
                "status": "success",
                "pred_box": ego_pred_box.tolist() if ego_pred_box is not None else None,
            }

            return data

        @self.app.route('/a2a/execute', methods=['POST'])
        def execute_a2a():
            message = request.json
            logger.info("Received message: %s", message)

            request_message = A2AMessage(**message)
            task_request = A2ATaskRequest(**request_message.payload)

            # 优先使用 metadata 中的 nats_subject/nats_durable
            metadata = getattr(task_request, 'metadata', {}) or {}
            if 'nats_subject' in metadata and metadata.get('nats_subject'):
                nats_subject = metadata['nats_subject']
                # 如果 metadata 中有 nats_durable 则使用，否则用 nats_subject 的 '.' 替换为 '-' 作为 durable
                nats_durable = metadata.get('nats_durable') or nats_subject.replace('.', '-')
            else:
                # metadata 中没有指定 nats_subject，则回退到默认的 SUBJECT 和 DURABLE
                nats_subject = NATS_SUBJECT
                nats_durable = NATS_DURABLE

            result = forward(nats_subject=nats_subject, nats_durable=nats_durable)

            task_response = A2ATaskResponse(
                task_id=task_request.task_id,
                status=result.get("status", "unknown"),
                result=json.dumps(result)
            )

            response_message = A2AMessage(
                sender_id="CooperativeFeatureFusionDetectionVizAgent",
                receiver_id=request_message.sender_id,
                message_type="response",
                payload=task_response.dict()
            )
            
            return response_message.dict()

        @self.app.route('/temp/post_data', methods=['POST'])
        def post_data():
            data = request.json

            if data.get("status") != "success":
                return jsonify(status='error', message='Status is not success'), 400

            self.data_queue.append(data)
            return jsonify(status='success', message='Data received')

        @self.app.route('/get_id')
        def get_id():
            return jsonify(id='TEST')

    def _encode_image_to_base64(self, img):
        """辅助函数：将OpenCV图像转换为Base64字符串"""
        if img is None:
            return ""
        # 1. 转为 JPEG
        ret, buffer = cv2.imencode('.jpg', img)
        if not ret:
            return ""
        # 2. 转为 Base64 字节 -> 解码为 字符串
        b64_str = base64.b64encode(buffer).decode('utf-8')
        return b64_str

    def _update_images(self, pcd_img=None, ego_comm_mask=None, others_comm_mask=None, ego_feature=None, fused_feature=None, pred_photos=None):
        """更新缓存中的图片数据"""
        # 准备数据包
        data_payload = {}

        if pcd_img is not None and pcd_img.size != 0:
            pcd_img = pcd_img.copy()
            pcd_img *= 255
            pcd_img = cv2.cvtColor(pcd_img, cv2.COLOR_RGB2BGR)
            data_payload['pcd_img'] = self._encode_image_to_base64(pcd_img)

        if ego_comm_mask is not None and ego_comm_mask.size != 0:   # ego_comm_mask.shape = (48, 176)
            ego_comm_mask = ego_comm_mask.copy()
            request_map = 1 - ego_comm_mask.squeeze()  # request_map.shape = (48, 176)

            # request_map = np.flipud(request_map)        # 上下翻转

            request_map_img = np.zeros((*request_map.shape, 3), dtype=np.uint8)

            color_0 = [255, 0, 255]  # 紫色
            color_1 = [0, 255, 255]  # 黄色

            request_map_img[request_map == 0] = color_0
            request_map_img[request_map == 1] = color_1

            original_height, original_width = request_map_img.shape[:2]

            scale_factor = 5
            new_width = original_width * scale_factor
            new_height = original_height * scale_factor

            request_map_img = cv2.resize(request_map_img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
            data_payload['request_map_img'] = self._encode_image_to_base64(request_map_img)

        if others_comm_mask is not None and others_comm_mask.size != 0:     # others_comm_mask.shape = (1, 1, 48, 176)
            others_comm_mask = others_comm_mask.copy()
            others_comm_mask = others_comm_mask.squeeze()

            # others_comm_mask = np.flipud(others_comm_mask)  # 上下翻转

            others_comm_mask_img = np.zeros((*others_comm_mask.shape, 3), dtype=np.uint8)

            color_0 = [255, 0, 255]  # 紫色
            color_1 = [0, 255, 255]  # 黄色

            others_comm_mask_img[others_comm_mask == 0] = color_0
            others_comm_mask_img[others_comm_mask == 1] = color_1

            original_height, original_width = others_comm_mask_img.shape[:2]

            scale_factor = 5
            new_width = original_width * scale_factor
            new_height = original_height * scale_factor

            others_comm_mask_img = cv2.resize(others_comm_mask_img, (new_width, new_height),
                                                interpolation=cv2.INTER_NEAREST)
            data_payload['others_comm_mask_img'] = self._encode_image_to_base64(others_comm_mask_img)

        if ego_feature is not None and ego_feature.size != 0:       # ego_feature.shape = (1, 256, 48, 176)
            ego_feature = ego_feature.copy()
            ego_feature = ego_feature.squeeze()

            ego_feature = ego_feature.sum(0)
            # ego_feature = np.flipud(ego_feature)  # 上下翻转

            # 将feature的值缩放到0到1之间
            # 找到最小值和最大值
            min_val = np.min(ego_feature)
            max_val = np.max(ego_feature)

            # 避免除以零的情况，如果所有值都相同
            if max_val == min_val:
                normalized_ego_feature = np.zeros_like(ego_feature, dtype=np.float32)
            else:
                normalized_ego_feature = (ego_feature - min_val) / (max_val - min_val)

            # BGR格式
            green = np.array([0, 255, 0], dtype=np.float32)  # 绿
            blue = np.array([255, 0, 0], dtype=np.float32)  # 蓝

            # 将 normalized_feature 从 (H, W) 扩展到 (H, W, 1)，以便与 (3,) 形状的颜色向量进行广播
            alpha_channel = normalized_ego_feature[:, :, np.newaxis]

            # 直接进行颜色插值
            interpolated_colors = (1 - alpha_channel) * green + alpha_channel * blue  # 越大越蓝

            # 确保颜色值在0-255范围内，并转换为np.uint8
            ego_feature_img = np.clip(interpolated_colors, 0, 255).astype(np.uint8)

            original_height, original_width = ego_feature_img.shape[:2]

            scale_factor = 5
            new_width = original_width * scale_factor
            new_height = original_height * scale_factor

            ego_feature_img = cv2.resize(ego_feature_img, (new_width, new_height),
                                            interpolation=cv2.INTER_NEAREST)
            data_payload['ego_feature_img'] = self._encode_image_to_base64(ego_feature_img)

        if fused_feature is not None and fused_feature.size != 0:   # fused_feature.shape = (1, 256, 48, 176)
            fused_feature = fused_feature.copy()
            fused_feature = fused_feature.squeeze()
            fused_feature = fused_feature.sum(0)
            # fused_feature = np.flipud(fused_feature)  # 上下翻转

            # 将feature的值缩放到0到1之间
            # 找到最小值和最大值
            min_val = np.min(fused_feature)
            max_val = np.max(fused_feature)

            # 避免除以零的情况，如果所有值都相同
            if max_val == min_val:
                normalized_fused_feature = np.zeros_like(fused_feature, dtype=np.float32)
            else:
                normalized_fused_feature = (fused_feature - min_val) / (max_val - min_val)

            # BGR格式
            green = np.array([0, 255, 0], dtype=np.float32)  # 绿
            blue = np.array([255, 0, 0], dtype=np.float32)  # 蓝

            # 将 normalized_feature 从 (H, W) 扩展到 (H, W, 1)，以便与 (3,) 形状的颜色向量进行广播
            alpha_channel = normalized_fused_feature[:, :, np.newaxis]

            # 直接进行颜色插值
            interpolated_colors = (1 - alpha_channel) * green + alpha_channel * blue  # 越大越蓝

            # 确保颜色值在0-255范围内，并转换为np.uint8
            fused_feature_img = np.clip(interpolated_colors, 0, 255).astype(np.uint8)

            original_height, original_width = fused_feature_img.shape[:2]

            scale_factor = 5
            new_width = original_width * scale_factor
            new_height = original_height * scale_factor

            fused_feature_img = cv2.resize(fused_feature_img, (new_width, new_height),
                                            interpolation=cv2.INTER_NEAREST)
            data_payload['fused_feature_img'] = self._encode_image_to_base64(fused_feature_img)

        if pred_photos is not None and len(pred_photos) == 4:
            # 处理第一张图 (idx 0)
            pred_img_0 = pred_photos[0].copy()
            pred_img_0 = cv2.cvtColor(pred_img_0, cv2.COLOR_RGB2BGR)
            data_payload['pred_img_0'] = self._encode_image_to_base64(pred_img_0)

            # 处理第二张图 (idx 3) - 假设你需要 idx 3
            pred_img_3 = pred_photos[3].copy()
            pred_img_3 = cv2.cvtColor(pred_img_3, cv2.COLOR_RGB2BGR)
            data_payload['pred_img_3'] = self._encode_image_to_base64(pred_img_3)
        else:
            # 发送空白图或保持上一帧（这里为了演示发送空字符串，前端需处理）
            pass

        # 如果有数据，一次性推送到前端事件 'update_frames'
        if data_payload:
            self.socketio.emit('update_frames', data_payload)

    def _register_socket_events(self):
        @self.socketio.on('connect')
        def handle_connect():
            print('Client connected')
