"""
Implementation of Where2comm fusion.
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import sys
from socket import *


from opencood.models.fuse_modules.self_attn import ScaledDotProductAttention


class Communication(nn.Module):
    def __init__(self, args):
        super(Communication, self).__init__()
        # Threshold of objectiveness
        self.threshold = args['threshold']
        if 'gaussian_smooth' in args:
            # Gaussian Smooth
            self.smooth = True
            kernel_size = args['gaussian_smooth']['k_size']
            c_sigma = args['gaussian_smooth']['c_sigma']
            self.gaussian_filter = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)
            self.init_gaussian_filter(kernel_size, c_sigma)
            self.gaussian_filter.requires_grad = False
        else:
            self.smooth = False

    def init_gaussian_filter(self, k_size=5, sigma=1.0):
        center = k_size // 2
        x, y = np.mgrid[0 - center: k_size - center, 0 - center: k_size - center]
        gaussian_kernel = 1 / (2 * np.pi * sigma) * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))

        self.gaussian_filter.weight.data = torch.Tensor(gaussian_kernel).to(
            self.gaussian_filter.weight.device).unsqueeze(0).unsqueeze(0)
        self.gaussian_filter.bias.data.zero_()

    # def forward(self, batch_confidence_maps, B):
    #     """
    #     Args:
    #         batch_confidence_maps: [(L1, H, W), (L2, H, W), ...]
    #     """
    #
    #     _, _, H, W = batch_confidence_maps[0].shape  # [2,2,48,176]
    #
    #     communication_masks = []
    #     communication_rates = []
    #     ego_comm_mask = None
    #     for b in range(B):
    #         # [2,1,48,176]
    #         ori_communication_maps, _ = batch_confidence_maps[b].sigmoid().max(dim=1, keepdim=True)
    #
    #         if self.smooth:
    #             communication_maps = self.gaussian_filter(ori_communication_maps)
    #         else:
    #             communication_maps = ori_communication_maps
    #         # communication_maps [2,1,48,176]
    #         L = communication_maps.shape[0]  # 2
    #
    #         if self.training:
    #             # Official training proxy objective
    #             K = int(H * W * random.uniform(0, 1))
    #             communication_maps = communication_maps.reshape(L, H * W)
    #             _, indices = torch.topk(communication_maps, k=K, sorted=False)
    #             communication_mask = torch.zeros_like(communication_maps).to(communication_maps.device)
    #             ones_fill = torch.ones(L, K, dtype=communication_maps.dtype, device=communication_maps.device)
    #             communication_mask = torch.scatter(communication_mask, -1, indices, ones_fill).reshape(L, 1, H, W)
    #         elif self.threshold:
    #             # ones_mask = torch.ones_like(communication_maps).to(communication_maps.device)
    #             # zeros_mask = torch.zeros_like(communication_maps).to(communication_maps.device)
    #             # communication_mask = torch.where(communication_maps > self.threshold, ones_mask, zeros_mask)
    #
    #             communication_mask = torch.zeros_like(communication_maps)
    #             indices = torch.topk(communication_maps.view(-1), k=24).indices
    #             communication_mask.view(-1)[indices] = 1
    #
    #         else:
    #             communication_mask = torch.ones_like(communication_maps).to(communication_maps.device)
    #
    #         communication_rate = communication_mask.sum() / (L * H * W)
    #         # Ego
    #         if b == 0:
    #             ego_comm_mask = communication_mask[0].clone().unsqueeze(0)
    #
    #         communication_mask[0] = 1
    #
    #         communication_masks.append(communication_mask)
    #         communication_rates.append(communication_rate)
    #     communication_rates = sum(communication_rates) / B
    #     communication_masks = torch.cat(communication_masks, dim=0)
    #     return communication_masks, communication_rates, ego_comm_mask

    def forward(self, confidence_maps):
        _, _, H, W = confidence_maps.shape  # [2,2,48,176]

        # [2,1,48,176]
        ori_communication_maps, _ = confidence_maps.sigmoid().max(dim=1, keepdim=True)

        if self.smooth:
            communication_maps = self.gaussian_filter(ori_communication_maps)
        else:
            communication_maps = ori_communication_maps
        # communication_maps [2,1,48,176]
        L = communication_maps.shape[0]  # 2

        if self.training:
            # Official training proxy objective
            K = int(H * W * random.uniform(0, 1))
            communication_maps = communication_maps.reshape(L, H * W)
            _, indices = torch.topk(communication_maps, k=K, sorted=False)
            communication_masks = torch.zeros_like(communication_maps).to(communication_maps.device)
            ones_fill = torch.ones(L, K, dtype=communication_maps.dtype, device=communication_maps.device)
            communication_masks = torch.scatter(communication_masks, -1, indices, ones_fill).reshape(L, 1, H, W)
        elif self.threshold:
            ones_mask = torch.ones_like(communication_maps).to(communication_maps.device)
            zeros_mask = torch.zeros_like(communication_maps).to(communication_maps.device)
            communication_masks = torch.where(communication_maps > self.threshold, ones_mask, zeros_mask)

        else:
            communication_masks = torch.ones_like(communication_maps).to(communication_maps.device)

        if L > 1:
            communication_rate = communication_masks[1:L].sum() / ((L - 1) * H * W)
        else:
            communication_rate = -1.0
        # Ego
        ego_comm_mask = communication_masks[0].clone().unsqueeze(0)

        communication_masks[0] = 1

        return communication_masks, communication_rate, ego_comm_mask

class AttentionFusion(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionFusion, self).__init__()
        self.att = ScaledDotProductAttention(feature_dim)

    def forward(self, x):
        cav_num, C, H, W = x.shape
        x = x.view(cav_num, C, -1).permute(2, 0, 1)  # (H*W, cav_num, C), perform self attention on each pixel
        x = self.att(x, x, x)
        x = x.permute(1, 2, 0).view(cav_num, C, H, W)[0]  # C, W, H before
        # 在最后的输出中，只返回第一个智能体的特征，可能意味着这个类的主要目的是将其他智能体的特征融合到第一个智能体中，以便它能够拥有综合的特征。
        return x


class Where2comm(nn.Module):
    def __init__(self, args):
        super(Where2comm, self).__init__()
        self.discrete_ratio = args['voxel_size'][0]
        self.downsample_rate = args['downsample_rate']

        self.fully = args['fully']
        if self.fully:
            print('constructing a fully connected communication graph')
        else:
            print('constructing a partially connected communication graph')

        self.multi_scale = args['multi_scale']
        if self.multi_scale:
            layer_nums = args['layer_nums']
            num_filters = args['num_filters'] # [ 64, 128, 256 ]
            self.num_levels = len(layer_nums)
            self.fuse_modules = nn.ModuleList()
            for idx in range(self.num_levels):
                fuse_network = AttentionFusion(num_filters[idx])
                self.fuse_modules.append(fuse_network)
        else:
            self.fuse_modules = AttentionFusion(args['in_channels'])  # 256

        self.naive_communication = Communication(args['communication'])

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    # def forward(self, x, psm_single, record_len, pairwise_t_matrix, backbone=None, comm_masked_features=None):
    def forward(self, x, psm_single, backbone=None, comm_masked_features=None):
        """
        Fusion forwarding.

        Parameters:
            x: Input data, (sum(n_cav), C, H, W).
            record_len: List, (B).
            pairwise_t_matrix: The transformation matrix from each cav to ego, (B, L, L, 4, 4).

        Returns:
            Fused feature.
        """

        _, C, H, W = x.shape
        # B = pairwise_t_matrix.shape[0]

        if self.multi_scale:
            # ego_x = x[0].clone().unsqueeze(0)
            #
            # ego_ups = []
            #
            # for i in range(self.num_levels):
            #     ego_x = backbone.blocks[i](ego_x)
            #     ego_x_fuse = [self.fuse_modules[i](ego_x)]
            #
            #     ego_x_fuse = torch.stack(ego_x_fuse)
            #
            #     # 4. Deconv
            #     if len(backbone.deblocks) > 0:
            #         ego_ups.append(backbone.deblocks[i](ego_x_fuse))
            #     else:
            #         ego_ups.append(ego_x_fuse)
            #
            # if len(ego_ups) > 1:
            #     ego_x = torch.cat(ego_ups, dim=1)
            # elif len(ego_ups) == 1:
            #     ego_x = ego_ups[0]
            #
            # if len(backbone.deblocks) > self.num_levels:
            #     ego_x = backbone.deblocks[-1](ego_x)

            ups = []

            for i in range(self.num_levels):
                x = backbone.blocks[i](x)

                # 1. Communication (mask the features)
                if i == 0:
                    if self.fully:
                        communication_rates = torch.tensor(1).to(x.device)
                    else:
                        # Prune
                        # batch_confidence_maps = self.regroup(psm_single, record_len)
                        confidence_maps = psm_single

                        # for i in range(len(batch_confidence_maps)):
                        #     print(batch_confidence_maps[i].shape)
                        # print("----------------")

                        # ----------------------
                        # send_data = pickle.dumps(batch_confidence_maps)
                        # send_data = zlib.compress(send_data)
                        # send_data="1
                        # tensor = torch.tensor([1.0, 2.0, 3.0])
                        # send_data = pickle.dumps(tensor)
                        # inference.tcp_socket.send(send_data)
                        # print("444444\n")
                        # from_server_msg = inference.tcp_socket.recv(4096)
                        # print("555555\n")
                        # from_server_msg = zlib.decompress(from_server_msg)
                        # batch_confidence_maps=pickle.loads(from_server_msg)
                        # -------------------------------

                        # communication_masks, communication_rates, ego_comm_mask = self.naive_communication(batch_confidence_maps, B)
                        communication_masks, communication_rate, ego_comm_mask = self.naive_communication(confidence_maps)

                        if x.shape[-1] != communication_masks.shape[-1]:
                            communication_masks = F.interpolate(communication_masks, size=(x.shape[-2], x.shape[-1]),
                                                                mode='bilinear', align_corners=False)
                        x = x * communication_masks

                        if comm_masked_features is not None and len(comm_masked_features) > 0:
                            features = []
                            communication_rate_sum = 0
                            for comm_masked_feature_dict in comm_masked_features:
                                comm_masked_feature = comm_masked_feature_dict['comm_masked_feature']
                                comm_mask = comm_masked_feature_dict['comm_mask']

                                communication_rate_sum += comm_mask.sum() / comm_mask.numel()

                                # 1. 插值掩码到目标尺寸
                                if x.shape[-1] != comm_mask.shape[-1]:
                                    comm_mask = F.interpolate(comm_mask,
                                                              size=(x.shape[-2], x.shape[-1]),
                                                              mode='bilinear', align_corners=False)     # [1, 1, H, W]

                                # 2. 获取掩码的非零位置索引
                                spatial_mask = comm_mask.squeeze(0).squeeze(0)  # [H, W]
                                non_zero_indices = torch.nonzero(spatial_mask != 0, as_tuple=False)  # [N, 2]

                                # 3. 创建全零特征图
                                c = comm_masked_feature.shape[0]
                                feature = torch.zeros((1, c, x.shape[-2], x.shape[-1]),
                                                      dtype=comm_masked_feature.dtype,
                                                      device=comm_masked_feature.device)  # [1, C, H, W]

                                # 4. 将提取的特征填充到非零位置
                                # 对于每个通道
                                for ch in range(c):
                                    # 在非零位置填充值
                                    feature[0, ch, non_zero_indices[:, 0], non_zero_indices[:, 1]] = comm_masked_feature[ch]

                                features.append(feature)

                            features.insert(0, x)
                            x = torch.cat(features, dim=0)

                            communication_rate = communication_rate_sum / len(comm_masked_features)

                # 2. Split the features
                # split_x: [(L1, C, H, W), (L2, C, H, W), ...]
                # For example [[2, 256, 48, 176], [1, 256, 48, 176], ...]
                # batch_node_features = self.regroup(x, record_len)
                node_features = x

                # 3. Fusion
                x_fuse = []

                # for b in range(B):
                #     neighbor_feature = batch_node_features[b]
                #     x_fuse.append(self.fuse_modules[i](neighbor_feature))
                neighbor_feature = node_features
                x_fuse.append(self.fuse_modules[i](neighbor_feature))

                x_fuse = torch.stack(x_fuse)

                # 4. Deconv
                if len(backbone.deblocks) > 0:
                    ups.append(backbone.deblocks[i](x_fuse))
                else:
                    ups.append(x_fuse)
            # for i in range(len(ups)):
            #     print(ups[i].shape)
            # print("------------------------------------")

            if len(ups) > 1:
                x_fuse = torch.cat(ups, dim=1)
            elif len(ups) == 1:
                x_fuse = ups[0]

            if len(backbone.deblocks) > self.num_levels:
                x_fuse = backbone.deblocks[-1](x_fuse)

            # return x_fuse, communication_rate, ego_comm_mask, ego_x
            return x_fuse, communication_rate, ego_comm_mask
        else:
            # TODO: 暂时没用到
            """
            # 1. Communication (mask the features)
            if self.fully:
                communication_rates = torch.tensor(1).to(x.device)
            else:
                # Prune
                batch_confidence_maps = self.regroup(psm_single, record_len)
                communication_masks, communication_rates, ego_comm_mask = self.naive_communication(batch_confidence_maps, B)
                x = x * communication_masks  # 特征与通信掩码相乘

                if comm_masked_features is not None:
                    features = []

                    for comm_masked_feature_dict in comm_masked_features:
                        comm_masked_feature = comm_masked_feature_dict['feature']
                        comm_mask = comm_masked_feature_dict['comm_mask']

                        # 1. 插值掩码到目标尺寸
                        if x.shape[-1] != comm_mask.shape[-1]:
                            comm_mask = F.interpolate(comm_mask,
                                                      size=(x.shape[-2], x.shape[-1]),
                                                      mode='bilinear', align_corners=False)  # [1, 1, H, W]

                        # 2. 获取掩码的非零位置索引
                        spatial_mask = comm_mask.squeeze(0).squeeze(0)  # [H, W]
                        non_zero_indices = torch.nonzero(spatial_mask != 0, as_tuple=False)  # [N, 2]

                        # 3. 创建全零特征图
                        c = comm_masked_feature.shape[0]
                        feature = torch.zeros((1, c, x.shape[-2], x.shape[-1]),
                                              dtype=comm_masked_feature.dtype,
                                              device=comm_masked_feature.device)  # [1, C, H, W]

                        # 4. 将提取的特征填充到非零位置
                        # 对于每个通道
                        for ch in range(c):
                            # 在非零位置填充值
                            feature[0, ch, non_zero_indices[:, 0], non_zero_indices[:, 1]] = comm_masked_feature[ch]

                        features.append(feature)

                    features.insert(0, x)
                    x = torch.cat(features, dim=0)

            # 2. Split the features
            # split_x: [(L1, C, H, W), (L2, C, H, W), ...]
            # For example [[2, 256, 48, 176], [1, 256, 48, 176], ...]
            batch_node_features = self.regroup(x, record_len)

            # 3. Fusion
            x_fuse = []
            for b in range(B):
                neighbor_feature = batch_node_features[b]
                x_fuse.append(self.fuse_modules(neighbor_feature))
            x_fuse = torch.stack(x_fuse)
            """
            return -1, -1, -1
        # return x_fuse, communication_rates, ego_comm_mask

    # def spatial_feature_to_comm_mask(self, spatial_feature, psm_single, record_len, pairwise_t_matrix, backbone=None):
    def spatial_feature_to_comm_mask(self, spatial_feature, psm_single, backbone=None):
        _, C, H, W = spatial_feature.shape  # [1, 64, 192, 704]
        # B = pairwise_t_matrix.shape[0]

        if self.multi_scale:
            x = backbone.blocks[0](spatial_feature)
            # Prune
            # batch_confidence_maps = self.regroup(psm_single, record_len)
            confidence_maps = psm_single

            # communication_masks, communication_rates, ego_comm_mask = self.naive_communication(batch_confidence_maps, B)
            communication_masks, communication_rate, ego_comm_mask = self.naive_communication(confidence_maps)
        else:
            x = spatial_feature
            # Prune
            # batch_confidence_maps = self.regroup(psm_single, record_len)
            # communication_masks, communication_rates, ego_comm_mask = self.naive_communication(batch_confidence_maps, B)
            confidence_maps = psm_single
            communication_masks, communication_rate, ego_comm_mask = self.naive_communication(confidence_maps)

        comm_mask_tensor = ego_comm_mask
        return comm_mask_tensor

    # def spatial_feature_to_comm_masked_feature(self, spatial_feature, psm_single, record_len, pairwise_t_matrix, backbone=None, request_map=None):
    def spatial_feature_to_comm_masked_feature(self, spatial_feature, psm_single, backbone=None, request_map=None):

        _, C, H, W = spatial_feature.shape  # [1, 64, 192, 704]

        if self.multi_scale:
            x = backbone.blocks[0](spatial_feature)
            # Prune
            # batch_confidence_maps = self.regroup(psm_single, record_len)
            confidence_maps = psm_single

            # communication_masks, communication_rates, ego_comm_mask = self.naive_communication(batch_confidence_maps, B)
            communication_masks, communication_rate, ego_comm_mask = self.naive_communication(confidence_maps)

            if request_map is not None:
                ego_comm_mask = ego_comm_mask * request_map

                # TODO: 因带宽限制设置的使用topk选取communication_maps中最高K位进行传输
                # ###########################################################################
                # communication_masks = torch.zeros_like(communication_maps)
                # indices = torch.topk(communication_maps.view(-1), k=24).indices
                # communication_masks.view(-1)[indices] = 1
                # ###########################################################################

            comm_mask_tensor = ego_comm_mask

            # 1. 插值掩码到特征图尺寸
            if x.shape[-1] != ego_comm_mask.shape[-1]:
                ego_comm_mask = F.interpolate(ego_comm_mask, size=(x.shape[-2], x.shape[-1]),
                                                    mode='bilinear', align_corners=False)
        else:
            x = spatial_feature
            # Prune
            # batch_confidence_maps = self.regroup(psm_single, record_len)
            # communication_masks, communication_rates, ego_comm_mask = self.naive_communication(batch_confidence_maps, B)
            confidence_maps = psm_single
            communication_masks, communication_rate, ego_comm_mask = self.naive_communication(confidence_maps)

            if request_map is not None:
                ego_comm_mask = ego_comm_mask * request_map

            comm_mask_tensor = ego_comm_mask

        # 2. 应用掩码
        x = x * ego_comm_mask

        # 3. 获取掩码的非零位置索引 (空间位置)
        # 掩码形状: [1, 1, H, W]
        spatial_mask = ego_comm_mask.squeeze(0).squeeze(0)  # [H, W]
        non_zero_indices = torch.nonzero(spatial_mask != 0, as_tuple=False)  # [N, 2] (h, w)

        # 4. 提取特征 (每个通道在非零位置的值)
        # x形状: [1, C, H, W] -> [C, H, W]
        x_squeezed = x.squeeze(0)  # [C, H, W]

        # 提取特征: [C, N]
        comm_masked_feature_tensor = x_squeezed[:, non_zero_indices[:, 0], non_zero_indices[:, 1]]

        return comm_masked_feature_tensor, comm_mask_tensor

    def fuse_features(self, features: list[torch.Tensor], backbone=None) -> torch.Tensor:
        x = torch.cat(features, dim=0)  # [N, C, H, W]

        if self.multi_scale:
            ups = []

            for i in range(self.num_levels):
                if i > 0:
                    x = backbone.blocks[i](x)
                x_fuse = [self.fuse_modules[i](x)]
                x_fuse = torch.stack(x_fuse)

                if len(backbone.deblocks) > 0:
                    ups.append(backbone.deblocks[i](x_fuse))
                else:
                    ups.append(x_fuse)

            if len(ups) > 1:
                x_fuse = torch.cat(ups, dim=1)
            elif len(ups) == 1:
                x_fuse = ups[0]

            if len(backbone.deblocks) > self.num_levels:
                x_fuse = backbone.deblocks[-1](x_fuse)

            return x_fuse
        else:
            # TODO: 暂时没用到
            return -1
