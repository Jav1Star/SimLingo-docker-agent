# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>, Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
import time
from tqdm import tqdm

import torch
import open3d as o3d
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import vis_utils
import matplotlib.pyplot as plt

from socket import *
import pickle


def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', required=True, type=str,
                        default='late',
                        help='late, early or intermediate')
    parser.add_argument('--show_vis', action='store_true',
                        help='whether to show image visualization result')
    parser.add_argument('--show_sequence', action='store_true',
                        help='whether to show video visualization result.'
                             'it can note be set true with show_vis together ')
    parser.add_argument('--save_vis', action='store_true',
                        help='whether to save visualization result')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy_test file')
    parser.add_argument('--global_sort_detections', action='store_true',
                        help='whether to globally sort detections by confidence score.'
                             'If set to True, it is the mainstream AP computing method,'
                             'but would increase the tolerance for FP (False Positives).')
    opt = parser.parse_args()
    return opt

def print_tensor_paths_and_shapes(structure, prefix=""):
    """
    递归统计嵌套结构中的 Tensor，输出它们的路径和形状。
    :param structure: 嵌套结构 (dict, list, tensor)
    :param prefix: 当前路径的前缀
    """
    if isinstance(structure, dict):
        for key, value in structure.items():
            # 递归处理字典
            current_path = f"{prefix}.{key}" if prefix else key
            print_tensor_paths_and_shapes(value, current_path)
    elif isinstance(structure, list):
        for idx, value in enumerate(structure):
            # 递归处理列表
            current_path = f"{prefix}[{idx}]"
            print_tensor_paths_and_shapes(value, current_path)
    elif isinstance(structure, torch.Tensor):
        # 遇到 Tensor，输出路径和形状
        print(f"Tensor found at path: '{prefix}', shape: {structure.shape}")
    else:
        # 其他类型不处理
        pass


class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        """
        ModelWrapper 用于将多个张量作为输入参数传递到模型，并组装成嵌套字典结构。
        
        Args:
            model (nn.Module): 实际的模型实例。
        """
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(
        self,
        object_bbx_mask,
        voxel_features,
        voxel_coords,
        voxel_num_points,
        record_len,
        targets,
        pos_equal_one,
        neg_equal_one,
        object_ids,
        prior_encoding,
        spatial_correction_matrix,
        pairwise_t_matrix,
        origin_lidar,
        anchor_box,
        transformation_matrix,
    ):
        """
        将输入张量组装成嵌套字典结构，并传递给实际模型。
        
        Args:
            object_bbx_mask (Tensor): Shape [1, 100]
            voxel_features (Tensor): Shape [15401, 32, 4]
            voxel_coords (Tensor): Shape [15401, 4]
            voxel_num_points (Tensor): Shape [15401]
            record_len (Tensor): Shape [1]
            targets (Tensor): Shape [1, 48, 176, 14]
            pos_equal_one (Tensor): Shape [1, 48, 176, 2]
            neg_equal_one (Tensor): Shape [1, 48, 176, 2]
            object_ids (Tensor): Shape [8]
            prior_encoding (Tensor): Shape [1, 5, 3]
            spatial_correction_matrix (Tensor): Shape [1, 5, 4, 4]
            pairwise_t_matrix (Tensor): Shape [1, 5, 5, 4, 4]
            origin_lidar (Tensor): Shape [1, 109231, 4]
            anchor_box (Tensor): Shape [48, 176, 2, 7]
            transformation_matrix (Tensor): Shape [4, 4]
        
        Returns:
            模型的输出结果。
        """
        # 将输入张量组装为嵌套字典结构
        inputs = {
            "object_bbx_mask": object_bbx_mask,
            "processed_lidar": {
                "voxel_features": voxel_features,
                "voxel_coords": voxel_coords,
                "voxel_num_points": voxel_num_points,
            },
            "record_len": record_len,
            "label_dict": {
                "targets": targets,
                "pos_equal_one": pos_equal_one,
                "neg_equal_one": neg_equal_one,
            },
            "object_ids": object_ids,
            "prior_encoding": prior_encoding,
            "spatial_correction_matrix": spatial_correction_matrix,
            "pairwise_t_matrix": pairwise_t_matrix,
            "origin_lidar": origin_lidar,
            "anchor_box": anchor_box,
            "transformation_matrix": transformation_matrix,
        }
        # 将组装的字典传递给实际模型
        return self.model(inputs)


def main():
    opt = test_parser()
    assert opt.fusion_method in ['late', 'early', 'intermediate']
    assert not (opt.show_vis and opt.show_sequence), 'you can only visualize ' \
                                                    'the results in single ' \
                                                    'image mode or video mode'

    hypes = yaml_utils.load_yaml(None, opt)

    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    print(f"{len(opencood_dataset)} samples found.")
    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             num_workers=0,
                             collate_fn=opencood_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    _, model = train_utils.load_saved_model(saved_path, model)

    model = ModelWrapper(model)

    for batch_data in data_loader:
        # batch_data = train_utils.to_device(batch_data, device)

        object_bbx_mask = batch_data['ego']['object_bbx_mask']
        voxel_features = batch_data['ego']['processed_lidar']['voxel_features']
        voxel_coords = batch_data['ego']['processed_lidar']['voxel_coords']
        voxel_num_points = batch_data['ego']['processed_lidar']['voxel_num_points']
        record_len = batch_data['ego']['record_len']
        targets = batch_data['ego']['label_dict']['targets']
        pos_equal_one = batch_data['ego']['label_dict']['pos_equal_one']
        neg_equal_one = batch_data['ego']['label_dict']['neg_equal_one']
        object_ids = batch_data['ego']['object_ids']
        prior_encoding = batch_data['ego']['prior_encoding']
        spatial_correction_matrix = batch_data['ego']['spatial_correction_matrix']
        pairwise_t_matrix = batch_data['ego']['pairwise_t_matrix']
        origin_lidar = batch_data['ego']['origin_lidar']
        anchor_box = batch_data['ego']['anchor_box']
        transformation_matrix = batch_data['ego']['transformation_matrix']

        torch.onnx.export(model, 
                        (object_bbx_mask, voxel_features, voxel_coords, voxel_num_points, record_len, targets, pos_equal_one, neg_equal_one, object_ids, prior_encoding, spatial_correction_matrix, pairwise_t_matrix, origin_lidar, anchor_box, transformation_matrix,), 
                        "point_pillar_where2comm_2024_10_28_24_50.onnx", 
                        input_names=("object_bbx_mask", "voxel_features", "voxel_coords", "voxel_num_points", "record_len", "targets", "pos_equal_one", "neg_equal_one", "object_ids", "prior_encoding", "spatial_correction_matrix", "pairwise_t_matrix", "origin_lidar", "anchor_box", "transformation_matrix"),
                        output_names=["output"], opset_version=11)
        break



if __name__ == '__main__':

    main()
