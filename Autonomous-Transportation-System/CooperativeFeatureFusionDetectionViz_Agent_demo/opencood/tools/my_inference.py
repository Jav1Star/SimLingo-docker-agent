# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>, Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import argparse
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

import cProfile
import pstats


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


def main():
    opt = test_parser()
    assert opt.fusion_method in ['late', 'early', 'intermediate']
    assert not (opt.show_vis and opt.show_sequence), 'you can only visualize ' \
                                                    'the results in single ' \
                                                    'image mode or video mode'

    type(opt).__hash__ = lambda ns : id(ns)

    hypes = yaml_utils.load_yaml(None, opt)

    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    print(f"{len(opencood_dataset)} samples found.")
    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             num_workers=1,
                             collate_fn=opencood_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    _, model = train_utils.load_saved_model(saved_path, model)
    model.eval()

    # Create the dictionary for evaluation.
    # also store the confidence score for each prediction
    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
                   0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
                   0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}

    start_time1 = time.time()
    start_time2 = time.time()
    start_time3 = time.time()

    # profile = cProfile.Profile()
    # profile.enable()

    # batch_datas = [batch_data for batch_data in data_loader]
    # next(iter(data_loader))

    # profile.disable()
    # profile.dump_stats('prof.prof')
    # stats = pstats.Stats(profile)
    # stats.sort_stats('cumtime')
    # stats.print_stats()

    for i, batch_data in tqdm(enumerate(data_loader)):

        start_time1 = time.time()
        # print(f"t1 = {start_time1 - start_time3}s")

        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            
            if opt.fusion_method == 'late':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_late_fusion(batch_data,
                                                          model,
                                                          opencood_dataset)
            elif opt.fusion_method == 'early':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_early_fusion(batch_data,
                                                           model,
                                                           opencood_dataset)
            elif opt.fusion_method == 'intermediate':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_intermediate_fusion(batch_data,
                                                                  model,
                                                                  opencood_dataset)
            else:
                raise NotImplementedError('Only early, late and intermediate'
                                          'fusion is supported.')

            start_time2 = time.time()
            # print(f"t2 = {start_time2 - start_time1}s")

            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.3)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.5)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.7)
            print(time.time() - start_time3)
            start_time3 = time.time()
            # print(f"{start_time3 - start_time2}s")

    eval_utils.eval_final_results(result_stat,
                                  opt.model_dir,
                                  opt.global_sort_detections)



if __name__ == '__main__':
    # --------------建立与协同对象的连接---
    # tcp_socket = socket(AF_INET,SOCK_STREAM)
    
    # -------------------- 

    # 使用cProfile对函数进行性能分析


    main()

