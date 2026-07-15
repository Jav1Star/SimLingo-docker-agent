import torch
from torch import nn
from typing import List, Optional
from transformers import AutoModel

from simlingo_adaption_training.models.encoder.visual_token_pruner import VisualTokenPruner



class LingoInternVLModel(nn.Module):
    def __init__(self, variant, *args, **kwargs):
        super().__init__()
        self.model = AutoModel.from_pretrained(variant, trust_remote_code=True) # <class 'transformers_modules.OpenGVLab.InternVL2-1B.0d75ccd166b1d0b79446ae6c5d1a4a667f1e6187.modeling_internvl_chat.InternVLChatModel'>
        try:
            self.num_embeddings = self.model.language_model.model.embed_tokens.num_embeddings
        except:
            self.num_embeddings = self.model.language_model.vocab_size
        self.use_global_img = None
        self.processor = None
        # 视觉 token prune 默认开启为轻量比例，eval 可通过外部配置覆盖。
        self.visual_token_pruner = VisualTokenPruner(
            mode="prune2drive",
            prune_ratio=0.1,
            min_keep=1,
        )
        

        
    def replace_placeholder_tokens(
        adaptor_dict: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        placeholder_values: Optional[List[dict]] = None,
        image_encoder: Optional[nn.Module] = None,
        wp_encoder: Optional[nn.Module] = None,
        scheduer: Optional[nn.Module] = None, # scheduler作为参数传入
        budget: Optional[float] = None,  # [新增参数] 接收外部传入的 Budget 目标
        labels: Optional[torch.LongTensor] = None, # [新增参数] 接收 Labels 用于同步对齐
    ):
        
        if 'tokenizer' in image_encoder.processor.__dict__:
            image_encoder.tokenizer = image_encoder.processor.tokenizer
        else:
            image_encoder.tokenizer = image_encoder.processor

        IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        img_context_token_id = image_encoder.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        image_encoder.img_context_token_id = img_context_token_id
        # ================= [新增] 统一预处理 Budget =================
        # 将 Budget 提前转为 1-d Tensor，供后续所有步骤复用
        budget_tensor = None
        if budget is not None:
            # 参考其他输入 Tensor 对齐 device 和 dtype.
            ref_tensor = adaptor_dict.get('language_inputs', None)
            if ref_tensor is None and 'language__ids' in adaptor_dict:
                print(f"internvl2_model.py bug: language_inputs is None, using language__ids as ref_tensor") 
                ref_tensor = adaptor_dict['language__ids']
            if ref_tensor is None:
                print(f"internvl2_model.py: ref_tensor is None, 'language__ids' is not in adaptor_dict")
            
            device = ref_tensor.device if ref_tensor is not None else image_encoder.device
            dtype = ref_tensor.dtype if ref_tensor is not None and ref_tensor.is_floating_point() else torch.float32

            # 确保转为 1-d Tensor [Batch_Size]
            # 假设 input_ids 存在，我们可以用它的 batch size
            bs = adaptor_dict['language__ids'].shape[0]

            if not isinstance(budget, torch.Tensor):
                budget_tensor = torch.full((bs,), budget, device=device, dtype=dtype)
            else:
                # 如果已经是 Tensor，确保维度和设备正确
                if budget.ndim == 0:
                    budget_tensor = budget.expand(bs).to(device).to(dtype)
                else:
                    budget_tensor = budget.to(device).to(dtype)
        # ===========================================================
        
        output_attentions = output_attentions if output_attentions is not None else image_encoder.model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else image_encoder.model.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else image_encoder.model.config.use_return_dict

        if inputs_embeds is None:
            # 1. Extract the input embeddings
            inputs_embeds = adaptor_dict['language_inputs']
            input_ids = adaptor_dict['language__ids']
            inputs_embeds_dtype = inputs_embeds.dtype
            inputs_embeds_work = inputs_embeds.to(dtype=torch.float32)
            
            # 2a replace placeholder (Waypoint 逻辑保持不变)
            smallest_added_id = image_encoder.tokenizer.additional_special_tokens_ids[0]
            special_ids = torch.tensor(list(set(input_ids[(input_ids >= smallest_added_id)].tolist())), device=input_ids.device)
            special_ids = special_ids.view(-1, 1, 1)
            batch_size, seq_len = input_ids.shape

            if special_ids.size(0) > 0 and len(placeholder_values) > 0:
                wp_encoder_dtype = wp_encoder.mlp[0].weight.dtype
                mask = input_ids == special_ids
                cumsum_mask = torch.cumsum(mask.float(), dim=2)
                first_occurrence_mask = (cumsum_mask == 1) & mask
                first_occurrences = torch.argmax(first_occurrence_mask.float(), dim=2)
                first_occurrences = first_occurrences.transpose(0, 1)
                special_token_pos = first_occurrences.nonzero()

                coords = [torch.tensor(placeholder_values[b_id][special_ids[key_id].item()], device=input_ids.device, dtype=wp_encoder_dtype) for key_id, b_id in zip(special_token_pos[:, 1], special_token_pos[:, 0])]
                coords_length_org = [len(coord) for coord in coords]
                coords = torch.cat(coords)
                wp_embeds = wp_encoder(coords.unsqueeze(0)).squeeze(0)
                wp_embeds = torch.split(wp_embeds, coords_length_org)

                first_occurrences_filtered = [first_occurrences[i] for i in special_token_pos[:, 0]]

                with torch.autocast(device_type=inputs_embeds_work.device.type, enabled=False):
                    for i, (pos, first_occurrence) in enumerate(zip(special_token_pos, first_occurrences_filtered)):
                        start = first_occurrence[pos[1]]
                        end = start + coords_length_org[i]
                        inputs_embeds_work[pos[0], start:end] = wp_embeds[i].to(dtype=inputs_embeds_work.dtype)

            inputs_embeds = inputs_embeds_work.to(dtype=inputs_embeds_dtype)

            # 2. Merge text and images (修改核心：显式拼接 Budget Token)
            if pixel_values is not None and input_ids.shape[1] != 1 and pixel_values.size(0) > 0:
                all_pixel_values = [pixel_values]
                    
                all_image_features = []
                _, N_embed, C_embed = inputs_embeds.shape
                
                for pixel_values_tmp in all_pixel_values:
                    BS, T, NP, C, H, W = pixel_values_tmp.shape
                    assert T == 1, "Only one frame is supported for now"
                    pixel_values_tmp = pixel_values_tmp.view(BS, NP, C, H, W)

                    if pixel_values_tmp.dim() == 5:
                        pixel_values_tmp = pixel_values_tmp.reshape(BS*NP, C, H, W)
                    elif pixel_values_tmp.dim() != 4:
                        raise ValueError(f"pixel_values of shape {pixel_values_tmp.shape}, expect to be of 4 or 5 dimensions")
                    
                    image_features = image_encoder.model.extract_feature(pixel_values_tmp)
                    image_features = image_features.reshape(-1, C_embed)
                    all_image_features.append(image_features)

                vit_embeds = torch.cat(all_image_features, dim=0)
                
                # === [Step A] 准备 Budget Embedding ===
                # 时延
                budget_embed = None
                if budget_tensor is not None: # <--- 改用 budget_tensor 判断
                    # 生成 Embedding: [BS, Hidden]
                    # 直接传处理好的 Tensor 给 scheduler
                    budget_embed = image_encoder.scheduler.budget_encoding(budget_tensor)
                """ if budget is not None:
                    print(f"DEBUG_CTX [2/3] LLM Input: type={type(budget)}")
                    # 确保转为 Tensor [Batch_Size]
                    if not isinstance(budget, torch.Tensor):
                        budget_tensor = torch.tensor([budget] * BS, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                    else:
                        print(f"DEBUG_CTX [2/3] LLM Input Shape: {budget.shape}, dim={budget.ndim}")
                        budget_tensor = budget.to(inputs_embeds.device).to(inputs_embeds.dtype)
                    
                    # 生成 Embedding: [BS, Hidden]
                    budget_embed = image_encoder.scheduler.budget_encoding(budget_tensor) """
                
                # === [Step B] 准备重构序列 ===
                # 文本
                inputs_embeds = inputs_embeds.reshape(BS, N_embed, C_embed)
                input_ids = input_ids.reshape(BS, N_embed)
                # 图像
                vit_embeds = vit_embeds.reshape(BS, -1, C_embed)

                visual_keep_mask = None
                visual_scores = None
                if hasattr(image_encoder, "visual_token_pruner") and image_encoder.visual_token_pruner is not None:
                    # 关键调用点：embedding 置零和 attention mask 必须同步，否则被剪 token 仍会参与注意力。
                    vit_embeds, visual_keep_mask, visual_scores = image_encoder.visual_token_pruner(vit_embeds)
                    adaptor_dict["visual_token_keep_mask"] = visual_keep_mask
                    adaptor_dict["visual_token_scores"] = visual_scores
                
                # 获取原始 Mask (关键！)
                # 优先用 language_inputs_mask，因为它通常是最准确的 attention mask
                old_mask = adaptor_dict.get('language_inputs_mask', adaptor_dict.get('inputs_mask'))
                if old_mask is None:
                    old_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
                
                new_inputs_embeds_list = []
                new_labels_list = []
                new_masks_list = []  # [新增] 用于存储重构后的 Mask
                budget_token_positions = [] 

                for b in range(BS):
                    mask_indices = (input_ids[b] == image_encoder.img_context_token_id)
                    
                    if mask_indices.any():
                        indices = torch.nonzero(mask_indices).squeeze()
                        if indices.dim() == 0: indices = indices.unsqueeze(0)
                        
                        start_idx = indices[0].item()
                        end_idx = indices[-1].item() + 1
                        
                        # 1. 切分 (Embeddings, Labels, AND Masks)
                        prefix = inputs_embeds[b, :start_idx]
                        suffix = inputs_embeds[b, end_idx:]
                        
                        parts_emb = [prefix, vit_embeds[b], suffix]
                        
                        # Labels 切分
                        if labels is not None:
                            prefix_label = labels[b, :start_idx]
                            vision_label = torch.full((vit_embeds[b].shape[0],), -100, dtype=labels.dtype, device=labels.device)
                            suffix_label = labels[b, end_idx:]
                            parts_label = [prefix_label, vision_label, suffix_label]
                            
                        # === [修正点] Mask 切分与重构 ===
                        if old_mask is not None:
                            prefix_mask = old_mask[b, :start_idx]
                            if visual_keep_mask is not None:
                                vision_mask = visual_keep_mask[b].to(dtype=old_mask.dtype, device=old_mask.device)
                            else:
                                vision_mask = torch.ones((vit_embeds[b].shape[0],), dtype=old_mask.dtype, device=old_mask.device)
                            suffix_mask = old_mask[b, end_idx:]
                            parts_mask = [prefix_mask, vision_mask, suffix_mask]
                        else:
                            # 如果没有 mask，默认为全 1 (极少情况)
                            parts_mask = [] # 后续处理
                        
                        # 2. 插入 Budget Token (在最后)
                        if budget_embed is not None:
                            # Embedding
                            parts_emb.append(budget_embed[b].unsqueeze(0))
                            
                            # Label
                            if labels is not None:
                                parts_label.append(torch.tensor([-100], dtype=labels.dtype, device=labels.device))
                            
                            # Mask (补 1)
                            if old_mask is not None:
                                parts_mask.append(torch.tensor([1], dtype=old_mask.dtype, device=old_mask.device))
                            
                            # Position
                            pos = prefix.shape[0] + vit_embeds[b].shape[0] + suffix.shape[0]
                            budget_token_positions.append(pos)
                        else:
                            budget_token_positions.append(0)
                            
                        # 3. 拼接
                        new_inputs_embeds_list.append(torch.cat(parts_emb, dim=0))
                        if labels is not None:
                            new_labels_list.append(torch.cat(parts_label, dim=0))
                        if old_mask is not None:
                            new_masks_list.append(torch.cat(parts_mask, dim=0))
                            
                    else:
                        # 纯文本情况
                        new_inputs_embeds_list.append(inputs_embeds[b])
                        if labels is not None:
                            new_labels_list.append(labels[b])
                        
                        # 纯文本 Mask 处理
                        parts_mask = [old_mask[b]] if old_mask is not None else []
                        
                        # Budget 插入
                        if budget_embed is not None:
                            parts_emb = [inputs_embeds[b], budget_embed[b].unsqueeze(0)]
                            new_inputs_embeds_list[-1] = torch.cat(parts_emb, dim=0) # 更新刚才 append 的
                            
                            pos = inputs_embeds[b].shape[0]
                            budget_token_positions.append(pos)

                            if labels is not None:
                                parts_label = [labels[b], torch.tensor([-100], dtype=labels.dtype, device=labels.device)]
                                new_labels_list[-1] = torch.cat(parts_label, dim=0)
                            
                            if old_mask is not None:
                                parts_mask.append(torch.tensor([1], dtype=old_mask.dtype, device=old_mask.device))
                        else:
                            budget_token_positions.append(0)
                        
                        if old_mask is not None:
                            new_masks_list.append(torch.cat(parts_mask, dim=0))

                # === [Step C] 更新回 adaptor_dict ===
                inputs_embeds = torch.stack(new_inputs_embeds_list, dim=0)
                adaptor_dict['language_inputs'] = inputs_embeds
                
                if labels is not None:
                    adaptor_dict['labels'] = torch.stack(new_labels_list, dim=0)
                
                # === [Step D 修正版] 更新 Mask ===
                if old_mask is not None:
                    new_mask = torch.stack(new_masks_list, dim=0)
                    if 'language_inputs_mask' in adaptor_dict:
                        adaptor_dict['language_inputs_mask'] = new_mask
                    if 'inputs_mask' in adaptor_dict:
                        adaptor_dict['inputs_mask'] = new_mask

                # === [Step E] 打包 AdaLLaVA 参数 ===
                if budget is not None:
                    adaptor_dict['budget_token_position'] = torch.tensor(
                        budget_token_positions, device=inputs_embeds.device
                    )
                    # 修改前：存入原始数据 (可能是标量)
                    # adaptor_dict['budget'] = budget 
                    # 修改后：存入韩式开头已经处理好的 1-d Tensor
                    adaptor_dict['budget'] = budget_tensor
                    adaptor_dict['scheduler'] = image_encoder.scheduler.forward

            # pixel_values is not None but is empty ---> text only cases
            elif pixel_values is not None and input_ids.shape[1] != 1 and pixel_values.size(0) == 0:
                pass
            
            # 这里不再处理adaptor_dict['inputs']
            
        return adaptor_dict
