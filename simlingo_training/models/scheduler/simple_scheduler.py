import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .scheduler_utils import *


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
    def forward(self, x):
        return self.net(x)


class SimpleScheduler_L(nn.Module):
    def __init__(self, config, tau=5, is_hard=True, threshold=0.5, bias=True):
        super().__init__()
        self.num_prefix_layers = config.num_prefix_layers
        self.num_hidden_layers = config.num_hidden_layers
        num_sub_layer = config.num_hidden_layers - config.num_prefix_layers
        self.num_attention_heads = config.num_attention_heads
        self.is_hard = is_hard
        self.tau = tau

        self.mlp_head = nn.Linear(config.hidden_size, num_sub_layer, bias=bias)
        self.scheduler_up_proj = FeedForward(256, config.hidden_size, config.hidden_size)

    def set_tau(self, tau):
        self.tau = tau

    def latency_encoding(self, latency):
        quantized_latency = latency_quantizing(latency, self.num_prefix_layers, self.num_hidden_layers)[1]

        # Scale the batch latency to a range of [0, 2π]
        scaled_values = quantized_latency * 2 * torch.pi  # Shape: [batch_size]

        # Generate frequency indices to create a diverse range of sine and cosine values
        frequencies = 1 / (10000 ** (torch.arange(128).to(quantized_latency.device) / 128))  # 128 frequencies for each sin and cos

        # Expand dimensions to compute sine and cosine for each value in the batch with each frequency
        # `scaled_values[:, None]` adds a dimension to match (batch_size, 1) with (128), resulting in (batch_size, 128)
        sin_values = torch.sin(scaled_values[:, None] * frequencies)  # Shape: [batch_size, 128]
        cos_values = torch.cos(scaled_values[:, None] * frequencies)  # Shape: [batch_size, 128]

        # Concatenate sin and cos values along the last dimension to get a tensor of size [batch_size, 256]
        latency_emb = torch.cat((sin_values, cos_values), dim=1).to(quantized_latency.dtype)  # Shape: [batch_size, 256]
        
        latency_emb = self.scheduler_up_proj(latency_emb)
        return latency_emb

    def forward(self, x, latency):
        latency = latency_quantizing(latency, self.num_prefix_layers, self.num_hidden_layers)[0]
        logits = self.mlp_head(x)
        output_samples = []
        for logits_, latency_ in zip (logits, latency):
            sample = n_times_gumbel_softmax(logits_, latency_.item(), self.tau, self.is_hard, training=self.training)
            output_samples.append(sample)
        
        output_samples = torch.stack(output_samples)
        output_samples = output_samples[:,:,None,None].repeat(1, 1, 2, self.num_attention_heads)

        prefix_execution_plan = self.get_prefix_execution_plan(output_samples)

        output_samples = torch.cat([prefix_execution_plan, output_samples], 1)
        return output_samples
    
    def get_prefix_execution_plan(self, output_samples):
        new_shape = list(output_samples.shape)
        new_shape[1] = self.num_prefix_layers

        prefix_execution_plan = torch.ones(new_shape, device=output_samples.device, dtype=output_samples.dtype)
        return prefix_execution_plan

    def get_random_latency(self, batch_size):
        latency = torch.randint(self.num_prefix_layers, self.num_hidden_layers + 1, (batch_size,)) / self.num_hidden_layers
        return latency
    

class SimpleScheduler_H(nn.Module):
    def __init__(self, config, tau=5, is_hard=True, threshold=0.5, bias=True):
        super().__init__()
        self.num_prefix_layers = config.num_prefix_layers
        self.num_hidden_layers = config.num_hidden_layers
        self.num_sub_layer = config.num_hidden_layers - config.num_prefix_layers
        self.num_attention_heads = config.num_attention_heads
        self.is_hard = is_hard
        self.tau = tau
        self.rank = config.scheduler_rank

        self.mlp_head = nn.Linear(config.hidden_size, self.num_sub_layer * self.num_attention_heads * 2, bias=bias)
        self.scheduler_up_proj = FeedForward(256, config.hidden_size, config.hidden_size)

    def set_tau(self, tau):
        self.tau = tau

    def latency_encoding(self, latency):
        quantized_latency = latency_quantizing(latency, 
                                               self.num_prefix_layers * self.num_attention_heads // self.rank, 
                                               self.num_hidden_layers * self.num_attention_heads // self.rank)[1]

        # Scale the batch latency to a range of [0, 2π]
        scaled_values = quantized_latency * 2 * torch.pi  # Shape: [batch_size]

        # Generate frequency indices to create a diverse range of sine and cosine values
        frequencies = 1 / (10000 ** (torch.arange(128).to(quantized_latency.device) / 128))  # 128 frequencies for each sin and cos

        # Expand dimensions to compute sine and cosine for each value in the batch with each frequency
        # `scaled_values[:, None]` adds a dimension to match (batch_size, 1) with (128), resulting in (batch_size, 128)
        sin_values = torch.sin(scaled_values[:, None] * frequencies)  # Shape: [batch_size, 128]
        cos_values = torch.cos(scaled_values[:, None] * frequencies)  # Shape: [batch_size, 128]

        # Concatenate sin and cos values along the last dimension to get a tensor of size [batch_size, 256]
        latency_emb = torch.cat((sin_values, cos_values), dim=1).to(quantized_latency.dtype)  # Shape: [batch_size, 256]
        
        latency_emb = self.scheduler_up_proj(latency_emb)
        return latency_emb
    
    def forward(self, x, latency):
        latency = latency_quantizing(latency, 
                                     self.num_prefix_layers * self.num_attention_heads // self.rank, 
                                     self.num_hidden_layers * self.num_attention_heads // self.rank)[0]
        logits = self.mlp_head(x)
        output_samples = []
        for logits_, latency_ in zip (logits, latency):
            logits_ = logits_.view(self.num_sub_layer, self.num_attention_heads * 2)
            sample = n_times_gumbel_softmax_head_version(logits_, latency_.item(), self.tau, self.is_hard, r=self.rank, training=self.training)
            output_samples.append(sample)
        
        output_samples = torch.stack(output_samples)
        output_samples = output_samples.view(output_samples.size(0), self.num_sub_layer, 2, -1)

        prefix_execution_plan = self.get_prefix_execution_plan(output_samples)

        output_samples = torch.cat([prefix_execution_plan, output_samples], 1)
        return output_samples
    
    def get_prefix_execution_plan(self, output_samples):
        new_shape = list(output_samples.shape)
        new_shape[1] = self.num_prefix_layers

        prefix_execution_plan = torch.ones(new_shape, device=output_samples.device, dtype=output_samples.dtype)
        return prefix_execution_plan

    def get_random_latency(self, batch_size):
        latency = torch.randint(self.num_prefix_layers * self.num_attention_heads // self.rank, 
                                self.num_hidden_layers * self.num_attention_heads // self.rank + 1, (batch_size,)) / (self.num_hidden_layers * self.num_attention_heads // self.rank)
        return latency