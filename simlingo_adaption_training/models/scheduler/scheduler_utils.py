import torch

        
def masked_gumbel_softmax(logits, masks, tau = 1, hard = False, eps = 1e-10, dim = -1, training = True):
    if training:
        gumbels = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
            .exponential_()
            .log()
        )  # ~Gumbel(0,1)
        logits = (logits + gumbels) / tau  # ~Gumbel(logits,tau)

    logits = logits.masked_fill(masks.bool(), float('-inf')) # mask out the already picked items
    y_soft = logits.softmax(dim)

    y_hard = y_soft.masked_fill(masks.bool(), -1.0) # mask out the already picked items
    index = y_hard.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(
        logits, memory_format=torch.legacy_contiguous_format
    ).scatter_(dim, index, 1.0)

    if hard:
        if training:
            return y_hard - y_soft.detach() + y_soft
        else:
            return y_hard
    else:
        return y_soft

def n_times_gumbel_softmax(logits, n = 1, tau = 1, hard = False, eps = 1e-10, dim = -1, training = True):
    cumulate_mask = torch.zeros_like(logits, device=logits.device)
    for i in range(int(n)):
        mask = masked_gumbel_softmax(logits, cumulate_mask, tau, hard=True, dim=dim, training=training)
        cumulate_mask = cumulate_mask + mask
    return cumulate_mask
    
def masked_gumbel_softmax_topk(logits, masks, k = 8, tau = 1, hard = False, eps = 1e-10, dim = -1, training = True):
    if training:
        gumbels = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
            .exponential_()
            .log()
        )  # ~Gumbel(0,1)
        logits = (logits + gumbels) / tau  # ~Gumbel(logits,tau)

    logits = logits.masked_fill(masks.bool(), float('-inf')) # mask out the already picked items
    y_soft = logits.softmax(dim)

    y_hard = y_soft.masked_fill(masks.bool(), -1.0) # mask out the already picked items
    index = y_hard.topk(dim=dim, k=k, largest=True)[1]
    y_hard = torch.zeros_like(
        logits, memory_format=torch.legacy_contiguous_format
    ).scatter_(dim, index, 1.0)

    if hard:
        if training:
            return y_hard - y_soft.detach() + y_soft
        else:
            return y_hard
    else:
        return y_soft


def n_times_gumbel_softmax_head_version(logits, n = 1, tau = 1, hard = False, eps = 1e-10, dim = -1, r = 8, training = True):
    cumulate_mask = torch.zeros_like(logits, device=logits.device)
    while n > 0:
        layer_mask = masked_gumbel_softmax(logits.view(-1), cumulate_mask.view(-1), tau, hard=True, dim=dim, training=training)
        layer_mask = layer_mask.view(logits.size()).sum(1)
        mask = (cumulate_mask * layer_mask.unsqueeze(1)).sum(0).view(2, -1)
        sub_logits = (logits * layer_mask.unsqueeze(1)).sum(0).view(2, -1)
        mask = [masked_gumbel_softmax_topk(sub_logits[_], masks=mask[_], k=r, tau=tau, hard=True, dim=dim, training=training) for _ in range(2)]
        mask = torch.stack(mask, 0).view(-1)
        cumulate_mask = cumulate_mask + layer_mask.unsqueeze(1) * mask.unsqueeze(0)
        n -= 1
    return cumulate_mask

def posemb_sincos_1d(latency_num, dim=256, temperature=10000, dtype=torch.float32):
    n = latency_num

    n = torch.arange(n)
    assert (dim % 2) == 0, 'feature dimension must be multiple of 2 for sincos emb'
    omega = torch.arange(dim // 2) / (dim // 2 - 1)
    omega = 1. / (temperature**omega)

    n = n.flatten()[:, None] * omega[None, :]
    pe = torch.cat((n.sin(), n.cos()), dim=1)
    return pe
    
def latency_quantizing(latency, num_prefix_units=16, num_total_units=32):
    if not torch.all((latency >= 0) & (latency <= 1)):
        raise ValueError("Latency must be between 0 and 1.")
    units = torch.floor(num_total_units * latency) - num_prefix_units
    units = torch.relu(units)
    return units, units / (num_total_units - num_prefix_units + 1)