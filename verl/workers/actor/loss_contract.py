"""Action-token masking and prompt means preserved across DP/SP and microbatches."""
from collections import Counter
from typing import Sequence

import torch
import torch.distributed as dist


def _gather_objects(value: object) -> list:
    if not dist.is_available() or not dist.is_initialized():
        return [value]
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, value)
    return gathered


def encode_prompt_uids(uids: Sequence[str], device: torch.device) -> torch.Tensor:
    """Use the same uid -> integer mapping on every rank, never positional metadata."""
    local_uids = [str(uid) for uid in uids]
    all_uids = _gather_objects(local_uids)
    mapping = {uid: i for i, uid in enumerate(sorted({u for rank in all_uids for u in rank}))}
    return torch.tensor([mapping[uid] for uid in local_uids], dtype=torch.long, device=device)


def prompt_balanced_loss_weights(prompt_ids: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    """Weights for one global optimizer minibatch: token -> sequence -> prompt mean.

    Gather counts before micro-splitting. SP-replicated rows appear repeatedly in
    the counts; scaling by world size compensates for FSDP's rank-mean reduction.
    An empty-action sequence contributes neither a mean nor a group count.
    The resulting microbatch contributions must be SUMMED, not averaged again.
    """
    if prompt_ids.ndim != 1 or len(prompt_ids) != len(loss_mask):
        raise ValueError("prompt IDs must align with loss-mask rows")
    token_counts = loss_mask.float().sum(-1)
    valid = token_counts > 0
    local_counts = Counter(prompt_ids[valid].detach().cpu().tolist())
    per_rank_counts = _gather_objects(dict(local_counts))
    global_counts = Counter()
    for counts in per_rank_counts:
        global_counts.update(counts)
    if not global_counts:
        return torch.zeros_like(loss_mask, dtype=torch.float32)
    rank_scale = len(per_rank_counts)
    num_prompts = len(global_counts)
    sequence_weights = torch.tensor(
        [rank_scale / (num_prompts * global_counts[pid]) if global_counts[pid] else 0.0
         for pid in prompt_ids.detach().cpu().tolist()],
        device=loss_mask.device, dtype=torch.float32,
    )
    return loss_mask.float() * (sequence_weights / token_counts.clamp_min(1.0)).unsqueeze(-1)


def weighted_token_sum(values: torch.Tensor, loss_mask: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """An additive contribution to the already-normalized optimizer minibatch."""
    if values.shape != loss_mask.shape or values.shape != weights.shape:
        raise ValueError("values, action mask and weights must have matching shapes")
    return (torch.where(loss_mask.bool(), values, torch.zeros_like(values)) * weights).sum()
