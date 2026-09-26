"""Four CPU ranks exercise actual collectives and SP Gather.backward; no model run."""
import datetime
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from verl.workers.actor.loss_contract import encode_prompt_uids, prompt_balanced_loss_weights, weighted_token_sum
from verl.utils.ulysses import Gather


def distributed_worker(rank, rendezvous):
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=4,
                            timeout=datetime.timedelta(seconds=45))
    group0 = dist.new_group([0, 1])
    group1 = dist.new_group([2, 3])
    sp_group = group0 if rank < 2 else group1
    uids = ["A", "A"] if rank < 2 else ["A", "B"]
    ids = encode_prompt_uids(uids, torch.device("cpu"))
    assert ids.tolist() == ([0, 0] if rank < 2 else [0, 1])
    mask = torch.tensor([[1., 0.], [1., 1.]])
    weights = prompt_balanced_loss_weights(ids, mask)
    # Hand-derived global groups: A has sequence means 1,1,1; B has 9 -> mean 5.
    values = torch.tensor([[1., 1.], [1., 1.] if rank < 2 else [9., 9.]])
    for micro in (1, 2):
        param = torch.tensor(1., requires_grad=True)
        for v, m, w in zip(values.split(micro), mask.split(micro), weights.split(micro)):
            weighted_token_sum(param * v, m, w).backward()
        mean_gradient = param.grad.clone()
        dist.all_reduce(mean_gradient)
        mean_gradient /= 4
        torch.testing.assert_close(mean_gradient, torch.tensor(5.))

    # Emulate the real sequence-parallel logprob gather and FSDP gradient mean.
    # Each rank owns one token position, Gather.backward multiplies by SP=2.
    for micro in (1, 2):
        param = torch.tensor(1., requires_grad=True)
        for v, m, w in zip(values.split(micro), mask.split(micro), weights.split(micro)):
            local_token_values = v[:, rank % 2:rank % 2 + 1] * param
            full_values = Gather.apply(sp_group, local_token_values, 1, True)
            weighted_token_sum(full_values, m, w).backward()
        grad = param.grad.clone()
        dist.all_reduce(grad)
        grad /= 4
        torch.testing.assert_close(grad, torch.tensor(5.))
    # All-empty action masks must still execute matching collectives on all ranks.
    empty_weights = prompt_balanced_loss_weights(ids, torch.zeros_like(mask))
    assert empty_weights.count_nonzero().item() == 0
    dist.destroy_process_group()


class DistributedLossContract(unittest.TestCase):
    def test_dp2_sp2_group_mean_and_micro_partition(self):
        with tempfile.TemporaryDirectory(prefix="actor_loss_cpu_") as td:
            mp.spawn(distributed_worker, args=(td + "/rendezvous",), nprocs=4, join=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
