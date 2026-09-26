"""CPU integration tests of the real actor updater with a tiny differentiable model.

Only the CUDA transfer and expensive transformer forward are replaced. Splitting,
DataProto metadata, losses, backward, clipping and optimizer steps are real.
"""
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.workers.actor.dp_actor import DataParallelPPOActor


class ToyPolicy(torch.nn.Module):
    def __init__(self, rows, width):
        super().__init__()
        self.log_probs = torch.nn.Parameter(torch.full((rows, width), -2.0))


def make_actor(rows, width, micro=2, mini=None, kl=False, entropy=0.0, kind="mse"):
    actor = object.__new__(DataParallelPPOActor)
    actor.actor_module = ToyPolicy(rows, width)
    actor.actor_optimizer = torch.optim.SGD(actor.actor_module.parameters(), lr=0.1)
    actor.config = OmegaConf.create(dict(
        use_kl_loss=kl, kl_loss_type=kind, kl_loss_coef=0.001,
        entropy_coeff=entropy, clip_ratio=0.2, grad_clip=1000.0,
        use_dynamic_bsz=False, ppo_mini_batch_size=mini or rows,
        ppo_micro_batch_size_per_gpu=micro, otr_prompt_balanced_loss=True,
    ))
    actor.ulysses_sequence_parallel_size = 1

    def forward(self, micro_batch, temperature):
        ids = micro_batch["responses"][:, 0].long()
        lp = self.actor_module.log_probs[ids]
        # A differentiable entropy surrogate lets us detect unmasked entropy loss.
        return lp.square(), lp

    actor._forward_micro_batch = types.MethodType(forward, actor)
    return actor


def make_batch(uids, actions, advantages, ref=-3.0):
    rows, width = actions.shape
    ids = torch.arange(rows)[:, None].expand(rows, width)
    return DataProto.from_dict(tensors={
        "responses": ids, "input_ids": torch.zeros(rows, width + 1, dtype=torch.long),
        "attention_mask": torch.ones(rows, width + 1),
        "position_ids": torch.zeros(rows, width + 1, dtype=torch.long),
        "old_log_probs": torch.full((rows, width), -2.0),
        "ref_log_prob": torch.full((rows, width), ref),
        "advantages": advantages * actions, "loss_mask": actions,
    }, non_tensors={
        "uid": np.array(uids, dtype=object),
        # Deliberately misleading old metadata: uid is the canonical GRPO group.
        "supporting_facts": np.array([{"original_prompt_id": 0} for _ in uids], dtype=object),
    }, meta_info={"temperature": 0.6})


def update(actor, batch):
    with patch.object(TensorDict, "cuda", lambda self, *a, **k: self):
        metrics = actor.update_policy(batch)
    return actor.actor_module.log_probs.detach(), metrics


class ActorLossContract(unittest.TestCase):
    def test_warmup_can_keep_its_explicit_observation_regularization(self):
        actions = torch.tensor([[1., 0.], [1., 0.]])
        actor = make_actor(2, 2, micro=1, kl=True, kind="kl")
        actor.config.otr_prompt_balanced_loss = False
        actor.config.use_action_loss_mask = False
        params, _ = update(actor, make_batch(["A", "B"], actions, torch.zeros(2, 2)))
        self.assertTrue((params[:, 1] < -2).all())

    def test_balanced_profile_cannot_silently_drop_action_mask(self):
        actor = make_actor(2, 2)
        actor.config.use_action_loss_mask = False
        with self.assertRaisesRegex(ValueError, "action loss_mask"):
            update(actor, make_batch(["A", "B"], torch.ones(2, 2), torch.ones(2, 2)))

    def test_observation_has_zero_kl_and_entropy_gradient(self):
        # Break caught: falling back to attention_mask for KL or entropy.
        actions = torch.tensor([[1., 0., 0., 1.], [1., 0., 0., 1.]])
        actor = make_actor(2, 4, micro=1, kl=True, entropy=0.001)
        params, _ = update(actor, make_batch(["A", "B"], actions, torch.zeros(2, 4)))
        torch.testing.assert_close(params[:, 1:3], torch.full((2, 2), -2.0), rtol=0, atol=0)
        self.assertFalse(torch.equal(params[:, 0], torch.full((2,), -2.0)))

    def test_uid_groups_survive_microbatch_split(self):
        # Break caught: dropping uid or normalizing each micro independently.
        actions = torch.ones(4, 1)
        adv = torch.tensor([[1.], [1.], [1.], [9.]])
        actor = make_actor(4, 1, micro=2)
        params, _ = update(actor, make_batch(["A", "A", "A", "B"], actions, adv))
        expected = torch.tensor([[-2 + .1/6], [-2 + .1/6], [-2 + .1/6], [-2 + .45]])
        torch.testing.assert_close(params, expected)

    def test_action_lengths_and_micro_partition_do_not_change_question_weight(self):
        actions = torch.tensor([[1., 0., 0., 0.], [1., 0., 0., 0.], [1., 1., 1., 1.], [1., 1., 1., 1.]])
        adv = torch.tensor([[1.], [1.], [3.], [3.]]).expand(4, 4)
        outputs = []
        for micro in (1, 2, 4):
            actor = make_actor(4, 4, micro=micro)
            params, _ = update(actor, make_batch(["A", "A", "B", "B"], actions, adv))
            outputs.append(params)
        torch.testing.assert_close(outputs[0], outputs[1])
        torch.testing.assert_close(outputs[0], outputs[2])
        expected = torch.tensor([[-1.975, -2., -2., -2.], [-1.975, -2., -2., -2.], [-1.98125]*4, [-1.98125]*4])
        torch.testing.assert_close(outputs[0], expected)

    def test_reorder_preserves_uid_alignment(self):
        actions = torch.ones(4, 1)
        batch = make_batch(["A", "A", "A", "B"], actions, torch.tensor([[1.], [1.], [1.], [9.]]))
        batch.reorder(torch.tensor([3, 0, 2, 1]))
        params, _ = update(make_actor(4, 1), batch)
        torch.testing.assert_close(params, torch.tensor([[-2+.1/6], [-2+.1/6], [-2+.1/6], [-1.55]]))

    def test_zero_action_sequence_is_finite_and_has_zero_gradient(self):
        actions = torch.tensor([[0., 0.], [1., 1.]])
        params, metrics = update(make_actor(2, 2, micro=1, kl=True), make_batch(["A", "B"], actions, torch.ones(2, 2)))
        self.assertTrue(torch.isfinite(params).all())
        torch.testing.assert_close(params[0], torch.tensor([-2., -2.]))

    def test_k2_reference_change_changes_restoring_gradient(self):
        # Existing supported k2 behavior used by the new launch configuration.
        actions = torch.ones(2, 1)
        low, _ = update(make_actor(2, 1, kl=True), make_batch(["A", "B"], actions, torch.zeros(2, 1), ref=-3.))
        high, _ = update(make_actor(2, 1, kl=True), make_batch(["A", "B"], actions, torch.zeros(2, 1), ref=-1.))
        self.assertTrue((low < -2).all())
        self.assertTrue((high > -2).all())

    def test_two_optimizer_minibatches_keep_distinct_uid_weights(self):
        actions = torch.ones(8, 1)
        adv = torch.tensor([[1.], [1.], [1.], [9.], [5.], [2.], [2.], [2.]])
        actor = make_actor(8, 1, micro=2, mini=4)
        params, _ = update(actor, make_batch(["A", "A", "A", "B", "C", "D", "D", "D"], actions, adv))
        expected = torch.tensor([[-2+.1/6]]*3 + [[-1.55], [-1.75]] + [[-2+.2/6]]*3)
        torch.testing.assert_close(params, expected)

    def test_masked_nonfinite_reference_does_not_poison_kl_gradient(self):
        actions = torch.tensor([[1., 0.], [1., 0.]])
        batch = make_batch(["A", "B"], actions, torch.zeros(2, 2))
        batch.batch["ref_log_prob"][:, 1] = float("inf")
        params, _ = update(make_actor(2, 2, micro=1, kl=True), batch)
        self.assertTrue(torch.isfinite(params).all())
        torch.testing.assert_close(params[:, 1], torch.tensor([-2., -2.]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
