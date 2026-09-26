"""Resolve real shell-launcher arguments without importing vLLM or starting jobs."""
import os
from pathlib import Path
import subprocess
import unittest

from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]


def config_from_launcher(script, extra_env=None, arguments=(), config_name="ppo_trainer"):
    env = dict(os.environ)
    for key in ("OBJECTIVE_PROFILE", "KL_LOSS_TYPE", "TRAIN_DATA_PATH", "VAL_DATA_PATH",
                "WARMUP_CHECKPOINT_PATH", "RETRIEVER_URL", "VAL_RETRIEVER_URL",
                "SAVE_FREQ", "TEST_FREQ", "LR_WARMUP_RATIO"):
        env.pop(key, None)
    env.update(WANDB_MODE="offline", CUDA_VISIBLE_DEVICES="-1")
    env.update(extra_env or {})
    # Replace only the python invocation; the actual launcher builds its arguments.
    command = 'python3() { printf "%s\\0" "$@"; }; export -f python3; bash "$@"'
    result = subprocess.run(["bash", "-c", command, "capture", str(ROOT / script), *arguments],
                            env=env, check=True, capture_output=True)
    argv = result.stdout.decode().rstrip("\0").split("\0")
    assert argv[:2] in (["-m", "verl.trainer.main_ppo"], ["-m", "verl.trainer.main_generation"])
    with initialize_config_dir(config_dir=str(ROOT / "verl/trainer/config"), version_base=None):
        return compose(config_name=config_name, overrides=argv[2:])


class MainTrainingLauncher(unittest.TestCase):
    def test_corrected_defaults(self):
        cfg = config_from_launcher("scripts/train.sh")
        actor = cfg.actor_rollout_ref.actor
        self.assertEqual(actor.kl_loss_type, "mse")
        self.assertTrue(actor.use_action_loss_mask)
        self.assertTrue(actor.otr_prompt_balanced_loss)
        self.assertEqual(actor.optim.lr, 1e-6)
        self.assertEqual(actor.optim.lr_warmup_steps_ratio, 0)
        self.assertEqual((cfg.trainer.save_freq, cfg.trainer.test_freq), (10, 10))
        self.assertEqual(cfg.trainer.total_training_steps, 215)

    def test_removed_switches_do_not_change_corrected_settings(self):
        cfg = config_from_launcher("scripts/train.sh", {
            "OBJECTIVE_PROFILE": "legacy", "KL_LOSS_TYPE": "kl",
        })
        self.assertEqual(cfg.actor_rollout_ref.actor.kl_loss_type, "mse")
        self.assertTrue(cfg.actor_rollout_ref.actor.use_action_loss_mask)
        self.assertTrue(cfg.actor_rollout_ref.actor.otr_prompt_balanced_loss)

    def test_cli_cannot_disable_corrected_actor_settings(self):
        cfg = config_from_launcher("scripts/train.sh", arguments=[
            "actor_rollout_ref.actor.use_kl_loss=False",
            "actor_rollout_ref.actor.kl_loss_type=kl",
            "actor_rollout_ref.actor.use_action_loss_mask=False",
            "actor_rollout_ref.actor.otr_prompt_balanced_loss=False",
        ])
        self.assertTrue(cfg.actor_rollout_ref.actor.use_kl_loss)
        self.assertEqual(cfg.actor_rollout_ref.actor.kl_loss_type, "mse")
        self.assertTrue(cfg.actor_rollout_ref.actor.use_action_loss_mask)
        self.assertTrue(cfg.actor_rollout_ref.actor.otr_prompt_balanced_loss)

    def test_paths_and_last_cli_override(self):
        cfg = config_from_launcher("scripts/train.sh", {
            "TRAIN_DATA_PATH": "/tmp/constructed.parquet", "VAL_DATA_PATH": "/tmp/dev.parquet",
            "WARMUP_CHECKPOINT_PATH": "/tmp/warmup",
        }, ["actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1"])
        self.assertEqual(cfg.data.train_files, "/tmp/constructed.parquet")
        self.assertEqual(cfg.data.val_files, "/tmp/dev.parquet")
        self.assertEqual(cfg.actor_rollout_ref.model.path, "/tmp/warmup")
        self.assertEqual(cfg.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu, 1)

    def test_warmup_keeps_historical_objective_and_accepts_overrides(self):
        cfg = config_from_launcher("scripts/on_off_policy_warmup.sh", arguments=["trainer.total_training_steps=10"])
        self.assertFalse(cfg.actor_rollout_ref.actor.use_kl_loss)
        self.assertFalse(cfg.actor_rollout_ref.actor.use_action_loss_mask)
        self.assertFalse(cfg.actor_rollout_ref.actor.otr_prompt_balanced_loss)
        self.assertEqual(cfg.trainer.total_training_steps, 10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
