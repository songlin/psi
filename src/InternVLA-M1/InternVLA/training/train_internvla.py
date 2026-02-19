"""
train.py
"""

# Standard Library
import argparse
import json
import os
from pathlib import Path
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Third-Party Libraries
import torch
import torch.distributed as dist
import wandb
import yaml
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# Local Modules
from InternVLA.training.trainer_utils.metrics import normalize_dotlist_args
from InternVLA.model.framework import build_framework
from InternVLA.training.trainer_utils.metrics import TrainerUtils
from InternVLA.training.trainer_utils.metrics import build_param_lr_groups
import time

deepspeed_plugin = DeepSpeedPlugin()
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
accelerator.print(accelerator.state)

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Initialize Overwatch =>> Wraps `logging.Logger`
from accelerate.logging import get_logger

logger = get_logger(__name__)


def load_fast_tokenizer():
    fast_tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)
    return fast_tokenizer


def setup_directories(cfg) -> Path:
    """create output directory with unique timestamp and save config"""
    timestamp = time.strftime("%Y%m%d_%H%M%S") # year, month, day, hour, minute, second
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id, timestamp)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        # create output directory and checkpoint directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

        # save config
        OmegaConf.save(cfg, output_dir / "config.yaml")
        with open(output_dir / "config.yaml", "r") as f_yaml, open(output_dir / "config.json", "w") as f_json:
            yaml_cfg = yaml.safe_load(f_yaml)
            json.dump(yaml_cfg, f_json, indent=2)

    return output_dir


def build_model(cfg) -> torch.nn.Module:
    """build model framework"""
    logger.info(f"Loading Base VLM `{cfg.framework.qwenvl.base_vlm}` from ID/Path")
    model = build_framework(cfg)

    return model


# here changes need to 📦 encapsulate Dataloader
from InternVLA.dataloader import build_dataloader


def prepare_data(cfg, accelerator, output_dir) -> Tuple[DataLoader, DataLoader]:
    """prepare training data"""
    # VLA data loader
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)

    accelerator.dataloader_config.dispatch_batches = False
    dist.barrier()

    return vla_train_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """set optimizer and scheduler"""
    # initialize optimizer
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )

    # print optimizer group info
    if dist.is_initialized() and dist.get_rank() == 0:
        for i, group in enumerate(optimizer.param_groups):
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    # initialize learning rate scheduler
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs,  # minimum learning rate
    )

    return optimizer, lr_scheduler


class VLATrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        # training status tracking
        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        # load pretrained weights
        if hasattr(self.config.trainer, "pretrained_checkpoint") and self.config.trainer.pretrained_checkpoint:
            pretrained_checkpoint = self.config.trainer.pretrained_checkpoint
            reload_modules = (
                self.config.trainer.reload_modules if hasattr(self.config.trainer, "reload_modules") else None
            )
            skip_reload_modules = (
                self.config.trainer.skip_reload_modules if hasattr(self.config.trainer, "skip_reload_modules") else None
            )
            self.model = self.load_pretrained_backbones(
                self.model, pretrained_checkpoint, 
                reload_modules=reload_modules, 
                skip_reload_modules=skip_reload_modules
            )

        # freeze parameters
        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)

        #  print model trainable parameters:
        self.print_trainable_parameters(self.model)

        # initialize distributed training components
        self.model, self.optimizer, self.vla_train_dataloader = self.setup_distributed_training(
            self.accelerator,  # must be the first param
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
            # self.vlm_train_dataloader
        )

        self._init_wandb()
        self._init_checkpointing()

    def _calculate_total_batch_size(self):
        """calculate global batch size"""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        """initialize Weights & Biases"""
        if self.accelerator.is_main_process:
            wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group="vla-train",
            )

    def _init_checkpointing(self):
        """initialize checkpoint directory"""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)

        # resume training state
        if pretrained_checkpoint and is_resume:
            self._load_checkpoint(self.config.resume_from_checkpoint)

    def _load_checkpoint(self, checkpoint_path):
        """load checkpoint"""
        self.accelerator.load_state(checkpoint_path)
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

    def _save_checkpoint(self):
        """save current training state"""

        if accelerator.is_main_process:

            checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")
            # save model state
            state_dict = self.accelerator.get_state_dict(self.model)
            torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")

            # save training metadata
            summary_data = {
                "steps": self.completed_steps,
            }
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")
        accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        """record training metrics"""
        if self.completed_steps % self.config.trainer.logging_frequency == 0:
            if dist.get_rank() == 0:
                # add learning rate
                metrics["learning_rate"] = self.lr_scheduler.get_last_lr()[0]

                # add epoch info
                metrics["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 2)

                # record to W&B
                wandb.log(metrics, step=self.completed_steps)
                # debug output
                logger.info(f"Step {self.completed_steps}, Loss: {metrics})")

    def _create_data_iterators(self):
        """create data iterators"""
        self.vla_iter = iter(self.vla_train_dataloader)
        # self.vlm_iter = iter(self.vlm_train_dataloader)

    def _get_next_batch(self):
        """get next batch (automatically handle data loop)"""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)

        return batch_vla

    def train(self):
        """execute training loop"""
        # print training config
        self._log_training_config()

        # prepare data iterators
        self._create_data_iterators()

        # create progress bar
        progress_bar = tqdm(
            range(self.config.trainer.max_train_steps), disable=not self.accelerator.is_local_main_process
        )

        # main training loop
        while self.completed_steps < self.config.trainer.max_train_steps:
            # get data batch
            batch_vla = self._get_next_batch()

            # execute training step
            step_metrics = self._train_step(batch_vla)

            # update progress
            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1

            # evaluate model
            if self.completed_steps % self.config.trainer.eval_interval == 0:
                step_metrics = self.eval_action_model(step_metrics)

            # record metrics
            self._log_metrics(step_metrics)

            # save checkpoint
            if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                self._save_checkpoint()

            # check termination condition
            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        # training end processing
        self._finalize_training()

        # execute evaluation step

    def eval_action_model(self, step_metrics: dict = None) -> float:
        """
        Evaluate the model on the given dataset using the specified metric function.

        :param eval_dataset: List of evaluation samples, each containing 'image', 'instruction', and 'action'.
        :param metric_fn: Function to compute the distance between predicted and ground truth actions.
        :return: Average metric score across the evaluation dataset.
        """

        if self.accelerator.is_main_process:

            examples = self._get_next_batch()

            score = 0.0
            num_samples = len(examples)

            batch_images = [example["image"] for example in examples]
            instructions = [example["lang"] for example in examples]  # [B, str]
            actions = [example["action"] for example in examples]  # label

            # Predict actions using the model
            output_dict = self.model.predict_action(
                batch_images=batch_images, instructions=instructions, use_ddim=True, num_ddim_steps=20
            )

            normalized_actions = output_dict["normalized_actions"]  # B, T, D

            actions = np.array(actions)  # convert actions to numpy.ndarray

            # denormalize actions:
            # FIXME
            def denormalize(actions_:np.ndarray):
                action_norm_stats = {
                        "mean": [
                    -0.6072647571563721,
                    0.7457647323608398,
                    0.8788257837295532,
                    -0.20688210427761078,
                    -0.3595120310783386,
                    -0.23220641911029816,
                    -0.36432409286499023,
                    -0.524077296257019,
                    -0.7612780928611755,
                    -0.7937824726104736,
                    0.2940281331539154,
                    0.4177097678184509,
                    0.3164995014667511,
                    0.40769699215888977,
                    -0.27801796793937683,
                    0.1345491111278534,
                    0.02578454092144966,
                    0.2565060257911682,
                    -0.024487663060426712,
                    -0.3724501132965088,
                    -0.164189875125885,
                    -0.31349989771842957,
                    -0.20494051277637482,
                    -0.08927679806947708,
                    0.08941172808408737,
                    0.06368149816989899,
                    -0.2469901591539383,
                    0.1160014271736145,
                    -0.01555413007736206,
                    -0.039668165147304535,
                    0.06010082736611366,
                    0.7120963931083679,
                    0.036779407411813736,
                    -0.001909215934574604,
                    -0.027210412546992302,
                    -0.6833463907241821
                ],
                "std": [
                    0.23424562811851501,
                    0.15335652232170105,
                    0.21423333883285522,
                    0.28928956389427185,
                    0.3334279954433441,
                    0.2579157054424286,
                    0.25160476565361023,
                    0.17685383558273315,
                    0.1545531004667282,
                    0.18313363194465637,
                    0.23420511186122894,
                    0.25718382000923157,
                    0.20897071063518524,
                    0.19025404751300812,
                    0.479464054107666,
                    0.2640666961669922,
                    0.2750664949417114,
                    0.622584879398346,
                    0.31611987948417664,
                    0.4374791085720062,
                    0.3139559030532837,
                    0.40725094079971313,
                    0.3220720589160919,
                    0.33236566185951233,
                    0.5598737001419067,
                    0.27561673521995544,
                    0.40926530957221985,
                    0.3357323408126831,
                    0.050867754966020584,
                    0.12385918200016022,
                    0.13145403563976288,
                    0.07437460124492645,
                    0.10808036476373672,
                    0.06163855269551277,
                    0.11503823101520538,
                    0.8225098848342896
                ],
                "min": [
                    -1.0481975078582764,
                    0.06307568401098251,
                    0.46587154269218445,
                    -1.4067349433898926,
                    -1.7463293075561523,
                    -1.38773775100708,
                    -1.6817723512649536,
                    -1.0481975078582764,
                    -0.9210000038146973,
                    -1.7463293075561523,
                    -0.0010000000474974513,
                    -0.0010000000474974513,
                    -0.0010000000474974513,
                    -0.0010000000474974513,
                    -1.8166981935501099,
                    -0.7797250747680664,
                    -0.9026529788970947,
                    -1.0471996068954468,
                    -1.2519687414169312,
                    -1.6144294738769531,
                    -1.3473931550979614,
                    -1.9426480531692505,
                    -1.1327459812164307,
                    -1.153347373008728,
                    -1.0471996068954468,
                    -1.3340427875518799,
                    -1.614429235458374,
                    -0.5743077993392944,
                    -0.31130483746528625,
                    -0.4641092121601105,
                    -0.3989304006099701,
                    0.34272482991218567,
                    -0.3499999940395355,
                    -0.5,
                    -0.5,
                    -2.0399999618530273
                ],
                "max": [
                    0.3988589644432068,
                    0.9210000038146973,
                    1.6892091035842896,
                    0.0010000000474974513,
                    0.0010000000474974513,
                    0.0010000000474974513,
                    0.0010000000474974513,
                    0.2635820806026459,
                    0.006984861101955175,
                    -0.43955960869789124,
                    1.5375014543533325,
                    1.7463293075561523,
                    1.4949489831924438,
                    1.7463293075561523,
                    0.9220698475837708,
                    1.124887228012085,
                    0.9969016909599304,
                    1.3985852003097534,
                    1.4562329053878784,
                    1.1667526960372925,
                    1.5457602739334106,
                    0.8206027746200562,
                    0.44797906279563904,
                    0.6963806748390198,
                    1.382392406463623,
                    1.5501763820648193,
                    0.935970664024353,
                    1.4882785081863403,
                    0.197315514087677,
                    0.41176825761795044,
                    0.8875675201416016,
                    0.7815188765525818,
                    0.3499999940395355,
                    0.5,
                    0.5,
                    0.0
                ],
                "q01": [
                    -1.0122302639484406,
                    0.3247970303893089,
                    0.5390077209472657,
                    -0.9648897647857666,
                    -1.6577193474769591,
                    -0.9134321182966232,
                    -1.445114164352417,
                    -1.0160934162139892,
                    -0.9210000038146973,
                    -1.1945471847057343,
                    -0.0010000000474974513,
                    -0.0010000000474974513,
                    -0.0010000000474974513,
                    0.05139310285449028,
                    -1.3317369318008423,
                    -0.2566007047891617,
                    -0.5315321964025498,
                    -0.9475072801113129,
                    -0.7175824987888336,
                    -1.4082000708580018,
                    -0.9849310135841369,
                    -1.220305984020233,
                    -0.877159241437912,
                    -0.8107936817407608,
                    -0.9586893695592881,
                    -0.8070452159643173,
                    -1.3410334050655366,
                    -0.3712126424908638,
                    -0.12484664030373097,
                    -0.3410728871822357,
                    -0.26549966126680374,
                    0.40623320430517196,
                    0.0,
                    -0.3499999940395355,
                    -0.5,
                    -1.9179999828338623
                ],
                "q99": [
                    0.06078679017722612,
                    0.9210000038146973,
                    1.2450403034687043,
                    0.0010000000474974513,
                    0.0010000000474974513,
                    0.0010000000474974513,
                    0.0009497498976998031,
                    -0.08942579187452793,
                    -0.37620131015777586,
                    -0.5473435187339782,
                    0.9840427029132847,
                    1.5757278883457189,
                    0.9208912372589112,
                    1.3351197028160098,
                    0.6973721981048584,
                    0.8147528767585756,
                    0.6551885795593262,
                    1.2582692217826843,
                    0.8366875994205484,
                    0.5123758268356324,
                    0.38507036834955216,
                    0.6165469551086428,
                    0.26588116645812987,
                    0.5687608057260515,
                    1.20501633644104,
                    0.6575165885686874,
                    0.5493933695554742,
                    0.9626659679412845,
                    0.11506167359650135,
                    0.26826522886753085,
                    0.5420023810863495,
                    0.7638321095705033,
                    0.3499999940395355,
                    0.0,
                    0.0,
                    0.0
                ]
                }
                action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
                mask = action_high != action_low


                actions_ = np.clip(actions_, -1, 1)
                denormalized_ = np.where(
                    mask,
                    0.5 * (actions_ + 1) * (action_high - action_low) + action_low,
                    actions_,
                )

                return denormalized_

            normalized_actions = denormalize(normalized_actions)
            actions = denormalize(actions)
            # B, Chunk, dim = actions.shape
            # num_pots = np.prod(actions.shape)
            # # Compute the metric score
            # score = TrainerUtils.euclidean_distance(normalized_actions, actions)
            # step_metrics["l2_norm_normalized"] = score / num_pots  # 重命名以反映真实含义
            
            # Define dimension splits: hand_joints(14) + arm_joints(14) + rpy(3) + height(1) = 32
            hand_joints_start, hand_joints_end = 0, 14
            arm_joints_start, arm_joints_end = 14, 28
            rpy_start, rpy_end = 28, 31
            height_start, height_end = 31, 32
            torso_vx_start, torso_vx_end = 32, 33
            torso_vy_start, torso_vy_end = 33, 34
            torso_vyaw_start, torso_vyaw_end = 34, 35
            torso_dyaw_start, torso_dyaw_end = 35, 36
            
            # Compute L1 loss for each component
            # hand_joints
            hand_l1 = np.abs(normalized_actions[..., hand_joints_start:hand_joints_end] - 
                           actions[..., hand_joints_start:hand_joints_end]).mean()
            step_metrics["l1_hand_joints"] = hand_l1
            
            # arm_joints
            arm_l1 = np.abs(normalized_actions[..., arm_joints_start:arm_joints_end] - 
                          actions[..., arm_joints_start:arm_joints_end]).mean()
            step_metrics["l1_arm_joints"] = arm_l1
            
            # rpy (roll, pitch, yaw)
            rpy_l1 = np.abs(normalized_actions[..., rpy_start:rpy_end] - 
                          actions[..., rpy_start:rpy_end]).mean()
            step_metrics["l1_rpy"] = rpy_l1
            
            # height
            height_l1 = np.abs(normalized_actions[..., height_start:height_end] - 
                             actions[..., height_start:height_end]).mean()
            step_metrics["l1_height"] = height_l1
            
            # torso_vx
            torso_vx_l1 = np.abs(normalized_actions[..., torso_vx_start:torso_vx_end] - 
                             actions[..., torso_vx_start:torso_vx_end]).mean()
            step_metrics["l1_torso_vx"] = torso_vx_l1
            
            # torso_vy
            torso_vy_l1 = np.abs(normalized_actions[..., torso_vy_start:torso_vy_end] - 
                             actions[..., torso_vy_start:torso_vy_end]).mean()
            step_metrics["l1_torso_vy"] = torso_vy_l1
            
            # torso_vyaw
            torso_vyaw_l1 = np.abs(normalized_actions[..., torso_vyaw_start:torso_vyaw_end] - 
                             actions[..., torso_vyaw_start:torso_vyaw_end]).mean()
            step_metrics["l1_torso_vyaw"] = torso_vyaw_l1
            
            # torso_dyaw
            torso_dyaw_l1 = np.abs(normalized_actions[..., torso_dyaw_start:torso_dyaw_end] - 
                             actions[..., torso_dyaw_start:torso_dyaw_end]).mean()
            step_metrics["l1_target_yaw"] = torso_dyaw_l1
            
            # Overall L1 loss (average of all dimensions)
            overall_l1 = np.abs(normalized_actions - actions).mean()
            step_metrics["l1_overall"] = overall_l1

            # MSE (Mean Squared Error)
            mse = np.mean((normalized_actions - actions) ** 2)
            # step_metrics["mse"] = mse
            
            # RMSE (Root Mean Squared Error) - this is the true L2 distance metric
            rmse = np.sqrt(mse)
            step_metrics["rmse"] = rmse
        pass
        dist.barrier()  # ensure all processes are synchronized
        return step_metrics

    def _log_training_config(self):
        """record training config"""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.config.trainer.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")

    def _train_step(self, batch_vla, batch_vlm=None):
        """execute single training step"""
        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()

            # VLA task forward propagation
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(batch_vla)

                action_loss = output_dict["action_loss"]
                total_loss = action_loss

            # VLA backward propagation
            self.accelerator.backward(total_loss)

            # gradient clipping
            if self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

            # optimizer step
            self.optimizer.step()
            self.lr_scheduler.step()

        return {
            "action_dit_loss": action_loss.item(),
        }

    def _finalize_training(self):
        """training end processing"""
        # save final model
        if self.accelerator.is_main_process:
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            state_dict = self.accelerator.get_state_dict(self.model)
            torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")

        # close W&B
        if self.accelerator.is_main_process:
            wandb.finish()

        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    logger.info("VLA Training :: Warming Up")

    # create output directory and save config
    output_dir = setup_directories(cfg=cfg)
    # build model
    vla = build_framework(cfg)
    # prepare data
    vla_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)

    # set optimizer and scheduler
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    # create trainer
    # Run VLA Training
    trainer = VLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    # execute training preparation
    trainer.prepare_training()
    # execute training
    trainer.train()

    # And... we're done!
    logger.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="InternVLA/config/training/internvla_cotrain_custom.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    # Load YAML config & Convert CLI overrides to dotlist config
    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)  # Normalize CLI args to dotlist format
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # if cfg.is_debug:
    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)
