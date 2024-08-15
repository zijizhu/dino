import math
import re
from functools import partial

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as v2
from lightning.pytorch.cli import LightningCLI
from timm.loss.cross_entropy import SoftTargetCrossEntropy
from torch import nn
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.classification.accuracy import Accuracy

from data import DataModule
from vision_transformer import VisionTransformer


def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def block_expansion_dino(state_dict: dict[str, torch.Tensor], n_splits: int = 3):
    """Perform Block Expansion on a ViT described in https://arxiv.org/abs/2404.17245"""
    block_keys = set(re.search("^blocks.(\d+).", key).group(0) for key in state_dict if key.startswith("blocks."))
    n_blocks = len(block_keys)
    
    block_indices = np.arange(0, n_blocks).reshape((n_splits, -1,))
    block_indices = np.concatenate([block_indices, block_indices[:, -1:]], axis=-1)
    
    n_splits, n_block_per_split = block_indices.shape
    new_block_indices = list((i + 1) * n_block_per_split - 1 for i in range(n_splits))
    
    expanded_state_dict = dict()
    learnable_param_names = []
    
    for dst_idx, src_idx in enumerate(block_indices.flatten()):
        src_keys = [k for k in state_dict if f"blocks.{src_idx}" in k]
        dst_keys = [k.replace(f"blocks.{src_idx}", f"blocks.{dst_idx}") for k in src_keys]
        
        block_state_dict = dict()
        
        for src_k, dst_k in zip(src_keys, dst_keys):
            if ("mlp.fc2" in dst_k or "attn.proj" in dst_k) and (dst_idx in new_block_indices):
                block_state_dict[dst_k] = torch.zeros_like(state_dict[src_k])
            else:
                block_state_dict[dst_k] = state_dict[src_k]

        expanded_state_dict.update(block_state_dict)

        if dst_idx in new_block_indices:
            learnable_param_names += dst_keys

    expanded_state_dict.update({k: v for k, v in state_dict.items() if "block" not in k})
    
    return expanded_state_dict, len(block_indices.flatten()), learnable_param_names


vit_small_config = dict(embed_dim=384, num_heads=6)
vit_base_config = dict(embed_dim=768, num_heads=12)
shared_config = dict(depth=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

MODEL_DICT = {
    "dino_vits16": partial(VisionTransformer, patch_size=16, **vit_small_config, **shared_config),
    "dino_vits8": partial(VisionTransformer, patch_size=8, **vit_small_config, **shared_config),
    "dino_vitb16": partial(VisionTransformer, patch_size=16, **vit_base_config, **shared_config),
    "dino_vitb8": partial(VisionTransformer, patch_size=8, **vit_base_config, **shared_config),
}


class DINOFinetuning(L.LightningModule):
    def __init__(
        self,
        n_splits: int,
        model_name: str,
        pretrained_ckpt_path: str,
        *,
        n_classes: int = 200,
        loss: str = "soft_xe",
        training_mode: str = "block",
        
        mixup_alpha: float = 1.0,
        cutmix_alpha: float = 1.0,
        mix_prob: float = 1.0,
        label_smoothing: float = 0.0,
        
        optimizer: str = "sgd",
        lr: float = 1e-2,
        betas: tuple[float, float] = (0.9, 0.999),
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        scheduler: str = "cosine",
        warmup_steps: int = 0,
        
        image_size: int = 224
    ):
        """Classification Model

        Args:
            model_name: Name of model checkpoint. List found in src/model.py
            optimizer: Name of optimizer. One of [adam, adamw, sgd]
            lr: Learning rate
            betas: Adam betas parameters
            momentum: SGD momentum parameter
            weight_decay: Optimizer weight decay
            scheduler: Name of learning rate scheduler. One of [cosine, none]
            warmup_steps: Number of warmup steps
            n_classes: Number of target class
            mix_prob: Probability of applying mixup or cutmix (applies when mixup_alpha and/or
                cutmix_alpha are >0)
            label_smoothing: Amount of label smoothing
            image_size: Size of input images
            training_mode: Fine-tuning mode. One of ["full", "linear", "lora"]
        """
        super().__init__()
        self.save_hyperparameters()
        self.n_splits = n_splits
        self.model_name = model_name
        self.pretrained_ckpt_path = pretrained_ckpt_path
        self.training_mode = training_mode
        self.loss = loss
        
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mix_prob = mix_prob
        self.label_smoothing = label_smoothing
    
        self.optimizer = optimizer
        self.lr = lr
        self.betas = betas
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.warmup_steps = warmup_steps
        self.n_classes = n_classes

        self.image_size = image_size

        arch = MODEL_DICT[model_name]
        state_dict = torch.load(self.pretrained_ckpt_path, map_location="cpu")
        if self.training_mode == "linear":
            self.net = arch(num_classes=0)
            self.net.load_state_dict(state_dict)
            for name, param in self.net.named_parameters():
                param.requires_grad = False
        elif self.training_mode == "block":
            expanded_state_dict, n_blocks, learnable_param_names = block_expansion_dino(
                state_dict=state_dict,
                n_splits=self.n_splits)
            self.net = arch(num_classes=0, depth=n_blocks)
            self.net.load_state_dict(expanded_state_dict)

            for name, param in self.net.named_parameters():
                param.requires_grad = name in learnable_param_names
        else:
            raise ValueError(f"{self.training_mode} is not a valid mode. Use one of ['full', 'linear']")
        self.head = nn.Linear(self.net.embed_dim, self.n_classes)

        if self.loss == "soft_xe":
            self.loss_fn = SoftTargetCrossEntropy()
        elif self.loss == "xe":
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            raise ValueError

        self.train_acc = Accuracy(num_classes=self.n_classes, task="multiclass", average="micro", top_k=1)
        self.val_acc = Accuracy(num_classes=self.n_classes, task="multiclass", average="micro", top_k=1)
        self.test_acc = Accuracy(num_classes=self.n_classes, task="multiclass", average="micro", top_k=1)
        
        if self.cutmix_alpha > 0 and self.mixup_alpha > 0:
            cutmix = v2.CutMix(alpha=self.cutmix_alpha, num_classes=self.n_classes)
            mixup = v2.MixUp(alpha=self.mixup_alpha, num_classes=self.n_classes)
            self.cutmix_or_mixup = v2.RandomChoice([cutmix, mixup], p=[self.mix_prob, 1-self.mix_prob])
        else:
            self.cutmix_or_mixup = None
        
    def _check(self):
        self.print("Learnable parameters:")
        for name, param in self.net.named_parameters():
            if param.requires_grad:
                self.print(name)
    
    def on_train_start(self) -> None:
        self._check()

    def forward(self, x: torch.Tensor):
        x = self.net(x)
        x = self.head(x)
        return x

    def shared_step(self, batch: tuple[torch.Tensor, ...], mode: str = "train"):
        x, labels = batch
        x = self(x)

        y = F.one_hot(labels, num_classes=self.n_classes) if (self.loss == "soft_xe" and labels.ndim == 1) else labels
        loss = self.loss_fn(x, y)
        acc = getattr(self, f"{mode}_acc")(x, labels)

        self.log(f"{mode}_loss", loss, on_epoch=True)
        self.log(f"{mode}_acc", acc, on_epoch=True)

        return loss

    def training_step(self, batch, _):
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)
        
        if self.cutmix_or_mixup is not None:
            batch = self.cutmix_or_mixup(*batch)

        return self.shared_step(batch, "train")

    def validation_step(self, batch, _):
        return self.shared_step(batch, "val")

    def test_step(self, batch, _):
        return self.shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = SGD(
            self.net.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_training_steps=int(self.trainer.estimated_stepping_batches),
            num_warmup_steps=self.warmup_steps)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
        

def cli_main():
    LightningCLI(DINOFinetuning, DataModule)


if __name__ == "__main__":
    cli_main()
