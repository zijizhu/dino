import re
from functools import partial

import numpy as np
import pytorch_lightning as pl
import torch
from timm.loss.cross_entropy import SoftTargetCrossEntropy
from torch import nn
from torch.optim import SGD
from torchmetrics.classification.accuracy import Accuracy
from transformers.optimization import get_cosine_schedule_with_warmup

from vision_transformer import VisionTransformer
import torchvision.transforms.v2 as v2


def block_expansion_dino(state_dict: dict[str, torch.Tensor], n_splits: int = 3):
    block_keys = set(re.search("^blocks.(\d+).", key).group(0) for key in state_dict if key.startswith("blocks."))
    n_blocks = len(block_keys)
    
    block_indices = np.arange(0, n_blocks).reshape((n_splits, -1,))
    block_indices = np.concatenate([block_indices, block_indices[:, -1:]], axis=-1)
    
    n_splits, n_block_per_split = block_indices.shape
    new_block_indices = list((i + 1) * n_block_per_split - 1 for i in range(n_splits))
    
    expanded_state_dict = dict()
    learable_param_names = []
    
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
            learable_param_names += dst_keys

    expanded_state_dict.update({k: v for k, v in state_dict.items() if "block" not in k})
    
    return expanded_state_dict, len(block_indices.flatten()), learable_param_names


def vit_small(patch_size=16, depth=12, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=depth, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(patch_size=16, depth=12, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=depth, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

MODEL_DICT = {
    "dino_vits16": partial(vit_small, patch_size=16, depth=12),
    "dino_vits8": partial(vit_small, patch_size=8, depth=12),
    "dino_vitb16": partial(vit_base, patch_size=16, depth=12),
    "dino_vitb8": partial(vit_base, patch_size=8, depth=12),
}


class DINOFinetuning(pl.LightningModule):
    def __init__(
        self,
        n_splits: int,
        model_name: str,
        ckpt_path: str,
        n_classes: int = 200,
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
        self.ckpt_path = ckpt_path
        self.training_mode = training_mode
        
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

        # Prepare model depending on fine-tuning mode
        arch = MODEL_DICT[model_name]
        state_dict = torch.load(self.ckpt_path, map_location="cpu")
        if self.training_mode == "linear":
            self.net = arch(num_classes=0)
            self.net.load_state_dict(state_dict)
            for name, param in self.net.named_parameters():
                param.requires_grad = "head" in name

        elif self.training_mode == "block":
            

            expanded_state_dict, n_blocks, learable_param_names = block_expansion_dino(state_dict=state_dict,
                                                                                       n_splits=self.n_splits)
            self.net = arch(num_classes=0, depth=n_blocks)
            self.net.load_state_dict(expanded_state_dict)

            for name, param in self.net.named_parameters():
                param.requires_grad = param in learable_param_names
        else:
            raise ValueError(f"{self.training_mode} is not a valid mode. Use one of ['full', 'linear']")

        self.head = nn.Linear(self.net.embed_dim, self.n_classes)

        # Define loss
        self.loss_fn = SoftTargetCrossEntropy()

        # Define metrics
        self.train_acc = Accuracy(num_classes=self.n_classes, task="multiclass", top_k=1)
        self.val_acc = Accuracy(num_classes=self.n_classes, task="multiclass", top_k=1)
        self.test_acc =Accuracy(num_classes=self.n_classes, task="multiclass", top_k=1)
        
        if self.cutmix_alpha > 0 and self.mixup_alpha > 0:
            cutmix = v2.CutMix(alpha=self.cutmix_alpha, num_classes=self.n_classes)
            mixup = v2.MixUp(alpha=self.mixup_alpha, num_classes=self.n_classes)
            self.cutmix_or_mixup = v2.RandomChoice([cutmix, mixup], p=self.mix_prob)
        else:
            self.cutmix_or_mixup = None


    def forward(self, x: torch.Tensor):
        x = self.net(x)
        x = self.head(x)
        return x

    def shared_step(self, batch: tuple[torch.Tensor, ...], mode="train"):
        if mode == "train" and self.cutmix_or_mixup is not None:
            batch = self.cutmix_or_mixup(*batch)

        x, y = batch
        x = self(x)

        loss = self.loss_fn(x, y)

        labels = y if len(y.shape == 1) else y.argmax(-1)
        acc = getattr(self, f"{mode}_metrics")(x, labels)

        self.log(f"{mode}_loss", loss, on_epoch=True)
        self.log(f"{mode}_acc", acc, on_epoch=True)

        return loss

    def training_step(self, batch, _):
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)
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
