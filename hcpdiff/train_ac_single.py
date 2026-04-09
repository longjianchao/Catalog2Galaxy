import argparse
import os
import sys
from functools import partial

import torch
from accelerate import Accelerator
from loguru import logger

from hcpdiff.train_ac import Trainer, RatioBucket, load_config_with_cli, set_seed, get_sampler

# ⚠️ 仅在环境变量 DEBUG_ANOMALY=1 时开启异常检测，正式训练不要开（慢30-50%）
if os.environ.get('DEBUG_ANOMALY'):
    torch.autograd.set_detect_anomaly(True)


class TrainerSingleCard(Trainer):
    def init_context(self, cfgs_raw):
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.cfgs.train.gradient_accumulation_steps,
            mixed_precision=self.cfgs.mixed_precision,
            step_scheduler_with_optimizer=False,
        )

        self.local_rank = 0
        self.world_size = self.accelerator.num_processes

        set_seed(self.cfgs.seed + self.local_rank)

    @property
    def unet_raw(self):
        # 统一处理：无论单卡还是 DDP 包装，先剥掉 .module 再取 .unet
        base = self.TE_unet.module if hasattr(self.TE_unet, 'module') else self.TE_unet
        return base.unet

    @property
    def TE_raw(self):
        # 统一处理：无论单卡还是 DDP 包装，先剥掉 .module 再取 .TE
        base = self.TE_unet.module if hasattr(self.TE_unet, 'module') else self.TE_unet
        return base.TE

    def get_loss(self, model_pred, target, timesteps, *args, **kwargs):
        # 1. 先让 hcpdiff 算完标准扩散去噪 Loss
        loss_dict = super().get_loss(model_pred, target, timesteps, *args, **kwargs)

        # 2. 拦截 CatalogTextEncoder 挂载的物理重建 Loss
        TE = self.TE_raw
        recon_loss = getattr(TE, 'recon_loss', None)

        # 取完立即清零，防止 forward 异常跳过时残留旧值污染下一步
        TE.recon_loss = None

        self.current_loss_recon = None
        if isinstance(recon_loss, torch.Tensor):
            # lambda_recon 建议从 0.05 开始：
            # diffusion loss 通常在 0.1~0.2 量级，0.05 权重下 recon 贡献约 10~20%
            # 若进度条里 loss_recon 远大于 loss，适当调小；反之可适当调大
            lambda_recon = 0.05
            self.current_loss_recon = recon_loss.item() * lambda_recon
            if isinstance(loss_dict, dict):
                # hcpdiff 会自动 sum() 所有键的 loss 并反传，
                # 'loss_recon' 也会出现在终端进度条里，方便监控
                loss_dict['loss_recon'] = recon_loss * lambda_recon
            else:
                # 极端兜底：框架版本不同返回单一 Tensor 时直接相加
                loss_dict = loss_dict + recon_loss * lambda_recon

        return loss_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stable Diffusion Training')
    parser.add_argument('--cfg', type=str, default='cfg/train/demo.yaml')
    args, cfg_args = parser.parse_known_args()

    conf = load_config_with_cli(args.cfg, args_list=cfg_args)  # skip --cfg
    trainer = TrainerSingleCard(conf)
    trainer.train()