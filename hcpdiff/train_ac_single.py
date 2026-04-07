import argparse
import sys
from functools import partial

import torch
from accelerate import Accelerator
from loguru import logger

from hcpdiff.train_ac import Trainer, RatioBucket, load_config_with_cli, set_seed, get_sampler

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
        # 如果 self.TE_unet 有 module 属性，说明是多卡 DDP 包装过的，取 .module.unet
        if hasattr(self.TE_unet, 'module'):
            return self.TE_unet.module.unet if self.train_TE else self.TE_unet.unet.module
        # 否则说明是单卡，维持原样
        return self.TE_unet.unet

    @property
    def TE_raw(self):
        if hasattr(self.TE_unet, 'module'):
            return self.TE_unet.module.TE if self.train_TE else self.TE_unet.TE
        return self.TE_unet.TE

    # 🚀 核心修复：拦截 get_loss，完美融入 hcpdiff 生态！
    def get_loss(self, model_pred, target, timesteps, *args, **kwargs):
        # 1. 先让 hcpdiff 算完标准的扩散去噪 Loss (默认返回一个字典)
        loss_dict = super().get_loss(model_pred, target, timesteps, *args, **kwargs)
        
        # 2. 提取我们在 CatalogTextEncoder 里挂载好的物理重建误差
        TE = self.TE_raw
        if hasattr(TE, 'recon_loss') and isinstance(TE.recon_loss, torch.Tensor):
            lambda_recon = 0.5  # 物理重建的正则化权重
            
            if isinstance(loss_dict, dict):
                # 极其优雅：直接将 loss_recon 塞进字典。
                # hcpdiff 会自动执行 sum() 反向传播，并把 'loss_recon' 打印到你的终端进度条里！
                loss_dict['loss_recon'] = TE.recon_loss * lambda_recon
            else:
                # 极端防爆措施：如果框架版本不同返回了单一张量，直接相加
                loss_dict = loss_dict + TE.recon_loss * lambda_recon
                
        return loss_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stable Diffusion Training')
    parser.add_argument('--cfg', type=str, default='cfg/train/demo.yaml')
    args, cfg_args = parser.parse_known_args()

    conf = load_config_with_cli(args.cfg, args_list=cfg_args)  # skip --cfg
    trainer = TrainerSingleCard(conf)
    trainer.train()