import torch
from torch import nn
import itertools
from transformers import CLIPTextModel
from hcpdiff.utils import pad_attn_bias

class TEUnetWrapper(nn.Module):
    def __init__(self, unet, TE, train_TE=False):
        super().__init__()
        self.unet = unet
        self.TE = TE
        self.train_TE = train_TE

    def forward(self, prompt_ids, noisy_latents, timesteps, attn_mask=None, position_ids=None, plugin_input={}, **kwargs):
        input_all = dict(prompt_ids=prompt_ids, noisy_latents=noisy_latents, timesteps=timesteps,
                         position_ids=position_ids, attn_mask=attn_mask, **plugin_input)

        if hasattr(self.TE, 'input_feeder'):
            for feeder in self.TE.input_feeder:
                feeder(input_all)

        from hcpdiff.models.textencoder_catalog import CatalogTextEncoder
        if isinstance(self.TE, CatalogTextEncoder):
            # 通道A：cross-attention token序列，和原来完全一样
            encoder_hidden_states = self.TE(
                prompt_ids,
                position_ids=position_ids,
                attention_mask=attn_mask,
                output_hidden_states=True
            )[0]  # [B, 22, 768]

            if attn_mask is not None:
                encoder_hidden_states, attn_mask = pad_attn_bias(encoder_hidden_states, attn_mask)

            input_all['encoder_hidden_states'] = encoder_hidden_states
            if hasattr(self.unet, 'input_feeder'):
                for feeder in self.unet.input_feeder:
                    feeder(input_all)

            # 通道B：AdaGN注入
            # TE.forward()已经把adagn_emb挂载到self.TE.last_adagn_emb上了
            adagn_emb = getattr(self.TE, 'last_adagn_emb', None)

            if adagn_emb is not None:
                # 用一次性hook在time_embedding层输出之后加上物理向量
                # 这样物理信号会随time_emb一起流进每个ResBlock的AdaGN层
                _adagn = adagn_emb.to(dtype=next(self.unet.parameters()).dtype,
                                      device=noisy_latents.device)

                def _adagn_hook(module, inp, out):
                    if torch.rand(1).item() < 0.001:  # 千分之一概率打印，不影响速度
                        print(f"[AdaGN Hook] out.norm={out.norm():.4f}, "
                            f"adagn.norm={_adagn.norm():.4f}")
                    return out + _adagn

                handle = self.unet.time_embedding.register_forward_hook(_adagn_hook)
                model_pred = self.unet(
                    noisy_latents, timesteps, encoder_hidden_states,
                    encoder_attention_mask=attn_mask
                ).sample
                handle.remove()  # 立即移除，不影响下一次forward
            else:
                # adagn_emb不存在时的兜底（不应该发生，但防御性保留）
                model_pred = self.unet(
                    noisy_latents, timesteps, encoder_hidden_states,
                    encoder_attention_mask=attn_mask
                ).sample

        else:
            # 非CatalogTextEncoder的原有逻辑，完全不变
            encoder_hidden_states = self.TE(
                prompt_ids,
                position_ids=position_ids,
                attention_mask=attn_mask,
                output_hidden_states=True
            )[0]

            if attn_mask is not None:
                encoder_hidden_states, attn_mask = pad_attn_bias(encoder_hidden_states, attn_mask)

            input_all['encoder_hidden_states'] = encoder_hidden_states
            if hasattr(self.unet, 'input_feeder'):
                for feeder in self.unet.input_feeder:
                    feeder(input_all)

            model_pred = self.unet(
                noisy_latents, timesteps, encoder_hidden_states,
                encoder_attention_mask=attn_mask
            ).sample

        return model_pred

    def prepare(self, accelerator):
        if self.train_TE:
            return accelerator.prepare(self)
        else:
            self.unet = accelerator.prepare(self.unet)
            return self

    def enable_gradient_checkpointing(self):
        def grad_ckpt_enable(m):
            if hasattr(m, 'gradient_checkpointing'):
                m.training = True

        self.unet.enable_gradient_checkpointing()
        if self.train_TE:
            from hcpdiff.models.textencoder_catalog import CatalogTextEncoder
            if not isinstance(self.TE, CatalogTextEncoder) and hasattr(self.TE, 'gradient_checkpointing_enable'):
                self.TE.gradient_checkpointing_enable()
            self.apply(grad_ckpt_enable)
        else:
            self.unet.apply(grad_ckpt_enable)

    def trainable_parameters(self):
        if self.train_TE:
            return itertools.chain(self.unet.parameters(), self.TE.parameters())
        else:
            return self.unet.parameters()


class SDXLTEUnetWrapper(TEUnetWrapper):
    def forward(self, prompt_ids, noisy_latents, timesteps, attn_mask=None, position_ids=None,
                crop_info=None, plugin_input={}, **kwargs):
        input_all = dict(prompt_ids=prompt_ids, noisy_latents=noisy_latents, timesteps=timesteps,
                         position_ids=position_ids, attn_mask=attn_mask, **plugin_input)

        if hasattr(self.TE, 'input_feeder'):
            for feeder in self.TE.input_feeder:
                feeder(input_all)

        encoder_hidden_states, pooled_output = self.TE(
            prompt_ids,
            position_ids=position_ids,
            attention_mask=attn_mask,
            output_hidden_states=True
        )

        added_cond_kwargs = {"text_embeds": pooled_output[-1], "time_ids": crop_info}
        if attn_mask is not None:
            encoder_hidden_states, attn_mask = pad_attn_bias(encoder_hidden_states, attn_mask)

        input_all['encoder_hidden_states'] = encoder_hidden_states
        if hasattr(self.unet, 'input_feeder'):
            for feeder in self.unet.input_feeder:
                feeder(input_all)

        model_pred = self.unet(
            noisy_latents, timesteps, encoder_hidden_states,
            encoder_attention_mask=attn_mask,
            added_cond_kwargs=added_cond_kwargs
        ).sample
        return model_pred