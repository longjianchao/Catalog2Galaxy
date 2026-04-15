import torch
from torch import nn
import itertools
from transformers import CLIPTextModel
from hcpdiff.utils import pad_attn_bias
from hcpdiff.models.morphology_renderer import MorphologyPriorInjector, expand_unet_input_channel


class TEUnetWrapper(nn.Module):
    def __init__(
        self,
        unet,
        TE,
        train_TE=False,
        use_morphology_prior=False,
        latent_size=24
    ):
        super().__init__()
        self.unet = unet
        self.TE = TE
        self.train_TE = train_TE
        
        # ========== 形态学双轨注入初始化 ==========
        self.use_morphology_prior = use_morphology_prior
        if use_morphology_prior:
            self.morphology_injector = MorphologyPriorInjector(latent_size=latent_size)
            expand_unet_input_channel(unet, new_in_channels=5, old_in_channels=4)
        # ==========================================

    def forward(self, prompt_ids, noisy_latents, timesteps, attn_mask=None, position_ids=None, plugin_input={}, **kwargs):
        input_all = dict(prompt_ids=prompt_ids, noisy_latents=noisy_latents, timesteps=timesteps,
                         position_ids=position_ids, attn_mask=attn_mask, **plugin_input)

        if hasattr(self.TE, 'input_feeder'):
            for feeder in self.TE.input_feeder:
                feeder(input_all)

        from hcpdiff.models.textencoder_catalog import CatalogTextEncoder
        if isinstance(self.TE, CatalogTextEncoder):
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

            # ========== 形态学先验拼接 ==========
            if self.use_morphology_prior:
                # 根本不需要从 TE 身上去取数据！
                # 传进来的 prompt_ids 就是底层 Dataset 吐出来的 22维 物理向量！
                geom_params = prompt_ids[:, self.geom_param_indices]
                
                # 渲染 + 强制类型对齐 + 拼接
                augmented_latents = self.morphology_injector(geom_params, noisy_latents)
            else:
                augmented_latents = noisy_latents
            # ====================================

            # ========== AdaGN 时间步注入 ==========
            adagn_emb = getattr(self.TE, 'last_adagn_emb', None)
            if adagn_emb is not None:
                _adagn = adagn_emb.to(dtype=next(self.unet.parameters()).dtype,
                                      device=noisy_latents.device)

                def _adagn_hook(module, inp, out):
                    # if torch.rand(1).item() < 0.001: 
                    #     print(f"[AdaGN Hook] out.norm={out.norm():.4f}, adagn.norm={_adagn.norm():.4f}")
                    return out + _adagn

                handle = self.unet.time_embedding.register_forward_hook(_adagn_hook)
                model_pred = self.unet(
                    augmented_latents, timesteps, encoder_hidden_states,
                    encoder_attention_mask=attn_mask
                ).sample
                handle.remove() 
            else:
                model_pred = self.unet(
                    augmented_latents, timesteps, encoder_hidden_states,
                    encoder_attention_mask=attn_mask
                ).sample

        else:
            # 非 CatalogTextEncoder 的原有逻辑
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

            # ========== 依然支持非 CatalogTE 时的形态注入 ==========
            if self.use_morphology_prior:
                geom_params = plugin_input.get('geom_params', None)
                if geom_params is not None:
                    augmented_latents = self.morphology_injector(geom_params, noisy_latents)
                else:
                    raise ValueError("use_morphology_prior=True 时，必须在 plugin_input 中提供 'geom_params'")
            else:
                augmented_latents = noisy_latents

            model_pred = self.unet(
                augmented_latents, timesteps, encoder_hidden_states,
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
        """
        SDXL 版本的 Wrapper，同样支持形态学先验注入
        """
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

        # ========== 形态学先验注入 (双轨注入方案) ==========
        if self.use_morphology_prior:
            geom_params = plugin_input.get('geom_params', None)
            
            if geom_params is not None:
                augmented_latents = self.morphology_injector(geom_params, noisy_latents)
            else:
                raise ValueError(
                    "use_morphology_prior=True 时，必须在 plugin_input 中提供 'geom_params'"
                )
        else:
            augmented_latents = noisy_latents
        # ==================================================

        model_pred = self.unet(
            augmented_latents, timesteps, encoder_hidden_states,
            encoder_attention_mask=attn_mask,
            added_cond_kwargs=added_cond_kwargs
        ).sample
        return model_pred