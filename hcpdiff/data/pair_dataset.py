"""
pair_dataset.py
====================
    :Name:        text-image pair dataset
    :Author:      Dong Ziyi
    :Affiliation: HCP Lab, SYSU
    :Created:     10/03/2023
    :Licence:     Apache-2.0
"""

import os.path
from argparse import Namespace

import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from typing import Tuple

from hcpdiff.utils.caption_tools import *
from hcpdiff.utils.utils import get_file_name, get_file_ext
from .bucket import BaseBucket
from .source import DataSource, ComposeDataSource

class TextImagePairDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(self, tokenizer, tokenizer_repeats: int = 1, att_mask_encode: bool = False,
                 bucket: BaseBucket = None, source: Dict[str, DataSource] = None, return_path: bool = False,
                 cache_path:str=None, encoder_attention_mask=False, **kwargs):
        self.return_path = return_path

        self.tokenizer = tokenizer
        self.tokenizer_repeats = tokenizer_repeats
        self.bucket: BaseBucket = bucket
        self.att_mask_encode = att_mask_encode
        self.source = ComposeDataSource(source)
        self.latents = None  # Cache latents for faster training. Works only without image argumentations.
        self.cache_path = cache_path
        self.encoder_attention_mask = encoder_attention_mask

    def load_data(self, path:str, data_source:DataSource, size:Tuple[int]):
        image_dict = data_source.load_image(path)
        image = image_dict['image']
        att_mask = image_dict.get('att_mask', None)
        if att_mask is None:
            data, crop_coord = self.bucket.crop_resize({"img":image}, size)
            image = data_source.procees_image(data['img'])  # resize to bucket size
            att_mask = torch.ones((size[1]//8, size[0]//8))
        else:
            data, crop_coord = self.bucket.crop_resize({"img":image, "mask":att_mask}, size)
            image = data_source.procees_image(data['img'])
            att_mask = torch.tensor(cv2.resize(data['mask'], (size[0]//8, size[1]//8), interpolation=cv2.INTER_LINEAR))
        return {'img':image, 'mask':att_mask}

    @torch.no_grad()
    def cache_latents(self, vae, weight_dtype, device, show_prog=True):
        if not self.cache_path:
            return

        # 1. 核心改进：确保 cache_path 是一个目录
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path, exist_ok=True)
        
        # 如果它是个文件（之前那个1.56GB的旧文件），提示用户手动删除
        if os.path.isfile(self.cache_path):
            raise IOError(f"检测到缓存路径 {self.cache_path} 是一个文件。请先手动删除该文件，本程序将创建同名目录。")

        self.latents = {}
        self.bucket.rest(0)

        print(f"正在同步/生成 Latents 缓存至目录: {self.cache_path}")
        
        for (path, data_source), size in tqdm(self.bucket, disable=not show_prog):
            img_name = data_source.get_image_name(path)
            # 每个星系拥有独立的缓存文件，例如：12345.pt
            latent_file = os.path.join(self.cache_path, img_name + ".pt")

            # 2. 检查该星系是否已经缓存过
            if os.path.exists(latent_file):
                try:
                    # 如果已存在，直接从独立文件加载
                    self.latents[img_name] = torch.load(latent_file, map_location='cpu')
                    continue
                except Exception:
                    print(f"警告：缓存文件 {latent_file} 损坏，将重新生成。")

            # 3. 如果没缓存，则调用 VAE 编码
            data = self.load_data(path, data_source, size)
            image = data['img'].unsqueeze(0).to(device, dtype=weight_dtype)
            
            # VAE 编码过程
            latents = vae.encode(image).latent_dist.sample().squeeze(0)
            data['img'] = (latents * vae.config.scaling_factor).cpu()
            
            # 4. 存入内存字典，并同步保存到独立文件
            self.latents[img_name] = data
            torch.save(data, latent_file)

        print(f"✅ 成功加载/缓存了 {len(self.latents)} 个星系特征。")

    def __len__(self):
        return len(self.bucket)

    def __getitem__(self, index):
        (path, data_source), size = self.bucket[index]
        img_name = data_source.get_image_name(path)

        if self.latents is None:
            data = self.load_data(path, data_source, size)
        else:
            data = self.latents[img_name].copy()

        prompt_ist = data_source.load_caption(img_name)

        # tokenize Sp or (Sn, Sp)
        tokens = self.tokenizer(prompt_ist, truncation=True, padding="max_length", return_tensors="pt",
                                    max_length=self.tokenizer.model_max_length*self.tokenizer_repeats)
        data['prompt'] = tokens.input_ids.squeeze()
        if self.encoder_attention_mask and 'attention_mask' in tokens:
            data['attn_mask'] = tokens.attention_mask.squeeze()
        if 'position_ids' in tokens:
            data['position_ids'] = tokens.position_ids.squeeze()

        if self.return_path:
            return data, path
        else:
            return data

    @staticmethod
    def collate_fn(batch):
        '''
        batch: [{img:tensor, prompt:str, ..., plugin_input:{...}},{}]
        '''
        has_plugin_input = 'plugin_input' in batch[0]
        if has_plugin_input:
            plugin_input = {k:[] for k in batch[0]['plugin_input'].keys()}

        datas = {k:[] for k in batch[0].keys() if k != 'plugin_input' and k != 'prompt'}
        sn_list, sp_list = [], []

        for data in batch:
            if has_plugin_input:
                for k, v in data.pop('plugin_input').items():
                    plugin_input[k].append(v)

            prompt = data.pop('prompt')
            if len(prompt.shape) == 2:
                sn_list.append(prompt[0])
                sp_list.append(prompt[1])
            else:
                sp_list.append(prompt)

            for k, v in data.items():
                datas[k].append(v)

        for k, v in datas.items():
            datas[k] = torch.stack(v)
            if k == 'mask':
                datas[k] = datas[k].unsqueeze(1)

        sn_list += sp_list
        datas['prompt'] = torch.stack(sn_list)
        if has_plugin_input:
            datas['plugin_input'] = {k:torch.stack(v) for k, v in plugin_input.items()}

        return datas
