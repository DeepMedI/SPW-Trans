# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math
from torch.nn import Dropout, Softmax, Linear, Conv3d, LayerNorm
from torch.nn.modules.utils import _pair, _triple
from torch.distributions.normal import Normal
from torch.autograd import Variable
import numpy as np
from torch import nn, einsum
from einops import rearrange, repeat
#from models import CONFIGS as CONFIGS_ViT_seg
import configs3d as configs
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from einops.layers.torch import Rearrange
import torch.utils.checkpoint as checkpoint

import time


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'cswin_224': _cfg(),
    'cswin_384': _cfg(
        crop_pct=1.0
    ),

}


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LePEAttention(nn.Module):
    def __init__(self, dim, resolution, idx, split_size=7, dim_out=None, num_heads=12, attn_drop=0., proj_drop=0.,
                 qk_scale=None):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        if idx == -1:
            H_sp, W_sp, G_sp = self.resolution, self.resolution, self.resolution
        elif idx == 0:
            H_sp, W_sp, G_sp = self.resolution, self.resolution, self.resolution
        elif idx == 1:
            W_sp, H_sp, G_sp = self.resolution, self.resolution, self.resolution
        else:
          #  print("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp
        self.G_sp = G_sp
        stride = 1
        self.get_v = nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.attn_drop = nn.Dropout(attn_drop)

    def im2cswin(self, x):
        B, N, C = x.shape
        H = W = G = int(pow(N, 1/3))+1
    
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W, G)
        x = img2windows(x, self.H_sp, self.W_sp, self.G_sp)
        x = x.reshape(-1, self.H_sp * self.W_sp * self.G_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
       # print(x.size(),'xx')  (1 1 64 256)--(1 4 64 64)--(1 16 64 32) 
        return x

    def get_lepe(self, x, func):
        B, N, C = x.shape
        H = W = G = int(pow(N, 1/3))+1
        
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W, G)

        H_sp, W_sp, G_sp = self.H_sp, self.W_sp, self.G_sp
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp, G // G_sp, G_sp)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous().reshape(-1, C, H_sp, W_sp, G_sp)  ### B', C, H', W', G'

        lepe = func(x)  ### B', C, H', W' G
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp * G_sp).permute(0, 1, 3, 2).contiguous()

        x = x.reshape(-1, self.num_heads, C // self.num_heads, self.H_sp * self.W_sp * self.G_sp).permute(0, 1, 3, 2).contiguous()
        return x, lepe

    def forward(self, qkv):
        """
        x: B L C
        """
        q, k, v = qkv[0], qkv[1], qkv[2]

        ### Img2Window
        H = W = G = self.resolution
        
        B, L, C = q.shape
        assert L == H * W * G, "flatten img_tokens has wrong size"

        q = self.im2cswin(q)
        k = self.im2cswin(k)
        v, lepe = self.get_lepe(v, self.get_v)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v) + lepe
        x = x.transpose(1, 2).reshape(-1, self.H_sp * self.W_sp, self.G_sp, C)  # B head N N @ B head N C

        ### Window2Img
        x = windows2img(x, self.H_sp, self.W_sp, self.G_sp, H, W, G).view(B, -1, C)  # B H' W' C

        return x

class SPWtransBlock(nn.Module):

    def __init__(self, dim, reso, num_heads,
                 split_size=7, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 last_stage=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patches_resolution = reso
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm1 = norm_layer(dim)

        if self.patches_resolution == split_size:
            last_stage = True
        if last_stage:
            self.branch_num = 1
        else:
            self.branch_num = 3
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        if last_stage:
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim, resolution=self.patches_resolution, idx=-1,
                    split_size=split_size, num_heads=num_heads, dim_out=dim,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])
        else:
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim // 3, resolution=self.patches_resolution, idx=i,
                    split_size=split_size, num_heads=num_heads // 3, dim_out=dim // 3,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                       drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        """
        x: B, H*W, C  # 1 64 512    1 1*8*8 512
        """
       # print(x.size(),'input')
        H = W = G = self.patches_resolution  #8
      #  print(H)
     
        
      
        #print(H)
        B, L, C = x.shape
     #   print(x.size())
    #    print(L)
        assert L == H * W * G, "flatten img_tokens has wrong size"
        img = self.norm1(x)
        qkv = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3)   # 3 1 64 512
        #print(qkv.size(),'qkv')

        if self.branch_num == 2:
            x1 = self.attns[0](qkv[:, :, :, :C // 3])
            x2 = self.attns[1](qkv[:, :, :, C // 3: 2* C // 3])
            x3 = self.attns[2](qkv[:, :, :, 2* C // 3: C])
            attened_x = torch.cat([x1, x2, x3], dim=2)
        else:
            attened_x = self.attns[0](qkv)     # 1 64 512
      #  print(attened_x.size(),'attened_x')
        attened_x = self.proj(attened_x)
     #   print(attened_x.size(),'attened_x')   1 64 512
        x = x + self.drop_path(attened_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
      #  print(x.size())
        

        return x

class SPWtransBlock(nn.Module):

    def __init__(self, dim, reso, num_heads,
                 split_size=7, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 last_stage=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patches_resolution = reso
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm1 = norm_layer(dim)

        if self.patches_resolution == split_size:
            last_stage = True
        if last_stage:
            self.branch_num = 1
        else:
            self.branch_num = 3
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        if last_stage:
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim, resolution=self.patches_resolution, idx=-1,
                    split_size=split_size, num_heads=12, dim_out=dim,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])
        else:
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim // 3, resolution=self.patches_resolution, idx=i,
                    split_size=split_size, num_heads=num_heads // 3, dim_out=dim // 3,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                       drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        """
        x: B, H*W, C  # 1 64 512    1 1*8*8 512
        """
       # print(x.size(),'input')
        H = W = G = self.patches_resolution  #8
      #  print(H)
     
        
      
        #print(H)
        B, L, C = x.shape
        #print(x.size())
    #    print(L)
        assert L == H * W * G, "flatten img_tokens has wrong size"
        img = self.norm1(x)
        #print(img.size(),'img')
        qkv = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3)   # 3 1 64 512
        #print(qkv.size(),'qkv')
        
        if self.branch_num == 2:
            x1 = self.attns[0](qkv[:, :, :, :C // 3])
            #print(x1.size(),'x1')
            x2 = self.attns[1](qkv[:, :, :, C // 3: C // 3 * 2])
            #print(x2.size(),'x2')
            x3 = self.attns[2](qkv[:, :, :, C // 3 * 2:])
            #print(x3.size(),'x3')
            attened_x = torch.cat([x1, x2, x3], dim=2)
        else:
            attened_x = self.attns[0](qkv)     # 1 64 512
        
        
            
            
            
       # if self.branch_num == 2:
          #  x1 = self.attns[0](qkv[:, :, :, :C // 2])
        #    x2 = self.attns[1](qkv[:, :, :, C // 2:])
        #    attened_x = torch.cat([x1, x2], dim=2)
      #  else:
           # attened_x = self.attns[0](qkv)     # 1 64 512

        attened_x = self.proj(attened_x)
     #   print(attened_x.size(),'attened_x')   1 64 512
        x = x + self.drop_path(attened_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
      #  print(x.size())
        

        return x
        


def img2windows(img, H_sp, W_sp, G_sp):
    """
    img: B C H W
    zG  B C H W G
    """
    B, C, H, W, G = img.shape
    
    #ZG
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp, G // G_sp, G_sp)
    img_perm = img_reshape.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous().reshape(-1, H_sp * W_sp * G_sp, C)
    
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, G_sp, H, W, G):
    """
    img_splits_hw: B' H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W * G/ H_sp / W_sp/ G_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, G // G_sp, H_sp, W_sp, G_sp, -1)
    img = img.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, H, W, G, -1)
    return img


class Merge_Block(nn.Module):
    def __init__(self, dim, dim_out, norm_layer=nn.LayerNorm):
        super().__init__()
        self.conv = nn.Conv3d(dim, dim_out, 3, 2, 1)
        self.norm = norm_layer(dim_out)

    def forward(self, x):
        B, new_HW, C = x.shape
        #print(x.shape)
        H = W = G = int(pow(new_HW, 1/3))+1
        
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W, G)
        x = self.conv(x)
        B, C = x.shape[:2]
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        x = self.norm(x)

        return x
        
class upMerge_Block(nn.Module):
    def __init__(self, dim, dim_out, norm_layer=nn.LayerNorm):
        super().__init__()
        self.conv = nn.Conv3d(dim, dim_out, 3, 1, 1)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.norm = norm_layer(dim_out)

    def forward(self, x):
        B, new_HW, C = x.shape
        #print(x.shape)
        H = W = G = int(pow(new_HW, 1/3))+1
        
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W, G)
        x = self.up(x)
        x = self.conv(x)
        B, C = x.shape[:2]
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        x = self.norm(x)

        return x
        
 
       
class DecoderBlock1(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels, 
            skip_channels,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv3dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv3dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

    def forward(self, x, skip):
        B, N, C = x.shape
        H = W = G = int(pow(N, 1/3))+1        
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W, G)      
        

        
        
        B, n1, C = skip.shape
        H1 = W1 = G1 = int(pow(n1, 1/3))+1     
        skip = skip.transpose(-2, -1).contiguous().view(B, C, H1, W1, G1)  
        x = self.up(x)   
     #   print(x.size(),'up')  
      #  print(skip.size(),'skip')
        
        
        if skip is not None:
            #print(x.shape)
            #print(skip.shape)
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
      
        x = x.view(B, C, -1).transpose(-1,-2).contiguous()
        #print('DecoderBlock')c
        return x
        

class SPWTransformer(nn.Module):


    def __init__(self, img_size=256, patch_size=16, in_chans=512, num_classes=1000, embed_dim=32, depth=[2, 2, 6, 2],
                 split_size=[3, 5, 7],
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, use_chk=False):
        super().__init__()
        self.use_chk = use_chk
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        heads = num_heads
        
        self.merge0 = Merge_Block(48, 48 * 2)

        curr_dim = embed_dim
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, int(np.sum(depth)))]  # stochastic depth decay rule
        self.stage1 = nn.ModuleList([
            SPWtransBlock(
                dim=96, num_heads=heads[0], reso=16, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[0],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer,last_stage=True)
            for i in range(depth[0])])

        self.merge1 = Merge_Block(96+2, 192)
        #curr_dim = curr_dim * 2
        self.stage2 = nn.ModuleList(
            [SPWtransBlock(
                dim=192, num_heads=heads[1], reso=8, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[1],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:1]) + i], norm_layer=norm_layer,last_stage=True)
                for i in range(depth[1])])

        self.merge2 = Merge_Block(192+2, 384)
        #curr_dim = curr_dim * 2
        temp_stage3 = []
        temp_stage3.extend(
            [SPWtransBlock(
                dim=384, num_heads=heads[2], reso=4, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[2],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:2]) + i], norm_layer=norm_layer, last_stage=True)
                for i in range(depth[2])])

        self.stage3 = nn.ModuleList(temp_stage3)

        self.upmerge1 = upMerge_Block(384, 192)
        #curr_dim = curr_dim * 2
        self.stage4 = nn.ModuleList(
            [SPWtransBlock(
                dim=192, num_heads=heads[1], reso=8, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[1],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:-1]) + i], norm_layer=norm_layer, last_stage=True)
                for i in range(depth[-1])])
                
        self.upmerge2 = upMerge_Block(192, 96)
        #curr_dim = curr_dim * 2
        self.stage5 = nn.ModuleList(
            [SPWtransBlock(
                dim=96, num_heads=heads[2], reso=16, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[0],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:-1]) + i], norm_layer=norm_layer, last_stage=True)
                for i in range(depth[-1])])
                
        self.decoder2 = DecoderBlock1(192, 96, 96) 
        self.decoder1 = DecoderBlock1(384, 192, 192)     
                

        self.norm = norm_layer(48)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        if self.num_classes != num_classes:
         #   print('reset head to', num_classes)
            self.num_classes = num_classes
            self.head = nn.Linear(self.out_dim, num_classes) if num_classes > 0 else nn.Identity()
            self.head = self.head.cuda()
            trunc_normal_(self.head.weight, std=.02)
            if self.head.bias is not None:
                nn.init.constant_(self.head.bias, 0)

    def forward_features(self, x, yuan):
        B = x.shape[0]   # 1 32*32*32 32
        
        
        x = self.merge0(x)   # 1 16*16*16 64
        
        #print(x.size(),'x')
        
        for blk in self.stage1:
            if self.use_chk:
                x = checkpoint.checkpoint(blk, x)
            else:
                x_4 = blk(x)  # 1 16*16*16 64
        x_41 = x_4.transpose(-1,-2).view(-1, 96, 16, 16, 16)      
        yuan = nn.MaxPool3d(8)(yuan)
        x_41 = torch.cat((yuan, x_41), dim=1)
        x_41 = x_41.view(-1,98,16*16*16).transpose(-1,-2)
                
                
        x = self.merge1(x_41)          
        for pre in self.stage2:
            x_8 = pre(x)  #  1 16*16*16 128   
        
        x_81 = x_8.transpose(-1,-2).view(-1, 192,8, 8, 8)      
        yuan = nn.MaxPool3d(2)(yuan)
        x_81 = torch.cat((yuan, x_81), dim=1)
        x_81 = x_81.view(-1,194,8*8*8).transpose(-1,-2)
            
            
        x = self.merge2(x_81)   
        for pre in self.stage3:
            x = pre(x)  #1 8*8*8 256
            
            
      #  x = self.upmerge1(x)     # 1 16*16*16 128
        x = self.decoder1(x,x_8)
        for pre in self.stage4:
            x = pre(x)
        
            
            
    #    x = self.upmerge2(x)  #  1 32*32*32 64
        x = self.decoder2(x,x_4)
        for pre in self.stage5:
            x = pre(x)
        
        

        return x

    def forward(self, x,yuan):
        x = self.forward_features(x,yuan)
        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict







class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size):
        super(Embeddings, self).__init__()
        self.config = config
        down_factor = config.down_factor
        patch_size = _triple(config.patches["size"])
        n_patches = int((img_size[0]/2**down_factor// patch_size[0]) * (img_size[1]/2**down_factor// patch_size[1]) * (img_size[2]/2**down_factor// patch_size[2]))
        self.hybrid_model = CNNEncoder(config, n_channels=2)
        in_channels = config['encoder_channels'][-1]
        self.patch_embeddings = Conv3d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1,64, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        x, features = self.hybrid_model(x)  #1 64 32 32 32
        a,b,c,d,e = x.shape
        x=x.view(a,b,c*d*e)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        return x, features


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        #print('Block')
        return x, weights

class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = StageModule(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        #print('Encoder')
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = SPWTransformer(patch_size=4, embed_dim=512, depth=[2, 2, 6, 2],
                             split_size=[1, 2, 7, 7], num_heads=[3, 6, 12, 24], mlp_ratio=4.)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)  # b 
    
        #print(embedding_output.shape)
        encoded = self.encoder(embedding_output, input_ids)
        #encoded, attn_weights = self.encoder(embedding_output) # (B, n_patch, hidden)
        #return encoded, attn_weights, features
        return encoded, features


class Conv3dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm3d(out_channels)

        super(Conv3dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels, 
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv3dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv3dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
          #  print(x.shape,'x')
         #   print(skip.shape,'s')
            x = torch.cat([x, skip], dim=1)
      #  print(x.size())
        x = self.conv1(x)
        x = self.conv2(x)
     #   print(x.size(),'sss')
        return x

class DecoderCup(nn.Module):
    def __init__(self, config, img_size):
        super().__init__()
        self.config = config
        self.down_factor = config.down_factor
        head_channels = config.conv_first_channel
        self.img_size = img_size
        self.conv_more = Conv3dReLU(
            64,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        
        skip_channels = self.config.skip_channels
        
        
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels
        self.patch_size = _triple(config.patches["size"])
        skip_channels = self.config.skip_channels
        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)
        
        self.inc1 = DoubleConv(24, 24)
        self.inc2 = DoubleConv(2, 24)
        self.inc3 = DoubleConv(48, 24)
        self.middecoder = DecoderBlock(24,24,24)
        self.enddecoder = DecoderBlock(24,24,24)

    def forward(self, hidden_states, features, xend):

        B, n, hidden = hidden_states.size()# reshape from (B, n_patch, hidden) to (B, h, w, hidden)     b 16*16*16 64
   
        
        l, h, w = (self.img_size[0]//2**self.down_factor//self.patch_size[0]), (self.img_size[1]//2**self.down_factor//self.patch_size[0]), (self.img_size[2]//2**self.down_factor//self.patch_size[0])
     #   print(l)
   #     print(w)

        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, l, h, w)  #B  64 32 32 32

      #  x = self.conv_more(x)
       # print(x.size(),'ssssssss')
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            
            x = decoder_block(x, skip=skip)

        mid = self.inc1(x)    # 1 16 64 64 64 
        
        xend = self.inc2(xend) #16 128 128 128     
           
     #   feats_down = nn.MaxPool3d(2)(xend)  # 16 64 64 64        
        x = self.middecoder(mid,xend) # 16 128 128 128

        x = torch.cat((x, xend),dim=1)
     #   print(x.size(),'x')
        x = self.inc3(x)
     #   print(x.size(),'x')
   
        return x

class SpatialTransformer(nn.Module):


    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
      #  print(self.grid.shape,'grud')
        #print(flow.shape)
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]
        #print('SpatialTransformer')

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        #print('DoubleConv')
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        #print('Down')
        return self.maxpool_conv(x)

        
class CNNEncoder(nn.Module):
    def __init__(self, config, n_channels=2):
        super(CNNEncoder, self).__init__()
        self.n_channels = n_channels
        decoder_channels = config.decoder_channels
        encoder_channels = config.encoder_channels
        self.down_num = config.down_num
        self.inc = DoubleConv(2, 24)
        self.down1 = Down(encoder_channels[0]+2, encoder_channels[1])
        self.down2 = Down(encoder_channels[1]+2, encoder_channels[2])
        self.width = encoder_channels[-1]
        
        self.down_num = config.down_num
        
        
    def forward(self, x):
        features = []
        yuan = x
      #  print(x.size())
        x1 = self.inc(x)  # 1 16 128 128 128
     #   print(x1.size(),'1')
        features.append(x1)
        x1 = torch.cat((x1,yuan),dim=1)
        x2 = self.down1(x1) #z 16 64 64 64
        features.append(x2)
    #    print(x2.size(),'2')
        yuan = nn.MaxPool3d(2)(yuan)
        x2 = torch.cat((x2,yuan),dim=1)
        feats = self.down2(x2) # 1 32 32 32
        features.append(feats)  
        
      #  fconv = self.inc(x)  # 1 16 128 128 128
    #    features.append(fconv) 
       
      #  print(feats.size(), 'fes')  1 32 32 32 32

        return feats, features[::-1]

class RegistrationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        conv3d.weight = nn.Parameter(Normal(0, 1e-5).sample(conv3d.weight.shape))
        conv3d.bias = nn.Parameter(torch.zeros(conv3d.bias.shape))
        super().__init__(conv3d)

class ViTVNet(nn.Module):
    def __init__(self, config,img_size=(128, 128, 128), int_steps=7, vis=False):
    #def __init__(self):#, vis=False
        super(ViTVNet, self).__init__()
        self.transformer = Transformer(config, img_size, vis)#vis
        self.decoder = DecoderCup(config, img_size)
        self.reg_head = RegistrationHead(
            in_channels=config.decoder_channels[-1],
            out_channels=config['n_dims'],
            kernel_size=3,
        )
       # img_size1=(128, 128, 128)
        self.spatial_trans = SpatialTransformer(img_size)
        self.config = config
        #self.integrate = VecInt(img_size, int_steps)
    def forward(self, x):
        xmid = x
        source = x[:,0:1,:,:]
        x,  features = self.transformer(x)   # 1 4096 64
        #print(x.shape,'x')
        x = self.decoder(x, features, xmid)
        #print(x.shape,'xx')
        flow = self.reg_head(x)
    #    print(flow.shape,'xxx')
    #    print(source.shape,'xxxx')
        out = self.spatial_trans(source, flow)
        #print(out.shape,'xxxx')
        return out, flow

class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.

    Obtained from https://github.com/voxelmorph/voxelmorph
    """

    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec



CONFIGS = {'ViT-V-Net': configs.get_3DReg_config(),}
config_vit = CONFIGS['ViT-V-Net']



if __name__ == "__main__":

    model = ViTVNet(config_vit)#img_size=(32, 256, 256),config_vit
    dummy_x = Variable(torch.randn(1, 1, 128, 128, 128))
    dummy_y = Variable(torch.randn(1, 1, 128, 128, 128))
    data = torch.cat((dummy_x,dummy_y),1)
    warp, flow_X = model(data)  # (1,3)
    # print(net)
    # print(X.size())
  #  print(warp.size())
  #  print(flow_X.size())

