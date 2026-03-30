import os 
import cv2 
import glob
from pathlib import Path
import utils
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from utils import convert_state_dict
from models import restormer_arch
from data.preprocess.crop_merge_image import stride_integral

os.sys.path.append('./data/MBD/')
from data.MBD.infer import net1_net2_infer_single_im

# Safetensors support
try:
    from safetensors.torch import load_file as load_safetensors
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False
    load_safetensors = None


##########################################################################
## DocRes Architecture (EXACT copy from docres_arch.py for safetensors)
##########################################################################

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))
    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))
    def forward(self, x):
        return self.body(x)

class DocResArch(nn.Module):
    """DocRes architecture from docres_arch.py - for safetensors models"""
    def __init__(self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[2,3,3,4],
        num_refinement_blocks=4,
        heads=[1,2,4,8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias',
        dual_pixel_task=True 
    ):
        super(DocResArch, self).__init__()

        # Auto-detect logic
        self.use_auto_coords = False
        embed_channels = inp_channels

        if inp_channels == 3:
            print("DocResArch: inp_channels is 3. Enabling Auto-Coordinate Injection (Internal input = 5).")
            self.use_auto_coords = True
            embed_channels = 5 # 3 RGB + 2 Coords

        self.patch_embed = OverlapPatchEmbed(embed_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim*2**1))
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2))
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim*2**3))
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim*2**2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim*2**1))
        self.reduce_chan_level1 = nn.Conv2d(int(dim*2**1), int(dim), kernel_size=1, bias=bias)
        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        self.output = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x, *args, **kwargs):
        # Handle Padding for odd-sized images
        h, w = x.shape[2], x.shape[3]
        factor = 8
        H, W = ((h + factor - 1) // factor) * factor, ((w + factor - 1) // factor) * factor
        pad_h = H - h
        pad_w = W - w

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), 'reflect')

        inp_img = x

        # Inject Coordinates if inputs are just images (now using Padded size)
        if self.use_auto_coords:
            b, c, hh, ww = inp_img.shape
            y_coords = torch.linspace(-1, 1, hh, device=inp_img.device)
            x_coords = torch.linspace(-1, 1, ww, device=inp_img.device)
            mesh_y, mesh_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
            coords = torch.stack((mesh_x, mesh_y), dim=0).unsqueeze(0).repeat(b, 1, 1, 1)
            inp_img = torch.cat([inp_img, coords], dim=1)

        # Main Architecture Pass
        x = self.patch_embed(inp_img)
        encoder_l1 = self.encoder_level1(x)
        x = self.down1_2(encoder_l1)

        encoder_l2 = self.encoder_level2(x)
        x = self.down2_3(encoder_l2)

        encoder_l3 = self.encoder_level3(x)
        x = self.down3_4(encoder_l3)

        latent = self.latent(x)

        x = self.up4_3(latent)
        x = torch.cat([x, encoder_l3], 1)
        x = self.reduce_chan_level3(x)
        x = self.decoder_level3(x)

        x = self.up3_2(x)
        x = torch.cat([x, encoder_l2], 1)
        x = self.reduce_chan_level2(x)
        x = self.decoder_level2(x)

        x = self.up2_1(x)
        x = torch.cat([x, encoder_l1], 1)
        x = self.reduce_chan_level1(x)
        x = self.decoder_level1(x)

        x = self.refinement(x)

        out = self.output(x)

        # Remove Padding
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :h, :w]

        return out


##########################################################################
## Helper Functions (ORIGINAL from your working inference.py)
##########################################################################

def dewarp_prompt(img):
    mask = net1_net2_infer_single_im(img,'models/mbd.pkl')
    base_coord = utils.getBasecoord(256,256)/256
    img[mask==0]=0
    mask = cv2.resize(mask,(256,256))/255
    return img,np.concatenate((base_coord,np.expand_dims(mask,-1)),-1)

def deshadow_prompt(img):
    h,w = img.shape[:2]
    img = cv2.resize(img,(1024,1024))
    rgb_planes = cv2.split(img)
    result_planes = []
    result_norm_planes = []
    bg_imgs = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        bg_imgs.append(bg_img)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)
    bg_imgs = cv2.merge(bg_imgs)
    bg_imgs = cv2.resize(bg_imgs,(w,h))
    result_norm = cv2.merge(result_norm_planes)
    result_norm[result_norm==0]=1
    shadow_map = np.clip(img.astype(float)/result_norm.astype(float)*255,0,255).astype(np.uint8)
    shadow_map = cv2.resize(shadow_map,(w,h))
    shadow_map = cv2.cvtColor(shadow_map,cv2.COLOR_BGR2GRAY)
    shadow_map = cv2.cvtColor(shadow_map,cv2.COLOR_GRAY2BGR)
    return bg_imgs

def deblur_prompt(img):
    x = cv2.Sobel(img,cv2.CV_16S,1,0)  
    y = cv2.Sobel(img,cv2.CV_16S,0,1)  
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)  
    high_frequency = cv2.addWeighted(absX,0.5,absY,0.5,0)
    high_frequency = cv2.cvtColor(high_frequency,cv2.COLOR_BGR2GRAY)
    high_frequency = cv2.cvtColor(high_frequency,cv2.COLOR_GRAY2BGR)
    return high_frequency

def appearance_prompt(img):
    h,w = img.shape[:2]
    img = cv2.resize(img,(1024,1024))
    rgb_planes = cv2.split(img)
    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)
    result_norm = cv2.merge(result_norm_planes)
    result_norm = cv2.resize(result_norm,(w,h))
    return result_norm

def binarization_promptv2(img):
    result,thresh = utils.SauvolaModBinarization(img)
    thresh = thresh.astype(np.uint8)
    result[result>155]=255
    result[result<=155]=0

    x = cv2.Sobel(img,cv2.CV_16S,1,0)  
    y = cv2.Sobel(img,cv2.CV_16S,0,1)  
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)  
    high_frequency = cv2.addWeighted(absX,0.5,absY,0.5,0)
    high_frequency = cv2.cvtColor(high_frequency,cv2.COLOR_BGR2GRAY)
    return np.concatenate((np.expand_dims(thresh,-1),np.expand_dims(high_frequency,-1),np.expand_dims(result,-1)),-1)

def dewarping(model,im_path,memory_fix=0):
    INPUT_SIZE=256
    im_org = cv2.imread(im_path)

    h_orig, w_orig = im_org.shape[:2]
    was_resized = False

    limit_map = {1: 1500, 2: 2000, 3: 3000}
    max_dim = limit_map.get(memory_fix, 0)

    if max_dim > 0 and max(h_orig, w_orig) > max_dim:
        scale = float(max_dim) / max(h_orig, w_orig)
        new_w, new_h = int(w_orig * scale), int(h_orig * scale)
        im_org = cv2.resize(im_org, (new_w, new_h))
        was_resized = True

    im_masked, prompt_org = dewarp_prompt(im_org.copy())

    h,w = im_masked.shape[:2]
    im_masked = im_masked.copy()
    im_masked = cv2.resize(im_masked,(INPUT_SIZE,INPUT_SIZE))
    im_masked = im_masked / 255.0
    im_masked = torch.from_numpy(im_masked.transpose(2,0,1)).unsqueeze(0)
    im_masked = im_masked.float().to(DEVICE)

    prompt = torch.from_numpy(prompt_org.transpose(2,0,1)).unsqueeze(0)
    prompt = prompt.float().to(DEVICE)

    in_im = torch.cat((im_masked,prompt),dim=1)

    base_coord = utils.getBasecoord(INPUT_SIZE,INPUT_SIZE)/INPUT_SIZE
    model = model.float()
    with torch.no_grad():
        pred = model(in_im)
        pred = pred[0][:2].permute(1,2,0).cpu().numpy()
        pred = pred+base_coord
    for i in range(15):
        pred = cv2.blur(pred,(3,3),borderType=cv2.BORDER_REPLICATE) 
    pred = cv2.resize(pred,(w,h))*(w,h)
    pred = pred.astype(np.float32)
    out_im = cv2.remap(im_org,pred[:,:,0],pred[:,:,1],cv2.INTER_LINEAR)

    prompt_org = (prompt_org*255).astype(np.uint8)
    prompt_org = cv2.resize(prompt_org,im_org.shape[:2][::-1])

    if was_resized:
        out_im = cv2.resize(out_im, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        prompt_org = cv2.resize(prompt_org, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

    return prompt_org[:,:,0],prompt_org[:,:,1],prompt_org[:,:,2],out_im

def appearance(model, im_path, memory_fix=0):
    MAX_SIZE = 1600

    if memory_fix == 1:
        MAX_SIZE = 1500
    elif memory_fix == 2:
        MAX_SIZE = 2000
    elif memory_fix == 3:
        MAX_SIZE = 3000

    im_org = cv2.imread(im_path)
    orig_h, orig_w = im_org.shape[:2]
    prompt = appearance_prompt(im_org)
    in_im = np.concatenate((im_org, prompt), -1)

    resized = False
    if max(orig_w, orig_h) > MAX_SIZE:
        scale = float(MAX_SIZE) / max(orig_w, orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        in_im = cv2.resize(in_im, (new_w, new_h))
        resized = True
    else:
        new_h, new_w = orig_h, orig_w

    factor = 8
    pad_h = (factor - new_h % factor) % factor
    pad_w = (factor - new_w % factor) % factor

    if pad_h != 0 or pad_w != 0:
        in_im = cv2.copyMakeBorder(in_im, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

    in_im = in_im / 255.0
    in_im = torch.from_numpy(in_im.transpose(2, 0, 1)).unsqueeze(0)
    in_im = in_im.half().to(DEVICE)

    model = model.half()
    model.eval()

    with torch.no_grad():
        pred = model(in_im)
        pred = torch.clamp(pred, 0, 1)
        pred = pred[0].permute(1, 2, 0).cpu().numpy()
        pred = (pred * 255).astype(np.uint8)

    pred = pred[:new_h, :new_w]

    if resized:
        pred[pred == 0] = 1
        shadow_map = cv2.resize(im_org, (new_w, new_h)).astype(float) / pred.astype(float)
        shadow_map = cv2.resize(shadow_map, (orig_w, orig_h))
        shadow_map[shadow_map == 0] = 0.00001
        out_im = np.clip(im_org.astype(float) / shadow_map, 0, 255).astype(np.uint8)
    else:
        out_im = pred

    return prompt[:, :, 0], prompt[:, :, 1], prompt[:, :, 2], out_im

def deshadowing(model,im_path,memory_fix=0):
    MAX_SIZE=1600

    if memory_fix == 1:
        MAX_SIZE = 1500
    elif memory_fix == 2:
        MAX_SIZE = 2000
    elif memory_fix == 3:
        MAX_SIZE = 3000

    im_org = cv2.imread(im_path)
    h,w = im_org.shape[:2]
    prompt = deshadow_prompt(im_org)
    in_im = np.concatenate((im_org,prompt),-1)

    if max(w,h) < MAX_SIZE:
        in_im,padding_h,padding_w = stride_integral(in_im,8)
    else:
        in_im = cv2.resize(in_im,(MAX_SIZE,MAX_SIZE))

    in_im = in_im / 255.0
    in_im = torch.from_numpy(in_im.transpose(2,0,1)).unsqueeze(0)

    in_im = in_im.half().to(DEVICE)
    model = model.half()
    with torch.no_grad():
        pred = model(in_im)
        pred = torch.clamp(pred,0,1)
        pred = pred[0].permute(1,2,0).cpu().numpy()
        pred = (pred*255).astype(np.uint8)

        if max(w,h) < MAX_SIZE:
            out_im = pred[padding_h:,padding_w:]
        else:
            pred[pred==0]=1
            shadow_map = cv2.resize(im_org,(MAX_SIZE,MAX_SIZE)).astype(float)/pred.astype(float)
            shadow_map = cv2.resize(shadow_map,(w,h))
            shadow_map[shadow_map==0]=0.00001
            out_im = np.clip(im_org.astype(float)/shadow_map,0,255).astype(np.uint8)

    return prompt[:,:,0],prompt[:,:,1],prompt[:,:,2],out_im

def deblurring(model,im_path,memory_fix=0):
    im_org = cv2.imread(im_path)

    h_orig, w_orig = im_org.shape[:2]
    was_resized = False

    limit_map = {1: 1500, 2: 2000, 3: 3000}
    max_dim = limit_map.get(memory_fix, 0)

    if max_dim > 0 and max(h_orig, w_orig) > max_dim:
        scale = float(max_dim) / max(h_orig, w_orig)
        new_w, new_h = int(w_orig * scale), int(h_orig * scale)
        im_org = cv2.resize(im_org, (new_w, new_h))
        was_resized = True

    in_im,padding_h,padding_w = stride_integral(im_org,8)
    prompt = deblur_prompt(in_im)
    in_im = np.concatenate((in_im,prompt),-1)
    in_im = in_im / 255.0
    in_im = torch.from_numpy(in_im.transpose(2,0,1)).unsqueeze(0)
    in_im = in_im.half().to(DEVICE)  
    model.to(DEVICE)
    model.eval()
    model = model.half()
    with torch.no_grad():
        pred = model(in_im)
        pred = torch.clamp(pred,0,1)
        pred = pred[0].permute(1,2,0).cpu().numpy()
        pred = (pred*255).astype(np.uint8)
        out_im = pred[padding_h:,padding_w:]

    if was_resized:
        out_im = cv2.resize(out_im, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        prompt = cv2.resize(prompt, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

    return prompt[:,:,0],prompt[:,:,1],prompt[:,:,2],out_im


def binarization(model,im_path,memory_fix=0,is_docres_arch=False):
    """
    Binarization task.
    For DocResArch (safetensors): expects 3-channel RGB input (coords auto-injected)
    For Restormer (pkl): expects 6-channel input (3 RGB + 3 prompt)
    """
    im_org = cv2.imread(im_path)

    h_orig, w_orig = im_org.shape[:2]
    was_resized = False

    limit_map = {1: 1500, 2: 2000, 3: 3000}
    max_dim = limit_map.get(memory_fix, 0)

    if max_dim > 0 and max(h_orig, w_orig) > max_dim:
        scale = float(max_dim) / max(h_orig, w_orig)
        new_w, new_h = int(w_orig * scale), int(h_orig * scale)
        im_org = cv2.resize(im_org, (new_w, new_h))
        was_resized = True

    im,padding_h,padding_w = stride_integral(im_org,8)
    prompt = binarization_promptv2(im)
    h,w = im.shape[:2]

    # Architecture-specific input handling
    if is_docres_arch:
        # DocResArch: only RGB (3 channels), coords auto-injected
        in_im = im.copy()
    else:
        # Restormer: RGB + 3 prompt channels = 6 channels
        in_im = np.concatenate((im,prompt),-1)

    in_im = in_im / 255.0
    in_im = torch.from_numpy(in_im.transpose(2,0,1)).unsqueeze(0)
    in_im = in_im.to(DEVICE)

    # FIX: Ensure model and input have same dtype
    if next(model.parameters()).dtype == torch.float16:
        in_im = in_im.half()
    else:
        in_im = in_im.float()
        model = model.float()

    with torch.no_grad():
        pred = model(in_im)
        pred = pred[:,:2,:,:]
        pred = torch.max(torch.softmax(pred,1),1)[1]
        pred = pred[0].cpu().numpy()
        pred = (pred*255).astype(np.uint8)
        pred = cv2.resize(pred,(w,h))
        out_im = pred[padding_h:,padding_w:]

    if was_resized:
        out_im = cv2.resize(out_im, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        prompt = cv2.resize(prompt, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

    return prompt[:,:,0],prompt[:,:,1],prompt[:,:,2],out_im

def get_args():
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--model_path', nargs='?', type=str, default='models/docres.pkl',help='Path of the saved checkpoint')
    parser.add_argument('--im_path', nargs='?', type=str, default='./distorted/',
                        help='Path of input document image')
    parser.add_argument('--out_folder', nargs='?', type=str, default='./output/',
                        help='Folder of the output images')
    parser.add_argument('--task', nargs='?', type=str, default='dewarping', 
                        help='task that need to be executed')
    parser.add_argument('--save_dtsprompt', nargs='?', type=int, default=0, 
                        help='Width of the input image')
    parser.add_argument('--memory_fix', nargs='?', type=int, default=0, 
                        help='1=1500px, 2=2000px, 3=3000px limit on long edge to avoid OOM.')
    args = parser.parse_args()
    possible_tasks = ['dewarping','deshadowing','appearance','deblurring','binarization','end2end']
    assert args.task in possible_tasks, 'Unsupported task, task must be one of '+', '.join(possible_tasks)
    return args

def model_init(args):
    model_path = args.model_path

    # Check if file is safetensors format
    if model_path.endswith('.safetensors'):
        if not HAS_SAFETENSORS:
            raise ImportError("safetensors package is required to load .safetensors files. Install with: pip install safetensors")

        print(f"Loading safetensors model using DocResArch from: {model_path}")

        # DocResArch for safetensors (from docres_arch.py)
        model = DocResArch(
            inp_channels=3,  # RGB only, coords auto-injected internally
            out_channels=3,
            dim=48,
            num_blocks=[2,3,3,4],
            num_refinement_blocks=4,
            heads=[1,2,4,8],
            ffn_expansion_factor=2.66,
            bias=False,
            LayerNorm_type='WithBias',
            dual_pixel_task=True
        )

        # Load safetensors file
        if DEVICE.type == 'cpu':
            state = load_safetensors(model_path, device='cpu')
        else:
            state = load_safetensors(model_path, device='cuda:0')

        print(f"Loaded {len(state)} keys from safetensors")

        # Load state dict directly - keys should match docres_arch.py format
        missing, unexpected = model.load_state_dict(state, strict=False)

        if missing:
            print(f"WARNING: Missing keys ({len(missing)}): {list(missing)[:5]}...")
        if unexpected:
            print(f"WARNING: Unexpected keys ({len(unexpected)}): {list(unexpected)[:5]}...")

        if not missing and not unexpected:
            print("All keys loaded successfully!")
        else:
            print(f"Model loaded with {len(missing)} missing and {len(unexpected)} unexpected keys")

        model.eval()
        model = model.to(DEVICE)
        return model, True  # True = is_docres_arch

    else:
        # Original Restormer model for .pkl and .pth files (UNCHANGED from your working version)
        print(f"Loading pickle model using Restormer architecture from: {model_path}")

        model = restormer_arch.Restormer( 
            inp_channels=6,  # 3 RGB + 3 prompt channels
            out_channels=3, 
            dim = 48,
            num_blocks = [2,3,3,4], 
            num_refinement_blocks = 4,
            heads = [1,2,4,8],
            ffn_expansion_factor = 2.66,
            bias = False,
            LayerNorm_type = 'WithBias',
            dual_pixel_task = True        
        )

        if DEVICE.type == 'cpu':
            state = convert_state_dict(torch.load(model_path, map_location='cpu')['model_state'])
        else:
            state = convert_state_dict(torch.load(model_path, map_location='cuda:0')['model_state'])    

        model.load_state_dict(state)

        model.eval()
        model = model.to(DEVICE)
        return model, False  # False = is_docres_arch

def inference_one_im(model,im_path,args,is_docres_arch=False):
    task = args.task
    memory_fix = args.memory_fix

    if task=='dewarping':
        prompt1,prompt2,prompt3,output = dewarping(model,im_path,memory_fix)
    elif task=='deshadowing':
        prompt1,prompt2,prompt3,output = deshadowing(model,im_path,memory_fix)
    elif task=='appearance':
        prompt1,prompt2,prompt3,output = appearance(model,im_path,memory_fix)
    elif task=='deblurring':
        prompt1,prompt2,prompt3,output = deblurring(model,im_path,memory_fix)
    elif task=='binarization':
        prompt1,prompt2,prompt3,output = binarization(model,im_path,memory_fix,is_docres_arch)
    elif task=='end2end':
        prompt1,prompt2,prompt3,output = dewarping(model,im_path,memory_fix)
        cv2.imwrite('output/step1.jpg',output)
        prompt1,prompt2,prompt3,output = deshadowing(model,'output/step1.jpg',memory_fix)
        cv2.imwrite('output/step2.jpg',output)
        prompt1,prompt2,prompt3,output = appearance(model,'output/step2.jpg',memory_fix)

    return prompt1,prompt2,prompt3,output


def save_results(
    img_path: str,
    out_folder: str,
    task: str,
    save_dtsprompt: bool,
    prompt1, prompt2, prompt3, output
):
    os.makedirs(out_folder, exist_ok=True)
    im_name = os.path.split(img_path)[-1]
    im_format = '.'+im_name.split('.')[-1]
    save_path = os.path.join(out_folder, im_name.replace(im_format, '_' + task + im_format))
    cv2.imwrite(save_path, output)
    if save_dtsprompt:
        cv2.imwrite(save_path.replace(im_format, '_prompt1' + im_format), prompt1)
        cv2.imwrite(save_path.replace(im_format, '_prompt2' + im_format), prompt2)
        cv2.imwrite(save_path.replace(im_format, '_prompt3' + im_format), prompt3)


if __name__ == '__main__':

    ## model init
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = get_args()
    model, is_docres_arch = model_init(args)

    img_source = args.im_path

    if Path(img_source).is_dir():
        img_paths = glob.glob(os.path.join(img_source, '*'))
        for img_path in img_paths:
            ## inference
            prompt1,prompt2,prompt3,output = inference_one_im(model,img_path,args,is_docres_arch)

            ## results saving
            save_results(
                img_path=img_path,
                out_folder=args.out_folder,
                task=args.task,
                save_dtsprompt=args.save_dtsprompt,
                prompt1=prompt1,
                prompt2=prompt2,
                prompt3=prompt3,
                output=output
            )

    else:
        ## inference
        prompt1,prompt2,prompt3,output = inference_one_im(model,img_source,args,is_docres_arch)

        ## results saving
        save_results(
            img_path=img_source,
            out_folder=args.out_folder,
            task=args.task,
            save_dtsprompt=args.save_dtsprompt,
            prompt1=prompt1,
            prompt2=prompt2,
            prompt3=prompt3,
            output=output
        )
