import os 
import cv2 
import glob
from pathlib import Path
import utils
import argparse
import numpy as np

import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from utils import convert_state_dict
from models import restormer_arch
from data.preprocess.crop_merge_image import stride_integral

# Import safetensors for loading .safetensors files
try:
    from safetensors.torch import load_file as safetensors_load
except ImportError:
    print("Warning: safetensors not installed. Install with: pip install safetensors")
    safetensors_load = None

# Import DocRes architecture
try:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from docres_arch import DocRes, docres_base, docres_large
    DOCRES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: DocRes architecture not available: {e}")
    DOCRES_AVAILABLE = False

os.sys.path.append('./data/MBD/')
from data.MBD.infer import net1_net2_infer_single_im


def dewarp_prompt(img):
    mask = net1_net2_infer_single_im(img,'models/mbd.pkl')
    base_coord = utils.getBasecoord(256,256)/256
    img[mask==0]=0
    mask = cv2.resize(mask,(256,256))/255
    return img,np.concatenate((base_coord,np.expand_dims(mask,-1)),-1)

def deshadow_prompt(img):
    h,w = img.shape[:2]
    # img = cv2.resize(img,(128,128))
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
    # result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)
    result_norm[result_norm==0]=1
    shadow_map = np.clip(img.astype(float)/result_norm.astype(float)*255,0,255).astype(np.uint8)
    shadow_map = cv2.resize(shadow_map,(w,h))
    shadow_map = cv2.cvtColor(shadow_map,cv2.COLOR_BGR2GRAY)
    shadow_map = cv2.cvtColor(shadow_map,cv2.COLOR_GRAY2BGR)
    # return shadow_map
    return bg_imgs

def deblur_prompt(img):
    x = cv2.Sobel(img,cv2.CV_16S,1,0)  
    y = cv2.Sobel(img,cv2.CV_16S,0,1)  
    absX = cv2.convertScaleAbs(x)   # 转回uint8  
    absY = cv2.convertScaleAbs(y)  
    high_frequency = cv2.addWeighted(absX,0.5,absY,0.5,0)
    high_frequency = cv2.cvtColor(high_frequency,cv2.COLOR_BGR2GRAY)
    high_frequency = cv2.cvtColor(high_frequency,cv2.COLOR_GRAY2BGR)
    return high_frequency

def appearance_prompt(img):
    h,w = img.shape[:2]
    # img = cv2.resize(img,(128,128))
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
    absX = cv2.convertScaleAbs(x)   # 转回uint8  
    absY = cv2.convertScaleAbs(y)  
    high_frequency = cv2.addWeighted(absX,0.5,absY,0.5,0)
    high_frequency = cv2.cvtColor(high_frequency,cv2.COLOR_BGR2GRAY)
    return np.concatenate((np.expand_dims(thresh,-1),np.expand_dims(high_frequency,-1),np.expand_dims(result,-1)),-1)

def dewarping(model,im_path,memory_fix=0):
    INPUT_SIZE=256
    im_org = cv2.imread(im_path)

    # OOM Fix: Resize down input based on level only if image is larger than limit
    h_orig, w_orig = im_org.shape[:2]
    was_resized = False

    limit_map = {1: 1500, 2: 2000, 3: 3000}
    max_dim = limit_map.get(memory_fix, 0)

    # Only resize if max_dim is set AND image is larger than max_dim
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

    # inference
    base_coord = utils.getBasecoord(INPUT_SIZE,INPUT_SIZE)/INPUT_SIZE
    model = model.float()
    with torch.no_grad():
        pred = model(in_im)
        pred = pred[0][:2].permute(1,2,0).cpu().numpy()
        pred = pred+base_coord
    ## smooth
    for i in range(15):
        pred = cv2.blur(pred,(3,3),borderType=cv2.BORDER_REPLICATE) 
    pred = cv2.resize(pred,(w,h))*(w,h)
    pred = pred.astype(np.float32)
    out_im = cv2.remap(im_org,pred[:,:,0],pred[:,:,1],cv2.INTER_LINEAR)

    prompt_org = (prompt_org*255).astype(np.uint8)
    prompt_org = cv2.resize(prompt_org,im_org.shape[:2][::-1])

    # OOM Fix: Resize back output to original
    if was_resized:
        out_im = cv2.resize(out_im, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        prompt_org = cv2.resize(prompt_org, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

    return prompt_org[:,:,0],prompt_org[:,:,1],prompt_org[:,:,2],out_im

def appearance(model, im_path, memory_fix=0, model_type='restormer'):

    MAX_SIZE = 1600

    if memory_fix == 1:
        MAX_SIZE = 1500
    elif memory_fix == 2:
        MAX_SIZE = 2000
    elif memory_fix == 3:
        MAX_SIZE = 3000

    # -------------------------
    # Load image
    # -------------------------
    im_org = cv2.imread(im_path)
    orig_h, orig_w = im_org.shape[:2]

    # Generate prompt
    prompt = appearance_prompt(im_org)

    # DocRes uses 3 channels (auto-coordinate injection)
    # Restormer uses 6 channels (image + prompt)
    if model_type == 'docres':
        in_im = im_org  # DocRes handles coordinates internally
    else:
        in_im = np.concatenate((im_org, prompt), -1)

    # -------------------------
    # Resize if larger than MAX_SIZE
    # -------------------------
    resized = False
    if max(orig_w, orig_h) > MAX_SIZE:
        scale = float(MAX_SIZE) / max(orig_w, orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        in_im = cv2.resize(in_im, (new_w, new_h))
        resized = True
    else:
        new_h, new_w = orig_h, orig_w

    # -------------------------
    # CRITICAL FIX
    # Pad to multiple of 8
    # -------------------------
    factor = 8

    pad_h = (factor - new_h % factor) % factor
    pad_w = (factor - new_w % factor) % factor

    if pad_h != 0 or pad_w != 0:
        in_im = cv2.copyMakeBorder(
            in_im,
            0, pad_h,
            0, pad_w,
            cv2.BORDER_REFLECT
        )

    padded_h, padded_w = in_im.shape[:2]

    # -------------------------
    # Normalize + tensor
    # -------------------------
    in_im = in_im / 255.0
    in_im = torch.from_numpy(in_im.transpose(2, 0, 1)).unsqueeze(0)

    # DocRes uses fp32 for coordinate injection, Restormer can use fp16
    if model_type == 'docres':
        in_im = in_im.float().to(DEVICE)
        model = model.float()
    else:
        in_im = in_im.half().to(DEVICE)
        model = model.half()

    model.eval()

    # -------------------------
    # Inference
    # -------------------------
    with torch.no_grad():
        pred = model(in_im)
        pred = torch.clamp(pred, 0, 1)
        pred = pred[0].permute(1, 2, 0).cpu().numpy()
        pred = (pred * 255).astype(np.uint8)

    # -------------------------
    # Remove padding
    # -------------------------
    pred = pred[:new_h, :new_w]

    # -------------------------
    # Restore to original size
    # -------------------------
    if resized:
        pred[pred == 0] = 1
        shadow_map = cv2.resize(im_org, (new_w, new_h)).astype(float) / pred.astype(float)
        shadow_map = cv2.resize(shadow_map, (orig_w, orig_h))
        shadow_map[shadow_map == 0] = 0.00001
        out_im = np.clip(im_org.astype(float) / shadow_map, 0, 255).astype(np.uint8)
    else:
        out_im = pred

    return prompt[:, :, 0], prompt[:, :, 1], prompt[:, :, 2], out_im

def deshadowing(model,im_path,memory_fix=0, model_type='restormer'):
    MAX_SIZE=1600 # Default default

    if memory_fix == 1:
        MAX_SIZE = 1500
    elif memory_fix == 2:
        MAX_SIZE = 2000
    elif memory_fix == 3:
        MAX_SIZE = 3000

    # obtain im and prompt
    im_org = cv2.imread(im_path)
    h,w = im_org.shape[:2]
    prompt = deshadow_prompt(im_org)

    # DocRes uses 3 channels (auto-coordinate injection)
    # Restormer uses 6 channels (image + prompt)
    if model_type == 'docres':
        in_im = im_org  # DocRes handles coordinates internally
    else:
        in_im = np.concatenate((im_org,prompt),-1)

    # constrain the max resolution 
    if max(w,h) < MAX_SIZE:
        in_im,padding_h,padding_w = stride_integral(in_im,8)
    else:
        in_im = cv2.resize(in_im,(MAX_SIZE,MAX_SIZE))

    # normalize
    in_im = in_im / 255.0
    in_im = torch.from_numpy(in_im.transpose(2,0,1)).unsqueeze(0)

    # inference
    # DocRes uses fp32 for coordinate injection, Restormer can use fp16
    if model_type == 'docres':
        in_im = in_im.float().to(DEVICE)
        model = model.float()
    else:
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


def deblurring(model,im_path,memory_fix=0, model_type='restormer'):
    # setup image
    im_org = cv2.imread(im_path)

    # OOM Fix: Resize down input
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

    # DocRes uses 3 channels (auto-coordinate injection)
    # Restormer uses 6 channels (image + prompt)
    if model_type == 'docres':
        in_im = in_im  # Just use the image, DocRes handles coordinates
    else:
        in_im = np.concatenate((in_im,prompt),-1)

    in_im = in_im / 255.0
    in_im = torch.from_numpy(in_im.transpose(2,0,1)).unsqueeze(0)

    # DocRes uses fp32 for coordinate injection, Restormer can use fp16
    if model_type == 'docres':
        in_im = in_im.float().to(DEVICE)
        model = model.float()
    else:
        in_im = in_im.half().to(DEVICE)
        model = model.half()

    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        pred = model(in_im)
        pred = torch.clamp(pred,0,1)
        pred = pred[0].permute(1,2,0).cpu().numpy()
        pred = (pred*255).astype(np.uint8)
        out_im = pred[padding_h:,padding_w:]

    # OOM Fix: Resize back output to original
    if was_resized:
        out_im = cv2.resize(out_im, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        prompt = cv2.resize(prompt, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

    return prompt[:,:,0],prompt[:,:,1],prompt[:,:,2],out_im



def binarization(model,im_path,memory_fix=0, model_type='restormer'):
    im_org = cv2.imread(im_path)

    # OOM Fix: Resize down input
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

    # DocRes uses 3 channels (auto-coordinate injection)
    # Restormer uses 6 channels (image + prompt)
    if model_type == 'docres':
        in_im = im  # Just use the image, DocRes handles coordinates
    else:
        in_im = np.concatenate((im,prompt),-1)

    in_im = in_im / 255.0
    in_im = torch.from_numpy(in_im.transpose(2,0,1)).unsqueeze(0)
    in_im = in_im.to(DEVICE)

    # DocRes uses fp32 for coordinate injection, Restormer can use fp16
    if model_type == 'docres':
        model = model.float()
        in_im = in_im.float()
    else:
        model = model.half()
        in_im = in_im.half()

    with torch.no_grad():
        pred = model(in_im)
        # For binarization, DocRes outputs 3 channels, we need to handle this
        if model_type == 'docres':
            # DocRes outputs RGB [B, 3, H, W], convert to grayscale for binarization
            # Take first channel and squeeze batch dim
            pred = pred[:,0:1,:,:]  # Take first channel, keep dim as [B, 1, H, W]
            pred = pred.squeeze(0).squeeze(0)  # Remove batch and channel dims -> [H, W]
        else:
            pred = pred[:,:2,:,:]
            pred = torch.max(torch.softmax(pred,1),1)[1]
            pred = pred.squeeze(0)  # Remove batch dim -> [H, W]

        pred = pred.cpu().numpy()
        pred = (pred*255).astype(np.uint8)

        # pred should now be 2D [H, W]
        if len(pred.shape) == 2:
            pred = cv2.resize(pred,(w,h))
            out_im = pred[padding_h:,padding_w:]
        else:
            # If still has channel dim, squeeze it
            pred = pred.squeeze()
            pred = cv2.resize(pred,(w,h))
            out_im = pred[padding_h:,padding_w:]

    # OOM Fix: Resize back output to original
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
    parser.add_argument('--model_type', nargs='?', type=str, default='auto',
                        help='Model architecture: restormer, docres, or auto (auto-detect from filename)')
    args = parser.parse_args()
    possible_tasks = ['dewarping','deshadowing','appearance','deblurring','binarization','end2end']
    assert args.task in possible_tasks, 'Unsupported task, task must be one of '+', '.join(possible_tasks)
    return args

def detect_model_type(model_path):
    """Auto-detect model type from filename and extension"""
    filename = os.path.basename(model_path).lower()

    # Check for safetensors extension
    if filename.endswith('.safetensors'):
        return 'docres'

    # Check for common DocRes filename patterns
    docres_patterns = ['docres', 'net_g', 'net_g_ema', 'ema']
    for pattern in docres_patterns:
        if pattern in filename:
            return 'docres'

    # Default to restormer for .pkl files
    if filename.endswith('.pkl'):
        return 'restormer'

    return 'restormer'

def load_checkpoint(model, checkpoint_path, model_type='restormer'):
    """Load checkpoint from either .pkl or .safetensors file"""

    if checkpoint_path.endswith('.safetensors'):
        if safetensors_load is None:
            raise ImportError("safetensors is required to load .safetensors files. Install with: pip install safetensors")

        print(f"Loading safetensors checkpoint: {checkpoint_path}")
        state_dict = safetensors_load(checkpoint_path)

        # Handle potential prefix in state dict keys
        new_state_dict = {}
        for k, v in state_dict.items():
            # Remove common prefixes
            if k.startswith('module.'):
                k = k[7:]
            new_state_dict[k] = v

        model.load_state_dict(new_state_dict, strict=False)

    elif checkpoint_path.endswith('.pkl') or checkpoint_path.endswith('.pth'):
        print(f"Loading pickle checkpoint: {checkpoint_path}")
        if DEVICE.type == 'cpu':
            state = convert_state_dict(torch.load(checkpoint_path, map_location='cpu')['model_state'])
        else:
            state = convert_state_dict(torch.load(checkpoint_path, map_location='cuda:0')['model_state'])
        model.load_state_dict(state)
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}. Use .safetensors, .pkl, or .pth")

    return model

def model_init(args):
    # Determine model type
    if args.model_type == 'auto':
        model_type = detect_model_type(args.model_path)
        print(f"Auto-detected model type: {model_type}")
    else:
        model_type = args.model_type
        print(f"Using specified model type: {model_type}")

    # Store model_type for later use
    args.model_type_actual = model_type

    if model_type == 'docres':
        if not DOCRES_AVAILABLE:
            raise ImportError("DocRes architecture not available. Make sure docres_arch.py is in the same directory.")

        print("Initializing DocRes model...")
        # DocRes with auto-coordinate injection (inp_channels=3 triggers auto-coords)
        model = DocRes(
            inp_channels=3,  # 3 triggers auto-coordinate injection -> 5 internal
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

        model = load_checkpoint(model, args.model_path, model_type='docres')
        model = model.to(DEVICE)
        model.eval()

    else:
        print("Initializing Restormer model...")
        # prepare model (Restormer - original)
        model = restormer_arch.Restormer( 
            inp_channels=6, 
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

        model = load_checkpoint(model, args.model_path, model_type='restormer')
        model = model.to(DEVICE)
        model.eval()

    return model

def inference_one_im(model,im_path,args):
    task = args.task
    memory_fix = args.memory_fix
    model_type = getattr(args, 'model_type_actual', 'restormer')

    if task=='dewarping':
        prompt1,prompt2,prompt3,output = dewarping(model,im_path,memory_fix)
    elif task=='deshadowing':
        prompt1,prompt2,prompt3,output = deshadowing(model,im_path,memory_fix, model_type)
    elif task=='appearance':
        prompt1,prompt2,prompt3,output = appearance(model,im_path,memory_fix, model_type)
    elif task=='deblurring':
        prompt1,prompt2,prompt3,output = deblurring(model,im_path,memory_fix, model_type)
    elif task=='binarization':
        prompt1,prompt2,prompt3,output = binarization(model,im_path,memory_fix, model_type)
    elif task=='end2end':
        prompt1,prompt2,prompt3,output = dewarping(model,im_path,memory_fix)
        cv2.imwrite('output/step1.jpg',output)
        prompt1,prompt2,prompt3,output = deshadowing(model,'output/step1.jpg',memory_fix, model_type)
        cv2.imwrite('output/step2.jpg',output)
        prompt1,prompt2,prompt3,output = appearance(model,'output/step2.jpg',memory_fix, model_type)
        # os.remove('output/step1.jpg')
        # os.remove('output/step2.jpg')

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
    model = model_init(args)

    img_source = args.im_path

    if Path(img_source).is_dir():
        img_paths = glob.glob(os.path.join(img_source, '*'))
        for img_path in img_paths:
            ## inference
            prompt1,prompt2,prompt3,output = inference_one_im(model,img_path,args)

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
        prompt1,prompt2,prompt3,output = inference_one_im(model,img_source,args)

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
