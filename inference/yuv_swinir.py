# Modified from https://github.com/JingyunLiang/SwinIR
import argparse
import cv2
import glob
import numpy as np
import os
import torch
from torch.nn import functional as F

from basicsr.archs.swinir_arch import SwinIR
from yuv_utils import *
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='datasets/swin.yuv', help='input yuv file')
    parser.add_argument('--output', type=str, default='results/swin.yuv', help='output yuv file')
    parser.add_argument('--num_frames', type=int, default=64, help='number of frames to process')
    parser.add_argument('--width', type=int, default=960, help='width of the frames')
    parser.add_argument('--height', type=int, default=544, help='height of the frames')
    parser.add_argument('--patch_size', type=int, default=64, help='training patch size')
    parser.add_argument('--scale', type=int, default=4, help='scale factor: 2, 4, 8')
    args = parser.parse_args()

    model_path = f"experiments/pretrained_models/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x{args.scale}.pth"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    model = define_model(args, model_path)
    model.eval()
    model = model.to(device)

    window_size = 8

    frames_np = load_yuv_frames(
        video_file_path=args.input,
        start_idx=0,
        num_frames=args.num_frames,
        width=args.width,
        height=args.height,
        bit_depth=10,
        pixel_format='yuv420p'
    )
    frames_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float().unsqueeze(0)

    frames_list = []
    for idx in tqdm(range(frames_tensor.shape[1]), desc='Processing frames', leave=False):
        frame = frames_tensor[:, idx].to(device)
        # inference
        with torch.no_grad():
            # pad input image to be a multiple of window_size
            mod_pad_h, mod_pad_w = 0, 0
            _, _, h, w = frame.shape
            if h % window_size != 0:
                mod_pad_h = window_size - h % window_size
            if w % window_size != 0:
                mod_pad_w = window_size - w % window_size
            frame = F.pad(frame, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

            output = model(frame)
            _, _, h, w = output.size()
            # remove padding
            output = output[:, :, 0:h - mod_pad_h * args.scale, 0:w - mod_pad_w * args.scale]

        # save image
        output = output.data.squeeze().float().cpu()
        frames_list.append(output)

    tqdm.write('Saving video')
    video = torch.stack(frames_list, dim=0)
    video = video.permute(0, 2, 3, 1)
    tqdm.write(f'Upsampled video shape: {video.shape}')
    rgb_to_yuv420p10bit(video, args.output)
    tqdm.write('Done!')

def define_model(args, model_path):
    # 001 classical image sr
    model = SwinIR(
        upscale=args.scale,
        in_chans=3,
        img_size=args.patch_size,
        window_size=8,
        img_range=1.,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='pixelshuffle',
        resi_connection='1conv')


    loadnet = torch.load(model_path)
    if 'params_ema' in loadnet:
        keyname = 'params_ema'
    else:
        keyname = 'params'
    model.load_state_dict(loadnet[keyname], strict=True)

    return model


if __name__ == '__main__':
    main()
