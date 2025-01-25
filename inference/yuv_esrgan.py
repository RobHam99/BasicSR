import argparse
import cv2
import glob
import numpy as np
import os
import torch

from basicsr.archs.rrdbnet_arch import RRDBNet
from yuv_utils import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        default=  # noqa: E251
        'experiments/pretrained_models/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth'  # noqa: E501
    )
    parser.add_argument('--input', type=str, default='datasets/Set14/LRbicx4', help='input test image folder')
    parser.add_argument('--output', type=str, default='results/ESRGAN', help='output folder')
    parser.add_argument('--num_frames', type=int, default=64, help='number of frames to process')
    parser.add_argument('--width', type=int, default=960, help='width of the frames')
    parser.add_argument('--height', type=int, default=544, help='height of the frames')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    checkpoint = torch.load(args.model_path)

    if args.model_path == 'experiments/pretrained_models/esrgan/RealESRGAN_x4plus.pth' or args.model_path == 'experiments/pretrained_models/esrgan/RealESRGAN_x2plus.pth':
        param_key = 'params_ema'
    else:
        param_key = 'params'

    model.load_state_dict(torch.load(args.model_path)[param_key], strict=True)
    model.eval()
    model = model.to(device)

    frames_np = load_yuv_frames(
        video_file_path=args.input,
        start_idx=0,
        num_frames=args.num_frames,
        width=args.width,
        height=args.height,
        bit_depth=10,
        pixel_format='yuv420p'
        )
    frames_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float()
    frames_tensor = frames_tensor.unsqueeze(0).to(device)

    frame_list = []
    for i in range(frames_tensor.shape[1]):
        frame = frames_tensor[:, i]
        # inference
        try:
            with torch.no_grad():
                output = model(frame)
        except Exception as error:
            print('Error', error, i)
        else:
            # save image
            output = output.data.squeeze().cpu()
            frame_list.append(output)

    print('Saving video...')
    video = torch.stack(frame_list, dim=0)
    video = video.permute(0, 2, 3, 1)
    print('Upsampled video shape: ', video.shape)
    rgb_to_yuv420p10bit(video, args.output)
    print('Done!')


if __name__ == '__main__':
    main()
