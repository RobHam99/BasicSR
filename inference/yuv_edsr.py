import argparse
import cv2
import glob
import os
import shutil
import torch
from tqdm import tqdm

from basicsr.archs.edsr_arch import EDSR
from yuv_utils import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
    '--input', type=str, default='datasets/REDS4/sharp_bicubic/000', help='input test image folder')
    parser.add_argument('--output', type=str, default='results/edsr/test.yuv', help='save image path')
    parser.add_argument('--num_frames', type=int, default=60, help='Number of frames to process')
    parser.add_argument('--width', type=int, default=960, help='Width of the video')
    parser.add_argument('--height', type=int, default=544, help='Height of the video')
    parser.add_argument('--scale', type=int, default=4, help='Scaling factor')

    args = parser.parse_args()

    if args.scale == 4:
        model_path = '/home/sk24938/source/sr/BasicSR/experiments/pretrained_models/edsr/EDSR_Lx4_f256b32_DIV2K_official-76ee1c8f.pth'
    elif args.scale == 2:
        model_path = '/home/sk24938/source/sr/BasicSR/experiments/pretrained_models/edsr/EDSR_Lx2_f256b32_DIV2K_official-be38e77d.pth'
    else:
        raise ValueError('Scale factor not supported')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set up model
    model = EDSR(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=256,
        num_block=32,
        upscale=args.scale,
        res_scale=0.1,
        img_range=1.0,
        rgb_mean=(0.4488, 0.4371, 0.4040)
    )
    model.load_state_dict(torch.load(model_path)['params'], strict=True)
    model.eval()
    model = model.to(device)

    # want to process yuv input frames
    # load yuv frames in rgb format
    # convert to tensor
    frames_np = load_yuv_frames(
        video_file_path=args.input,
        start_idx=0,
        num_frames=args.num_frames,
        width=args.width,
        height=args.height,
        bit_depth=10,
        pixel_format='yuv420p',
        convert_to_rgb=True
        )

    frames_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float()
    frames_tensor = frames_tensor.to(device)

    frame_list = []
    for i in tqdm(range(frames_tensor.shape[0]), desc='Processing frames'):
        frame = frames_tensor[i, :, :, :]
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
