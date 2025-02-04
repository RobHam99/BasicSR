import argparse
import cv2
import glob
import os
import shutil
import torch
from tqdm import tqdm

from basicsr.archs.basicvsrpp_arch import BasicVSRPlusPlus
from basicsr.data.data_util import read_img_seq
from basicsr.utils.img_util import tensor2img
from yuv_utils import *


def inference(frames_tensor, model, save_path):
    with torch.no_grad():
        outputs = model(frames_tensor)
    # save imgs
    outputs = outputs.squeeze()
    outputs = outputs.permute(0, 2, 3, 1)
    outputs = outputs.cpu()
    rgb_to_yuv420p10bit(outputs, save_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='', help='input yuv video')
    parser.add_argument('--output', type=str, default='results/BasicVSRPP', help='save image path')
    parser.add_argument('--num_frames', type=int, default=60, help='Number of frames to process')
    parser.add_argument('--width', type=int, default=960, help='Width of the video')
    parser.add_argument('--height', type=int, default=544, help='Height of the video')
    parser.add_argument('--interval', type=int, default=15, help='interval size')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = '/home/sk24938/source/sr/BasicSR/experiments/pretrained_models/basic_vsr_pp/basicvsr_plusplus_reds4.pth'

    # set up model
    model = BasicVSRPlusPlus(mid_channels=64, num_blocks=7, spynet_path='/home/sk24938/source/sr/BasicSR/experiments/pretrained_models/spynet_20210409-c6c1bd09.pth')
    model.load_state_dict(torch.load(model_path, weights_only=True)['state_dict'], strict=True)
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
        pixel_format='yuv420p'
        )
    frames_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float()

    # load data and inference
    if args.num_frames <= args.interval:  # too many images may cause CUDA out of memory
        frames_tensor = frames_tensor.unsqueeze(0).to(device)
        inference(frames_tensor, model, args.output)
    else:
        for idx in tqdm(range(0, args.num_frames, args.interval), desc='BasicVSR++'):
            interval = min(args.interval, args.num_frames - idx)
            frames_tensor = frames_tensor.unsqueeze(0).to(device)
            inference(frames_tensor, model, args.output)


if __name__ == '__main__':
    main()
