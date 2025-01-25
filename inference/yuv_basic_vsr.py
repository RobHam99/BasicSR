import argparse
import cv2
import glob
import os
import shutil
import torch

from basicsr.archs.basicvsr_arch import BasicVSR
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
    print(outputs.shape)
    rgb_to_yuv420p10bit(outputs, save_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='experiments/pretrained_models/BasicVSR_REDS4.pth')
    parser.add_argument(
        '--input_path', type=str, default='datasets/REDS4/sharp_bicubic/000', help='input test image folder')
    parser.add_argument('--save_path', type=str, default='results/BasicVSR', help='save image path')
    parser.add_argument('--interval', type=int, default=15, help='interval size')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set up model
    model = BasicVSR(num_feat=64, num_block=30)
    model.load_state_dict(torch.load(args.model_path)['params'], strict=True)
    model.eval()
    model = model.to(device)

    # want to process yuv input frames
    # load yuv frames in rgb format
    # convert to tensor
    frames_np = load_yuv_frames(
        video_file_path=args.input_path,
        start_idx=0,
        num_frames=12,
        width=256,
        height=256,
        bit_depth=10,
        pixel_format='yuv420p'
        )
    print(frames_np.shape)
    frames_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float()
    print(frames_tensor.shape)
    # load data and inference
    num_frames = len(frames_tensor)
    if num_frames <= args.interval:  # too many images may cause CUDA out of memory
        frames_tensor = frames_tensor.unsqueeze(0).to(device)
        inference(frames_tensor, model, args.save_path)
    else:
        for idx in range(0, num_frames, args.interval):
            interval = min(args.interval, num_frames - idx)
            frames_tensor = frames_tensor.unsqueeze(0).to(device)
            inference(frames_tensor, model, args.save_path)


if __name__ == '__main__':
    main()
