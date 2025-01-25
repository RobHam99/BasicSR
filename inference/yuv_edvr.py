import argparse
import cv2
import glob
import os
import shutil
import torch
from tqdm import tqdm

from basicsr.archs.edvr_arch import EDVR
from yuv_utils import *


def inference(frames_tensor, model, save_path):
    with torch.no_grad():
        outputs = model(frames_tensor)
    # save imgs
    outputs = outputs.permute(0, 2, 3, 1).squeeze()
    outputs = outputs.cpu()
    return outputs


def pad_frames(frames_tensor, pad_size):
    # Pad the frames by repeating the first and last frames
    start_padding = frames_tensor[:, :1, :, :, :].expand(-1, pad_size, -1, -1, -1)
    end_padding = frames_tensor[:, -1:, :, :, :].expand(-1, pad_size, -1, -1, -1)
    padded_frames = torch.cat([start_padding, frames_tensor, end_padding], dim=1)
    return padded_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
    '--input', type=str, default='datasets/REDS4/sharp_bicubic/000', help='input test image folder')
    parser.add_argument('--output', type=str, default='results/edsr/test.yuv', help='save image path')
    parser.add_argument('--num_frames', type=int, default=60, help='Number of frames to process')
    parser.add_argument('--width', type=int, default=960, help='Width of the video')
    parser.add_argument('--height', type=int, default=544, help='Height of the video')
    parser.add_argument('--interval', type=int, default=5, help='Interval size for processing frames in chunks')

    args = parser.parse_args()

    model_path = '/home/sk24938/source/sr/BasicSR/experiments/pretrained_models/edvr/EDVR_M_x4_SR_REDS_official-32075921.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set up model
    model = EDVR(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_frame=args.interval,
        deformable_groups=8,
        num_extract_block=5,
        num_reconstruct_block=10,
        hr_in=False,
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
    frames_tensor = frames_tensor.unsqueeze(0)
    pad_size = args.interval // 2
    frames_tensor = pad_frames(frames_tensor, pad_size)

    frame_list = []
    for idx in tqdm(range(0 + pad_size, args.num_frames + pad_size, 1), desc='Processing frames'):
        start_idx = idx - pad_size
        end_idx = idx + pad_size
        frames_tensor_chunk = frames_tensor[:, start_idx:end_idx+1, :, :, :].to(device)
        output = inference(frames_tensor_chunk, model, args.output)
        frame_list.append(output)
        torch.cuda.empty_cache()

    print('Saving video...')
    video = torch.stack(frame_list, dim=0)
    print('Upsampled video shape: ', video.shape)
    rgb_to_yuv420p10bit(video, args.output)
    print('Done!')


if __name__ == '__main__':
    main()
