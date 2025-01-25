import os
import subprocess
import argparse
from tqdm import tqdm

def esrgan_dir(input_dir, output_dir, model, scaling_factor=4):
    files = os.listdir(input_dir)

    # Chose model and file identification letter
    if model == 'RealESRGAN':
        scale_dict = {2: ['RealESRGAN_x2plus.pth', 'B'], 4: ['RealESRGAN_x4plus.pth', 'C']}
    else:
        scale_dict = {2: ['DF2KOST_official-ff704c30.pth', 'B'], 4: ['DF2KOST_official-ff704c30.pth', 'C']}

    if scaling_factor == 2:
        width, height = 1920, 1088
    elif scaling_factor == 4:
        width, height = 960, 544

    output_dir = os.path.join(output_dir, model, 'x' + str(scaling_factor))

    for file in tqdm(files, desc='Processing'):

        if file[0] != scale_dict[scaling_factor][1]:
            continue

        input_file = os.path.join(input_dir, file)
        output_file = os.path.join(output_dir, file)
        cmd = [
            'python3', 'inference/esrgan_yuv.py',
            '--model_path', 'experiments/pretrained_models/esrgan/{0}'.format(scale_dict[scaling_factor][0]),
            '--input', input_file,
            '--output', output_file,
            '--num_frames', '64',
            '--width', str(width),
            '--height', str(height)
        ]

        subprocess.run(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='datasets/Set14/LRbicx4', help='input test image folder')
    parser.add_argument('--output', type=str, default='results/ESRGAN', help='output folder')
    parser.add_argument('--scaling_factor', type=int, default=4, help='scaling factor')
    parser.add_argument('--model', type=str, default='RealESRGAN', help='model to use')
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    esrgan_dir(args.input, args.output, args.model, args.scaling_factor)