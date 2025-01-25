import os
import subprocess
import argparse
from tqdm import tqdm


def swinir_dir(input_dir, output_dir, scaling_factor=4):
    files = os.listdir(input_dir)

    # Chose model and file identification letter

    if scaling_factor == 2:
        width, height = 1920, 1088
        letter = 'B'
        model = 'experiments/pretrained_models/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth'
    elif scaling_factor == 4:
        width, height = 960, 544
        letter = 'C'
        model = 'experiments/pretrained_models/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth'
    elif scaling_factor == 8:
        width, height = 480, 272
        letter = 'D'
        model = 'experiments/pretrained_models/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x8.pth'

    for file in tqdm(files, desc='Processing videos'):

        if file[0] != letter:
            continue

        input_file = os.path.join(input_dir, file)
        output_file = os.path.join(output_dir, file)
        tqdm.write('Processing {0}'.format(file))
        cmd = [
            'python3', 'inference/swin_yuv.py',
            '--model_path', model,
            '--input', input_file,
            '--output', output_file,
            '--num_frames', '64',
            '--width', str(width),
            '--height', str(height),
            '--scale', str(scaling_factor)
        ]

        subprocess.run(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='datasets/swinir.yuv', help='input test image folder')
    parser.add_argument('--output', type=str, default='results/swinir.yuv', help='output folder')
    parser.add_argument('--scaling_factor', type=int, default=4, help='scaling factor')
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    swinir_dir(args.input, args.output, args.scaling_factor)