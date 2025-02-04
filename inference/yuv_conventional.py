from PIL import Image
from yuv_utils import *
import cv2
import argparse
from tqdm import tqdm


def write_yuv_file(y, u, v, output_file_path):
    """
    Writes YUV frames to a .yuv file in 4:2:0 format.
    """
    with open(output_file_path, "wb") as f:
        for i in range(len(y)):
            f.write(y[i].tobytes())
            f.write(u[i].tobytes())
            f.write(v[i].tobytes())


def rescale_frame(frame, scaling_factor=2, bit_depth=10, method='bicubic'):
    interpolation_methods = {
        'nearest': cv2.INTER_NEAREST,
        'bicubic': cv2.INTER_CUBIC
    }

    if method not in interpolation_methods:
        raise ValueError("Invalid method. Choose 'nearest' or 'bicubic'.")
    interpolation = interpolation_methods[method]
    height, width = frame.shape[:2]
    # Convert to uint16 for 10-bit data if necessary
    frame = np.clip(frame, 0, (2 ** bit_depth) - 1)

    # Resize Y (luma) channel
    Y = frame[:, :, 0]
    Y_resized = cv2.resize(
        Y.astype(np.uint16),  # Ensure correct type (uint16 for 10-bit)
        (int(width * scaling_factor), int(height * scaling_factor)),
        interpolation=cv2.INTER_CUBIC
    )

    # Resize U and V (chroma) channels
    U = frame[:, :, 1]
    V = frame[:, :, 2]
    U_resized = cv2.resize(
        U.astype(np.uint16),
        (int(width * scaling_factor / 2), int(height * scaling_factor / 2)),
        interpolation=interpolation
    )
    V_resized = cv2.resize(
        V.astype(np.uint16),
        (int(width * scaling_factor / 2), int(height * scaling_factor / 2)),
        interpolation=interpolation
    )

    # Clip resized channels to keep values within 10-bit range
    Y_resized = np.clip(Y_resized, 0, (2 ** bit_depth) - 1)
    U_resized = np.clip(U_resized, 0, (2 ** bit_depth) - 1)
    V_resized = np.clip(V_resized, 0, (2 ** bit_depth) - 1)

    return Y_resized, U_resized, V_resized


def main():
    arg_parser = argparse.ArgumentParser(description='Rescale YUV video')
    arg_parser.add_argument('--input', help='Path to the YUV video file')
    arg_parser.add_argument('--output', help='Path to the output YUV video file')
    arg_parser.add_argument('--num_frames', type=int, default=60, help='Number of frames to process')
    arg_parser.add_argument('--width', type=int, default=3840, help='Width of the video')
    arg_parser.add_argument('--height', type=int, default=2176, help='Height of the video')
    arg_parser.add_argument('--scale', type=int, default=4, help='Scaling factor')
    arg_parser.add_argument('--method', type=str, default='bicubic', help='Interpolation method')

    args = arg_parser.parse_args()

    # Load the YUV frame
    yuv_frame = load_yuv_frames(
        video_file_path=args.input,
        start_idx=0,
        num_frames=args.num_frames,
        width=args.width,
        height=args.height,
        bit_depth=10,
        pixel_format='yuv420p',
        convert_to_rgb=False
        )

    # Rescale the YUV frame
    y_arr = []
    u_arr = []
    v_arr = []
    for i, frame in tqdm(enumerate(yuv_frame), desc=f'{args.method}'):
        y, u, v = rescale_frame(
            frame=frame,
            scaling_factor=args.scale,
            bit_depth=10,
            method=args.method
            )
        y_arr.append(y)
        u_arr.append(u)
        v_arr.append(v)

    # Write the rescaled YUV video to a file
    write_yuv_file(y_arr, u_arr, v_arr, args.output)


if __name__ == '__main__':
    main()