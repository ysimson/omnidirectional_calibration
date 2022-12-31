import os

import numpy as np
from shutil import copy

# Create the warped images only for the right rectified image

if __name__ == "__main__":
    image_size = (1280, 720)
    principal_point = (611.3747, 366.7739)
    focal_length = (346.9404, 346.9404)
    # folders = [r'C:\Users\ysimson\work\data\realsense_images\Intel_RealSense_D465\20221002_141742',
    #            r'C:\Users\ysimson\work\data\realsense_images\Intel_RealSense_D465\20221002_160154',
    #            r'C:\Users\ysimson\work\data\realsense_images\Intel_RealSense_D465\20221002_172012',
    #            r'C:\Users\ysimson\work\data\realsense_images\Intel_RealSense_D465\20221003_062631',
    #            r'C:\Users\ysimson\work\data\realsense_images\Intel_RealSense_D465\20221031_144401',
    #            r'C:\Users\ysimson\work\data\realsense_images\Intel_RealSense_D465\20221102_105639',
    #            r'C:\Users\ysimson\work\data\realsense_images\Intel_RealSense_D465\20221102_110324',
    #            r'C:\Users\ysimson\work\data\realsense_images\Intel_RealSense_D465\20221106_091011',
    #            r'C:\Users\ysimson\work\data\realsense_images\Intel_RealSense_D465\20221107_122044',
    #            r'C:\Users\ysimson\work\data\realsense_images\Intel_RealSense_D465\20221107_122055',
    #            r'C:\Users\ysimson\work\data\realsense_images\Intel_RealSense_D465\20221117_052048',
    #            r'C:\Users\ysimson\work\data\realsense_images\Intel_RealSense_D465\20221117_054212']

    remote_folder = r"Z:\data\realsense_images"
    folders = [r'C:\Users\ysimson\work\data\realsense_images\Intel_RealSense_D465\20221031_144401',
               r'C:\Users\ysimson\work\data\realsense_images\Intel_RealSense_D465\20221102_105639',
               r'C:\Users\ysimson\work\data\realsense_images\Intel_RealSense_D465\20221102_110324',
               r'C:\Users\ysimson\work\data\realsense_images\Intel_RealSense_D465\20221106_091011',
               r'C:\Users\ysimson\work\data\realsense_images\Intel_RealSense_D465\20221107_122044',
               r'C:\Users\ysimson\work\data\realsense_images\Intel_RealSense_D465\20221107_122055',
               r'C:\Users\ysimson\work\data\realsense_images\Intel_RealSense_D465\20221117_052048',
               r'C:\Users\ysimson\work\data\realsense_images\Intel_RealSense_D465\20221117_054212']

    for folder in folders:
        suffix = r'bin\Traces\5_Rectified\Right'
        suffix_out = r'bin\Traces\5_RectifiedWarped\Right'

        image_path = os.path.join(folder, suffix)
        images_list = [os.path.join(image_path, x) for x in os.listdir(image_path) if x.endswith('.png')]
        test_name = os.path.basename(folder)
        right_image_fn = images_list[0]

        pitches = np.array([0.0001, 0.001, 0.01, 0.1, 0.5])
        pitches = np.concatenate((-np.flip(pitches), [0], pitches))
        yaws = pitches.copy()
        roll = 0
        for pitch in pitches:
            for yaw in yaws:
                output_folder = os.path.join(folder, suffix_out + f"_p{pitch:.04f}_y{yaw:.04f}")
                warped_output_fn = os.path.join(output_folder, os.path.basename(right_image_fn))
                print(f"Warped file: {warped_output_fn}")

                output_dest_folder = os.path.join(remote_folder, test_name, suffix_out + f"_p{pitch:.04f}_y{yaw:.04f}")
                print(f"Output dest: {output_dest_folder}")
                os.makedirs(output_dest_folder, exist_ok=True)
                copy(warped_output_fn, output_dest_folder)
