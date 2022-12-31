import os

import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
import cv2 as cv
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial.transform import Rotation as Rot

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

        right_image_fn = images_list[0]
        right_image = cv.imread(right_image_fn, cv.IMREAD_UNCHANGED)
        # plt.figure()
        # plt.imshow(right_image)
        # plt.show()

        K = np.array([[focal_length[0], 0, principal_point[0]],
                      [0, focal_length[1], principal_point[1]],
                      [0, 0, 1]])
        pitches = np.array([0.0001, 0.001, 0.01, 0.1, 0.5])
        pitches = np.concatenate((-np.flip(pitches), [0], pitches))
        yaws = pitches.copy()
        yaw = 0
        roll = 0
        for pitch in pitches:
            for yaw in yaws:
                R = Rot.from_euler(seq='XYZ', angles=[pitch, yaw, roll], degrees=True).as_matrix()
                H = K * R * np.linalg.inv(K)
                height, width = right_image.shape[:2]
                img_warp = cv.warpPerspective(right_image, H, (width, height), borderMode=cv.BORDER_REPLICATE)
                fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 4))
                axes[0].imshow(right_image)
                axes[0].set_title("Rectified")
                axes[1].imshow(img_warp)
                axes[1].set_title("Warped")
                plt.tight_layout()
                fig.suptitle(f"y: {yaw:.04f}, p: {pitch:.04f}, r: {roll:.04f}", y=0.95)
                plt.close()

                output_folder = os.path.join(folder, suffix_out + f"_p{pitch:.04f}_y{yaw:.04f}")
                os.makedirs(output_folder, exist_ok=True)
                warped_output_fn = os.path.join(output_folder, os.path.basename(right_image_fn))
                print(f"Output: {warped_output_fn}")
                cv.imwrite(warped_output_fn, img_warp)

