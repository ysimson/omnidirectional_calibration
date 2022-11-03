import os

import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import re
import cv2 as cv
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# Use Agg backend for canvas
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


if __name__ == "__main__":

    # deep_depth_dir = r'C:\Users\ysimson\work\data\oskar_data\sid_2022-08-09--17-22-24\DS5D580DepthWhiteWall\Traces\deep'
    # classic_depth_dir = r'C:\Users\ysimson\work\data\oskar_data\sid_2022-08-09--17-22-24\DS5D580DepthWhiteWall\Traces\6_Depth'

    deep_depth_dir = r'C:\Users\ysimson\work\data\realsense_images\Intel_RealSense_D465\20221102_110324\out'
    classic_depth_dir = r'C:\Users\ysimson\work\data\realsense_images\Intel_RealSense_D465\20221102_110324\bin\Traces\6_Depth'
    rectified_left_dir = r'C:\Users\ysimson\work\data\realsense_images\Intel_RealSense_D465\20221102_110324\bin\Traces\5_Rectified\Left'
    rectified_right_dir = r'C:\Users\ysimson\work\data\realsense_images\Intel_RealSense_D465\20221102_110324\bin\Traces\5_Rectified\Right'

    idx = 1
    # open and display classic depth
    classic_depth_files = [os.path.join(classic_depth_dir, x) for x in os.listdir(classic_depth_dir) if '.mat' in x]
    deep_depth_files = [os.path.join(deep_depth_dir, x) for x in os.listdir(deep_depth_dir) if '.mat' in x]
    left_rectified_files = [os.path.join(rectified_left_dir, x) for x in os.listdir(rectified_left_dir) if '.png' in x]
    right_rectified_files = [os.path.join(rectified_left_dir, x) for x in os.listdir(rectified_left_dir) if '.png' in x]

    classic_fn_dict = {}
    for fn in classic_depth_files:
        m = re.search('\S+IR(\d+)_depth_conf.mat', fn)
        classic_fn_dict[int(m.group(1))] = fn

    deep_fn_dict = {}
    for fn in deep_depth_files:
        m = re.search('\S+IR(\d+).mat', fn)
        deep_fn_dict[int(m.group(1))] = fn

    rectified_left_dict = {}
    for fn in left_rectified_files:
        m = re.search('\S+IR(\d+).png', fn)
        rectified_left_dict[int(m.group(1))] = fn

    rectified_right_dict = {}
    for fn in right_rectified_files:
        m = re.search('\S+IR(\d+).png', fn)
        rectified_right_dict[int(m.group(1))] = fn

    common_keys = list(set(deep_fn_dict.keys()) & set(classic_fn_dict.keys()) & set(rectified_left_dict.keys()))
    common_keys.sort()
    video_name = os.path.join(r'C:\Users\ysimson\work\data\realsense_images\Intel_RealSense_D465\20221102_110324', 'video.mp4')
    video = cv.VideoWriter(video_name, cv.VideoWriter_fourcc(*'mp4v'), 15, (1600, 800))

    for idx in common_keys:

        results_dict = io.loadmat(classic_fn_dict[idx])
        disparity = results_dict["disparity"]
        confidence_depth = results_dict["confidence_depth"]
        baseline = results_dict["baseline"]
        focal_length = results_dict["focal_length"]

        # plt.figure()
        # plt.imshow(disparity)

        results_deep_dict = io.loadmat(deep_fn_dict[idx])
        deep_disparity = results_deep_dict["disp"]
        # plt.figure()
        # plt.imshow(deep_disparity)
        rectified_left_img = cv.imread(rectified_left_dict[idx], cv.IMREAD_UNCHANGED)

        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(16, 8))
        im0 = axes[0, 0].imshow(disparity)
        axes[0, 0].set_title("Disparity classic")
        plt.colorbar(im0, ax=axes[0, 0])

        im1 = axes[0, 1].imshow(deep_disparity)
        axes[0, 1].set_title("Disparity deep")
        plt.colorbar(im1, ax=axes[0, 1])

        im2 = axes[1, 0].imshow(rectified_left_img)
        axes[1, 0].set_title("Rectified left")

        disparity_diff = deep_disparity - disparity
        disparity_diff[confidence_depth < 1] = np.nan
        im3 = axes[1, 1].imshow(disparity_diff)
        axes[1, 1].set_title("deep_disparity - classic disparity")
        plt.colorbar(im3, ax=axes[1, 1])

        plt.tight_layout()

        canvas = FigureCanvas(fig)
        canvas.draw()
        mat = np.array(canvas.renderer._renderer)
        mat = cv.cvtColor(mat, cv.COLOR_RGB2BGR)

        # write frame to video
        video.write(mat)
        plt.close()

    video.release()
    plt.show()
    print("Done")
