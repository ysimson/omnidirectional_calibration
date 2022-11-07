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
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches


if __name__ == "__main__":

    base_dir = r'C:\Users\ysimson\work\data\oskar_data\sid_2022-08-09--17-22-24'

    output_dir = os.path.join(base_dir, 'compare')
    deep_depth_dir = os.path.join(base_dir, 'deep')
    classic_depth_dir = os.path.join(base_dir, r'DS5D580DepthWhiteWall\Traces\6_Depth')
    rectified_left_dir = os.path.join(base_dir, r'DS5D580DepthWhiteWall\Traces\5_Rectified\Left')
    rectified_right_dir = os.path.join(base_dir, r'DS5D580DepthWhiteWall\Traces\5_Rectified\Right')

    write_video = False
    show_depth = True
    # open and display classic depth
    classic_depth_files = [os.path.join(classic_depth_dir, x) for x in os.listdir(classic_depth_dir) if '.mat' in x]
    deep_depth_files = [os.path.join(deep_depth_dir, x) for x in os.listdir(deep_depth_dir) if '.mat' in x]
    left_rectified_files = [os.path.join(rectified_left_dir, x) for x in os.listdir(rectified_left_dir) if '.png' in x]
    right_rectified_files = [os.path.join(rectified_left_dir, x) for x in os.listdir(rectified_left_dir) if '.png' in x]

    classic_fn_dict = {}
    for fn in classic_depth_files:
        m = re.search('\S+IR_(\d+).*_depth_conf.mat', fn)
        classic_fn_dict[int(m.group(1))] = fn

    deep_fn_dict = {}
    for fn in deep_depth_files:
        m = re.search('\S+IR_(\d+)_.*.mat', fn)
        deep_fn_dict[int(m.group(1))] = fn

    rectified_left_dict = {}
    for fn in left_rectified_files:
        m = re.search('\S+IR_(\d+)_.*.png', fn)
        rectified_left_dict[int(m.group(1))] = fn

    rectified_right_dict = {}
    for fn in right_rectified_files:
        m = re.search('\S+IR_(\d+)_.*.png', fn)
        rectified_right_dict[int(m.group(1))] = fn

    common_keys = list(set(deep_fn_dict.keys()) & set(classic_fn_dict.keys()) & set(rectified_left_dict.keys()))
    common_keys.sort()

    os.makedirs(output_dir, exist_ok=True)

    for idx in common_keys:
        print(idx)
        results_dict = io.loadmat(classic_fn_dict[idx])
        disparity = results_dict["disparity"]
        confidence_depth = results_dict["confidence_depth"]
        baseline = results_dict["baseline"]
        focal_length = results_dict["focal_length"]

        results_deep_dict = io.loadmat(deep_fn_dict[idx])
        deep_disparity = results_deep_dict["disp"]

        if show_depth:
            disparity_classic = disparity.copy()
            disparity_classic[disparity_classic == 0] = np.nan
            depth = (baseline * focal_length) / disparity_classic
            depth[confidence_depth < 1] = np.nan
            depth[depth > 10000] = np.nan
            # depth = (baseline * focal_length) / deep_disparity  # <-- deep
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            im = axes[0].imshow(depth)
            axes[1].axis('equal')
            divider = make_axes_locatable(axes[0])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)

            x1 = 300
            x2 = 900
            y1 = 200
            y2 = 600
            # Create a Rectangle patch, add the patch to the Axes
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
            axes[0].add_patch(rect)

            im = axes[1].imshow(depth[y1:y2, x1:x2], cmap='inferno')
            axes[1].axis('equal')
            divider = make_axes_locatable(axes[1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            plt.tight_layout()

            rms = np.sqrt(np.nanmean((idx - depth[y1:y2, x1:x2]) ** 2))
            median = np.nanmedian(idx - depth[y1:y2, x1:x2])
            std = np.nanstd(idx - depth[y1:y2, x1:x2])
            print(f'rms: {rms:.02f}cm, median: {median:.02f}cm, std: {std:.02f}cm')

        rectified_left_img = cv.imread(rectified_left_dict[idx], cv.IMREAD_UNCHANGED)

        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(16, 8))
        im0 = axes[0, 0].imshow(disparity)
        axes[0, 0].set_title("Disparity classic")
        plt.colorbar(im0, ax=axes[0, 0])

        im1 = axes[0, 1].imshow(deep_disparity)
        axes[0, 1].set_title("Disparity deep")
        plt.colorbar(im1, ax=axes[0, 1])

        im2 = axes[1, 0].imshow(rectified_left_img, cmap='gray')
        axes[1, 0].set_title("Rectified left")

        disparity_diff = deep_disparity - disparity
        disparity_diff[confidence_depth < 1] = np.nan
        im3 = axes[1, 1].imshow(disparity_diff)
        axes[1, 1].set_title("deep_disparity - classic disparity")
        plt.colorbar(im3, ax=axes[1, 1])

        plt.tight_layout()

        plt.savefig(os.path.join(output_dir, f'debug_{idx:04d}.png'))

        canvas = FigureCanvas(fig)
        canvas.draw()
        mat = np.array(canvas.renderer._renderer)
        mat = cv.cvtColor(mat, cv.COLOR_RGB2BGR)

        plt.close()

    plt.show()
    print("Done")
