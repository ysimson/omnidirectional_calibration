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


def convert_depth_to_display(disp: np.ndarray):
    """

    """
    disp_vis = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
    disp_vis = disp_vis.astype("uint8")
    disp_vis = cv.applyColorMap(disp_vis, cv.COLORMAP_INFERNO)

    return disp_vis


if __name__ == "__main__":

    movie_code = '20221117_054212'   # '20221031_144401'  # '20221102_105639' # '20221107_122055'  # '20221107_122044'  # '20221106_091011'  # '20221102_110324' 20221102_105639
    base_dir = os.path.join(r'C:\Users\ysimson\work\data\realsense_images\Intel_RealSense_D465', movie_code)

    output_dir = os.path.join(base_dir, 'compare')  # r'C:\Users\ysimson\work\data\realsense_images\Intel_RealSense_D465\20221102_110324\compare'
    deep_depth_dir = os.path.join(base_dir, 'out')  # r'C:\Users\ysimson\work\data\realsense_images\Intel_RealSense_D465\20221102_110324\out'
    classic_depth_dir = os.path.join(base_dir, r'bin\Traces\6_Depth')  # r'C:\Users\ysimson\work\data\realsense_images\Intel_RealSense_D465\20221102_110324\bin\Traces\6_Depth'
    rectified_left_dir = os.path.join(base_dir, r'bin\Traces\5_Rectified\Left')  # r'C:\Users\ysimson\work\data\realsense_images\Intel_RealSense_D465\20221102_110324\bin\Traces\5_Rectified\Left'
    rectified_right_dir = os.path.join(base_dir, r'bin\Traces\5_Rectified\Right')  # r'C:\Users\ysimson\work\data\realsense_images\Intel_RealSense_D465\20221102_110324\bin\Traces\5_Rectified\Right'

    write_video = True
    show_depth = True
    focus_on = False
    rectangle = False
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
    if write_video:
        os.makedirs(output_dir, exist_ok=True)
        video_name = os.path.join(output_dir, 'video.mp4')
        video = cv.VideoWriter(video_name, cv.VideoWriter_fourcc(*'mp4v'), 15, (1280, 720))

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
            depth_dict = {"classic": depth}
            depth = (baseline * focal_length) / deep_disparity  # <-- deep
            depth_dict["deep"] = depth

            # invalid = depth_dict["classic"] > 2000
            # depth_dict["classic"][invalid] = np.nan

            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            im = axes[0, 0].imshow(depth_dict["classic"])
            divider = make_axes_locatable(axes[0, 0])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            axes[0, 0].set_title('Classic')

            x1 = 585
            x2 = 620
            y1 = 330
            y2 = 370
            # Create a Rectangle patch, add the patch to the Axes
            if rectangle:
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
                axes[0, 0].add_patch(rect)

            im = axes[1, 0].imshow(depth_dict["classic"][y1:y2, x1:x2], cmap='inferno')
            divider = make_axes_locatable(axes[1, 0])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)

            # Deep
            im = axes[0, 1].imshow(depth_dict["deep"])
            divider = make_axes_locatable(axes[0, 1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            axes[0, 1].set_title('Deep')

            # Create a Rectangle patch, add the patch to the Axes
            if rectangle:
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
                axes[0, 1].add_patch(rect)

            im = axes[1, 1].imshow(depth_dict["deep"][y1:y2, x1:x2], cmap='inferno')
            divider = make_axes_locatable(axes[1, 1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"depth_debug_{idx:04d}.png"))

            depth_dict["classic"][np.isnan(depth_dict["classic"])] = 0.
            cv.imwrite(os.path.join(output_dir, f"classic_depth_{idx:04d}.tiff"), (depth_dict["classic"]).astype(np.uint16))
            cv.imwrite(os.path.join(output_dir, f"deep_depth_{idx:04d}.tiff"), (depth_dict["deep"]).astype(np.uint16))

            disp_vis = convert_depth_to_display(deep_disparity)

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
        plt.savefig(os.path.join(output_dir, f'disparity_{idx:04d}.png'))

        if focus_on and show_depth:
            x1 = 570
            x2 = 740
            y1 = 390
            y2 = 470
            fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(16, 8))
            im0 = axes[0, 0].imshow(depth_dict["classic"][y1:y2, x1:x2])
            axes[0, 0].set_title("Depth classic")
            plt.colorbar(im0, ax=axes[0, 0])

            im1 = axes[0, 1].imshow(depth_dict["deep"][y1:y2, x1:x2])
            axes[0, 1].set_title("Depth deep")
            plt.colorbar(im1, ax=axes[0, 1])

            im2 = axes[1, 0].imshow(rectified_left_img[y1:y2, x1:x2], cmap='gray')
            axes[1, 0].set_title("Rectified left")

            depth_diff = depth_dict["deep"] - depth_dict["classic"]
            im3 = axes[1, 1].imshow(depth_diff[y1:y2, x1:x2])
            axes[1, 1].set_title("deep depth - classic deep")

            plt.colorbar(im3, ax=axes[1, 1])
            plt.tight_layout()

            plt.savefig(os.path.join(output_dir, f'focus_depth_{idx:04d}.png'))

        canvas = FigureCanvas(fig)
        canvas.draw()
        mat = np.array(canvas.renderer._renderer)
        mat = cv.cvtColor(mat, cv.COLOR_RGB2BGR)

        # write frame to video
        if write_video:
            video.write(disp_vis)   #(mat)
        plt.close("all")

    if write_video:
        video.release()
    plt.show()
    print("Done")
