import os

import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
import cv2 as cv
from mpl_toolkits.axes_grid1 import make_axes_locatable


if __name__ == "__main__":

    gdc_depth_fn = r"C:\Users\ysimson\work\data\realsense_images\Intel_RealSense_D465\20221117_054212\bin\Traces\6_Depth\CalibrationLeftIR0010_depth_conf.mat"
    rs_depth_fn = r"C:\Users\ysimson\work\data\realsense_images\Intel_RealSense_D465\20221117_054212\bin\Traces\6_Depth\CalibrationLeftIR0010_depth_conf_reference.mat"

    gdc_depth_image = io.loadmat(gdc_depth_fn)
    rs_depth_image = io.loadmat(rs_depth_fn)

    gdc_disp = gdc_depth_image['disparity']
    rs_disp = rs_depth_image['disparity']

    gdc_conf_name = r"C:\Users\ysimson\work\data\realsense_images\Intel_RealSense_D465\20221117_054212\bin\Traces\6_Depth\CalibrationLeftIR0010_conf.bmp"
    rs_conf_name = r"C:\Users\ysimson\work\data\realsense_images\Intel_RealSense_D465\20221117_054212\bin\Traces\6_Depth\CalibrationLeftIR0010_conf_reference.bmp"
    gdc_conf = cv.imread(gdc_conf_name, cv.IMREAD_UNCHANGED)
    rs_conf = cv.imread(rs_conf_name, cv.IMREAD_UNCHANGED)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
    im = axes[0, 0].imshow(gdc_disp)
    axes[0, 0].set_title("GDC rectification -> SCP disparity")
    divider = make_axes_locatable(axes[0, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    im = axes[0, 1].imshow(rs_disp)
    axes[0, 1].set_title("RS rectification -> SCP disparity")
    divider = make_axes_locatable(axes[0, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    im = axes[1, 0].imshow(gdc_conf)
    axes[1, 0].set_title("GDC confidence")

    im = axes[1, 1].imshow(rs_conf)
    axes[1, 1].set_title("RS confidence")

    mutual_conf = np.zeros(gdc_conf.shape, np.uint8)
    mutual_conf[np.logical_and(gdc_conf == 255, rs_conf == 255)] = 255

    err_image = gdc_disp - rs_disp
    err_image[mutual_conf == 0] = np.nan
    fig, axes = plt.subplots(1, 1, figsize=(8, 6), sharex=True, sharey=True)
    im = axes.imshow(err_image)
    axes.set_title("Error in confidence area")
    divider = make_axes_locatable(axes)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    plt.show()


