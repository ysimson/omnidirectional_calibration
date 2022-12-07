import os

import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
import cv2 as cv
from mpl_toolkits.axes_grid1 import make_axes_locatable


if __name__ == "__main__":

    gdc_depth_fn = r'C:\Users\ysimson\work\data\realsense_images\Intel_RealSense_D465\20221117_054212\gdc\00001_out_Matlab.mat'
    rs_depth_fn = r"C:\Users\ysimson\work\data\realsense_images\Intel_RealSense_D465\20221117_054212\out\CalibrationLeftIR0010.mat"

    gdc_depth_image = io.loadmat(gdc_depth_fn)
    rs_depth_image = io.loadmat(rs_depth_fn)

    gdc_disp = gdc_depth_image['disp']
    rs_disp = rs_depth_image['disp']

    fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
    im = axes[0, 0].imshow(gdc_disp)
    axes[0, 0].set_title("GDC rectification -> disparity")
    divider = make_axes_locatable(axes[0, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    im = axes[0, 1].imshow(rs_disp)
    axes[0, 1].set_title("RS rectification -> disparity")
    divider = make_axes_locatable(axes[0, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    err_image = gdc_disp.astype(np.int16) - rs_disp.astype(np.int16)
    err_image_sub5 = err_image.copy()
    err_image_sub5[np.abs(err_image_sub5) >= 5] = 0
    im = axes[1, 0].imshow(err_image_sub5)
    axes[1, 0].set_title("|GDC - RS Error| < 5")
    divider = make_axes_locatable(axes[1, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    err_image_sup5 = err_image.copy()
    err_image_sup5[np.abs(err_image_sup5) < 5] = 0
    im = axes[1, 1].imshow(err_image_sup5)
    axes[1, 1].set_title("|GDC - RS Error| >= 5")
    divider = make_axes_locatable(axes[1, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()


