import os

import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
import cv2 as cv
from mpl_toolkits.axes_grid1 import make_axes_locatable


if __name__ == "__main__":

    gdc_rectified_fn = r'C:\Users\ysimson\work\GDC7_FP_reference_tests\00001_out_Matlab.png'
    rs_rectified_fn = r'C:\Users\ysimson\work\data\realsense_images\Intel_RealSense_D465\20221117_054212\bin\Traces\5_Rectified\Left\CalibrationLeftIR0010.png'

    gdc_rectified_image = cv.imread(gdc_rectified_fn, cv.IMREAD_UNCHANGED)
    rs_rectified_image = cv.imread(rs_rectified_fn, cv.IMREAD_UNCHANGED)

    gdc_rectified_image = gdc_rectified_image[:, :, 0].astype(np.uint16) * 4
    rs_rectified_image = rs_rectified_image.astype(np.uint16) // 64

    fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
    axes[0, 0].imshow(gdc_rectified_image)
    axes[0, 0].set_title("GDC")

    axes[0, 1].imshow(rs_rectified_image)
    axes[0, 1].set_title("RS")

    err_image = gdc_rectified_image.astype(np.int16) - rs_rectified_image.astype(np.int16)
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


