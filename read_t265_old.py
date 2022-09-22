#!/usr/bin/python
# -*- coding: utf-8 -*-
## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2019 Intel Corporation. All Rights Reserved.
# Python 2/3 compatibility
from __future__ import print_function

"""
This example shows how to use T265 intrinsics and extrinsics in OpenCV to
asynchronously compute depth maps from T265 fisheye images on the host.

T265 is not a depth camera and the quality of passive-only depth options will
always be limited compared to (e.g.) the D4XX series cameras. However, T265 does
have two global shutter cameras in a stereo configuration, and in this example
we show how to set up OpenCV to undistort the images and compute stereo depth
from them.

Getting started with python3, OpenCV and T265 on Ubuntu 16.04:

First, set up the virtual enviroment:

$ apt-get install python3-venv  # install python3 built in venv support
$ python3 -m venv py3librs      # create a virtual environment in pylibrs
$ source py3librs/bin/activate  # activate the venv, do this from every terminal
$ pip install opencv-python     # install opencv 4.1 in the venv
$ pip install pyrealsense2      # install librealsense python bindings

Then, for every new terminal:

$ source py3librs/bin/activate  # Activate the virtual environment
$ python3 t265_stereo.py        # Run the example
"""

# First import the library
import pyrealsense2 as rs

# Import OpenCV and numpy
import cv2 as cv
import numpy as np
import os
import argparse
from time import gmtime, strftime

# Set up a mutex to share data between threads
from threading import Lock

frame_mutex = Lock()
frame_data = {"left": None,
              "right": None,
              "timestamp_ms": None
              }

"""
This callback is called on a separate thread, so we must use a mutex
to ensure that data is synchronized properly. We should also be
careful not to do much work on this thread to avoid data backing up in the
callback queue.
"""


def callback(frame):
    global frame_data
    if frame.is_frameset():
        frameset = frame.as_frameset()
        f1 = frameset.get_fisheye_frame(1).as_video_frame()
        f2 = frameset.get_fisheye_frame(2).as_video_frame()
        left_data = np.asanyarray(f1.get_data())
        right_data = np.asanyarray(f2.get_data())
        ts = frameset.get_timestamp()
        frame_mutex.acquire()
        frame_data["left"] = left_data
        frame_data["right"] = right_data
        frame_data["timestamp_ms"] = ts
        frame_mutex.release()


if __name__ == "__main__":
    # Create object for parsing command-line options
    parser = argparse.ArgumentParser(description="")
    # Add argument which takes path to a bag file as an input
    parser.add_argument("-i", "--input", type=str, help="Path to the bag file. If noe given read live stream")
    parser.add_argument("-b", "--bin_format", type=bool, default=False, help="Save image as bin")
    parser.add_argument("-d", "--downsample_rate", type=int, default=1, help="Down-sample output images")
    parser.add_argument("-o", "--output_folder", type=str, help="Path to the output images",
                        default=r"C:\Users\ysimson\work\data\realsense_images")

    # Parse the command line arguments to an object
    args = parser.parse_args()
    # Safety if no parameter have been given
    if not args.input:
        print("No input parameter have been given. Reading device live")
    # Check if the given file have bag extension
    elif os.path.splitext(args.input)[1] != ".bag":
        print("The given file is not of correct file format.")
        print("Only .bag files are accepted")

    base_output_folder = args.output_folder

    # Declare RealSense pipeline, encapsulating the actual device and sensors
    pipe = rs.pipeline()

    # Build config object and stream everything
    cfg = rs.config()
    ctx = rs.context()

    # Start streaming with our callback
    pipe.start(cfg, callback)

    try:
        # Set up an OpenCV window to visualize the results
        WINDOW_TITLE = 'Realsense'
        cv.namedWindow(WINDOW_TITLE, cv.WINDOW_NORMAL)

        # Retrieve the stream and intrinsic properties for both cameras
        profiles = pipe.get_active_profile()
        print(profiles.get_device())
        streams = {"left": profiles.get_stream(rs.stream.fisheye, 1).as_video_stream_profile(),
                   "right": profiles.get_stream(rs.stream.fisheye, 2).as_video_stream_profile()}
        intrinsics = {"left": streams["left"].get_intrinsics(),
                      "right": streams["right"].get_intrinsics()}

        # Print information about both cameras
        print("Left camera:", intrinsics["left"])
        print("Right camera:", intrinsics["right"])

        device = profiles.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        device_model = str(device.get_info(rs.camera_info.name))
        print(f"device_model: {device_model}")

        device_model = device_model.replace(" ", "_")
        timestamp = strftime("%Y%m%d_%H%M%S", gmtime())
        output_folder = os.path.join(base_output_folder, device_model, timestamp)
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)

        left_cam_folder = os.path.join(output_folder, 'cam0')
        right_cam_folder = os.path.join(output_folder, 'cam1')
        os.makedirs(left_cam_folder, exist_ok=True)
        os.makedirs(right_cam_folder, exist_ok=True)

        while True:
            # Check if the camera has acquired any frames
            frame_mutex.acquire()
            valid = frame_data["timestamp_ms"] is not None
            frame_mutex.release()

            # If frames are ready to process
            if valid:
                # Hold the mutex only long enough to copy the stereo frames
                frame_mutex.acquire()
                frame_copy = {"left": frame_data["left"].copy(),
                              "right": frame_data["right"].copy()}
                frame_mutex.release()
                cv.imshow(WINDOW_TITLE, np.hstack((frame_copy["left"], frame_copy["right"])))

            key = cv.waitKey(1)
            if key == 32 or cv.getWindowProperty(WINDOW_TITLE, cv.WND_PROP_VISIBLE) < 1:
                break

            if valid:
                frame_ts_ns = int(1e6 * frame_data["timestamp_ms"])

                left_filename = os.path.join(left_cam_folder, f'{frame_ts_ns}.png')
                right_filename = os.path.join(right_cam_folder, f'{frame_ts_ns}.png')
                if isinstance(frame_copy["left"][0, 0], np.uint8):
                    cv.imwrite(left_filename, frame_copy["left"])
                    cv.imwrite(right_filename, frame_copy["right"])
    finally:
        pipe.stop()
