#####################################################
##               Read bag from file                ##
#####################################################


# First import library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
# Import argparse for command-line options
import argparse
# Import os.path for file path manipulation
import os.path
import math


def get_extrinsics(src, dst):
    """
    Returns R, T transform from src to dst
    """
    extrinsics = src.get_extrinsics_to(dst)
    R = np.reshape(extrinsics.rotation, [3, 3]).T
    T = np.array(extrinsics.translation)
    return (R, T)


def camera_matrix(intrinsics):
    """
    Returns a camera matrix K from librealsense intrinsics
    """
    return np.array([[intrinsics.fx, 0, intrinsics.ppx],
                     [0, intrinsics.fy, intrinsics.ppy],
                     [0, 0, 1]])


def fisheye_distortion(intrinsics):
    """
    Returns the fisheye distortion from librealsense intrinsics
    """
    return np.array(intrinsics.coeffs[:4])


if __name__ == "__main__":
    # Create object for parsing command-line options
    parser = argparse.ArgumentParser(description="Read recorded bag file and display depth stream in jet colormap.\
                                    Remember to change the stream fps and format to match the recorded.")
    # Add argument which takes path to a bag file as an input
    parser.add_argument("-i", "--input", type=str, help="Path to the bag file")
    # Parse the command line arguments to an object
    args = parser.parse_args()
    # Safety if no parameter have been given
    if not args.input:
        print("No input paramater have been given.")
        print("For help type --help")
        exit()
    # Check if the given file have bag extension
    if os.path.splitext(args.input)[1] != ".bag":
        print("The given file is not of correct file format.")
        print("Only .bag files are accepted")
        exit()

    # Configure the OpenCV stereo algorithm. See
    # https://docs.opencv.org/3.4/d2/d85/classcv_1_1StereoSGBM.html for a
    # description of the parameters
    window_size = 5
    min_disp = 0
    # must be divisible by 16
    num_disp = 112 - min_disp
    max_disp = min_disp + num_disp
    stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                   numDisparities=num_disp,
                                   blockSize=16,
                                   P1=8 * 3 * window_size ** 2,
                                   P2=32 * 3 * window_size ** 2,
                                   disp12MaxDiff=1,
                                   uniquenessRatio=10,
                                   speckleWindowSize=100,
                                   speckleRange=32)

    try:
        # Create pipeline
        pipeline = rs.pipeline()

        # Create a config object
        config = rs.config()

        # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
        rs.config.enable_device_from_file(config, args.input)

        config.enable_stream(rs.stream.pose)
        config.enable_stream(rs.stream.fisheye, 1)
        config.enable_stream(rs.stream.fisheye, 2)
        # Start streaming from file
        pipeline.start(config)

        WINDOW_TITLE = 'Realsense'
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)

        profiles = pipeline.get_active_profile()
        streams = {"left": profiles.get_stream(rs.stream.fisheye, 1).as_video_stream_profile(),
                   "right": profiles.get_stream(rs.stream.fisheye, 2).as_video_stream_profile()}
        intrinsics = {"left": streams["left"].get_intrinsics(),
                      "right": streams["right"].get_intrinsics()}

        # Translate the intrinsics from librealsense into OpenCV
        K_left = camera_matrix(intrinsics["left"])
        D_left = fisheye_distortion(intrinsics["left"])
        K_right = camera_matrix(intrinsics["right"])
        D_right = fisheye_distortion(intrinsics["right"])
        (width, height) = (intrinsics["left"].width, intrinsics["left"].height)

        # Get the relative extrinsics between the left and right camera
        (R, T) = get_extrinsics(streams["left"], streams["right"])

        # We need to determine what focal length our undistorted images should have
        # in order to set up the camera matrices for initUndistortRectifyMap.  We
        # could use stereoRectify, but here we show how to derive these projection
        # matrices from the calibration and a desired height and field of view

        # We calculate the undistorted focal length:
        #
        #         h
        # -----------------
        #  \      |      /
        #    \    | f  /
        #     \   |   /
        #      \ fov /
        #        \|/
        stereo_fov_rad = 90 * (math.pi / 180)  # 90 degree desired fov
        stereo_height_px = 300  # 300x300 pixel stereo output
        stereo_focal_px = stereo_height_px / 2 / math.tan(stereo_fov_rad / 2)

        # We set the left rotation to identity and the right rotation
        # the rotation between the cameras
        R_left = np.eye(3)
        R_right = R

        # The stereo algorithm needs max_disp extra pixels in order to produce valid
        # disparity on the desired output region. This changes the width, but the
        # center of projection should be in the center of the cropped image
        stereo_width_px = stereo_height_px + max_disp
        stereo_size = (stereo_width_px, stereo_height_px)
        stereo_cx = (stereo_height_px - 1) / 2 + max_disp
        stereo_cy = (stereo_height_px - 1) / 2

        # Construct the left and right projection matrices, the only difference is
        # that the right projection matrix should have a shift along the x axis of
        # baseline*focal_length
        P_left = np.array([[stereo_focal_px, 0, stereo_cx, 0],
                           [0, stereo_focal_px, stereo_cy, 0],
                           [0, 0, 1, 0]])
        P_right = P_left.copy()
        P_right[0][3] = T[0] * stereo_focal_px

        # Construct Q for use with cv2.reprojectImageTo3D. Subtract max_disp from x
        # since we will crop the disparity later
        Q = np.array([[1, 0, 0, -(stereo_cx - max_disp)],
                      [0, 1, 0, -stereo_cy],
                      [0, 0, 0, stereo_focal_px],
                      [0, 0, -1 / T[0], 0]])

        # Create an undistortion map for the left and right camera which applies the
        # rectification and undoes the camera distortion. This only has to be done
        # once
        m1type = cv2.CV_32FC1
        (lm1, lm2) = cv2.fisheye.initUndistortRectifyMap(K_left, D_left, R_left, P_left, stereo_size, m1type)
        (rm1, rm2) = cv2.fisheye.initUndistortRectifyMap(K_right, D_right, R_right, P_right, stereo_size, m1type)
        undistort_rectify = {"left": (lm1, lm2),
                             "right": (rm1, rm2)}

        # Create opencv window to render image in
        cv2.namedWindow("Fisheye stream", cv2.WINDOW_AUTOSIZE)

        # Create colorizer object
        colorizer = rs.colorizer()

        # Streaming loop
        mode = "stack"
        while True:
            # Get frameset of depth
            frames = pipeline.wait_for_frames()

            f1 = frames.get_fisheye_frame(1).as_video_frame()
            left_data = np.asanyarray(f1.get_data())
            f2 = frames.get_fisheye_frame(2).as_video_frame()
            right_data = np.asanyarray(f2.get_data())

            frame_copy = {"left": left_data.copy(),
                          "right": right_data.copy()}
            # Undistort and crop the center of the frames
            center_undistorted = {"left": cv2.remap(src=frame_copy["left"],
                                                    map1=undistort_rectify["left"][0],
                                                    map2=undistort_rectify["left"][1],
                                                    interpolation=cv2.INTER_LINEAR),
                                  "right": cv2.remap(src=frame_copy["right"],
                                                     map1=undistort_rectify["right"][0],
                                                     map2=undistort_rectify["right"][1],
                                                     interpolation=cv2.INTER_LINEAR)}

            # compute the disparity on the center of the frames and convert it to a pixel disparity
            # (divide by DISP_SCALE=16)
            disparity = stereo.compute(center_undistorted["left"], center_undistorted["right"]).astype(
                np.float32) / 16.0

            # re-crop just the valid part of the disparity
            disparity = disparity[:, max_disp:]

            # convert disparity to 0-255 and color it
            disp_vis = 255*(disparity - min_disp)/ num_disp
            disp_color = cv2.applyColorMap(cv2.convertScaleAbs(disp_vis,1), cv2.COLORMAP_JET)
            color_image = cv2.cvtColor(center_undistorted["left"][:,max_disp:], cv2.COLOR_GRAY2RGB)

            if mode == "stack":
                cv2.imshow(WINDOW_TITLE, np.hstack((color_image, disp_color)))
            if mode == "overlay":
                ind = disparity >= min_disp
                color_image[ind, 0] = disp_color[ind, 0]
                color_image[ind, 1] = disp_color[ind, 1]
                color_image[ind, 2] = disp_color[ind, 2]
                cv2.imshow(WINDOW_TITLE, color_image)

            cv2.imshow(WINDOW_TITLE, np.hstack((color_image, disp_color)))

            # Stack both images horizontally
            images = np.hstack((left_data, right_data))

            # Render image in opencv window
            cv2.imshow("Fisheye stream", images)
            key = cv2.waitKey(1)
            # if pressed escape exit program
            if key == 27:
                cv2.destroyAllWindows()
                break

    finally:
        pass
