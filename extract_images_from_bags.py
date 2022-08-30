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


def read_first_stereo(bag_filename: str):

    # Create pipeline
    pipeline = rs.pipeline()

    # Create a config object
    config = rs.config()

    # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
    rs.config.enable_device_from_file(config, bag_filename)

    config.enable_stream(rs.stream.pose)
    config.enable_stream(rs.stream.fisheye, 1)
    config.enable_stream(rs.stream.fisheye, 2)
    # Start streaming from file
    pipeline.start(config)
    frames = pipeline.wait_for_frames()

    f1 = frames.get_fisheye_frame(1).as_video_frame()
    left_data = np.asanyarray(f1.get_data())
    f2 = frames.get_fisheye_frame(2).as_video_frame()
    right_data = np.asanyarray(f2.get_data())

    return left_data, right_data


def write_binary(img: np.ndarray, bin_filename: str):
    img_u16 = 4 * img.astype(np.uint16)  # convert 8bits into 12bits
    with open(bin_filename, "wb") as f:
        f.write(img_u16.T.ravel())


def write_binary_stereo(img_left: np.ndarray, img_right: np.ndarray, bin_filename: str):
    img_left_u16 = 16 * img_left.astype(np.uint16).ravel()  # convert 8bits into 12bits
    img_right_u16 = 16 * img_right.astype(np.uint16).ravel()  # convert 8bits into 12bits

    img_left_right_u16 = np.vstack((img_left_u16, img_right_u16))
    with open(bin_filename, "wb") as f:
        f.write(img_left_right_u16)


if __name__ == "__main__":
    # Create object for parsing command-line options
    parser = argparse.ArgumentParser(description="Read recorded bag file and display depth stream in jet colormap.\
                                    Remember to change the stream fps and format to match the recorded.")
    # Add argument which takes path to a bag file as an input
    parser.add_argument("-i", "--input_folder", type=str, help="Path to the bags folder")
    parser.add_argument("-o", "--output_folder", type=str, help="Path to output view points")
    # Parse the command line arguments to an object
    args = parser.parse_args()
    # Safety if no parameter have been given
    if not args.input_folder:
        print("No input paramater have been given.")
        print("For help type --help")
        exit()

    list_of_ros_bags = [os.path.join(args.input_folder, x) for x in os.listdir(args.input_folder) if os.path.isfile(os.path.join(args.input_folder, x))]

    for bin_idx, rosbag_filename in enumerate(list_of_ros_bags):
        filename = os.path.basename(rosbag_filename)
        img_output_folder = os.path.join(args.input_folder, os.path.splitext(filename)[0])
        if not os.path.isdir(img_output_folder):
            os.makedirs(img_output_folder)

        left_img, right_img = read_first_stereo(rosbag_filename)

        left_filename = os.path.join(img_output_folder, "image_left.tiff")
        right_filename = os.path.join(img_output_folder, "image_right.tiff")
        cv2.imwrite(left_filename, left_img)
        cv2.imwrite(right_filename, right_img)

        left_right_binname = os.path.join(args.output_folder, f"trackingImage{bin_idx}.bin")
        write_binary_stereo(left_img, right_img, left_right_binname)
