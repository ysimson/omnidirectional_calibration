import time
import pyrealsense2 as rs
import numpy as np
import cv2 as cv
import argparse
import os
from time import gmtime, strftime

LASER_ON_CONST_TRUE = "14 00 ab cd 7f 00 00 00 01 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00"
LASER_ON_CONST_FALSE = "14 00 ab cd 7f 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00"


def write_binary_image(img: np.ndarray, bin_filename: str):
    img_u16 = img.astype(np.uint16).ravel()  # convert 16bits into 12bits

    with open(bin_filename, "wb") as f:
        f.write(img_u16)


def send_hardware_monitor_command(dev, command):
    command_input = []  # array of uint_8t
    command = command.lower()
    command = command.replace("0x", "").replace(" ", "").replace("\t", "")
    command = command.replace("x", "")
    current_uint8_t_string = ''
    for i in range(0, len(command)):
        current_uint8_t_string += command[i]
        if len(current_uint8_t_string) >= 2:
            command_input.append(int('0x' + current_uint8_t_string, 0))
            current_uint8_t_string = ''
    if current_uint8_t_string != '':
        command_input.append(int('0x' + current_uint8_t_string, 0))
    raw_result = rs.debug_protocol(dev).send_and_receive_raw_data(command_input)
    return raw_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    # Add argument which takes path to a bag file as an input
    parser.add_argument("-d", "--downsample_rate", type=int, default=1, help="Down-sample output images")
    parser.add_argument("-o", "--output_folder", type=str, help="Path to the output images",
                        default=r"C:\Users\ysimson\work\data\realsense_images")
    # Parse the command line arguments to an object
    args = parser.parse_args()
    base_output_folder = args.output_folder

    ctx = rs.context()
    devices = ctx.query_devices()
    dev0 = ctx.query_devices()[0]
    dev1 = ctx.query_devices()[1]

    device_model0 = str(dev0.get_info(rs.camera_info.name))
    print(f'Device model0: {device_model0}')  # Let's use the first device as the laser device
    print('First device profiles')
    sensor = dev0.first_depth_sensor()
    profiles = sensor.get_stream_profiles()
    for profile in profiles:
        print(profile.as_video_stream_profile())

    device_model1 = str(dev1.get_info(rs.camera_info.name))
    print(f'Device model1: {device_model1}')
    print('Second device profiles')
    sensor = dev1.first_depth_sensor()
    profiles = sensor.get_stream_profiles()
    for profile in profiles:
        print(profile.as_video_stream_profile())

    print('Config ... ')
    pipe1 = rs.pipeline()
    cfg1 = rs.config()
    cfg1.enable_device(dev0.get_info(rs.camera_info.serial_number))
    cfg1.enable_stream(rs.stream.infrared, 1, 424, 240, rs.format.y8, 6)
    cfg1.enable_stream(rs.stream.infrared, 2, 424, 240, rs.format.y8, 6)

    pipe2 = rs.pipeline()
    cfg2 = rs.config()
    cfg2.enable_device(dev1.get_info(rs.camera_info.serial_number))
    # cfg2.enable_stream(rs.stream.infrared, 0, 1280,  960, rs.format.uyvy, 30)
    cfg2.enable_stream(rs.stream.infrared, 1, 1600, 1300, rs.format.y16, 30)
    cfg2.enable_stream(rs.stream.infrared, 2, 1600, 1300, rs.format.y16, 30)
    # cfg2.enable_stream(rs.stream.infrared, 0, 1280, 960, rs.format.bgr8,  30)

    # Start streaming from both cameras
    pipe1.start(cfg1)
    pipe2.start(cfg2)

    print('Sending cmd to laser...')
    res = send_hardware_monitor_command(dev0, LASER_ON_CONST_TRUE)
    # res = send_hardware_monitor_command(dev, LASER_ON_CONST_FALSE)

    print('Streaming...')
    device_model_str = device_model1.replace(" ", "_")

    timestamp = strftime("%Y%m%d_%H%M%S", gmtime())
    output_folder = os.path.join(base_output_folder, device_model_str, timestamp)
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    left_cam_folder = os.path.join(output_folder, 'cam0')
    right_cam_folder = os.path.join(output_folder, 'cam1')
    os.makedirs(left_cam_folder, exist_ok=True)
    os.makedirs(right_cam_folder, exist_ok=True)
    bin_folder = os.path.join(output_folder, 'bin')
    os.makedirs(bin_folder, exist_ok=True)
    gray_folder = os.path.join(output_folder, 'gray')
    os.makedirs(gray_folder, exist_ok=True)

    try:
        while True:
            frames1 = pipe1.wait_for_frames()
            frame_ts = frames1.timestamp  # milliseconds

            f0 = frames1.get_infrared_frame(1).as_video_frame()
            f1 = frames1.get_infrared_frame(2).as_video_frame()
            if f0 and f1:
                small_left_data = np.asanyarray(f0.get_data())
                small_right_data = np.asanyarray(f1.get_data())
                images_laser = np.hstack((small_left_data, small_right_data))

                # Show images
                cv.namedWindow('RealSense D435', cv.WINDOW_NORMAL)
                cv.imshow('RealSense D435', images_laser)

            frames2 = pipe2.wait_for_frames()
            frame_ts = frames2.timestamp  # milliseconds

            f_left = frames2.get_infrared_frame(1).as_video_frame()
            f_right = frames2.get_infrared_frame(2).as_video_frame()
            # f_rgb = frames2.get_infrared_frame(0).as_video_frame()

            if f_left and f_right:
                left_data = np.asanyarray(f_left.get_data())
                right_data = np.asanyarray(f_right.get_data())
                # rgb_data = np.asanyarray(f_rgb.get_data())

                images = np.hstack((left_data, right_data))

                cv.namedWindow('RealSense D580', cv.WINDOW_NORMAL)
                cv.imshow('RealSense D580', images)

                frame_ts_ns = int(1e6 * frame_ts)
                left_filename = os.path.join(left_cam_folder, f'{frame_ts_ns}.png')
                right_filename = os.path.join(right_cam_folder, f'{frame_ts_ns}.png')
                cv.imwrite(left_filename, (left_data // 256).astype(np.uint8))
                cv.imwrite(right_filename, (right_data // 256).astype(np.uint8))

                small_left_filename = os.path.join(gray_folder, f'{frame_ts_ns}.png')
                cv.imwrite(small_left_filename, small_left_data)

                frame_index = frames2.frame_number

                if args.downsample_rate == 1 or frame_index % args.downsample_rate == 0:
                    bin_idx = frame_index // args.downsample_rate
                    left_bin_filename = os.path.join(bin_folder, f"CalibrationLeftIR{bin_idx:04d}.bin")
                    right_bin_filename = os.path.join(bin_folder, f"CalibrationRightIR{bin_idx:04d}.bin")

                    write_binary_image(left_data, left_bin_filename)
                    write_binary_image(right_data, right_bin_filename)

            k = cv.waitKey(1)
            if k == 32:
                print(f"Space bar pressed: {k}")
                break

    finally:
        print('stop')
        pipe1.stop()
        pipe2.stop()
