import cv2 as cv
import pyrealsense2 as rs
import numpy as np
import argparse
import os
from time import gmtime, strftime


def write_binary_stereo(img_left: np.ndarray, img_right: np.ndarray, bin_filename: str):
    img_left_u16 = (img_left / 16.).astype(np.uint16).ravel()  # convert 16bits into 12bits
    img_right_u16 = (img_right / 16.).astype(np.uint16).ravel()  # convert 16bits into 12bits

    img_left_right_u16 = np.vstack((img_left_u16, img_right_u16))
    with open(bin_filename, "wb") as f:
        f.write(img_left_right_u16)


if __name__ == "__main__":
    # Create object for parsing command-line options
    parser = argparse.ArgumentParser(description="")
    # Add argument which takes path to a bag file as an input
    parser.add_argument("-i", "--input", type=str, help="Path to the bag file. If noe given read live stream")
    parser.add_argument("-b", "--bin_format", type=bool, default=False, help="Save image as bin")
    parser.add_argument("-d", "--downsample_rate", type=int, default=1, help="Down-sample output images")
    parser.add_argument("-o", "--output_folder", type=str, help="Path to the output images",
                        default=r"C:\Users\ysimson\work\data\realsense_images")
    parser.add_argument('-l', '--laser', action='store_true', help='If activated and have two devices activate laser on second device')

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

    # start reading
    pipeline = rs.pipeline()
    config = rs.config()
    ctx = rs.context()
    dev = ctx.query_devices()[0]

    try:
        sensor = dev.first_depth_sensor()
        profiles = sensor.get_stream_profiles()
        for profile in profiles:
            print(profile.as_video_stream_profile())
    except Exception as e:
        print(e)
        print('\n'.join(map(str, dev.sensors[0].get_stream_profiles())))

    # Get device product line for setting a supporting resolution
    # try:
    #     pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    #     pipeline_profile = config.resolve(pipeline_wrapper)
    #     device = pipeline_profile.get_device()
    #     device_product_line = str(device.get_info(rs.camera_info.product_line))
    #     device_model = str(device.get_info(rs.camera_info.name))
    #     print(f"device_model: {device_model}")
    #     for s in device.sensors:
    #         print(s.get_info(rs.camera_info.name))
    # except Exception as e:
    device_model = str(dev.get_info(rs.camera_info.name))

    # Configuration for D435
    if 'D435' in device_model:
        config.enable_stream(rs.stream.infrared, 1, 1280, 800, rs.format.y8, 30)
        config.enable_stream(rs.stream.infrared, 2, 1280, 800, rs.format.y8, 30)

    # Configuration for D580!!! (On D465 board)
    elif 'D580' in device_model or 'D465' in device_model:
        config.enable_stream(rs.stream.infrared, 1, 1600, 1300, rs.format.y16, 30)
        config.enable_stream(rs.stream.infrared, 2, 1600, 1300, rs.format.y16, 30)

    # configuration for T265
    elif 'T265' in device_model:
        # config.enable_stream(rs.stream.fisheye, 1, 848, 800, rs.format.y8, 30)
        # config.enable_stream(rs.stream.fisheye, 2, 848, 800, rs.format.y8, 30)
        config.enable_stream(rs.stream.pose)
        config.enable_stream(rs.stream.fisheye, 1)
        config.enable_stream(rs.stream.fisheye, 2)
    else:
        print(f"Unfamiliar device!! {device_model}??")
        exit(1)

    device_model = device_model.replace(" ", "_")

    timestamp = strftime("%Y%m%d_%H%M%S", gmtime())
    output_folder = os.path.join(base_output_folder, device_model, timestamp)
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    if args.bin_format:
        bin_folder = os.path.join(output_folder, 'bin')
        os.makedirs(bin_folder, exist_ok=True)

        # For visualizing the images in binary format convert to 8bits
        vis_folder = os.path.join(output_folder, 'vis')
        os.makedirs(vis_folder, exist_ok=True)
    else:
        left_cam_folder = os.path.join(output_folder, 'cam0')
        right_cam_folder = os.path.join(output_folder, 'cam1')
        os.makedirs(left_cam_folder, exist_ok=True)
        os.makedirs(right_cam_folder, exist_ok=True)

    # Start streaming
    profile = pipeline.start(config)

    # Get intrinsics
    # profiles = pipeline.get_active_profile()
    # streams = {"left": profiles.get_stream(rs.stream.infrared, 1).as_video_stream_profile(),
    #            "right": profiles.get_stream(rs.stream.infrared, 2).as_video_stream_profile()}
    # intrinsics = {"left": streams["left"].get_intrinsics(),
    #               "right": streams["right"].get_intrinsics()}

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frame_ts = frames.timestamp  # milliseconds

            f1 = frames.get_infrared_frame(1).as_video_frame()
            left_data = np.asanyarray(f1.get_data())
            f2 = frames.get_infrared_frame(2).as_video_frame()
            right_data = np.asanyarray(f2.get_data())

            if isinstance(right_data[0, 0], np.uint8):
                images = np.hstack((left_data, right_data))
            else:
                images = (np.hstack((left_data, right_data)) / 256).astype(np.uint8)

            # Show images
            cv.namedWindow('RealSense', cv.WINDOW_NORMAL)
            cv.imshow('RealSense', images)
            k = cv.waitKey(1)
            if k == 32:
                print(f"key pressed: {k}")
                break

            frame_index = frames.frame_number
            if args.downsample_rate > 1 and frame_index % args.downsample_rate != 0:
                continue
            if args.bin_format:
                bin_idx = frame_index // args.downsample_rate
                bin_image_filename = os.path.join(bin_folder, f"trackingImage{bin_idx}.bin")
                write_binary_stereo(left_data, right_data, bin_image_filename)

                images = (np.hstack((left_data, right_data)) // 256).astype(np.uint8)
                frame_ts_ns = int(1e6 * frame_ts)
                stereo_filename = os.path.join(vis_folder, f'{frame_ts_ns}.png')
                cv.imwrite(stereo_filename, images)
            else:
                frame_ts_ns = int(1e6 * frame_ts)

                left_filename = os.path.join(left_cam_folder, f'{frame_ts_ns}.png')
                right_filename = os.path.join(right_cam_folder, f'{frame_ts_ns}.png')
                if isinstance(right_data[0, 0], np.uint8):
                    cv.imwrite(left_filename, left_data)
                    cv.imwrite(right_filename, right_data)
                else:
                    cv.imwrite(left_filename, (left_data // 256).astype(np.uint8))
                    cv.imwrite(right_filename, (right_data // 256).astype(np.uint8))
    finally:
        pipeline.stop()
