import time
import pyrealsense2 as rs
import numpy as np
import cv2 as cv

LASER_ON_CONST_TRUE = "14 00 ab cd 7f 00 00 00 01 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00"
LASER_ON_CONST_FALSE = "14 00 ab cd 7f 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00"


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


ctx = rs.context()
devices = ctx.query_devices()
dev0 = ctx.query_devices()[0]  # first device connected to USB 2.0
dev1 = ctx.query_devices()[1]

sensor = dev0.first_depth_sensor()
profiles = sensor.get_stream_profiles()
for profile in profiles:
    print(profile.as_video_stream_profile())

device_model0 = str(dev0.get_info(rs.camera_info.name))
device_model1 = str(dev1.get_info(rs.camera_info.name))

print(f'device_model0: {device_model0}')  # Let's use the first device as the laser device
print(f'device_model1: {device_model1}')

print('Config ... ')
pipe1 = rs.pipeline()
cfg1 = rs.config()
cfg1.enable_device(dev0.get_info(rs.camera_info.serial_number))
cfg1.enable_stream(rs.stream.infrared, 1, 424, 240, rs.format.y8, 6)
cfg1.enable_stream(rs.stream.infrared, 2, 424, 240, rs.format.y8, 6)

pipe2 = rs.pipeline()
cfg2 = rs.config()
cfg2.enable_device(dev1.get_info(rs.camera_info.serial_number))
cfg2.enable_stream(rs.stream.infrared, 1, 1600, 1300, rs.format.y16, 30)
cfg2.enable_stream(rs.stream.infrared, 2, 1600, 1300, rs.format.y16, 30)

# Start streaming from both cameras
pipe1.start(cfg1)
pipe2.start(cfg2)

print('Sending cmd to laser...')
res = send_hardware_monitor_command(dev0, LASER_ON_CONST_TRUE)
# res = send_hardware_monitor_command(dev, LASER_ON_CONST_FALSE)

print('Streaming...')

try:
    while True:
        frames1 = pipe1.wait_for_frames()
        frame_ts = frames1.timestamp  # milliseconds

        f0 = frames1.get_infrared_frame(1).as_video_frame()
        f1 = frames1.get_infrared_frame(2).as_video_frame()
        if f0 and f1:
            left_data = np.asanyarray(f0.get_data())
            right_data = np.asanyarray(f1.get_data())
            images_laser = np.hstack((left_data, right_data))

            # Show images
            cv.namedWindow('RealSense D435', cv.WINDOW_NORMAL)
            cv.imshow('RealSense D435', images_laser)

        frames2 = pipe2.wait_for_frames()
        frame_ts = frames2.timestamp  # milliseconds

        f0 = frames2.get_infrared_frame(1).as_video_frame()
        f1 = frames2.get_infrared_frame(2).as_video_frame()
        if f0 and f1:
            left_data = np.asanyarray(f0.get_data())
            right_data = np.asanyarray(f1.get_data())
            images = np.hstack((left_data, right_data))

            cv.namedWindow('RealSense D580', cv.WINDOW_NORMAL)
            cv.imshow('RealSense D580', images)

        k = cv.waitKey(1)
        if k == 32:
            print(f"Space bar pressed: {k}")
            break

finally:
    print('stop')
    pipe1.stop()
    pipe2.stop()
