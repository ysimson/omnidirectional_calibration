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
dev = ctx.query_devices()[0] # first device connected to USB 2.0

sensor = dev.first_depth_sensor()
profiles = sensor.get_stream_profiles()
for profile in profiles:
    print(profile.as_video_stream_profile())

pipe = rs.pipeline()
device_model = str(dev.get_info(rs.camera_info.name))

print(f'device_model: {device_model}')

print('Config ... ')
cfg = rs.config()
# cfg.enable_stream(rs.stream.depth, 256, 144, rs.format.z16, 90)
cfg.enable_stream(rs.stream.infrared, 1, 424, 240, rs.format.y8, 6)
cfg.enable_stream(rs.stream.infrared, 2, 424, 240, rs.format.y8, 6)

try:
    pipeline_wrapper = rs.pipeline_wrapper(pipe)
    pipeline_profile = cfg.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))
    device_model = str(device.get_info(rs.camera_info.name))
    print(f"device_model: {device_model}")
    for s in device.sensors:
        print(s.get_info(rs.camera_info.name))
except Exception as e:
    print(e)

print('start stream')
pipe.start(cfg)
print('sending cmd')
res = send_hardware_monitor_command(dev, LASER_ON_CONST_TRUE)
# res = send_hardware_monitor_command(dev, LASER_ON_CONST_FALSE)

print('streaming..')

try:
    while True:
        frames = pipe.wait_for_frames()
        frame_ts = frames.timestamp  # milliseconds

        f0 = frames.get_infrared_frame(0).as_video_frame()
        f1 = frames.get_infrared_frame(1).as_video_frame()
        left_data = np.asanyarray(f1.get_data())
        #f2 = frames.get_infrared_frame(2).as_video_frame()
        right_data = np.asanyarray(f1.get_data())

        images = np.hstack((left_data, right_data))
        # Show images
        cv.namedWindow('RealSense', cv.WINDOW_NORMAL)
        cv.imshow('RealSense', images)
        k = cv.waitKey(1)
        if k == 32:
            print(f"Space bar pressed: {k}")
            break

finally:
    print('stop')
    pipe.stop()
