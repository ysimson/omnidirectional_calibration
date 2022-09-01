import cv2 as cv
import pyrealsense2 as rs
import numpy as np

# ctx = rs.context()
# dev = ctx.query_devices()[0]
# pipe = rs.pipeline()

# device = rs.context().devices[0]
# pipeline = rs.pipeline()
# config = rs.config()
ctx = rs.context()
dev = ctx.query_devices()[0]
sensor = dev.first_depth_sensor()
profiles = sensor.get_stream_profiles()
for profile in profiles:
    print(profile.as_video_stream_profile())

pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

for s in device.sensors:
    print(s.get_info(rs.camera_info.name))

cfg = rs.config()
# Configuration for D580!!!
config.enable_stream(rs.stream.infrared, 1, 1600, 1300, rs.format.y16, 30)
config.enable_stream(rs.stream.infrared, 2, 1600, 1300, rs.format.y16, 30)

# configuration for T2658


# Start streaming
profile = pipeline.start(config)
#print(device)
device.get_info(rs.camera_info.advanced_mode)
device.sensors[0].profiles

try:
    while True:
        frames = pipeline.wait_for_frames()

        f1 = frames.get_infrared_frame(1).as_video_frame()
        left_data = np.asanyarray(f1.get_data())
        f2 = frames.get_infrared_frame(2).as_video_frame()
        right_data = np.asanyarray(f2.get_data())

        images = np.hstack((left_data, right_data))

        # Show images
        cv.namedWindow('RealSense', cv.WINDOW_NORMAL)
        cv.imshow('RealSense', images)
        cv.waitKey(1)

finally:
    pipeline.stop()
