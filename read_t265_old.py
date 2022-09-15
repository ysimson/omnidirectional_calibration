import cv2 as cv
import pyrealsense2 as rs
import numpy as np

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.fisheye, 1)
config.enable_stream(rs.stream.fisheye, 2)

ctx = rs.context()
dev = ctx.query_devices()[0]
device_model = str(dev.get_info(rs.camera_info.name))

# Start streaming
profile = pipeline.start(config)
dev.get_info(rs.camera_info.advanced_mode)
dev.sensors[0].profiles

try:
    for i in range(0, 100):
        frames = pipeline.wait_for_frames()

        f1 = frames.get_fisheye_frame(1).as_video_frame()
        left_data = np.asanyarray(f1.get_data())
        f2 = frames.get_fisheye_frame(2).as_video_frame()
        right_data = np.asanyarray(f2.get_data())

        images = np.hstack((left_data, right_data))
        # Show images
        cv.namedWindow('RealSense', cv.WINDOW_NORMAL)
        cv.imshow('RealSense', images)
        k = cv.waitKey(1)
        if k == 32:
            print(f"key pressed: {k}")
            break

finally:
    pipeline.stop()
