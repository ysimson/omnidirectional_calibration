import bagpy
from bagpy import bagreader
import pandas as pd
import pyrealsense2 as rs

b = bagreader(r'C:\Users\ysimson\OneDrive - Intel Corporation\Documents\20220803_140014.bag')

# get the list of topics

fisheye1_csv = b.message_by_topic('/device_0/sensor_0/Fisheye_1/image/data')
df_img1 = pd.read_csv(fisheye1_csv)
image_stream = df_img1.loc[0, 'data']
stream_array = list(map(int, image_stream.split('\\')))
h, w = df_img1.loc[0, ['height', 'width']]
print(b.topic_table)
