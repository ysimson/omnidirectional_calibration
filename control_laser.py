import time
import pyrealsense2 as rs


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
dev = ctx.query_devices()[0]
pipe = rs.pipeline()


cfg = rs.config()
cfg.enable_stream(rs.stream.depth, 256, 144, rs.format.z16, 90)
print('start stream')
pipe.start(cfg)
print('sending cmd')
res = send_hardware_monitor_command(dev, LASER_ON_CONST_TRUE)


print('streaming..')
time.sleep(5)


print('stop')
pipe.stop()