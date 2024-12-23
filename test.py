import subprocess
import time

time.sleep(10)

command = 'cd Desktop/Data/mjpg-streamer/mjpg-streamer-experimental && ./mjpg_streamer -i "input_uvc.so -d /dev/video0 --resolution 1920x1080" -o "output_http.so -n -w /root/stream/www"'
subprocess.run(command, shell=True, check=True)

