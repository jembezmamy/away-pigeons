import io
import math
from functools import reduce
import operator
import picamera
from PIL import Image
import time
import subprocess
import os
import motion_detector

prior_image = None
file_number = 1
total_video_length = 0
last_video_started_at = 0
last_video_ended_at = time.time()

def detect_motion(camera):
    global prior_image
    stream = io.BytesIO()
    camera.capture(stream, format='jpeg', use_video_port=True)
    stream.seek(0)
    if prior_image is None:
        prior_image = Image.open(stream)
        return False
    else:
        current_image = Image.open(stream)
        # Compare current_image to prior_image to detect motion. This is
        # left as an exercise for the reader!
        any_motion = motion_detector.detect(current_image, prior_image, 128, 0.0003)
        # Once motion detection is done, make the prior image the current
        prior_image = current_image
        return any_motion

def write_video(stream, name):
    # Write the entire content of the circular buffer to disk. No need to
    # lock the stream here as we're definitely not writing to it
    # simultaneously
    with io.open(name, 'wb') as output:
        for frame in stream.frames:
            if frame.frame_type == picamera.PiVideoFrameType.sps_header:
                stream.seek(frame.position)
                break
        while True:
            buf = stream.read1()
            if not buf:
                break
            output.write(buf)
    # Wipe the circular stream once we're done
    stream.seek(0)
    stream.truncate()

def process_video():
    global file_number
    global total_video_length
    global prior_image

    print("Processing video")

    video_list = open("tmp/list.txt", "w")
    for i in range(1, file_number):
        video_list.write("file video%s.h264\n" % i)
    video_list.close()

    output_file_path = "output/%d.mp4" % time.time()
    subprocess.call(["ffmpeg", "-f", "concat", "-i", "tmp/list.txt", "-c", "copy", output_file_path])

    publish_video(output_file_path)

    file_number = 1
    total_video_length = 0
    prior_image = None

def publish_video(path):
    print("Publishing video")
    subprocess.call(["Dropbox-Uploader/dropbox_uploader.sh", "upload", path, "away-pigeons/" + path])
    print("Deleting file")
    os.remove(path)
    print("Done")


with picamera.PiCamera() as camera:
    camera.resolution = (1280, 720)
    # camera.hflip = True
    # camera.vflip = True
    stream = picamera.PiCameraCircularIO(camera, seconds=3)
    camera.start_recording(stream, format='h264')
    print('Started watching')
    try:
        while True:
            camera.wait_recording(1)
            if detect_motion(camera):
                print('Motion detected!')
                last_video_started_at = time.time()

                # As soon as we detect motion, split the recording to
                # record the frames "after" motion
                camera.split_recording("tmp/video%d.h264" % (file_number + 1))
                # Write the 10 seconds "before" motion to disk as well
                write_video(stream, "tmp/video%d.h264" % file_number)
                file_number += 2

                current_length = total_video_length

                # Wait until motion is no longer detected, then split
                # recording back to the in-memory circular buffer
                while current_length < 15 and detect_motion(camera):
                    current_length = total_video_length + time.time() - last_video_started_at
                    camera.wait_recording(1)

                print('Motion stopped!')
                last_video_ended_at = time.time()
                total_video_length += last_video_ended_at - last_video_started_at

                print("Total length: %ds" % total_video_length)

                camera.split_recording(stream)

                if total_video_length >= 15:
                    process_video()

            elif total_video_length > 0 and time.time() - last_video_ended_at > 30:
                process_video()

    finally:
        camera.stop_recording()
