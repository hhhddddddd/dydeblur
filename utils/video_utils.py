import time
import numpy as np
import os
import glob
import torch
import subprocess

def frame2video(framerate, frame_path, video_path):     
    command = "ffmpeg -framerate {0} -pattern_type glob -i '{1}' -c:v libx264 -pix_fmt yuv420p {2}".format(framerate, frame_path, video_path)
    subprocess.run(command, shell=True)
    print("frame2video is ok!")

def video2frame(framerate, frame_path, video_path):
    command = "ffmpeg -i '{0}' -r {1} {2}".format(video_path, framerate, frame_path)
    subprocess.run(command, shell=True)
    print('ok')

if __name__ == "__main__":
    # path = "/home/xuankai/dataset/Deblur4DGS/stereo_high_dataset/" # high
    # path = "/home/xuankai/dataset/Deblur4DGS/stereo_low_dataset/"  # low
    path = "/home/xuankai/code/dydeblur/data/Deblur4DGS/stereo_low_dataset/"  # low
    need_separate = False       # D2RF
    need_separate_4dgs = True   # Deblur4DGS
    # for scene in sorted(os.listdir(path)):
        # Separate left and right camera images
    scene = 'man'
    if need_separate:
        frame_path = path + scene + "/images_2/"
        new_frame_path = path + scene + "/video/"
        os.makedirs(new_frame_path, exist_ok=True)
        for frame in sorted(os.listdir(frame_path)):
            idx = frame.split('_')[0]
            note = frame.split('_')[1].split('.')[0]
            if note == 'left':
                source_path = frame_path + frame
                target_path = new_frame_path + idx + ".png"
                command = "cp {0} {1}".format(source_path, target_path)
                subprocess.run(command, shell=True)
    if need_separate_4dgs:
        frame_path = path + scene + "/x1/images/"
        new_frame_path = path + scene + "/x1/images_half/"
        os.makedirs(new_frame_path, exist_ok=True)
        for i, frame in enumerate(sorted(os.listdir(frame_path))):
            idx = frame.split('_')[0]
            if i % 2 == 0:
                source_path = frame_path + frame
                target_path = new_frame_path + idx
                command = "cp {0} {1}".format(source_path, target_path)
                subprocess.run(command, shell=True)

    frame_path = path + scene + "/x1/images_half/*.png"
    video_path = "/home/xuankai/code/dydeblur/assets/" + scene + ".mp4"
    # video_path = "/home/xuankai/code/dydeblur/assets/Deblur4DGS/high/" + scene + ".mp4"
    framerate = 10
    frame2video(framerate, frame_path, video_path)
# 1: 0,1,2,...
