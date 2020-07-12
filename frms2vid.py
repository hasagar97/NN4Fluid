import cv2
import numpy as np
import os
from os.path import isfile, join
pathIn= 'data/liquid3_x10_y10_f500/vimgs_l_big/'
pathOut = 'video.avi'
fps = 20
frame_array = []
files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
#for sorting the file names properly
files.sort(key = lambda x: x[5:-4])
files.sort()
frame_array = []
files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
#for sorting the file names properly
files.sort(key = lambda x: x[:-4])

img = cv2.imread(pathIn+files[0])
height, width, layers = img.shape
size = (width,height)


out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)


for i in range(len(files)):
    filename=pathIn + files[i]
    #reading each files
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    
    #inserting the frames into an image array
#     frame_array.append(img)

# for i in range(len(frame_array)):
#     # writing to a image array
    out.write(img)
out.release()