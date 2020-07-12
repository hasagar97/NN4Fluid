import cv2
import numpy as np
import os
from os.path import isfile, join

homePath = "GT_liq/"
homePath1 = "5_5/"
pathIn1= homePath1+'l_adv/'
pathIn= homePath+'l_adv/'
pathOut = 'video_combines.avi'
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
size = (width,2*height)


out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)


for i in range(len(files)):
	filename=pathIn + files[i]
	filename1 = pathIn1 +files[i]
	#reading each files
	img = cv2.imread(filename)
	img1 = cv2.imread(filename1)
	if(width == 256):
		img[img == 182 ] = 255
		img[img[:,:,0]<240] = (224,179,69)

		cv2.putText(img,'GT', 
			(256-50,25), 
			cv2.FONT_HERSHEY_SIMPLEX, 
			1,
			(0,0,0),
			2)


		img1[img1 == 182 ] = 255
		img1[img1[:,:,0]<240] = (224,179,69)

		cv2.putText(img1,'pred', 
			(256-50,25), 
			cv2.FONT_HERSHEY_SIMPLEX, 
			0.5,
			(0,0,0),
			1)

	img_c = np.concatenate((img,img1),axis=0)
	if(i%10==0):
		cv2.imwrite("report/"+str(i)+".png",img)
	# print(img1.shape,img.shape,img_c.shape)

	out.write(np.concatenate((img,img1),axis=0))
out.release()