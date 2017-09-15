import cv2
import os
import scipy
import numpy as np
import cnnconfig as cf

height = cf.height
width = cf.width
class_num = cf.class_num

class ReadFaceImg(path, height = height, width = width):
	def GetDataset(self, path):
		path = os.listdir('dataset')
		id = 0;
		for peoplename in path:
			peoplepath = os.listdir('dataset'+'/'+peoplename)
			for filename in peoplepath:
				img = cv2.imread('dataset'+'/'+peoplename +'/'+filename)
				img = cv2.resize(img,(height,width))
				imgs.append(img)
				label = np.zeros(class_num)
				label[id] = 1
				labels.append(label)
			id = id+1
		return imgs,labels

##	def ImgPreprocess():
		
##	def ReadInFace(self, img, height = height, width = width):
		