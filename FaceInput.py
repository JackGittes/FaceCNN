import cv2
import os
import numpy as np
import cnnconfig as cf

height = cf.height
width = cf.width
class_num = cf.class_num

class ReadFaceImg():
	def GetDataset():
		imgs=[]
		labels=[]
		peoplenames = []
		path = os.listdir('dataset')
		id = 0
		for peoplename in path:
			peoplenames.append(peoplename)
			peoplepath = os.listdir('dataset'+'/'+peoplename)
			for filename in peoplepath:
				if filename.endswith('.jpg') or filename.endswith('.tif'):
					img = cv2.imread('dataset'+'/'+peoplename +'/'+filename)
					img = cv2.resize(img,(height,width))
					imgs.append(img)
					label = np.zeros(class_num)
					label[id] = 1
					labels.append(label)
			id = id+1
	#	imgs = np.array(imgs)
	#	labels = np.array(labels)
		return imgs,labels,peoplenames

def GetOneImage(path):
	imgs = []
	img = cv2.imread(path)
	img = cv2.resize(img,(height,width))
	imgs.append(img)
	return imgs

def GetNextBatch(batch_size , imgs , labels):
	randarr = np.random.randint(0,len(imgs),batch_size)
	batch_imgs = []
	batch_labels = []
	for i in randarr:
		batch_imgs.append(imgs[i])
		batch_labels.append(labels[i])
	return batch_imgs,batch_labels

def GetFullDataset():
	return ReadFaceImg.GetDataset()

