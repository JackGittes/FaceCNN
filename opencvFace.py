import os
import cv2
import cnnconfig as cf

height = 80
width = 80
class ImgConvert():
    def GetDataset(self):
        path = os.listdir('dataset')
        for peoplename in path:
            peoplepath = os.listdir('dataset' + '/' + peoplename)
            print(peoplename)
            id =1
            for filename in peoplepath:
                if filename.endswith('.jpg') or filename.endswith('.jpeg'):
                    img = cv2.imread('dataset' + '/' + peoplename + '/' + filename)
                    img = cv2.resize(img, (height, width))
                    greyimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite('dataset'+'/'+ peoplename + '/' + peoplename + '_%d'%(id) + '.jpg', greyimg)
                id = id + 1
        return
ImgConvert().GetDataset()