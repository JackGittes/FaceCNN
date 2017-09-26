import cv2
from PIL import Image
import glob
import os
import numpy as np

def DetectFace(image, faceCascade, returnImage=False):
    min_size = (20,20)
    haar_scale = 1.1
    min_neighbors = 3
    haar_flags = 0
    cv2.equalizeHist(image, image)
    faces = cv2.HaarDetectObjects(
            image, faceCascade, cv.CreateMemStorage(0),
            haar_scale, min_neighbors, haar_flags, min_size
        )
    if faces and returnImage:
        for ((x, y, w, h), n) in faces:
            # Convert bounding box to two CvPoints
            pt1 = (int(x), int(y))
            pt2 = (int(x + w), int(y + h))
            cv.Rectangle(image, pt1, pt2, cv.RGB(255, 0, 0), 5, 8, 0)

    if returnImage:
        return image
    else:
        return faces
def pil2cvGrey(pil_im):
    pil_im = pil_im.convert('L')
    cv_im = cv.CreateImageHeader(pil_im.size, cv.IPL_DEPTH_8U, 1)
    cv.SetData(cv_im, pil_im.tostring(), pil_im.size[0]  )
    return cv_im

def cv2pil(cv_im):
    return Image.fromstring("L", cv.GetSize(cv_im), cv_im.tostring())

def imgCrop(image, cropBox, boxScale=1):
    xDelta=max(cropBox[2]*(boxScale-1),0)
    yDelta=max(cropBox[3]*(boxScale-1),0)
    PIL_box=[cropBox[0]-xDelta, cropBox[1]-yDelta, cropBox[0]+cropBox[2]+xDelta, cropBox[1]+cropBox[3]+yDelta]

    return image.crop(PIL_box)

path = 'F:\DeskFiles\Picture\Ye_JiaMeng'
def faceCrop(imagePattern,boxScale=1):
    # Select one of the haarcascade files:
    #   haarcascade_frontalface_alt.xml
    #   haarcascade_frontalface_alt2.xml
    #   haarcascade_frontalface_alt_tree.xml
    #   haarcascade_frontalface_default.xml
    #   haarcascade_profileface.xml
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    face_cascade.load('D:/OpenCV/opencv/sources/data/haarcascades_cuda/haarcascade_frontalface_alt.xml')
    isexist = os.path.exists(path+'/cropped')
    if not isexist:
        os.mkdir(path + '/cropped')
    pre = len(path)
    imgList=glob.glob(imagePattern)
    if len(imgList)<=0:
        print('No Images Found')
        return
    for img in imgList:
        imgname = img[pre:]
        pil_im = Image.open(img)
        if pil_im.mode == 'RGB':
            cvimg = cv2.cvtColor(np.asarray(pil_im),cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(cvimg, 1.2, 5)
            if len(faces)>0:
                n=1
                for face in faces:
                    croppedImage=imgCrop(pil_im, face,boxScale=boxScale)
                    fname,ext=os.path.splitext(imgname)
                    croppedImage.save(path+'/cropped'+fname+'_cropped'+str(n)+ext)
                    n+=1
            else:
                print('No faces found:'+ img)
        else:
            pil_im.close()
faceCrop(path+'/*.jpg',boxScale=1)