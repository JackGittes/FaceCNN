import cv2

videoCapture = cv2.VideoCapture('VideoCapture/001/oto.avi')

fps = videoCapture.get(cv2.cv.CV_CAP_PROP_FPS)
size = (int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))

success, frame = videoCapture.read()

id=1
while success and id<=200:
    cv2.imshow("Capturing Video Frame ...", frame)
    cv2.waitKey(1000 / int(fps))
    cv2.imwrite('VideoCapture'+'/'+'001'+'/'+'001_cap'+'/'+'001'+'_%d'%(id)+'.jpg',img)
    success, frame = videoCapture.read()
    id=id+1