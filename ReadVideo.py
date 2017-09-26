import cv2
import os
# vid = cv2.VideoCapture('‪F:/DeskFiles/20170922172306.mp4')
#
# n=1
# while True:
#     ret,frame =vid.read()
#     if ret:
#         frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
#         cv2.imwrite('Zhou_Hongtao_'+'%d'%(n)+'.jpg',frame)
#         n=n+1
#     else:
#         break
#
# vid.release()
# cv2.destroyAllWindows()
rootpath = 'F:\DeskFiles\Picture\Ye_JiaMeng\cropped'
imglist = os.listdir(rootpath)
imgnum = len(imglist)
i=1
while i<imgnum:
    os.remove(rootpath+'/'+imglist[i])
    i=i+4