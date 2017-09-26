import cv2
import numpy as np
import cnnconfig as cf
import Model
import tensorflow as tf
import FaceInput
import os
import time

height = cf.height
width = cf.width

output = Model.FaceNet()
ppimg,pplabels,peoplenames=FaceInput.ReadFaceImg.GetDataset()

saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, tf.train.latest_checkpoint('checkpoint/'))

def WhosFace(img):
    result = sess.run(output, feed_dict={Model.x : img, Model.keep_prob : 1.0})
    result = result[0]
    pos = np.argmax(result)
    return pos,result

cap = cv2.VideoCapture(0)
timeframe = 0
total_time = 0
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
face_cascade.load('D:/OpenCV/opencv/sources/data/haarcascades_cuda/haarcascade_frontalface_alt.xml')
while(1):
    t1 = time.time()
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    if len(faces)>0:
        (x0,y0,w0,h0) = faces[0]
        area = w0*h0
        for (x,y,w,h) in faces:
            if w*h>area:
                (x0,y0,w0,h0)=(x,y,w,h)
                area = w*h


        face=[]
        img = frame[x0:x0+w0,y0:y0+h0]
        img = cv2.resize(img,(height,width))
        grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(grayimg,cv2.COLOR_GRAY2BGR)
        face.append(img)
        Top_One,prob = WhosFace(face)
        Peoplename = peoplenames[Top_One]

        mygroup = [44,47,46]
        mygp_pos = np.argmax([prob[44],prob[47],prob[46]])
        mygroupname = peoplenames[mygroup[mygp_pos]]

        Top_Five = []
        Top_FivePos = []
        for i in range(5):
            pos = np.argmax(prob)
            Top_Five.append(prob[pos])
            Top_FivePos.append(pos)
            prob[pos] = -1
        t2 = time.time()
        total_time = t2-t1 + total_time
        if timeframe == 10:
            os.system('cls')
            print('\nFPS:%.4f'% (10.0/(total_time)))
            print('\n' + 'Top-1 prediction is '+Peoplename)
            print('\n' + 'Top-5 prediction: ' )
            for i in range(5):
                print('\n' + peoplenames[Top_FivePos[i]] + '=%.8f%%' % (100.0*Top_Five[i]))
            print('\n\n')
            timeframe = 0
            total_time = 0

        text = 'The face is '+ mygroupname
        cv2.putText(frame, text, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)
#        cv2.rectangle(frame, (x0, y0), (x0 + w0, y0 + h0), (0, 255, 0), 2)
        cv2.imshow("DeepCore for Demo", frame)

        timeframe = timeframe + 1
    else:
        cv2.imshow("DeepCore for Demo", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
sess.close()