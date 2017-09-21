import cv2
import numpy as np
import cnnconfig as cf
import Model
import tensorflow as tf
import FaceInput
import os
from PIL import Image, ImageTk
import time
import tkinter as tk

height = cf.height
width = cf.width

output = Model.FaceNet()
ppimg,pplabels,peoplenames=FaceInput.ReadFaceImg.GetDataset()

saver = tf.train.Saver()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
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

current_image = None  # current image from the camera

root = tk.Tk()  # initialize root window
root.title("DeepCore Demo")  # set window title
# self.destructor function gets fired when the window is closed
# root.protocol('WM_DELETE_WINDOW', self.destructor)
panel = tk.Label(root)  # initialize image panel
panel.pack(padx=10, pady=10)
root.config(cursor="arrow")

# create a button, that when pressed, will take the current frame and save it to file
btn = tk.Button(root, text="Finish Demo")
btn.pack(fill="both", expand=True, padx=10, pady=10)

def FaceRecog(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    if len(faces) > 0:
        (x0, y0, w0, h0) = faces[0]
        area = w0 * h0
        for (x, y, w, h) in faces:
            if w * h > area:
                (x0, y0, w0, h0) = (x, y, w, h)
                area = w * h
        cv2.rectangle(frame, (x0, y0), (x0 + w0, y0 + h0), (0, 255, 0), 2)

        face = []
        img = frame[x0:x0 + w0, y0:y0 + h0]
        img = cv2.resize(img, (height, width))
        grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(grayimg, cv2.COLOR_GRAY2BGR)
        face.append(img)
        Top_One, prob = WhosFace(face)

        mygroup = [2, 238, 240]
        mygp_pos = np.argmax([prob[2], prob[238], prob[240]])
        mygroupname = peoplenames[mygroup[mygp_pos]]

        text = 'The face is ' + mygroupname
        cv2.putText(frame, text, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)
    return frame

def video_loop():
    ret, frame = cap.read()
    if ret:
        key = cv2.waitKey(500)
    frame = FaceRecog(frame)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)  # convert colors from BGR to RGBA
    current_image = Image.fromarray(cv2image)  # convert image for PIL
            #self.current_image= self.current_image.resize([1280,1024],PIL.Image.ANTIALIAS)
    imgtk = ImageTk.PhotoImage(image=current_image)  # convert image for tkinter
    panel.imgtk = imgtk  # anchor imgtk so it does not be deleted by garbage-collector
    panel.config(image=imgtk)  # show the image
        #self.root.attributes("-fullscreen",True)
    root.after(50, video_loop)  # call the same function after 30 milliseconds

video_loop()
root.mainloop()

cap.release()
cv2.destroyAllWindows()
sess.close()