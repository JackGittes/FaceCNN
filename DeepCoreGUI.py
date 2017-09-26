import cv2
import numpy as np
import cnnconfig as cf
import Model
import tensorflow as tf
import FaceInput
from PIL import Image, ImageTk
import tkinter as tk
import time

height = cf.height
width = cf.width

threshold = 0.1

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

current_image = None

root = tk.Tk()
#root.geometry('600x800')
root.title("DeepCore for Demo")
panel = tk.Label(root)
panel.pack(side = 'left',padx=10, pady=10)
root.config(cursor="arrow")

def FaceRecog(frame):
    mygroupname = ''
    prob = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    if len(faces) > 0:
        (x0, y0, w0, h0) = faces[0]
        area = w0 * h0
        for (x, y, w, h) in faces:
            if w * h > area:
                (x0, y0, w0, h0) = (x, y, w, h)
                area = w * h
        face = []
        img = frame[x0:x0 + w0, y0:y0 + h0]
        img = cv2.resize(img, (height, width))
        grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(grayimg, cv2.COLOR_GRAY2BGR)
        face.append(img)
        Top_One, prob = WhosFace(face)

        mygroup = [11, 13, 14]
        mygp_pos = np.argmax([prob[11], prob[13], prob[14]])
        mygroupname = peoplenames[mygroup[mygp_pos]]

        if prob[mygroup[mygp_pos]]<threshold:
            mygroupname = 'Unrecognized'

        text = mygroupname
        cv2.rectangle(frame, (x0, y0), (x0 + w0, y0 + h0), (0, 255, 0), 2)
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)
    return frame,prob,mygroupname

def TopFivePred(prob):
    top_five_prob = []
    top_five_name = []
    for i in range(5):
        pos = np.argmax(prob)
        top_five_prob.append(prob[pos])
        top_five_name.append(peoplenames[pos])
        prob[pos] = -1
    return top_five_prob,top_five_name

frame = tk.Frame(root, width=100, height=50)
frame.place(x=700, y=0)

labelframe = tk.LabelFrame(root, text="Prediction:", font=('Arial', 12), width=15, height=2)
labelframe.pack(fill="both", expand="yes")

var = tk.StringVar()
left = tk.Label(labelframe, textvariable=var, font=('Arial', 12), width=15, height=2,bg='white', justify='left')
left.pack()

var_FPS = tk.StringVar()
fps_info = tk.Label(labelframe, textvariable=var_FPS, font=('Arial', 12), width=15, height=2,bg='white', justify='left')
fps_info.pack(side = 'right')

var_top1_prob = tk.StringVar()
top1_prob = tk.Label(labelframe, textvariable=var_top1_prob, font=('Arial', 12), width=15, height=2,bg='white', justify='left')
top1_prob.pack(side = 'right')

face_fps=0
top1 = 0.0
def video_loop():
    global top1,face_fps
    ret, frame = cap.read()
    if ret:
        key = cv2.waitKey(500)
    t1 = time.time()
    frame,prob,gpname = FaceRecog(frame)
    t2 = time.time()
    if ret:
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        cv2image = cv2.resize(cv2image,(400,300))
        current_image = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=current_image)
        panel.imgtk = imgtk
        panel.config(image=imgtk)
        face_fps = 1/(t2-t1)
        if gpname and gpname != 'Unrecognized':
            top1 = prob[peoplenames.index(gpname)]
    if len(prob)>0:
        var.set(gpname)
        var_top1_prob.set('%.2f%%'%(100*top1))
        var_FPS.set('%.2f'%(face_fps))

    root.after(30, video_loop)

click_video = False
def display_video():
    global click_video
    if click_video == False:
        click_video = True
        video_loop()
    else:
        click_video = True

pause_btn = tk.Button(root, text="Pause", command=display_video)
pause_btn.pack(fill="both", expand=True, padx=10, pady=10)

btn = tk.Button(root, text="Finish Demo",fg='red',command= root.quit)
btn.pack(fill="both", expand=True, padx=10, pady=10)
#video_loop()
root.mainloop()

cap.release()
cv2.destroyAllWindows()
sess.close()

