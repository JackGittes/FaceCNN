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
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
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
root.geometry('800x350')
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

lb_frame1 = tk.LabelFrame(root, text="Prediction:", font=('Arial', 12), width=15, height=2)
lb_frame1.pack(fill="both", expand="no")

lb_frame2 = tk.LabelFrame(root, text="Process FPS:", font=('Arial', 12), width=15, height=2)
lb_frame2.pack(fill="both", expand="no")

lb_frame3 = tk.LabelFrame(root, text="Probability:", font=('Arial', 12), width=15, height=2)
lb_frame3.pack(fill="both", expand="no")

var = tk.StringVar()
left = tk.Label(lb_frame1, textvariable=var, font=('Arial', 12), width=15, height=2, justify='left')
left.pack()

var_FPS = tk.StringVar()
fps_info = tk.Label(lb_frame2, textvariable=var_FPS, font=('Arial', 12), width=15, height=2, justify='left')
fps_info.pack()

var_top1_prob = tk.StringVar()
top1_prob = tk.Label(lb_frame3, textvariable=var_top1_prob, font=('Arial', 12), width=15, height=2, justify='left')
top1_prob.pack()

pause = True
face_fps=0
top1 = 0.0

canvas_width = 200
canvas_height = 100

rec_height = [0,0,0,0]
for i in range(4):
    rec_height[i]=40
    rec_width=40

w = tk.Canvas(root,width=canvas_width,height=50)
rec1 = w.create_rectangle(10,10,50,50,outline='blue',fill='red')
rec2 = w.create_rectangle(50,10,100,50,outline='blue',fill='red')
rec3 = w.create_rectangle(100,10,150,50,outline='blue',fill='red')
w.coords(rec1,(40,41,40+rec_width,40))
w.coords(rec2,(80,41,80+rec_width,40))
w.coords(rec3,(120,41,120+rec_width,40))

w.pack()

def video_loop():
    global top1,face_fps,pause,convas_width,convas_height,w,rec1,rec2,rec3
    ret, frame = cap.read()
    if ret and (not pause):
        t1 = time.time()
        frame, prob, gpname = FaceRecog(frame)
        t2 = time.time()
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        cv2image = cv2.resize(cv2image,(400,300))
        current_image = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=current_image)
        panel.imgtk = imgtk
        panel.config(image=imgtk)
        face_fps = 1/(t2-t1)
        rec_height = [0,0,0,0]
        if gpname and gpname != 'Unrecognized':
            top1 = prob[peoplenames.index(gpname)]
            mem1_prob = prob[11]
            mem2_prob = prob[13]
            mem3_prob = prob[14]
            rec_height[0] = int(mem1_prob * 40)
            rec_height[1] = int(mem2_prob * 40)
            rec_height[2] = int(mem3_prob * 40)
            w.coords(rec1, (40, 41 - rec_height[0], 40 + rec_width, 40))
            w.coords(rec2, (80, 41 - rec_height[1], 80 + rec_width, 40))
            w.coords(rec3, (120, 41 - rec_height[2] , 120 + rec_width, 40))
        if len(prob) > 0:
            var.set(gpname)
            var_top1_prob.set('%.2f%%' % (100 * top1))
            var_FPS.set('%.2f' % (face_fps))

    elif ret and pause:
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        cv2image = cv2.resize(cv2image, (400, 300))
        current_image = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=current_image)
        panel.imgtk = imgtk
        panel.config(image=imgtk)

        var.set('None')
        var_top1_prob.set('0.0')
        var_FPS.set('0.0')

    root.after(30, video_loop)

def display_video():
    global pause
    if pause == False:
        pause = True
    else:
        pause = False

pause_btn = tk.Button(root, text="START/PAUSE", command=display_video)
pause_btn.pack(fill="both", expand=True, padx=10, pady=10)

btn = tk.Button(root, text="Finish Demo",fg='red',command= root.quit)
btn.pack(fill="both", expand=True, padx=10, pady=10)
video_loop()

root.mainloop()

cap.release()
cv2.destroyAllWindows()
sess.close()
