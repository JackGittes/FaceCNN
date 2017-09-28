import cv2
import numpy as np
import cnnconfig as cf
import Model
import tensorflow as tf
import FaceInput
from PIL import Image, ImageTk
import tkinter as tk
import time
from tkinter import filedialog

# This defines the image's height and width that the CNN input requires.
height = cf.height
width = cf.width

# When probability is greater than the threshold, WhosFace returns a recognized name,
# otherwise WhosFace will return 'Unrecognized'
threshold = 0.1

# Load pre-trained CNN-based face recognition model and get all people names in the dataset.
output = Model.FaceNet()
ppimg,pplabels,peoplenames=FaceInput.ReadFaceImg.GetDataset()

saver = tf.train.Saver()

# Given that the Tkinter and OpenCV also need GPU memory, we give only 30% of GPU memory to CNN model,
# which can guarantee both the OpenCV, Tkinter and CNN model have enough memory.
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
saver.restore(sess, tf.train.latest_checkpoint('checkpoint/'))

# Give an image and infer its name.
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

Real_Time_Status = True

current_image = None

root = tk.Tk()
root.geometry('870x350')
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

        if Real_Time_Status:
            cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)
        cv2.rectangle(frame, (x0, y0), (x0 + w0, y0 + h0), (0, 255, 0), 2)
    else:
        mygroupname = 'No Face Detected'
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
lb_frame1.pack(fill="both", expand="no",padx=30)

lb_frame2 = tk.LabelFrame(root, text="Process FPS:", font=('Arial', 12), width=15, height=2)
lb_frame2.pack(fill="both", expand="no",padx=30)

lb_frame3 = tk.LabelFrame(root, text="Probability:", font=('Arial', 12), width=15, height=2)
lb_frame3.pack(fill="both", expand="no",padx=30)

var = tk.StringVar()
left = tk.Label(lb_frame1, textvariable=var, font=('Arial', 12), width=15, height=2, justify='left')
left.pack()

var_FPS = tk.StringVar()
fps_info = tk.Label(lb_frame2, textvariable=var_FPS, font=('Arial', 12), width=15, height=2, justify='left')
fps_info.pack()

var_top1_prob = tk.StringVar()
top1_prob = tk.Label(lb_frame3, textvariable=var_top1_prob, font=('Arial', 12), width=15, height=2, justify='left')
top1_prob.pack()

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
w.coords(rec1,(40,51,40+rec_width,50))
w.coords(rec2,(80,51,80+rec_width,50))
w.coords(rec3,(120,51,120+rec_width,50))
w.pack()


pause = True
face_fps=0
top1 = 0.0
file_image = []
background_image = cv2.imread('D:/Github/2/FaceCNN/logo.jpg')

def video_loop():
    global top1,face_fps,pause,convas_width,convas_height,w,rec1,rec2,rec3,file_image,background_image
    if Real_Time_Status:
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

            rec_height = [0,0,0]
            if gpname and gpname != 'Unrecognized' and gpname != 'No Face Detected':
                top1 = prob[peoplenames.index(gpname)]
                mem1_prob = prob[11]
                mem2_prob = prob[13]
                mem3_prob = prob[14]
                rec_height[0] = int(mem1_prob * 40)
                rec_height[1] = int(mem2_prob * 40)
                rec_height[2] = int(mem3_prob * 40)
                w.coords(rec1, (40, 51 - rec_height[0], 40 + rec_width, 50))
                w.coords(rec2, (80, 51 - rec_height[1], 80 + rec_width, 50))
                w.coords(rec3, (120, 51 - rec_height[2] , 120 + rec_width, 50))
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
    else:
        if len(file_image)==0:
            background_image = cv2.resize(background_image, (400, 300))
            current_image = Image.fromarray(background_image)
            imgtk = ImageTk.PhotoImage(image=current_image)
            panel.imgtk = imgtk
            panel.config(image=imgtk)

            var.set('None')
            var_top1_prob.set('0.0')
            var_FPS.set('0.0')
        else:
            disp_image = file_image

            gray_img = cv2.cvtColor(file_image, cv2.COLOR_BGR2GRAY)
            faces_img = face_cascade.detectMultiScale(gray_img, 1.2, 5)
            if len(faces_img) > 0:
                (x0, y0, w0, h0) = faces_img[0]
                area = w0 * h0
                for (x1, y1, w1, h1) in faces_img:
                    if w1 * h1 > area:
                        (x0, y0, w0, h0) = (x1, y1, w1, h1)
                        area = w1 * h1
                face = []
                img = file_image[x0:x0 + w0, y0:y0 + h0]
                img = cv2.resize(img, (height, width))
                grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.cvtColor(grayimg, cv2.COLOR_GRAY2BGR)
                face.append(img)
                Top1, prob = WhosFace(face)

                mygroup = [11, 13, 14]
                mygp_pos = np.argmax([prob[11], prob[13], prob[14]])
                gpname = peoplenames[mygroup[mygp_pos]]

                if prob[mygroup[mygp_pos]] < threshold:
                    gpname = 'Unrecognized'
                cv2.rectangle(disp_image, (x0, y0), (x0 + w0, y0 + h0), (0, 255, 0), 2)

            else:
                gpname = 'No Face Detected'

            img_show = cv2.cvtColor(disp_image, cv2.COLOR_BGR2RGBA)
            img_show = cv2.resize(img_show,(400,300))
            current_show = Image.fromarray(img_show)
            imgtk2 = ImageTk.PhotoImage(image=current_show)
            panel.imgtk = imgtk2
            panel.config(image=imgtk2)

            rec_height = [0, 0, 0]
            if gpname and gpname != 'Unrecognized' and gpname !='No Face Detected':
                top1 = prob[peoplenames.index(gpname)]
                mem1_prob = prob[11]
                mem2_prob = prob[13]
                mem3_prob = prob[14]
                rec_height[0] = int(mem1_prob * 40)
                rec_height[1] = int(mem2_prob * 40)
                rec_height[2] = int(mem3_prob * 40)
                w.coords(rec1, (40, 51 - rec_height[0], 40 + rec_width, 50))
                w.coords(rec2, (80, 51 - rec_height[1], 80 + rec_width, 50))
                w.coords(rec3, (120, 51 - rec_height[2] , 120 + rec_width, 50))
                var.set(gpname)
                var_top1_prob.set('%.2f%%'%(100*top1))
            elif gpname and gpname == 'Unrecognized':
                w.coords(rec1, (40, 51, 40 + rec_width, 50))
                w.coords(rec2, (80, 51, 80 + rec_width, 50))
                w.coords(rec3, (120, 51, 120 + rec_width, 50))
                var.set(gpname)
                var_top1_prob.set('0.0')
            elif gpname and gpname == 'No Face Detected':
                w.coords(rec1, (40, 51, 40 + rec_width, 50))
                w.coords(rec2, (80, 51, 80 + rec_width, 50))
                w.coords(rec3, (120, 51, 120 + rec_width, 50))
                var.set(gpname)
                var_top1_prob.set('0.0')
            else:
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

def OpenImage():
    global file_image
    filename = filedialog.askopenfilename(initialdir='D:/Github/2/FaceCNN/test_image')
    if filename.endswith('.jpg'):
        file_image = cv2.imread(filename)
        print(filename)
    else:
        return file_image
    return file_image

OpenFile_btn = tk.Button(root, text="Select an image", command=OpenImage)
OpenFile_btn.pack(padx=10,pady=10,side='left')

pause_btn = tk.Button(root, text="START/PAUSE", command=display_video)
pause_btn.pack(padx=10,pady=10,side='left')

def Single_Image_Mode():
    return

def Mode_Choose():
    global Real_Time_Status,file_image
    if Real_Time_Status == False:
        Real_Time_Status = True
    else:
        Real_Time_Status = False
        file_image = []

Mode_Choose_btn = tk.Button(root, text="Mode Switch", command=Mode_Choose)
Mode_Choose_btn.pack(side='left', padx=10, pady=10)

btn = tk.Button(root, text="Finish Demo",fg='red',command= root.quit)
btn.pack(padx=10,pady=10,side='left')

video_loop()
root.mainloop()
cap.release()
cv2.destroyAllWindows()
sess.close()
