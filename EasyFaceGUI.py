import tkinter as tk
from PIL import Image,ImageTk
import cv2
import numpy as np
import cnnconfig as cf
import Model
import tensorflow as tf
import FaceInput

height = cf.height
width = cf.width

cap = cv2.VideoCapture(0)

top = tk.Tk()
top.title('EasyFace GUI for Demo')
panel = tk.Label(master=top)
panel.pack(padx=10,pady=10)
top.config(cursor = 'arrow')

panel.grid(column=0, rowspan=4, padx=5, pady=5)

button1 = tk.Button(master=top, text='Function 1')
button1.grid(column=1, columnspan=2, row=0, padx=5, pady=5)
button2 = tk.Button(master=top, text='Function 2')
quit_button = tk.Button(master=top, text='Quit', bg="red3", fg="white")
quit_button.grid(column=1, row=3, padx=5, pady=5)

output = Model.FaceNet()
saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, tf.train.latest_checkpoint('checkpoint/'))
ppimg,pplabels,peoplenames=FaceInput.ReadFaceImg.GetDataset()

def WhosFace(img):
    result = sess.run(output, feed_dict={Model.x : img, Model.keep_prob : 1.0})
    result = result[0]
    result_copy = result
    Top5_Index = []
    for i in range(5):
        pos = np.argmax(result_copy)
        result_copy[pos] = -1
        Top5_Index.append(pos)
    return Top5_Index,result

while(1):
    # get a frame
    ret, frame = cap.read()
    cv2.rectangle(frame, (200, 200), (400, 400), (0, 255, 0), 2)

    face=[]
    img = frame[200:400,200:400]
    img = cv2.resize(img,(height,width))
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(grayimg,cv2.COLOR_GRAY2BGR)
    face.append(img)
    nameIndex,prob = WhosFace(face)
    Peoplename = peoplenames[nameIndex[0]]
    text = 'The face is '
    cv2.putText(frame, text, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)

    cvimg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    current_frame = Image.fromarray(cvimg)
    tkimg = ImageTk.PhotoImage(image=current_frame)
    panel.tkimg = tkimg
    panel.config(image=tkimg)

    panel.update()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
sess.close()