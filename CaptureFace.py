import cv2
import cnnconfig as cf

height = cf.height
width = cf.width

id=1
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
face_cascade.load('D:/OpenCV/opencv/sources/data/haarcascades_cuda/haarcascade_frontalface_alt.xml')

while(1):
    ret, frame = cap.read()
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

        img = frame[x0:x0 + w0, y0:y0 + h0]
        img = cv2.resize(img, (height, width))
        grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(grayimg,cv2.COLOR_GRAY2BGR)

        cv2.imwrite('capture'+'/'+'New'+'/'+'New'+'_%d'%(id)+'.jpg',img)
        id = id + 1
    cv2.imshow("capture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()