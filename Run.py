import cv2
import numpy as np
import face_recognition
import os
#from datetime import datetime
# from PIL import ImageGrab
path = 'C:\\Users\\zyedo\\Desktop\\Face_recognition\\images'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)
def findEncodings(images):
    encodeList = []
    for img in images:
     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
     encode = face_recognition.face_encodings(img)[0]
     encodeList.append(encode)
    return encodeList
#### FOR CAPTURING SCREEN RATHER THAN WEBCAM
# def captureScreen(bbox=(300,300,690+300,530+300)):
# capScr = np.array(ImageGrab.grab(bbox))
# capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
# return capScr
encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture("Test.mp4")
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))



frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('2ndop.avi', fourcc, fps, (frame_width, frame_height))


frame_number = 0

while True:
     success, img = cap.read()
     frame_number += 1
     if not success:
         break
 #img = captureScreen()
     imgS = cv2.resize(img,(0,0),None,0.25,0.25)
     imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
     facesCurFrame = face_recognition.face_locations(imgS)
     encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
     for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
         matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
         faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
         print(faceDis)
         matchIndex = np.argmin(faceDis)
         if matches[matchIndex]:
             name = classNames[matchIndex].upper()
             print(name)
             y1,x2,y2,x1 = faceLoc
             y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
             cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
             cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
     print("Writing frame {} / {}".format(frame_number, length))
     out.write(img)
     cv2.imshow('frame',img)

        
 
cap.release()
cv2.destroyAllWindows()

