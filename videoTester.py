import os
import cv2
import numpy as np
import faceRecognition as fr


#This module captures images via webcam and performs face recognitionqqqq
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('trainingData.yml')#Load saved training data

name = {0 : "Priyanka",1 : "Kangana"}

#to save a video on your system VideoWriter() is uesd
#but before that we required an other method that is cv2.VideoWriter_fourcc()
#four_cc is a color codes 4 byte ('m','p','g','n') or (*'mpgn')
cap=cv2.VideoCapture(0)
#some times cap is not initialized in order to check so we have to use cap.isopened()


while True:
    #it will return an boolean value True when the videocapture is on else will return false
    ret,test_img=cap.read()# captures frame and returns boolean value and captured image
    faces_detected,gray_img=fr.faceDetection(test_img)



    for (x,y,w,h) in faces_detected:
      cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)

    resized_img = cv2.resize(test_img, (1000, 700)) 
    cv2.imshow('face detection Tutorial ',resized_img)
    cv2.waitKey(10)
    #if waitKey is paased with 10 millisecond it will wait to 10 ms if any key is not pressed it will get stoped

    for face in faces_detected:
        (x,y,w,h)=face
        roi_gray=gray_img[y:y+w, x:x+h]
        label,confidence=face_recognizer.predict(roi_gray)#predicting the label of given image
        print("confidence:",confidence)
        print("label:",label)
        fr.draw_rect(test_img,face)
        predicted_name=name[label]
        if confidence < 39:#If confidence less than 37 then don't print predicted face text on screen
           fr.put_text(test_img,predicted_name,x,y)


    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('face recognition tutorial ',resized_img)
    if cv2.waitKey(10) == ord('q'):#wait until 'q' key is pressed
        break


cap.release()
cv2.destroyAllWindows

