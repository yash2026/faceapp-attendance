import cv2
import numpy as np
import face_recognition


imgyash = face_recognition.load_image_file('Imagesbasic/yash.jpg')
imgyash = cv2.cvtColor(imgyash,cv2.COLOR_BGR2RGB)
imgtest = face_recognition.load_image_file('Imagesbasic/test2.jpg')
imgtest = cv2.cvtColor(imgtest,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgyash)[0]
encodeyash = face_recognition.face_encodings(imgyash)[0]
cv2.rectangle(imgyash,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)
# print(faceLoc)

faceLoctest = face_recognition.face_locations(imgtest)[0]
encodetest = face_recognition.face_encodings(imgtest)[0]
cv2.rectangle(imgtest,(faceLoctest[3],faceLoctest[0]),(faceLoctest[1],faceLoctest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeyash],encodetest)
facedis = face_recognition.face_distance([encodeyash],encodetest)
print(results,facedis)
cv2.putText(imgtest,f'{results} {round(facedis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('yash',imgyash)
cv2.imshow('test1',imgtest)
cv2.waitKey(0)
