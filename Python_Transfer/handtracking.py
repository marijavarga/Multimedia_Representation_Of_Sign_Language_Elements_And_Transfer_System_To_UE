from multiprocessing.forkserver import write_signed
from multiprocessing.sharedctypes import Value
from cv2 import NORM_MINMAX, cvtColor, imshow, normalize
import numpy
import cv2
import numpy as np
import math

import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import json

#===============
import socket
import requests

url='http://localhost:3000/'

correct_payload = {'username': 'hello', 'password': '0'}





class Obelezje:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0

    def __init__(self, x,y,z):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return "Obelezje.....x: "+str(self.x)+" y: "+str(self.y)+" z: "+str(self.z)

class Ruka:
        


        wrist = Obelezje(0,0,0) #0

        thumb_cmc = Obelezje(0,0,0)  #1
        thumb_mcp = Obelezje(0,0,0)#2
        thumb_tip = Obelezje(0,0,0)#3
        thumb_ip = Obelezje(0,0,0)#4

        index_finger_mcp = Obelezje(0,0,0)#5
        index_finger_pip = Obelezje(0,0,0)#6
        index_finger_dip = Obelezje(0,0,0)#7
        index_finger_tip = Obelezje(0,0,0)#8

        middle_finger_mcp = Obelezje(0,0,0)#9
        middle_finger_pip = Obelezje(0,0,0)#10
        middle_finger_dip = Obelezje(0,0,0)#11
        middle_finger_tip = Obelezje(0,0,0)#12

        ring_finger_mcp = Obelezje(0,0,0)#13
        ring_finger_pip = Obelezje(0,0,0)#14
        ring_finger_dip = Obelezje(0,0,0)#15
        ring_finger_tip = Obelezje(0,0,0)#16

        pinky_mcp = Obelezje(0,0,0)#17
        pinky_pip = Obelezje(0,0,0)#18
        pinky_dip = Obelezje(0,0,0)#19
        pinky_tip = Obelezje(0,0,0)#20
        ###21 obelezja    
        gest_hand = ''
       
        


class HandEncoder(json.JSONEncoder):
        def default(self, o):
            return o.__dict__
        


# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils


# Load the gesture recognizer model
model = load_model('mp_hand_gesture')

# Load class names
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)

rem = 0


cap = cv2.VideoCapture(0) #kanal za video sekvencu
while(cap.isOpened()):

    
    
    ruka = Ruka()

    ret, img = cap.read() #ucitavanje slike
    img = cv2.flip(img, 1)

    xW, yH, c1 = img.shape
    #cv2.rectangle(img, (640,640 ), (0,0), (0,255,0), 0)
   # crop_img = img[0:640, 0:640] #detekcija pokreta se desava samo u ovom okviru 
    cv2.rectangle(img, (300,300 ), (100,100), (0,255,0), 0)
    crop_img = img[100:300, 100:300] #detekcija pokreta se desava samo u ovom okviru 

    grey_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY) # grayscale

    value = (35,35) #
    blurred = cv2.GaussianBlur(grey_img, value, 0)

    _, thresh_img = cv2.threshold(blurred, 127,255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) #otsu metod binarizacija

    cv2.imshow('ThresholdedImage', thresh_img)

    (version, _, _) = cv2.__version__.split('.')

    if version == '3':
        image, contours, hierarchy = cv2.findContours(thresh_img.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    
    elif version == '4':
        contours, hierarchy = cv2.findContours(thresh_img.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    cnt = max(contours, key = lambda x: cv2.contourArea(x)) #pretpostavka je da je najveca povrsina kontura ruke

    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(crop_img, (x,y), (x+w, y+h), (0,0,255),0) #BGR

    hull= cv2.convexHull(cnt) 

    drawing = np.zeros(crop_img.shape,np.uint8)
    cv2.drawContours(drawing, [cnt], 0, (0,255,0),0) #zelena
    cv2.drawContours(drawing, [hull], 0,(0,0,255),0) #crvena BGR model

    hull = cv2.convexHull(cnt, returnPoints=False)

    defects = cv2.convexityDefects(cnt, hull)
    count_defects = 0
    cv2.drawContours(thresh_img, contours, -1,(0,255,0), 3)

    imageHeight, imageWidth, _ = img.shape

    for i in range(defects.shape[0]):
        s, e, f, d = defects[i,0]

        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])

        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

        angle  = math.acos ((b**2 + c**2 - a**2)/(2*b*c)) * 57

        if angle <= 90:
            count_defects += 1
            cv2.circle(crop_img, far, 1,[0,255,0], -1)
        cv2.line(crop_img, start, end, [0,255,0], 2)

    if count_defects == 0:
        cv2.putText(img, "1 - jedan prst", (10,100),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    elif count_defects==2:
        cv2.putText(img, "3 - tri prsta", (10,100),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    elif count_defects ==3:
        cv2.putText(img, "4 - cetiri prsta", (10,100),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    elif count_defects ==4:
         cv2.putText(img, "5 - pet prstiju", (10,100),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    else:
         cv2.putText(img, "2 - dva prsta", (10,100),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    
# Get hand landmark prediction
    result = hands.process(img)

    print(result)
    
    className = ''

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = lmx = int(lm.x*xW)
                lmy = int(lm.y*yH )
                landmarks.append([lmx, lmy])
               

            for point in mpHands.HandLandmark:
     
                normalizedLandmark = handslms.landmark[point]
                pixelCoordinatesLandmark = mpDraw._normalized_to_pixel_coordinates(normalizedLandmark.x, normalizedLandmark.y, imageWidth, imageHeight)

                if (str(point).split("."))[1]=="WRIST":
                    ruka.wrist=Obelezje(normalizedLandmark.x, normalizedLandmark.y,normalizedLandmark.z )

                if (str(point).split("."))[1]=="THUMB_CMC":
                    ruka.thumb_cmc=Obelezje(normalizedLandmark.x, normalizedLandmark.y,normalizedLandmark.z )
                if (str(point).split("."))[1]=="THUMB_MCP":
                    ruka.thumb_mcp=Obelezje(normalizedLandmark.x, normalizedLandmark.y,normalizedLandmark.z )
                if (str(point).split("."))[1]=="THUMB_IP":
                    ruka.thumb_ip=Obelezje(normalizedLandmark.x, normalizedLandmark.y,normalizedLandmark.z )
                if (str(point).split("."))[1]=="THUMB_TIP":
                    ruka.thumb_tip=Obelezje(normalizedLandmark.x, normalizedLandmark.y,normalizedLandmark.z )
                
                if (str(point).split("."))[1]=="INDEX_FINGER_MCP":
                    ruka.index_finger_mcp=Obelezje(normalizedLandmark.x, normalizedLandmark.y,normalizedLandmark.z )
                if (str(point).split("."))[1]=="INDEX_FINGER_PIP":
                    ruka.index_finger_pip=Obelezje(normalizedLandmark.x, normalizedLandmark.y,normalizedLandmark.z )
                if (str(point).split("."))[1]=="INDEX_FINGER_DIP":
                    ruka.index_finger_dip=Obelezje(normalizedLandmark.x, normalizedLandmark.y,normalizedLandmark.z )
                if (str(point).split("."))[1]=="INDEX_FINGER_TIP":
                    ruka.index_finger_tip=Obelezje(normalizedLandmark.x, normalizedLandmark.y,normalizedLandmark.z )
                
                if (str(point).split("."))[1]=="MIDDLE_FINGER_MCP":
                    ruka.middle_finger_mcp=Obelezje(normalizedLandmark.x, normalizedLandmark.y,normalizedLandmark.z )
                if (str(point).split("."))[1]=="MIDDLE_FINGER_PIP":
                    ruka.middle_finger_pip=Obelezje(normalizedLandmark.x, normalizedLandmark.y,normalizedLandmark.z )
                if (str(point).split("."))[1]=="MIDDLE_FINGER_DIP":
                    ruka.middle_finger_dip=Obelezje(normalizedLandmark.x, normalizedLandmark.y,normalizedLandmark.z )
                if (str(point).split("."))[1]=="MIDDLE_FINGER_TIP":
                    ruka.middle_finger_tip=Obelezje(normalizedLandmark.x, normalizedLandmark.y,normalizedLandmark.z )
                
                if (str(point).split("."))[1]=="RING_FINGER_MCP":
                    ruka.ring_finger_mcp=Obelezje(normalizedLandmark.x, normalizedLandmark.y,normalizedLandmark.z )
                if (str(point).split("."))[1]=="RING_FINGER_PIP":
                    ruka.ring_finger_pip=Obelezje(normalizedLandmark.x, normalizedLandmark.y,normalizedLandmark.z )
                if (str(point).split("."))[1]=="RING_FINGER_DIP":
                    ruka.ring_finger_dip=Obelezje(normalizedLandmark.x, normalizedLandmark.y,normalizedLandmark.z )
                if (str(point).split("."))[1]=="RING_FINGER_TIP":
                    ruka.ring_finger_tip=Obelezje(normalizedLandmark.x, normalizedLandmark.y,normalizedLandmark.z )
                
                if (str(point).split("."))[1]=="PINKY_MCP":
                    ruka.pinky_mcp=Obelezje(normalizedLandmark.x, normalizedLandmark.y,normalizedLandmark.z )
                if (str(point).split("."))[1]=="PINKY_PIP":
                    ruka.pinky_pip=Obelezje(normalizedLandmark.x, normalizedLandmark.y,normalizedLandmark.z )
                if (str(point).split("."))[1]=="PINKY_DIP":
                    ruka.pinky_dip=Obelezje(normalizedLandmark.x, normalizedLandmark.y,normalizedLandmark.z )
                if (str(point).split("."))[1]=="PINKY_TIP":
                    ruka.pinky_tip=Obelezje(normalizedLandmark.x, normalizedLandmark.y,normalizedLandmark.z )
               


             
                
                with open('C:\\Users\\Marija\\Documents\\Unreal Projects\\HandTrack\\Content\\Files\\textDoc.txt', 'a') as f:
                    f.writelines(str(point)+ '\n' + str(normalizedLandmark) + '\n')  



            # Drawing landmarks on frames
            mpDraw.draw_landmarks(img, handslms, mpHands.HAND_CONNECTIONS)
            prediction = model.predict([landmarks])
            print(prediction)
            classID = np.argmax(prediction)
            className = classNames[classID]

            

    
    
    # show the prediction on the frame
    cv2.putText(img, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)


    cv2.imshow('Gesture', img)
    all_img = np.hstack((drawing, crop_img))
    cv2.imshow('Contours', all_img)
    
    
    
    if rem%10 == 0:
        rem=0
        ruka.gest_hand = className
        rem = rem+1
        
    else: 
        rem = rem+1
    
    rukaJson = HandEncoder().encode(ruka)
    print(rukaJson)

       
    correct_payload['password'] = rukaJson
    r = requests.post(url, data=correct_payload)
    

    print(correct_payload)

    k = cv2.waitKey(10)
    if k==27:
        break




