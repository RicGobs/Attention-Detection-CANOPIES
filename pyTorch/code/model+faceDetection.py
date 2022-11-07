from turtle import width
import numpy as np
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
import mediapipe as mp
mp_facedetector = mp.solutions.face_detection
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms.functional as TF


weights_path = '/home/rick/Desktop/model/pesi_pytorch/Soluzione_resnext50_noCrop_terzoGiro_screen.pth'

#OVAL=[10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
val=152
r=297 #12
l=67 #44

class Network(nn.Module):
    def __init__(self,num_classes=94):
        super().__init__()
        self.model_name='resnext50_32x4d'
        self.model=models.resnext50_32x4d(pretrained=False)
        self.model.conv1=nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc=nn.Linear(self.model.fc.in_features,num_classes)
        
    def forward(self, x):
        x=self.model(x)
        return x

best_network = Network()
best_network.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu'))) 
best_network.eval()        


face =  mp_facedetector.FaceDetection(
    model_selection=1, min_detection_confidence=0.5) #high distance, red

# capture the webcam
cap = cv2.VideoCapture(0) #, cv2.CAP_DSHOW)
if (cap.isOpened() == False):
    print('Error while trying to open webcam. Plese check again...')

fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
output = cv2.VideoWriter('/home/rick/Videos/output.avi',fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# get the frame width and height
ww = int(cap.get(3))
hh = int(cap.get(4))
#y=0
#x=0

import cv2
 
capture = cv2.VideoCapture(0)
 
#fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
#output = cv2.VideoWriter('/home/rick/Desktop/video.avi', fourcc, 30.0, (640,480))

while(cap.isOpened()):
    # capture each frame of the video
    ret, frame = cap.read()
    if ret == True:
        img = frame
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ris= face.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        all_landmarks = []
        if ris.detections:

            grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            for id, quadrato in enumerate(ris.detections):

                cont = quadrato.location_data.relative_bounding_box
                height_frame, width_frame, prof= frame.shape
                dis = int(cont.xmin * width_frame), int(cont.ymin * height_frame), int(cont.width * width_frame), int(cont.height * height_frame)
                #cv2.rectangle(frame,dis,(0,255,0),2)
                y=int(cont.ymin * height_frame)
                x=int(cont.xmin * width_frame)
                h=int(cont.height * height_frame)+30
                w=int(cont.width * width_frame)+30
                #display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = grayscale_image[y:y+h, x:x+w]

                image = TF.resize(Image.fromarray(image), size=(224, 224))
                image = TF.to_tensor(image)
                image = TF.normalize(image, [0.5], [0.5])
                with torch.no_grad():
                    landmarks = best_network(image.unsqueeze(0)) 

                landmarks = (landmarks.view(47,2).detach().numpy() + 0.5) * np.array([[w, h]]) + np.array([[x, y]])
                all_landmarks.append(landmarks)

                for landmarks in all_landmarks:
                    #for elem in landmarks:
                     #   print(elem)
                      #  cv2.circle(frame, (int(elem[0]), int(elem[1])), 1, (0, 0, 255), -1, cv2.LINE_AA)   
        
                    for i in range(47):
                        if i!=8 and i!=9 and i!=46:
                            cv2.circle(frame, (int(landmarks[i][0]), int(landmarks[i][1])), 1, (0, 0, 255), -1, cv2.LINE_AA)   

                        #if i==1:
                        #    cv2.circle(frame, (int(landmarks[i][0]), int(landmarks[i][1])), 3, (0, 5, 5), -1, cv2.LINE_AA)                        
                        #if i==12:
                        #    cv2.circle(frame, (int(landmarks[i][0]), int(landmarks[i][1])), 3, (0, 5, 5), -1, cv2.LINE_AA)                               
                        #if i==5:
                        #    cv2.circle(frame, (int(landmarks[i][0]), int(landmarks[i][1])), 3, (0, 255, 255), -1, cv2.LINE_AA)       
                        #if i==44:
                        #    cv2.circle(frame, (int(landmarks[i][0]), int(landmarks[i][1])), 3, (0, 255, 255), -1, cv2.LINE_AA)                              
 
                    x1=int(landmarks[8,0])
                    y1=int(landmarks[8,1])
                    x2=int(landmarks[9,0])
                    y2=int(landmarks[9,1])
                    distance_r=math.sqrt((landmarks[17,0].item()-landmarks[0,0].item())**2+(landmarks[17,1].item()-landmarks[0,1].item())**2)
                    distance_l=math.sqrt((landmarks[39,0].item()-landmarks[6,0].item())**2+(landmarks[39,1].item()-landmarks[6,1].item())**2)
                    dist=distance_r/distance_l
                    #cv2.circle(frame, (int(landmarks[0][0]), int(landmarks[0][1])), 3, (0, 255, 255), -1, cv2.LINE_AA)  
                    distance_down_r=math.sqrt((landmarks[1,0].item()-landmarks[12,0].item())**2+(landmarks[1,1].item()-landmarks[12,1].item())**2)
                    distance_down_l=math.sqrt((landmarks[5,0].item()-landmarks[44,0].item())**2+(landmarks[5,1].item()-landmarks[44,1].item())**2)
                    distance_down=math.sqrt((landmarks[10,0].item()-landmarks[28,0].item())**2+(landmarks[10,1].item()-landmarks[28,1].item())**2)
                    
                    dist_down_r=distance_down_r/distance_down
                    dist_down_l=distance_down_l/distance_down

                    distance_up=math.sqrt((landmarks[12,0].item()-landmarks[44,0].item())**2+(landmarks[12,1].item()-landmarks[44,1].item())**2)

                    if dist_down_r>0.26:
                        right="down"  
                        #print(dist_down_r)
                    else:
                        if distance_up<55:
                            right="up"  
                            print(distance_up,dist_down_r)
                        else:
                            right="att"  
                            print(distance_up,dist_down_r)  

                    if dist_down_l>0.24:
                        left="down"  
                        #print(dist_down_l)
                    else:
                        if distance_up<55:
                            left="up"  
                            print(distance_up,dist_down_l)
                        else:
                            left="att"  
                            print(distance_up,dist_down_l)  

                    if dist>1.7:
                        #print("geom dice right")
                        cen="right"  
                        #print(dist)
                    elif(dist<0.3):
                        #print("geom dice left")
                        cen="left"  
                        #print(dist)
                    else:
                        #print("geom dice attento")
                        cen="att"  
                        #print(dist)

                    cv2.putText(frame, f"o: {cen}",(x1,y1+90),cv2.FONT_HERSHEY_PLAIN,2,(255,0,150),1, cv2.LINE_AA)
                    cv2.putText(frame, f"r_v: {right}",(x1,y1+30),cv2.FONT_HERSHEY_PLAIN,2,(255,0,150),1, cv2.LINE_AA)   
                    cv2.putText(frame, f"l_v: {left}",(x1,y1+60),cv2.FONT_HERSHEY_PLAIN,2,(255,0,150),1, cv2.LINE_AA)

                    if cen=="att" and (right=="att" and left=="att"):
                        cv2.putText(frame, f"ATTENTO",(x1-60,y1-30),cv2.FONT_HERSHEY_PLAIN,2,(255,255,150),1, cv2.LINE_AA)

                    cv2.rectangle(frame, (x1,  y1), (x2, y2), (255, 200, 0), 2)
                    
        output.write(frame)
        #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cv2.imshow('Solution', frame)
        #output.write(frame)
        # press `q` to exit
        if cv2.waitKey(27) & 0xFF == ord('q'):
            break
 
    else: 
       break

# release VideoCapture()
cap.release()
output.release()
# close all frames and video windows
cv2.destroyAllWindows()
