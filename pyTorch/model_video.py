import numpy as np
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math

import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms.functional as TF

weights_path = '/home/rick/Desktop/model/pesi_pytorch/Soluzione_resnext50_noCrop_terzoGiro_screen.pth'

#OVAL=[10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

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

# capture the webcam
cap = cv2.VideoCapture(0) #, cv2.CAP_DSHOW)
if (cap.isOpened() == False):
    print('Error while trying to open webcam. Plese check again...')
 
# get the frame width and height
w = int(cap.get(3))
h = int(cap.get(4))
y=0
x=0
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
output = cv2.VideoWriter('/home/rick/Videos/output.avi',fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while(cap.isOpened()):
    # capture each frame of the video
    ret, frame = cap.read()
    if ret == True:
        image = frame

        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = grayscale_image[y:y+h, x:x+w]
        image = TF.resize(Image.fromarray(image), size=(224, 224))
        image = TF.to_tensor(image)
        image = TF.normalize(image, [0.5], [0.5])
            
        with torch.no_grad():
            landmarks = best_network(image.unsqueeze(0)) 
           
        landmarks = (landmarks.view(47,2).detach().numpy() + 0.5) * np.array([[w, h]]) + np.array([[x, y]])
 
        for elem in landmarks:
            cv2.circle(frame, (int(elem[0]), int(elem[1])), 1, (0, 0, 255), -1, cv2.LINE_AA)   
            
        """
        for i in range(47):
                        if i==1:
                            cv2.circle(frame, (int(landmarks[i][0]), int(landmarks[i][1])), 3, (0, 5, 5), -1, cv2.LINE_AA)                        
                        if i==12:
                            cv2.circle(frame, (int(landmarks[i][0]), int(landmarks[i][1])), 3, (0, 5, 5), -1, cv2.LINE_AA)                               

                        if i==5:
                            cv2.circle(frame, (int(landmarks[i][0]), int(landmarks[i][1])), 3, (0, 255, 255), -1, cv2.LINE_AA)       
                        if i==44:
                            cv2.circle(frame, (int(landmarks[i][0]), int(landmarks[i][1])), 3, (0, 255, 255), -1, cv2.LINE_AA)      
        
        q=0
        for i in range(10,46):    
            if q!=0: 
                cv2.line(frame, (int(landmarks[i][0]),int(landmarks[i][1])), (a,b), (0,255,255),4)
            a=int(landmarks[i][0])
            b=int(landmarks[i][1])
            q=1
        """
        x1=int(landmarks[8,0])
        y1=int(landmarks[8,1])
        x2=int(landmarks[9,0])
        y2=int(landmarks[9,1])
        distance_r=math.sqrt((landmarks[17,0].item()-landmarks[0,0].item())**2+(landmarks[17,1].item()-landmarks[0,1].item())**2)
        distance_l=math.sqrt((landmarks[39,0].item()-landmarks[6,0].item())**2+(landmarks[39,1].item()-landmarks[6,1].item())**2)
        dist=distance_r/distance_l

        distance_up_r=math.sqrt((landmarks[1,0].item()-landmarks[12,0].item())**2+(landmarks[1,1].item()-landmarks[12,1].item())**2)
        distance_up_l=math.sqrt((landmarks[5,0].item()-landmarks[44,0].item())**2+(landmarks[5,1].item()-landmarks[44,1].item())**2)
        distance=math.sqrt((landmarks[10,0].item()-landmarks[28,0].item())**2+(landmarks[10,1].item()-landmarks[28,1].item())**2)
                    
        dist_up_r=distance_up_r/distance
        dist_up_l=distance_up_l/distance

        if dist_up_r>0.25:
            #print("geom dice right")
            right="down right"  
            print(dist_up_r)
        else:
            #print("geom dice attento")
            right="attento right"  
            print(dist_up_r)

        if dist_up_l>0.25:
            #print("geom dice right")
            left="down left"  
            print(dist_up_l)
        else:
            #print("geom dice attento")
            left="attento left"  
            print(dist_up_l)
        """
                    if dist>1.7:
                        #print("geom dice right")
                        pytorch="right"  
                        print(dist)
                    elif(dist<0.3):
                        #print("geom dice left")
                        pytorch="left"  
                        print(dist)
                    else:
                        #print("geom dice attento")
                        pytorch="attento"  
                        print(dist)

                    if((landmarks[46,1]).item()>0):
                            print("modello dice attento")
                            mediapipe="attento"
                    else:  
                            print("modello dice di no")
                            mediapipe="no"
"""
                    #cv2.putText(frame, f"modello: {mediapipe}",(x1,y1),cv2.FONT_HERSHEY_PLAIN,1.2,(0,255,255),1, cv2.LINE_AA)
                    #cv2.putText(frame, f"g: {pytorch}",(x1,y1+30),cv2.FONT_HERSHEY_PLAIN,2,(255,0,150),1, cv2.LINE_AA)
                    
        cv2.putText(frame, f"g: {right}",(x1,y1+30),cv2.FONT_HERSHEY_PLAIN,2,(255,0,150),1, cv2.LINE_AA)   
        cv2.putText(frame, f"g: {left}",(x1,y1+60),cv2.FONT_HERSHEY_PLAIN,2,(255,0,150),1, cv2.LINE_AA)

        cv2.rectangle(frame, (x1,  y1), (x2, y2), (255, 200, 0), 2)
        cv2.imshow('Solution', frame)
        # press `q` to exit

        output.write(frame)

        if cv2.waitKey(27) & 0xFF == ord('q'):
            break
        
    else: 
       break

# release VideoCapture()
cap.release()
output.release()

# close all frames and video windows
cv2.destroyAllWindows()
