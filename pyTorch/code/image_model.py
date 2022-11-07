import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms.functional as TF

def load_images_from_folder(folder):
    images_dict = {}
    for filename in os.listdir(folder):
        images_dict[filename]= cv2.imread(folder+"/"+filename) 
    return images_dict

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

ims={}
ims=load_images_from_folder("/home/rick/Desktop/model/accuracy_error")  
#######################################################################
images={}
images=load_images_from_folder("/home/rick/Desktop/model/test_Att_vecchio")   
image_path = "/home/rick/Desktop/model/test_Att_vecchio/"
weights_path = '/home/rick/Desktop/model/Soluzione_resnext50_noCrop_terzoGiro_screen.pth'
#######################################################################
data_bbox = np.load('/home/rick/Desktop/model/data_bbox_mod.npy',)
data_names = np.load('/home/rick/Desktop/model/data_names_mod.npy')
data_attenction = np.load("/home/rick/Desktop/model/data_attenction_mod.npy")
#######################################################################
accuracy=0
acc=0
len=0

true_positive=0
false_positive=0
true_negative=0
false_negative=0

precision=0
recall=0
f1_score=0

total_mse=0
mse=0
 
total_mae=0
mae=0

total_deviation=0
deviation=0
#######################################################################

best_network = Network()
best_network.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu'))) 
best_network.eval()

for name, image in images.items():
  
  frame = image.copy()
  im=image_path+name
  image = cv2.imread(im)
  grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  height, width,_ = image.shape
  x=0
  y=0
  w=width
  h=height

  image = grayscale_image[y:y+h, x:x+w]
  image = TF.resize(Image.fromarray(image), size=(256, 256)) # 224 oppure 256
  image = TF.to_tensor(image)
  image = TF.normalize(image, [0.5], [0.5])

  with torch.no_grad():
      landmarks = best_network(image.unsqueeze(0)) 

  landmarks = (landmarks.view(47,2).detach().numpy() + 0.5) * np.array([[w, h]]) + np.array([[x, y]])
  #all_landmarks.append(landmarks)

  cv2.rectangle(frame, (int(landmarks[8,0]), int(landmarks[8,1])), (int(landmarks[9,0]), int(landmarks[9,1])), (0, 255, 0), 2)

  for i in range(47):
    if i!=46 and i!=8 and i!=9:
      cv2.circle(frame, (int(landmarks[i][0]), int(landmarks[i][1])), 2, (255, 255, 255), 3 ,cv2.LINE_AA)   

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
      #print(distance_up,dist_down_r)
    else:
      right="att"  
      #print(distance_up,dist_down_r)  
  if dist_down_l>0.24:
      left="down"  
      #print(dist_down_l)
  else:
      if distance_up<55:
        left="up"  
        #print(distance_up,dist_down_l)
      else:
        left="att"  
        #print(distance_up,dist_down_l)  

  if dist>1.6:
    #print("geom dice right")
    cen="right"  
    #print(dist)
  elif(dist<0.4):
    #print("geom dice left")
    cen="left"  
    #print(dist)
  else:
    #print("geom dice attento")
    cen="att"  
   #print(dist)

  #cv2.putText(frame, f"o: {cen}",(x1,y1+90),cv2.FONT_HERSHEY_PLAIN,2,(255,0,150),1, cv2.LINE_AA)
  #cv2.putText(frame, f"r_v: {right}",(x1,y1+30),cv2.FONT_HERSHEY_PLAIN,2,(255,0,150),1, cv2.LINE_AA)   
  #cv2.putText(frame, f"l_v: {left}",(x1,y1+60),cv2.FONT_HERSHEY_PLAIN,2,(255,0,150),1, cv2.LINE_AA)

  if cen=="att":
    cv2.putText(frame, f"ATTENTO",(x1-70,y1-30),cv2.FONT_HERSHEY_PLAIN,5,(255,255,150),3, cv2.LINE_AA)

  
  cv2.imwrite("/home/rick/Desktop/model/train/"+name, frame)

"""
print("######################\n")

mae=total_mae/len
mae=mae/256
print("mae -> ",mae)
print()

mse=total_mse/len
mse=mse/256
print("mse -> ",mse)
print()

deviation=total_deviation/len
deviation=deviation/256
print("deviation -> ",deviation)
print()

print("######################\n")

accuracy=acc/len
print("accuracy -> ",accuracy)
print()

precision=true_positive/(true_positive+false_positive)
print("precision -> ",precision)
print()

recall=true_positive/(true_positive+false_negative)
print("recall -> ",recall)
print()

f1_score=2*(recall*precision)/(recall+precision)
print("f1_score -> ",f1_score)
print()

# initialize list of lists
data = [[true_positive, false_positive], [false_negative, true_negative]]
# Create the pandas DataFrame
df = pd.DataFrame(data, columns=['Actual positive', 'Actual negative'],index=['Predicted positive', 'Predicted negative'])
print(df)
print()

print(len)
"""
