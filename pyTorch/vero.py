import cv2 as cv
import numpy as np
import mediapipe as mp
import math
import pandas as pd
import os

mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
draw_spec=mp_drawing.DrawingSpec(thickness=-1,circle_radius=1)

LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  

RIGHT_IRIS=[474,475,476,477]
LEFT_IRIS=[469,470,471,472]       

L_LEFT=[33]
L_LEFT_ESTERNO=[127]
L_RIGHT=[133]
L_UP=[159]
L_DOWN=[145]

R_RIGHT_ESTERNO=[356]
R_LEFT=[362]
R_RIGHT=[263]
R_UP=[386]
R_DOWN=[374]

NOSE=[19]


#-----------FUNCTIONS---------

def distance_x(p1,p2):
    x1,y1=p1.ravel()
    x2,y2=p2.ravel()
    distance=math.sqrt((x2-x1)**2+(y2-y1)**2) #così calcolo solo destra e sinistra
    if distance==0:
        distance=1
    return distance

def distance_y(p1,p2):
    x1,y1=p1.ravel()
    x2,y2=p2.ravel()
    distance=math.sqrt((y2-y1)**2+(x2-x1)**2) #così calcolo solo sopra e sotto
    if distance==0:
        distance=1
    return distance

def iris_right_horizontal(center,right,left):
    dist=distance_x(center,right)
    total_dist=distance_x(right,left)        
    val=dist/total_dist
    pos=""
    if val>0.41  and val<=0.53:
        pos="center"
        #pos="ATTENTO"
    elif val<=0.41:
        pos="right"
    else:
        pos="left"    
    return pos,val

def iris_left_horizontal(center,right,left):
    dist=distance_x(center,left)
    total_dist=distance_x(right,left)    
    val=dist/total_dist
    pos=""
    if val>0.41 and val<=0.55:
        pos="center"
        #pos="ATTENTO"
    elif val<=0.41:
        pos="left"
    else:
        pos="right"    
    return pos,val    

def iris_right_vertical(center,up,down):
    dist=distance_y(center,up)
    total_distance=distance_y(up,down)   
    val=dist/total_distance
    pos=""
    if val>0.40  and val<=0.54:
        pos="center"
        #pos="ATTENTO"
    elif val<=0.40:
        pos="up"
    else:
        pos="down"    
    return pos,val

def iris_left_vertical(center,up,down):
    dist=distance_y(center,up)
    total_distance=distance_y(up,down)   
    val=dist/total_distance
    pos=""
    if val>0.50  and val<=0.57:
        pos="center"
        #pos="ATTENTO"
    elif val<=0.40:
        pos="up"
    else:
        pos="down"    
    return pos,val

def i_function(center,right,left,esterno):
    dist1=distance_x(esterno,right)  
    dist2=distance_x(center,left)  
    pos=""
    if dist1>18  and dist1<=50:
        pos="fcentrale"
        #pos="ATTENTO"
    elif dist1<=18:
        pos="fdestra"
    else:
        pos="fsinistra"    
    return pos,dist1

def distanza(right,left):
    dist1=(right-left)*100
    #print(dist1)
    pos=""
    if dist1>-3.5  and dist1<=4:
        pos="fcentrale"
        #pos="ATTENTO"
    elif dist1<=-3.5:
        pos="fsinistra"
    else:
        pos="fdestra"    
    return pos,dist1
#---------------MAIN------------

bbox=[]
names=[]
a=0
attenction=[]

def create_annotation(name_obj,puntiR,puntiL,bound_box,att):
    puntiR=np.concatenate((puntiR, puntiL), axis=0)
    puntiR=np.concatenate((puntiR, bound_box), axis=0)
    bbox.append(puntiR)
    names.append(name_obj)
    if att=="":
        a=0
    else:
        a=1    
    attenction.append(a)
    


def resize_and_show(filename_img,img):
  #print(filename_img)
  cv.imwrite("/home/riccardo/Desktop/finale/train/"+filename_img+".png", img)
  #cv.imshow('image',img)
  cv.waitKey(0)

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, 
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

def load_images_from_folder(folder):
    images_dict = {}
    for filename in os.listdir(folder):
        file=filename
        images_dict[file]= cv.imread(folder+"/"+filename) 
    return images_dict

images={}
#images["/home/riccardo/Desktop/py/archive/train_2/image_0001.png"]=cv.imread("/home/riccardo/Desktop/py/archive/train_2/image_0001.png")
images=load_images_from_folder("/home/riccardo/Desktop/finale/train_13")   


for name, image in images.items():
    
    h,w=image.shape[:2]
    results = face_mesh.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    frame = image.copy()
    if results.multi_face_landmarks:
        i=0
        prof={}
        box_lista=[]
        att_lista=[]
        for face_landmarks in results.multi_face_landmarks:

           #mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

           h, w, c = frame.shape
           cx_min=  w
           cy_min = h
           cx_max= cy_max= 0
           for id, lm in enumerate(face_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                if cx<cx_min:
                    cx_min=cx
                if cy<cy_min:
                    cy_min=cy
                if cx>cx_max:
                    cx_max=cx
                if cy>cy_max:
                    cy_max=cy
           cv.rectangle(frame, (cx_min, cy_min), (cx_max, cy_max), (255, 255, 0), 2)
           bounding_box=np.array([[cx_min,cy_min],[cx_max, cy_max],[cx_max, cy_min],[cx_min, cy_max]])

           punti_3d=np.array([([a.x,a.y,a.z]) for a in face_landmarks.landmark ])
           #print(punti_3d)

           punti_mesh_face=np.array([np.multiply([p.x,p.y],[w,h]).astype(int) for p in results.multi_face_landmarks[i].landmark])
           #I take all landmarks of a face
           dist_mesh_face=np.array([([p.z]) for p in results.multi_face_landmarks[i].landmark])
           #print(punti_mesh_face)
           (l_x,l_y), l_radius=cv.minEnclosingCircle(punti_mesh_face[LEFT_IRIS])
           (r_x,r_y), l_radius=cv.minEnclosingCircle(punti_mesh_face[RIGHT_IRIS])
           #print([l_x,l_y],[r_x,l_y])
           center_left=np.array([l_x,l_y],dtype=np.int32) #dove sta guardando occhio sinistro
           center_right=np.array([r_x,r_y],dtype=np.int32) #dove sta guardando occhio destro

           cv.circle(frame,center_left,int(l_radius),(255,0,255),1,cv.LINE_AA)
           cv.circle(frame,center_right,int(l_radius),(255,0,255),1,cv.LINE_AA)        
           cv.circle(frame, center_left, radius=2, color=(0, 255, 0), thickness=-1)
           cv.circle(frame, center_right, radius=2, color=(0, 255, 0), thickness=-1)
           cv.circle(frame,punti_mesh_face[R_RIGHT][0],3,(255,255,255),-1,cv.LINE_AA)
           cv.circle(frame,punti_mesh_face[R_LEFT][0],3,(0,255,255),-1,cv.LINE_AA)
           cv.circle(frame,punti_mesh_face[R_RIGHT_ESTERNO][0],3,(0,255,255),-1,cv.LINE_AA)
           
           iris_pos1,val1=iris_right_horizontal(center_right,punti_mesh_face[R_RIGHT][0],punti_mesh_face[R_LEFT][0])
           iris_pos2,val2=iris_left_horizontal(center_left,punti_mesh_face[L_RIGHT][0],punti_mesh_face[L_LEFT][0])
           iris_pos3,val3=iris_right_vertical(center_right,punti_mesh_face[R_UP][0],punti_mesh_face[R_DOWN][0])
           iris_pos4,val4=iris_left_vertical(center_left,punti_mesh_face[L_UP][0],punti_mesh_face[L_DOWN][0])
           i+=1
  
           #mp_drawing.draw_landmarks(image=frame,landmark_list=face_landmarks,connections=mp_face_mesh.FACEMESH_CONTOURS,landmark_drawing_spec=None,connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
           
           #iris_pos,var1=i_function(center_right,punti_mesh_face[R_RIGHT][0],punti_mesh_face[R_LEFT][0],punti_mesh_face[R_RIGHT_ESTERNO][0])
           iris_pos,var1=distanza(dist_mesh_face[R_RIGHT][0][0],dist_mesh_face[L_LEFT][0][0])
           #print(dist_mesh_face[R_RIGHT][0])

        
           cv.putText(frame,f"h_r: {iris_pos1} {val1: .2f}",(punti_mesh_face[R_RIGHT_ESTERNO][0][0],punti_mesh_face[R_RIGHT_ESTERNO][0][1]),cv.FONT_HERSHEY_PLAIN,1.2,(0,255,0),1,cv.LINE_AA)
           cv.putText(frame,f"h_l: {iris_pos2} {val2: .2f}",(punti_mesh_face[R_RIGHT_ESTERNO][0][0],punti_mesh_face[R_RIGHT_ESTERNO][0][1]+30),cv.FONT_HERSHEY_PLAIN,1.2,(0,0,255),1,cv.LINE_AA)
           cv.putText(frame, f"v_r: {iris_pos3} {val3: .2f}",(punti_mesh_face[R_RIGHT_ESTERNO][0][0],punti_mesh_face[R_RIGHT_ESTERNO][0][1]+60), cv.FONT_HERSHEY_PLAIN, 1.2,(0,255,0), 1,cv.LINE_AA)
           cv.putText(frame, f"v_l: {iris_pos4} {val4: .2f}",(punti_mesh_face[R_RIGHT_ESTERNO][0][0],punti_mesh_face[R_RIGHT_ESTERNO][0][1]+90),cv.FONT_HERSHEY_PLAIN,1.2,(0,0,255),1, cv.LINE_AA)
           cv.putText(frame, f"v: {iris_pos} {var1: .2f} ",(punti_mesh_face[R_RIGHT_ESTERNO][0][0],punti_mesh_face[R_RIGHT_ESTERNO][0][1]+120),cv.FONT_HERSHEY_PLAIN,1.2,(0,255,255),1, cv.LINE_AA)
           #h_l=horizontal_left
           #v_r=vertical_right
           #v_l=vertical_left
           #v=view of the face
           attenzione=""
           if (iris_pos1=="center" and iris_pos2=="center")  and iris_pos=="fcentrale": 
                attenzione="attento"
                prof[float(dist_mesh_face[NOSE][0][0])]=face_landmarks
                cv.putText(frame,attenzione,(punti_mesh_face[R_RIGHT_ESTERNO][0][0],punti_mesh_face[R_RIGHT_ESTERNO][0][1]+150),cv.FONT_HERSHEY_PLAIN, 2, (255,0,255), 1,cv.LINE_AA)
                
           elif (iris_pos1=="right" or iris_pos2=="right") and iris_pos=="fsinistra": 
                attenzione="attento"
                prof[float(dist_mesh_face[NOSE][0][0])]=face_landmarks
                cv.putText(frame,attenzione,(punti_mesh_face[R_RIGHT_ESTERNO][0][0],punti_mesh_face[R_RIGHT_ESTERNO][0][1]+150),cv.FONT_HERSHEY_PLAIN, 2, (255,0,255), 1,cv.LINE_AA)
                
           elif (iris_pos1=="left" or iris_pos2=="left") and iris_pos=="fdestra" : 
                attenzione="attento"
                prof[float(dist_mesh_face[NOSE][0][0])]=face_landmarks
                cv.putText(frame,attenzione,(punti_mesh_face[R_RIGHT_ESTERNO][0][0]-30,punti_mesh_face[R_RIGHT_ESTERNO][0][1]+150),cv.FONT_HERSHEY_PLAIN, 3, (255,0,255), 1,cv.LINE_AA)
           
        #I have written "down" and "up", but they are not accurate. I am not using them.
        e=0.0
        if prof:
            for elem in prof:
                #print(elem)
                if e==0.0:
                    e=elem 
                else:
                    if e>elem:
                        e=elem
            r=prof[e] 

        #cv.putText(frame,f"PROFONDO",(r[R_RIGHT_ESTERNO][0][0],punti_mesh_face[R_RIGHT_ESTERNO][0][1]-30),cv.FONT_HERSHEY_PLAIN, 2, (255,255,255), 1,cv.LINE_AA)
        '''
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=r,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=draw_spec,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())    
        '''
            

        if cv.waitKey(5) & 0xFF == 27:
            break

        #resize_and_show(name,frame)
        create_annotation(name,punti_mesh_face[RIGHT_IRIS],punti_mesh_face[LEFT_IRIS],bounding_box,attenzione)
        #print(bbox)

data_names=np.array(names)
data_bbox=np.array(bbox)
data_attenction=np.array(attenction)

np.save("/home/riccardo/Desktop/finale/data_names13",data_names)
np.save("/home/riccardo/Desktop/finale/data_bbox13",data_bbox)
np.save("/home/riccardo/Desktop/finale/data_attenction13",data_attenction)
