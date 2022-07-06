import mediapipe as mp
import cv2 as cv
mp_facedetector = mp.solutions.face_detection

cap = cv.VideoCapture(0)
face1 =  mp_facedetector.FaceDetection( 
    model_selection=0, min_detection_confidence=0.5) #low distance, green

face2 =  mp_facedetector.FaceDetection(
    model_selection=1, min_detection_confidence=0.5) #high distance, red

while cap.isOpened():
        s, img = cap.read()
        #convert BGR in RGB (I understood that is better on online paper)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        ris1 = face1.process(img)
        ris2= face2.process(img)
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

        if ris1.detections and ris2.detections:
            for id, quadrato in enumerate(ris1.detections):
                print(id, quadrato)

                cont = quadrato.location_data.relative_bounding_box

                altezza, larghezza, prof= img.shape

                dis = int(cont.xmin * larghezza), int(cont.ymin * altezza), int(cont.width * larghezza), int(cont.height * altezza)
                cv.rectangle(img,dis,(0,255,0),2)
                cv.putText(img, f'{int(quadrato.score[0]*100)}%', (dis[0]+100, dis[1] - 20), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            for id, quadrato in enumerate(ris2.detections):
                print(id, quadrato)

                cont = quadrato.location_data.relative_bounding_box

                altezza, larghezza, prof= img.shape

                dis = int(cont.xmin * larghezza), int(cont.ymin * altezza), int(cont.width * larghezza), int(cont.height * altezza)
                cv.rectangle(img,dis,(0,0,255),2)
                cv.putText(img, f'{int(quadrato.score[0]*100)}%', (dis[0], dis[1] - 20), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
           
        elif ris1.detections and not ris2.detections:
            for id, quadrato in enumerate(ris1.detections):
                print(id, quadrato)

                cont = quadrato.location_data.relative_bounding_box

                altezza, larghezza, prof= img.shape

                dis = int(cont.xmin * larghezza), int(cont.ymin * altezza), int(cont.width * larghezza), int(cont.height * altezza)
                cv.rectangle(img,dis,(0,255,0),2)
                cv.putText(img, f'{int(quadrato.score[0]*100)}%', (dis[0]+100, dis[1] - 20), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        elif ris2.detections and not ris1.detections:
            for id, quadrato in enumerate(ris2.detections):
                print(id, quadrato)

                cont = quadrato.location_data.relative_bounding_box

                altezza, larghezza, prof= img.shape

                dis = int(cont.xmin * larghezza), int(cont.ymin * altezza), int(cont.width * larghezza), int(cont.height * altezza)
                cv.rectangle(img,dis,(0,0,255),2)
                cv.putText(img, f'{int(quadrato.score[0]*100)}%', (dis[0], dis[1] - 20), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)       

        cv.imshow('Solution', img)

        if cv.waitKey(5) & 0xFF == 27:
            break

cap.release()