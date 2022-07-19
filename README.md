# Face and Attention Detection

Il progetto è diviso in due parti:

1.implementare il sistema utilizzando il modello di mediapipe;

2.implementare il sistema creando un modello da zero su pytorch.


### 1. MEDIAPIPE 

L'implementazione dell'attention detection su Mediapipe utilizza il modello pre-addestrato di Mediapipe e in particolare il FaceMesh, sia per il riconoscimento del viso che degli occhi. Inizialmente si è partiti dal face Detection ma è stato compreso che non dava un contributo rilevante al progetto. Poichè non è stato usato anche il Mediapipe Pose per la stima della posizione del corpo e della faccia, si è calcolata la direzione di quest'ultima con l'utilizzo delle proporzioni facciali; lo stesso è stato fatto per la posizione delle iridi degli occhi; unendo i risultati si è ottenuto un sistema che rileva la direzione della faccia e dello sguardo, di conseguenza si è potuta identificare l'attenzione dei soggetti.

_Vantaggi:_ sistema affidabile nelle situazioni comuni, veloce, leggero.

_Svantaggi:_ sistema non affidabile in situazioni non comuni quali copertura del viso, direzione del viso troppo lontana dal centro della camera, ricerca di Mediapipe di trovare i landmark anche se coperti, inaffidabilità dei landmark e dello sguardo in situazioni molto particolari come angolature anormali.

Librerie usate: Mediapipe e OpenCV 

#### Entrambe le persone sono attente, la prima viene evidenziata
![Logo](https://github.com/RicGobs/LabVision/blob/main/mediapipe/EyeRecognition/solution1.jpg)

#### Solo la persona più vicina(la prima) è attenta, la prima viene evidenziata
![Logo](https://github.com/RicGobs/LabVision/blob/main/mediapipe/EyeRecognition/solution2.jpg)

#### Solo la persona più lonatana(la seconda) è attenta, la seconda viene evidenziata
![Logo](https://github.com/RicGobs/LabVision/blob/main/mediapipe/EyeRecognition/solution3.jpg)

### 2. PYTORCH
L'implementazione dell'attention detection non è completa, si sta ricercando al meglio i landmark del voto concentrandosi su quelli delle iridi degli occhi. Addestrato al meglio il modello per fare questo, si concluderà con la classificazione dell'attenzione dei soggetti. Il sistema usa un dataset creato grazie alla prima parte del progetto, l'inferenza permessa è sufficiente per osservare che il modello riesce a predire la zona vicino agli occhi ma la sua accuratezza è minore di quella di Mediapipe.
Le immagini vengono passate con tre numpy array: uno per i nomi delle immagini, uno con i landmarks ed uno con il 0/1 per la classificazione dell'attenzione. Viene ritagliata la faccia dall'immagine e viene data in pasto al modello. E' stata provata ResNet18 la quale è troppo piccola per risolvere il problema, e ResNet50 che svolge un lavoro mediocre. In futuro si proveranno ulteriori modelli.

_Vantaggi:_ sistema più strutturato sul quale si possono risolvere i problemi presentati in precedenza su Mediapipe.

_Svantaggi:_ online non sono presenti dataset per il riconoscimento dei 478 landmarks del viso e nemmeno quelli per le iridi degli occhi, sono presenti dei dataset che hanno dai 20 ai 68 landmark ma non permettono il riconoscimento dell'iride, solamente dell'occhio. Questo problema potrebbe essere superato con l'utilizzo di un pose estimation ma non è l'obbiettivo del progetto. Il dataset, osservando alcuni dei risultati ottenuti da Mediapipe è meno performante di quello che si riteneva. La scarsa accuratezza è dovuta quindi anche al dataset non completamene preciso.

Librerie usate: Pytorch, Pytorch models, Mediapipe per il dataset, OpenCV per rappresentare le immagini



----

The project is divided into two parts:

1. to implement the system using the mediapipe model;

2. implement the system by creating a model from scratch on pytorch.


### 1. MEDIAPIPE

The implementation of attention detection on Mediapipe uses the pre-trained model of Mediapipe and in particular the FaceMesh, both for face and eye recognition. Initially, we started with face detection but it was understood that it did not make a significant contribution to the project. Since the Mediapipe Pose was not used to estimate the position of the body and face, the direction of the latter was calculated using the facial proportions; the same was done for the position of the irises of the eyes; by combining the results, a system was obtained that detects the direction of the face and gaze, consequently it was possible to identify the attention of the subjects.

_Advantages: _ Reliable system in common situations, fast, light.

_Disadvantages: _ system not reliable in uncommon situations such as face coverage, direction of the face too far from the center of the camera, Mediapipe's search to find the landmarks even if they are covered, unreliability of landmarks and viewpoints in very particular situations such as abnormal angles.

Used libraries: Mediapipe and OpenCV


### 2. PYTORCH
The implementation of attention detection is not complete, we are researching the landmarks of the vote by focusing on those of the irises of the eyes. Trained the model best to do this, it will end with the classification of the subjects' attention. The system uses a dataset created thanks to the first part of the project, the allowed inference is sufficient to observe that the model is able to predict the area near the eyes but its accuracy is less than that of Mediapipe.
The images are passed with three numpy arrays: one for the image names, one with the landmarks and one with the 0/1 for the attention classification. The face is cut out of the image and fed to the model. We tried ResNet18 which is too small to solve the problem, and ResNet50 which does a mediocre job. Further models will be tested in the future.

_Advantages: _ more structured system on which the problems presented above on Mediapipe can be solved.

_Disadvantages: _ online there are no datasets for the recognition of the 478 landmarks of the face nor those for the irises of the eyes, there are datasets that have from 20 to 68 landmarks but do not allow the recognition of the iris, only of the eye. This problem could be overcome with the use of a pose estimation but it is not the aim of the project. The dataset, observing some of the results obtained by Mediapipe is less performing than what was thought. Poor accuracy is therefore also due to the dataset that is not completely accurate.

Libraries used: Pytorch, Pytorch models, Mediapipe for the dataset, OpenCV to represent the images

