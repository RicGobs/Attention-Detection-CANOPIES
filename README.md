# Face and Attention Detection

Il progetto è diviso in due parti:

1.implementare il sistema utilizzando il modello di mediapipe;

2.implementare il sistema creando un modello da zero su pytorch.


### 1. MEDIAPIPE 

L'implementazione dell'attention detection su Mediapipe utilizza il modello pre-addestrato di Mediapipe e in particolare il FaceMesh, sia per il riconoscimento del viso che degli occhi. Inizialmente si è partiti dal face Detection ma è stato compreso che non dava un contributo rilevante al progetto. Poichè non è stato usato anche il Mediapipe Pose per la stima della posizione del corpo e della faccia, si è calcolata la direzione di quest'ultima con l'utilizzo delle proporzioni facciali; lo stesso è stato fatto per la posizione delle iridi degli occhi; unendo i risultati si è ottenuto un sistema che rileva la direzione della faccia e dello sguardo, di conseguenza si è potuta identificare l'attenzione dei soggetti.

_Vantaggi:_ sistema affidabile nelle situazioni comuni, veloce, leggero.

_Svantaggi:_ sistema non affidabile in situazioni non comuni quali copertura del viso, direzione del viso troppo lontana dal centro della camera, ricerca di Mediapipe di trovare i landmark anche se coperti, inaffidabilità dei landmark e dello sguardo in situazioni molto particolari come angolature anormali.

Librerie usate: Mediapipe e OpenCV 

![Logo](https://github.com/RicGobs/LabVision/blob/main/mediapipe/EyeRecognition/solution1.jpg)

### 2. PYTORCH
L'implementazione dell'attention detection non è completa, si sta ricercando al meglio i landmark del voto concentrandosi su quelli delle iridi degli occhi. Addestrato al meglio il modello per fare questo, si concluderà con la classificazione dell'attenzione dei soggetti. Il sistema usa un dataset creato grazie alla prima parte del progetto, l'inferenza permessa è sufficiente per osservare che il modello riesce a predire la zona vicino agli occhi ma la sua accuratezza è minore di quella di Mediapipe.
Le immagini vengono passate con tre numpy array: uno per i nomi delle immagini, uno con i landmarks ed uno con il 0/1 per la classificazione dell'attenzione. Viene ritagliata la faccia dall'immagine e viene data in pasto al modello. E' stata provata ResNet18 la quale è troppo piccola per risolvere il problema, e ResNet50 che svolge un lavoro mediocre. In futuro si proveranno ulteriori modelli.

_Vantaggi:_ sistema più strutturato sul quale si possono risolvere i problemi presentati in precedenza su Mediapipe.

_Svantaggi:_ online non sono presenti dataset per il riconoscimento dei 478 landmarks del viso e nemmeno quelli per le iridi degli occhi, sono presenti dei dataset che hanno dai 20 ai 68 landmark ma non permettono il riconoscimento dell'iride, solamente dell'occhio. Questo problema potrebbe essere superato con l'utilizzo di un pose estimation ma non è l'obbiettivo del progetto. Il dataset, osservando alcuni dei risultati ottenuti da Mediapipe è meno performante di quello che si riteneva. La scarsa accuratezza è dovuta quindi anche al dataset non completamene preciso.

Librerie usate: Pytorch, Pytorch models, Mediapipe per il dataset, OpenCV per rappresentare le immagini



----

The project is divided into two parts:

- implement the system using the mediapipe model;

- implement the system by creating a model from scratch on pytorch.

For now I have implemented the face detection with mediapipe, but I am trying to improve the code in such a way as to use the best model based on the distance from the robot chamber. I am also working on eye tracking, starting from the mediapipe face mesh, to understand when the person is looking at the robot. Once these two parts have been completed, I will start working on the model from scratch in pytorch.

For the mediapipe folder, all programs are executable by downloading mediapipe and openCV.

