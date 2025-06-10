# Face Landmark and Attention Detection

The project is divided into two parts:

1. to implement the system using the mediapipe model; available in "Attention-Detection-CANOPIES-project/mediapipe/EyeRecognition/".

2. implement the system by creating a model from scratch on pytorch; available in "Attention-Detection-CANOPIES-project/pyTorch/model+faceDetection.py".


### 1. MEDIAPIPE

The implementation of attention detection on Mediapipe uses the pre-trained model of Mediapipe and in particular the FaceMesh, both for face and eye recognition. Initially, we started with face detection but it was understood that it did not make a significant contribution to the project. Since the Mediapipe Pose was not used to estimate the position of the body and face, the direction of the latter was calculated using the facial proportions; the same was done for the position of the irises of the eyes; by combining the results, a system was obtained that detects the direction of the face and gaze, consequently it was possible to identify the attention of the subjects.

_Advantages: _ Reliable system in common situations, fast, light.

_Disadvantages: _ system not reliable in uncommon situations such as face coverage, direction of the face too far from the center of the camera, Mediapipe's search to find the landmarks even if they are covered, unreliability of landmarks and viewpoints in very particular situations such as abnormal angles.

Used libraries: Mediapipe and OpenCV




### 2. PYTORCH
The implementation of attention detection is complete. The system uses a dataset created thanks to the first part of the project, the allowed inference is sufficient to observe that the model is able to predict the area near the eyes but its accuracy is less than that of Mediapipe.
The images are passed with three numpy arrays: one for the image names, one with the landmarks and one with the 0/1 for the attention classification. The face is cut out of the image and fed to the model. Several ResNet models have been tried:

ResNet18

ResNet34

ResNet50

ResNet101

ResNet152

It is believed that ResNet34 and ResNet50 perform well for this task, ResNet18 is too small as a network, ResNet101 and ResNet152 are models too complicated for the problem and its characteristics (such as the limited dataset)

_Advantages:_ more structured system on which the problems presented above on Mediapipe can be solved.

_Disadvantages:_ online there are no datasets for the recognition of the 478 landmarks of the face nor those for the irises of the eyes, there are datasets that have from 20 to 68 landmarks but do not allow the recognition of the iris, only of the eye. This problem could be overcome with the use of a pose estimation but it is not the aim of the project. The dataset, observing some of the results obtained by Mediapipe is less performing than what was thought. Poor accuracy is therefore also due to the dataset that is not completely accurate.

Libraries used: Pytorch, Pytorch models, Mediapipe for the dataset, OpenCV to represent the images

We started by recognizing the attention of people individually:

#### Person Attentive
![Logo](https://github.com/RicGobs/LabVision/blob/main/mediapipe/EyeRecognition/volto_attento.jpg)
#### Person NOT Attentive
![Logo](https://github.com/RicGobs/LabVision/blob/main/mediapipe/EyeRecognition/uomo_non_attento.jpg)

We analyzed and identified the facial structure:

#### Person NOT Attentive with facial structure
![Logo](https://github.com/RicGobs/LabVision/blob/main/mediapipe/EyeRecognition/solution.jpg)

We recognized the attention of multiple people in front of the camera:

#### Both people are attentive, the first is highlighted
![Logo](https://github.com/RicGobs/LabVision/blob/main/mediapipe/EyeRecognition/solution1.jpg)

#### Only the closest person (the first) is attentive, the first is highlighted
![Logo](https://github.com/RicGobs/LabVision/blob/main/mediapipe/EyeRecognition/solution2.jpg)

#### Only the furthest person (the second) is attentive, the second is highlighted
![Logo](https://github.com/RicGobs/LabVision/blob/main/mediapipe/EyeRecognition/solution3.jpg)


### 2. PYTORCH
The implementation of attention detection is complete. The system uses a dataset created thanks to the first part of the project, the allowed inference is sufficient to observe that the model is able to predict the area near the eyes but its accuracy is less than that of Mediapipe.
The images are passed with three numpy arrays: one for the image names, one with the landmarks and one with the 0/1 for the attention classification. The face is cut out of the image and fed to the model. Several ResNet models have been tried:

ResNet18

ResNet34

ResNet50

ResNet101

ResNet152

It is believed that ResNet34 and ResNet50 perform well for this task, ResNet18 is too small as a network, ResNet101 and ResNet152 are models too complicated for the problem and its characteristics (such as the limited dataset)

_Advantages:_ more structured system on which the problems presented above on Mediapipe can be solved.

_Disadvantages:_ online there are no datasets for the recognition of the 478 landmarks of the face nor those for the irises of the eyes, there are datasets that have from 20 to 68 landmarks but do not allow the recognition of the iris, only of the eye. This problem could be overcome with the use of a pose estimation but it is not the aim of the project. The dataset, observing some of the results obtained by Mediapipe is less performing than what was thought. Poor accuracy is therefore also due to the dataset that is not completely accurate.

Libraries used: Pytorch, Pytorch models, Mediapipe for the dataset, OpenCV to represent the images

