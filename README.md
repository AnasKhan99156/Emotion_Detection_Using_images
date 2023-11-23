# Emotion_Detection_Using_images

# What is an emotion? 

An emotion is a mental and physiological state which is subjective and private; it involves a lot of behaviors, actions, thoughts and feelings. 

# Face Detection

Given an image, detecting the presence of a human face is a complex task due to the possible variations of the face. The different sizes, angles and poses human face might have within the image cause this variation. The emotions which are deducible from the human face and different imaging conditions such as illumination and occlusions also affect facial Emotion Recognition with Image Processing and Neural Networks  appearances. In addition, the presence of spectacles, such as beard, hair and makeup have a considerable effect in the facial appearance as well.

# Feature Extraction 

Selecting a sufficient set of feature points which represent the important characteristics of the human face and which can be extracted easily is the main challenge a successful facial feature extraction approach has to answer. 

When using geometry and symmetry for the extraction of features, the human visual characteristics of the face are employed. The symmetry contained within the face is helpful in detecting the facial features irrespective of the differences in shape, size and structure of features.  

A good emotional classifier should be able to recognize emotions independent of gender, age, ethnic group, pose, lighting conditions, backgrounds, hair styles, glasses, beard and birth marks.

# EMOTION RECOGNITION

The prototype system for emotion recognition is divided into 3 stages: face detection, feature extraction and emotion classification. After locating the face with the use of a face detection algorithm, the knowledge in the symmetry and formation of the face combined with image processing techniques were used to process the enhanced face region to determine the feature locations. These feature areas were further processed to extract the feature points required for the emotion classification stage. 

From the feature points extracted, distances among the  features are calculated and given as input to the neural network to classify the emotion contained. The neural network was trained to recognize the 6 universal emotions. 

# Emotion classification 

The extracted feature points are processed to obtain the inputs for the neural network. The neural network has being trained so that the emotions happiness, sadness, anger, disgust, surprise and fear are recognized. Images from Facial expressions and emotion database are taken to train the network.

# CONCLUSION 

A neural network based solution combined with image processing was proposed to classify the six universal emotions: Happiness, Sadness, Anger, Disgust, Surprise and Fear. Initially a face detection step is performed on the input image. Afterwards an image processing based feature point extraction method is used to extract the feature points. Finally, a set of values obtained from processing the extracted feature points are given as input to a neural network to recognize the emotion contained. 
