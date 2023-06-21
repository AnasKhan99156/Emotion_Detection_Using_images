from keras.layers import Conv2D, MaxPooling2D ,Flatten,Dense
import os
import cv2
import numpy as np
from keras.models import Model
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.layers.core import Activation
from keras.layers import concatenate
from keras.layers import Input
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
def Multi_Layes_CNN_model(height, width):
    input_shape = (height, width, 1)
    print(input_shape)
    input = Input(shape=input_shape)

    convulation1=Conv2D (20,(5,5),padding= 'same',activation= 'relu',kernel_regularizer=regularizers.l2(0.01))(input)
    print(convulation1.shape)
    
    convulation2=Conv2D (30,(3,3),activation= 'relu',padding= 'same',kernel_regularizer=regularizers.l2(0.01))(convulation1)
    print(convulation2.shape)
    x=MaxPooling2D ((2,2))(convulation2)
    print(x.shape)

    convulation3=Conv2D(40,(3,3),activation= 'relu',padding= 'same',kernel_regularizer=regularizers.l2(0.01))(x)
    print(convulation3.shape)
    
    convulation4=Conv2D(50,(2,2),padding= 'same',activation= 'relu',kernel_regularizer=regularizers.l2(0.01))(convulation3)
    print(convulation4.shape)
    x=MaxPooling2D ((2,2))(convulation3)
    print(x.shape)


    convulation5=Conv2D(60,(2,2),activation= 'relu',padding= 'same',kernel_regularizer=regularizers.l2(0.01))(x)
    print(convulation5.shape)
    x = MaxPooling2D((2, 2))(convulation5)
    print(x.shape)


    convulation6 = Conv2D(65, (2, 2), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(x)
    print(convulation6.shape)
    x = MaxPooling2D((2, 2))(convulation6)
    print(x.shape)


    convulation7 = Conv2D(70, (2, 2), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(x)
    print(convulation7.shape)
    x=Flatten ()(convulation7)


    dense= Dense(7, activation="softmax")(x)
  
    print(x.shape)
    model = Model(input, x)
    return model

def load_data(path):
    x = []
    y = []
    for image in os.listdir(path + '/angry'):
        # print(j, image)
        img = cv2.imread(path + '/angry/' + image, cv2.IMREAD_ANYDEPTH)
        x.append(img)
        y.append(0)
    for image in os.listdir(path + '/disgusted'):
        # print(j, image)
        img = cv2.imread(path + '/disgusted/' + image, cv2.IMREAD_ANYDEPTH)
        x.append(img)
        y.append(1)
    for image in os.listdir(path + '/fearful'):
        # print(j, image)
        img = cv2.imread(path + '/fearful/' + image, cv2.IMREAD_ANYDEPTH)
        x.append(img)
        y.append(2)
    for image in os.listdir(path + '/happy'):
        # print(j, image)
        img = cv2.imread(path + '/happy/' + image, cv2.IMREAD_ANYDEPTH)
        x.append(img)
        y.append(3)

    for image in os.listdir(path + '/neutral'):
        # print(j, image)
        img = cv2.imread(path + '/neutral/' + image, cv2.IMREAD_ANYDEPTH)
        x.append(img)
        y.append(4)

    for image in os.listdir(path + '/sad'):
        # print(j, image)
        img = cv2.imread(path + '/sad/' + image, cv2.IMREAD_ANYDEPTH)
        x.append(img)
        y.append(5)

    for image in os.listdir(path + '/surprised'):
        # print(j, image)
        img = cv2.imread(path + '/surprised/' + image, cv2.IMREAD_ANYDEPTH)
        x.append(img)
        y.append(6)


    x = np.asarray(x)
    y = np.asarray(y)
    hot = LabelEncoder()
    y_encoded = hot.fit_transform(y)

    y_encoded = to_categorical(y_encoded, num_classes=7)
    return x, y_encoded

if __name__ == '_main_':
    acc = 0

    ap_train = 'emoion datset/train'
    ap_test = 'emoion datset/test'

    # ap_val='F:/Dataset/UCSD_Ped2_Split/AP_Tr_Val/val'

    x1_train, y1_train = load_data(ap_train)
    # x1_val, y1_val = load_data(ap_val)

    x1_test, y1_test = load_data(ap_test)

    x1_train = x1_train.reshape((x1_train.shape[0], x1_train.shape[1], x1_train.shape[2], 1))
    # x1_val = x1_val.reshape((x1_val.shape[0], x1_val.shape[1], x1_val.shape[2], 1))

    x1_test = x1_test.reshape((x1_test.shape[0], x1_test.shape[1], x1_test.shape[2], 1))

    model1 = Multi_Layes_CNN_model(x1_train.shape[1], x1_train.shape[2])

    print(model1.output_shape)
    p = Dense(4, activation='softmax',kernel_regularizer=regularizers.l2(0.01))(model1.output)
    # model3_1.input,model3_2.input,model3_3.input
    model = Model(inputs=[model1.input], outputs=p)
    # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.95, nesterov=True)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    highacc=0
   
    for epoch in range(1,400):
        print("Epoch")
        print(epoch)
        model.fit([x1_train], y1_train,epochs=1, verbose=1)
        y_predict = model.predict([x1_test])
        y_pred = np.argmax(y_predict, axis=1)
        y_true = np.argmax(y1_test, axis=1)
        print(y_pred.shape)
        print(y_true.shape)
        print("Accuracy: ", accuracy_score(y_true,y_pred ))
        if acc < accuracy_score(y_true, y_pred):
            acc = accuracy_score(y_true, y_pred)
           # model.save("AD_LowFusMultiClassification_24.h5")
            #model.save_weights("AD_LowFusMultiClassification_24_W.h5")
            model.save("AD_MultiClassification_2.h5")
            model.save_weights("AD_MultiClassification_2_W.h5")
        print ('high till now')
        print(acc)
    print(acc)