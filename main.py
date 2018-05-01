# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 21:04:58 2018

@author: ander
"""

import cv2
import keras
from keras.models import Model
from keras.datasets import cifar10
import numpy as np
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import scale
import os
import sys

os.chdir('C:/Users/ander/OneDrive/Desktop/DL/FaceRec')
img = cv2.imread('faces.jpg')
img = cv2.resize(img, (224,224))

detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


model = keras.applications.vgg19.VGG19(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)


layer_name = 'fc2'
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

img = np.expand_dims(img, axis=0)
intermediate_output = intermediate_layer_model.predict(img)

# Get negative face examples
print('Processing AT&T Faces...')
ATT_FACES_DIR = 'C:/Users/ander/OneDrive/Desktop/DL/FaceRec/AT&T_Faces/'
negatives = np.empty([363,4096])
count = 0
for folder in os.listdir(ATT_FACES_DIR):
    for path in os.listdir(ATT_FACES_DIR+folder):
        img = cv2.imread(ATT_FACES_DIR+folder+'/'+path)
        faces = detector.detectMultiScale(img)
        
        if len(faces) >0:
            x,y,w,h = faces[0]
            #cv2.rectangle(img,(x,y),(x+w,y+h),(123,74,90))
            crop = img[y:y+h,x:x+w]
            crop = cv2.resize(crop, (224,224))
            crop = (crop - np.mean(crop) ) / np.std(crop)
            #crop = cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
            crop = np.expand_dims(crop, axis=0)
            intermediate_output = intermediate_layer_model.predict(crop)
            negatives[count]=intermediate_output
            count +=1
 
#            cv2.imshow('face',img)
#            cv2.waitKey(1)

print('Processing cifar...')
# Get negative CIFAR examples
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
cifar_train = np.empty((len(x_train[:1000]), 4096))
for i,img in enumerate(x_train[:1000]):
    img = cv2.resize(img, (224,224))
    cifar_train[i] = intermediate_layer_model.predict(np.expand_dims(img,axis=0))
    if i % 1000 == 0:
        print(i)
# free up some memory...
del x_train, y_train, x_test, y_test

print('Processing Morgan Negatives...')
# Get negative MORGAN examples
MO_DIR = 'E:/MorganPics/'
Mo_positives = np.empty([73, 4096])
count = 0
for path in os.listdir(MO_DIR):
    img = cv2.imread(MO_DIR + path)
    faces = detector.detectMultiScale(img)
    if len(faces) >0:
        x,y,w,h = faces[0]
        crop = img[y:y+h,x:x+w]
        crop = cv2.resize(crop, (224,224))
        crop = (crop - np.mean(crop) ) / np.std(crop)
#        cv2.imshow('face',crop)
#        cv2.waitKey(1)
        #crop = cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
        crop = np.expand_dims(crop, axis=0)
        intermediate_output = intermediate_layer_model.predict(crop)
        Mo_positives[count]=intermediate_output
        count +=1
        
cv2.destroyAllWindows()
# Get positive examples
print('Processing Positives...')
POS_FACES_DIR = 'E:/FaceRecPics/'
positives = np.empty([362*2, 4096])
count = 0
for path in os.listdir(POS_FACES_DIR):
    img = cv2.imread(POS_FACES_DIR + path)
    faces = detector.detectMultiScale(img)
    if len(faces) >0:
        x,y,w,h = faces[0]
        cv2.rectangle(img,(x,y),(x+w,y+h),(123,74,200))
        crop = img[y:y+h,x:x+w]
        crop = cv2.resize(crop, (224,224))
        crop = (crop - np.mean(crop) ) / np.std(crop)
        #cv2.imshow('face',img)
        #cv2.waitKey(1)
        #crop = cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
        flipped = np.flip(crop,axis=1)
        flipped = np.expand_dims(flipped, axis=0)
        crop = np.expand_dims(crop, axis=0)
        intermediate_output = intermediate_layer_model.predict(crop)
        intermediate_flipped = intermediate_layer_model.predict(flipped)
        positives[count]=intermediate_output
        positives[count+1] = intermediate_flipped
        count +=2
        
cv2.destroyAllWindows()

print('Training SVM...')
# Generate labels: 0 for negative 1 for positive
labels =  [0] * (len(Mo_positives) + len(cifar_train) + len(negatives)) + [1] * len(positives)
data = np.vstack([Mo_positives, cifar_train, negatives, positives])
#labels =  [0] * ( len(negatives)) + [1] * len(positives)
#data = np.vstack([ negatives, positives])
clf = svm.SVC()
clf.fit(data,labels)

# Try on image
img = cv2.imread('E:/FaceRecPics/15.png')
img = cv2.resize(img, (224,224))

sys.exit()
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# TEST ON DIR OF IMAGES
TEST_DIR = 'E:/FaceRecTESTPics/'
TEST_DIR = 'E:/MOREfaceRecTestPics/'
i = 0
for path in os.listdir(TEST_DIR):
    img = cv2.imread(TEST_DIR + path)
    faces = detector.detectMultiScale(img)

    for face in faces:
        x,y,w,h = face
        crop = img[y:y+h,x:x+w]
        crop = cv2.resize(crop, (224,224))
        crop = (crop - np.mean(crop) ) / np.std(crop)
        crop = np.expand_dims(crop, axis=0)
        intermediate_output = intermediate_layer_model.predict(crop)
        res = clf.predict(intermediate_output)
        if res:
            cv2.rectangle(img, (x,y), (x+w, y+h),(0,255,0), 2)
        else:
            cv2.rectangle(img, (x,y), (x+w, y+h),(0,0,255),2)
            
    cv2.imshow('result',img);
    cv2.waitKey(1)
    #cv2.imwrite('C:/Users/ander/OneDrive/Desktop/IST597-MachineLearning_HOLD/demopics/{}.jpg'.format(i), img)
    i+=1
cv2.destroyAllWindows()





#TEST ON SINGLE IMAGE

IMAGE_PATH = 'C:/Users/ander/OneDrive/Desktop/DL/IST597/IST-597/fam.jpg'
img = cv2.imread(IMAGE_PATH)
img = cv2.resize(img, (2686//2,3356//2))
faces = detector.detectMultiScale(img)
for face in faces:
    x,y,w,h = face
    crop = img[y:y+h,x:x+w]
    crop = cv2.resize(crop, (224,224))
    crop = (crop - np.mean(crop) ) / np.std(crop)
    crop = np.expand_dims(crop, axis=0)
    intermediate_output = intermediate_layer_model.predict(crop)
    res = clf.predict(intermediate_output)
    if res:
        cv2.rectangle(img, (x,y), (x+w, y+h),(0,255,0), 2)
    else:
        cv2.rectangle(img, (x,y), (x+w, y+h),(0,0,255),2)
        
cv2.imshow('result',img);
cv2.waitKey()
cv2.destroyAllWindows()






import random
svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
for i in range(min(len(negatives),len(positives))):
    
    svd.fit(positives[i:i+1])
    svd2.fit(negatives[i:i+1])
    print(svd.singular_values_[0] - svd2.singular_values_[0])

svd2 = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
for i in range(20):
    svd2.fit(negatives[i:i+1])
    print(svd2.singular_values_[0])



for face in faces:
    x,y,w,h = face
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

pca_pos = PCA(n_components=2)
pca_pos.fit(positives)
    
pca_neg = PCA(n_components=2)
pca_neg.fit(negatives)




pos_hold = positives
x = []
y = []
for i in range(100):
    np.random.shuffle(pos_hold)
    temp = pos_hold[:10,]
    pca = PCA(n_components=2)
    pca.fit(temp)
    x.append(pca.explained_variance_ratio_[0])
    y.append(pca.explained_variance_ratio_[1])


pos_hold = negatives
x_neg = []
y_neg = []
for i in range(100):
    np.random.shuffle(pos_hold)
    temp = pos_hold[:10,]
    pca = PCA(n_components=2)
    pca.fit(temp)
    x_neg.append(pca.explained_variance_ratio_[0])
    y_neg.append(pca.explained_variance_ratio_[1])

plt.plot(x,y,'ro',x_neg,y_neg,'bo')
