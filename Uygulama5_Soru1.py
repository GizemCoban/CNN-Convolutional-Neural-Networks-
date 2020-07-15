# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 14:29:42 2020

@author:Gizem ÇOBAN
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")

#Sınıfların Belirlenmesi ve Etiketlenmesi
label_encoder=LabelEncoder().fit(train.species)
lables=label_encoder.transform(train.species)
classes=list(label_encoder.classes_) 

#Verilerin Hazırlanması ve Özellikle Sınıf Sayısının belirlenmesi
train=train.drop(["id","species"], axis=1)
test=test.drop(["id"], axis=1)
nb_features=192
nb_classes=len(classes)

#Eğitim verilerinin standartlaştırılması
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler().fit(train.values)
train=scaler.transform(train.values)

#Eğitim verisinin eğitim ve doğrulama için ayarlanması
from sklearn.model_selection import train_test_split
X_train,X_valid,y_train,y_valid=train_test_split(train,lables,test_size=0.1)

#Etiketlerin Kategorileştirilmesi
from tensorflow.keras.utils import to_categorical
y_train=to_categorical(y_train)
y_valid=to_categorical(y_valid)

X_train=np.array(X_train).reshape(891,192,1)
X_valid=np.array(X_valid).reshape(99,192,1)


#Modelin Oluşturulması
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Conv1D,Dropout,Flatten,MaxPooling1D

model=Sequential()
model.add(Conv1D(512,1,input_shape=(nb_features,1)))
model.add(Activation("relu"))
model.add(MaxPooling1D(2))
model.add(Conv1D(256,1))
model.add(Activation("relu"))
model.add(MaxPooling1D(2))
#verilerlin %25 atıyoruz
model.add(Dropout(0.25))
#verileri düzleştirme
model.add(Flatten())
#Yapay Sinir Ağı
model.add(Dense(2048,activation="relu"))
model.add(Dense(1024,activation="relu"))
#En son sonoflandırma yapalım. Sınıflandırma için softmax kullanılır
model.add(Dense(nb_classes,activation="softmax"))
model.summary()

#Modelin Derlenmesi
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

#Modelin Eğitilmesi
model.fit(X_train,y_train,epochs=75,validation_data=(X_valid,y_valid))

#gerekli değerlerin gösterilmesi
print ("Ortalama Eğitim Kaybı:",np.mean(model.history.history["loss"]))
print ("Ortalama Eğitim Başarımı:",np.mean(model.history.history["accuracy"]))
print ("Ortalama Doğrulama Kaybı:",np.mean(model.history.history["val_loss"]))
print ("Ortalama Doğrulama Başarımı:",np.mean(model.history.history["val_accuracy"]))


#Değerlerin Grafik Üzerinde Gösterilmesi


import matplotlib.pyplot as plt
fig,(ax1,ax2)=plt.subplots(2,1,figsize=(15,15))
ax1.plot(model.history.history['loss'],color='g',label='Eğitim Kaybı')
ax1.plot(model.history.history['val_loss'],color='y',label='Doğrulama Kaybı')
ax1.set_xticks(np.arange(15,75,15))

ax2.plot(model.history.history['accuracy'],color='g',label='Eğitim Başarımı')
ax2.plot(model.history.history['val_accuracy'],color='y',label='Doğrulama Başarımı')
ax2.set_xticks(np.arange(15,75,15))
plt.legend()
plt.show()