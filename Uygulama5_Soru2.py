# -- coding: utf-8 --
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


#Etiketlerin Kategorileştirilmesi
from tensorflow.keras.utils import to_categorical
lables=to_categorical(lables)

lables=np.array(lables).reshape(990,99,1)
train=np.array(train).reshape(990,192,1)

lables = lables[:,0]

#veriyi k kadar bölme CROSS VALIDATION
k=2  
val_data_samples =  len(train)//k
all_scores = []

#Modelin Oluşturulması
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Conv1D,Dropout,Flatten,MaxPooling1D

for i in range(k):
    print("işlem ", i)
    #test ve eğitim verilerini çapraz doğrulama ile ayırma
    val_data =train[i * val_data_samples: (i+1) * val_data_samples]
    val_targets = lables[i * val_data_samples: (i+1) * val_data_samples]
    
    part_train_data = np.concatenate([train[:i * val_data_samples], train[(i + 1) * val_data_samples:]],axis=0)
    part_train_targets = np.concatenate([lables[:i * val_data_samples], lables[(i + 1) * val_data_samples:]],axis=0)
    
 
    
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
    #En son sonoflandırma yapalım. Sınıflandırma için sofkopyala at sen tmax kullanılır
    model.add(Dense(nb_classes,activation="softmax"))
    model.summary()

    #Modelin Derlenmesi
    model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

    #Modelin Eğitilmesi
    model.fit(part_train_data,part_train_targets,epochs=5)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)
    
print (all_scores)

toplam=0
for i in range(len(all_scores)):
  toplam=toplam+all_scores[i]
  
ortalamabasari=toplam/(len(all_scores))
print("Ortalama Basari:",ortalamabasari)