# -- coding: utf-8 --
"""
Created on Sat May  2 23:43:16 2020

@author: ASUS
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


veri=pd.read_csv("diyabet.csv")

#sınıf sayısını belirle
label_encoder=LabelEncoder().fit(veri["class"])
labels=label_encoder.transform(veri["class"])
classes=list(label_encoder.classes_)


#Verilerin Hazırlanması ve Özellikle Sınıf Sayısının belirlenmesi
veri=veri.drop(["class"], axis=1)

nb_features=8
nb_classes=len(classes)


#Eğitim verilerinin standartlaştırılması
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler().fit(veri.values)
veri=scaler.transform(veri.values)



#Eğitim verisinin eğitim ve doğrulama için ayarlanması
from sklearn.model_selection import train_test_split
X_train,X_valid,y_train,y_valid=train_test_split(veri,labels,test_size=0.1)

X_train=np.array(X_train).reshape(691,8,1)
X_valid=np.array(X_valid).reshape(77,8,1)



#Modelin Oluşturulması
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Conv1D,Dropout,Flatten,MaxPooling1D

model=Sequential()

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
model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

#Modelin Eğitilmesi
model.fit(X_train,y_train,epochs=5,validation_data=(X_valid,y_valid))

yhat_probs = model.predict(X_valid)

yhat_classes = model.predict_classes(X_valid, verbose=0)



from sklearn.metrics import f1_score
f1 = f1_score(y_valid, yhat_classes ,average='micro')
print('F1 score: %f' % f1)

from sklearn.metrics import multilabel_confusion_matrix
cm= multilabel_confusion_matrix(y_valid, yhat_classes) 


sensitivity1 = cm[0,0,0]/(cm[0,0,0]+cm[0,1,0])
print('Sensitivity : ', sensitivity1 )

specificity1 = cm[1,1,1]/(cm[1,0,1]+cm[1,1,1])
print('Specificity : ', specificity1)

#Eğitim ve Doğrulama Başarımlarının Gösterilmesi
import matplotlib.pyplot as plt
plt.plot([f1, sensitivity1, specificity1])
plt.plot(model.history.history["accuracy"])
plt.plot(model.history.history["val_accuracy"])
plt.title("Model Başarımı")
plt.ylabel("Başarım")
plt.xlabel("Epak")
plt.legend(["F1, Sensivity, Specifity Score","Accuracy", "val_Accuracy"],loc="upper left")
plt.show