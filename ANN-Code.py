import pandas as pd
import numpy as np

data = pd.read_csv('C:/Users/JOGESH MISHRA/Desktop/Deep_Learning_A_Z/Volume 1 - Supervised Deep Learning/Part 1 - Artificial Neural Networks (ANN)/Artificial_Neural_Networks/Churn_Modelling.csv')

X = data[['CreditScore','Geography','Gender','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary']].values
y = data['Exited'].values 

#PRE-PROCESSING 

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

ct = ColumnTransformer(transformers=[('one_hot_1',OneHotEncoder(),[1])],remainder ='passthrough')
X=ct.fit_transform(X)
X= X[:,1:12]

labelencoder =LabelEncoder()
y=labelencoder.fit_transform(y)

#TEST_TRAIN SPLIT

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=40)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#BUILDING ANN LAYERS

import keras 
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#TRAINING ANN

classifier.fit(X_train,y_train,batch_size=10, epochs=100)

#PREDICTING ANN ON TEST_SET

y_pred = classifier.predict(X_test)

#CONFUSION MATRIX
from sklearn.metrics import confusion_matrix
y_pred =y_pred >0.5

cm= confusion_matrix(y_test,y_pred)

#EVALUATING THE ANN MODEL FORMED

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import  cross_val_score

def build_classifier():
    classifier = Sequential()
    
    classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
    classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
    classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
    classifier.compile(optimizer= 'adam',loss='binary_crossentropy',metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn=build_classifier,batch_size=10, epochs=100)

accuracies = cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10,n_jobs=-1)

accuracies.mean()
accuracies.std()

#TUNING THE MODEL

from sklearn.model_selection import GridSearchCV 
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import  cross_val_score

def build_classifier(optimizer):
    classifier = Sequential()
    
    classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
    classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
    classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
    classifier.compile(optimizer= optimizer,loss='binary_crossentropy',metrics=['accuracy'])
    return classifier


classifier = KerasClassifier(build_fn=build_classifier)
parameters = {'batch_size':[25,32],
              'epochs':[100,500],
              'optimizer':['adam','rmsprop']
              }
grid_search= GridSearchCV(estimator=classifier,param_grid=parameters,scoring='accuracy',cv=10)

grid_search=grid_search.fit(X_train,y_train)

#BEST PARAMETERS
grid_search.best_params_
#BEST ACCURACY
grid_search.best_score_

