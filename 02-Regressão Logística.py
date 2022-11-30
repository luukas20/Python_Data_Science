import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
#%matplotlib inline 

train = pd.read_csv('C:/Users/lucas/OneDrive/Documentos/Cursos/Python para Data Science/Dados/titanic_train.csv')

train.info()

train.head(10)

def input_idade(cols):
  Classe = cols[0]
  Idade = cols[1]
  
  if pd.isnull(Idade):
    if Classe == 1:
      return 37
    elif Classe == 2:
      return 29 
    else:
      return 24 
  else: 
    return Idade
  

train['Age'] = train[['Pclass','Age']].apply(input_idade,axis=1)


del train['Cabin']

train.dropna(inplace = True)

sex = pd.get_dummies(train['Sex'], drop_first = True)

embarque = pd.get_dummies(train['Embarked'], drop_first = True)

train.drop(['PassengerId','Name', 'Sex','Ticket','Embarked'], axis = 1, inplace=True)

train = pd.concat([train,sex,embarque], axis=1)

X = train[['Pclass', 'Age', 'SibSp','Parch', 'Fare','male','Q','S']]
Y = train['Survived'] 

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.4,random_state=101)

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(X_train,Y_train)

predic = logmodel.predict(X_test)


from sklearn.metrics import classification_report

print(classification_report(Y_test,predic))


from sklearn.metrics import confusion_matrix

print(confusion_matrix(Y_test,predic))
