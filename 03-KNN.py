
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('C:/Users/lucas/OneDrive/Documentos/Cursos/Python para Data Science/Dados/Classified Data.csv')

df.head

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(df.drop('TARGET CLASS', axis=1))

df_norm = scaler.transform(df.drop('TARGET CLASS', axis=1))

df_param = pd.DataFrame(df_norm, columns=df.columns[:-1])

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(df_param,df['TARGET CLASS'],test_size=0.3,random_state=101)

from sklearn.neighbors import KNeighborsClassifier

KNN = KNeighborsClassifier(n_neighbors=1)

KNN.fit(X_train,Y_train)

predic = KNN.predict(X_test)

mae=[]
for i in range(1,60):
  KNN = KNeighborsClassifier(n_neighbors=i)
  KNN.fit(X_train,Y_train)
  predic = KNN.predict(X_test)
  mae.append(np.mean(predic != Y_test)) 

plt.figure()
plt.plot(range(1,60), mae, color='darkblue', linestyle = 'dashed', marker = 'o')
plt.xlabel('K Vizinhos')
plt.ylabel('MAE')
plt.show()

KNN = KNeighborsClassifier(n_neighbors=30)
KNN.fit(X_train,Y_train)
predic = KNN.predict(X_test)


from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(predic,Y_test))

print(confusion_matrix(predic,Y_test))



