import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from k_nearest_neighbors import KNearestNeighbors
data=pd.read_csv("Social_Network_Ads.csv")

X=data.iloc[:,2:4].values
y=data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

# An object for KNN

knn=KNearestNeighbors(k=5)
knn.fit(X_train,y_train)
# accuracy Score
from sklearn.metrics import accuracy_score
y_pred = knn.predict(X_test)
print("Accuracy:",accuracy_score(y_test, y_pred))

# Confusion Matrix
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, y_pred))

def predict_new():    
    age=int(input("Enter your age"))  
    salary=int(input("Enter your salary"))     
    X_new=np.array([[age],[salary]]).reshape(1,2)
    X_new=scaler.transform(X_new)
    
    result=knn.predict(X_new)
    if result==0:
        print("Will not purchase")
    else:
        print("will purchase it")
        
predict_new()
# Classification Report
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


# Decision Boundary

a = np.arange(start=X_train[:,0].min()-1,stop=X_train[:,0].max()+1,step=0.01)
b = np.arange(start=X_train[:,1].min()-1, stop=X_train[:,1].max()+1, step=0.01)

XX, YY = np.meshgrid(a, b)
input_array = np.array([XX.ravel(), YY.ravel()]).T
labels = knn.predict(input_array)
plt.figure(figsize=(8,6))

plt.contourf(XX, YY,labels.reshape(XX.shape),alpha=0.35,cmap='winter')
plt.scatter(X_train[:,0],X_train[:,1],c=y_train,edgecolors='black',cmap='winter')

plt.xlabel("Age (Scaled)")
plt.ylabel("Estimated Salary (Scaled)")
plt.title("KNN Decision Boundary")
plt.show()