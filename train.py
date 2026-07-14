# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model.k_nearest_neighbors import KNearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

# Load Dataset
data=pd.read_csv("data/Social_Network_Ads.csv")

# Feature Selection
X=data[["Age", "EstimatedSalary"]].values 
y =data["Purchased"].values

# Train-Test Split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42,stratify=y)

# Feature Scaling
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
 
# Train Custom KNN
knn=KNearestNeighbors(k=5)
knn.fit(X_train,y_train)

# Model Evaluation
y_pred = knn.predict(X_test) 

print("\nModel Performance")

print(f"\nConfusion Matrix: {confusion_matrix(y_test, y_pred)}")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report")
print(classification_report(y_test, y_pred))

# Predict New User
def predict_new():    
    age=int(input("Enter your age: "))  
    salary=int(input("Enter your salary: "))     
    X_new=np.array([[age,salary]])
    X_new=scaler.transform(X_new)
    
    result=knn.predict(X_new)
    if result[0] == 0:
        print("Customer is NOT likely to purchase the product.")
    else:
        print("Customer is likely to purchase the product.")
if __name__ == "__main__":
    predict_new()

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
plt.savefig("outputs/decision_boundary.png", dpi=300,bbox_inches="tight")
plt.show() 

# Confusion Matrix Heatmap  
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))

plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks([0, 1], ["Not Purchased", "Purchased"])
plt.yticks([0, 1], ["Not Purchased", "Purchased"])

# Add values inside each cell
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j,i,cm[i, j],ha="center",va="center",color="black",fontsize=12)
plt.colorbar()
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png",dpi=300,bbox_inches="tight")
plt.show()

plt.close()