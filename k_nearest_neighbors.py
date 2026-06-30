import operator
from collections import Counter
import numpy as np

class KNearestNeighbors:
    def __init__(self,k):
        self.k=k
    def fit(self, X_train,y_train):
        self.X_train=X_train
        self.y_train=y_train
        print("Training Done")
        
    def predict(self, X_test):
        predictions = []

        for point in X_test:
        
            distance={}
            Counter = 0
            for i in self.X_train:
                distance[Counter]=((point[0]-i[0])**2 + (point[1]-i[1])**2)**0.5
                Counter=Counter+1
            
            distance=sorted(distance.items() , key= operator.itemgetter(1))
            
            result = self.classify(distance[:self.k])
        
            predictions.append(result)

        return np.array(predictions)
    
    
        
    def classify(self,distance):
        label=[] 
        
        for i in distance:
            label.append(self.y_train[i[0]])
            
        return Counter(label).most_common()[0][0]                  