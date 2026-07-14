from collections import Counter
import numpy as np

class KNearestNeighbors:
    def __init__(self, k):
        self.k=k
    def fit(self, X_train,y_train):
        self.X_train=X_train
        self.y_train=y_train
        print("Model initialized successfully")
        
    def euclidean_distance(self, point1, point2):
        return ((point1[0] - point2[0]) ** 2 +(point1[1] - point2[1]) ** 2) ** 0.5
        
    def predict(self, X_test):
        predictions = []

        for point in X_test:
        
            distances={}
            index = 0
            for train_point in self.X_train:
                distances[index]=self.euclidean_distance(point,train_point)
                index += 1
            
            distances=sorted(distances.items(), key=lambda x: x[1])
            
            result = self.classify(distances[:self.k])
        
            predictions.append(result)

        return np.array(predictions)
    
    def classify(self,distances):
        labels = [] 
        
        for  neighbor  in distances:
            labels.append(self.y_train[neighbor[0]])
            
        return Counter(labels).most_common(1)[0][0]                  