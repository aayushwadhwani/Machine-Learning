from turtle import st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, confusion_matrix

class SVM:
  def __init__(self,x,y,features) -> None:
    self.x = x
    self.y = y
    self.features = features

    self.get_training_and_testing()
    self.m = len(self.x_test)
    self.n = len(self.features)

    self.start()
  
  def get_training_and_testing(self):
    self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.3)

  def start(self):
    svc_obj = SVC()
    svc_obj.fit(self.x_train, self.y_train)

    y_pred = svc_obj.predict(self.x_test)

    for i in range(self.m):
      str = "For "
      for j in range(self.n):
        str += f"{self.features[j]} = {self.x_test[i][j]}, "
      str += f"The predicted score is {y_pred[i]} and the actual score is {self.y_test[i]}"
      print(str)
    
def main():
  df = load_breast_cancer()
  x = df["data"]
  y = df["target"]
  x_columns = df["feature_names"]

  SVM(x,y,x_columns)

  

if __name__ == "__main__":
  main()