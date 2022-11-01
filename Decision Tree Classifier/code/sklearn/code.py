import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

class DecisionTree:

  def get_training_and_testing(self):
    self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.x, self.y, test_size=0.2)

  def __init__(self, x, y, xdf) -> None:
    self.x = x
    self.y = y
    self.xdf = xdf

    self.get_training_and_testing()
    self.m = len(self.train_x)
    self.n = len(self.train_x[0])

    self.create()
  
  def create(self):
    obj = DecisionTreeClassifier()
    obj.fit(self.x, self.y)
    y_pred = obj.predict(self.test_x)
    accuracy = accuracy_score(self.test_y, y_pred)
    print(accuracy)
    print(confusion_matrix(self.test_y, y_pred))
    print(export_text(obj, feature_names=list(self.xdf)))
    

def main():
  df = pd.read_csv("./dataset/data.csv")
  x = df.drop("result", axis=1)
  x_column_names = x.columns.values

  for name in x_column_names:
    unique = x[name].unique()
    values = {}
    start = 1
    for i in unique:
      values[i] = start
      start+=1
    x[name] = x[name].replace(values)
  
  y = df["result"].values

  obj = DecisionTree(x.values, y, x.columns)

if __name__ == "__main__":
  main()