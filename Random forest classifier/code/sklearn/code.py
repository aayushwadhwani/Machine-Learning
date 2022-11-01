import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

class Random_Forest:
  def __init__(self, x, y, features) -> None:
    self.x = x
    self.y = y
    self.features = features

    self.get_test_train_data()

    self.m = len(self.x_train)
    self.n = len(self.features)

    self.start()
  
  def get_test_train_data(self):
    self.x_train, self.x_test, self.y_train, self.y_test =  train_test_split(self.x, self.y, test_size=0.3)
  
  def start(self):
    obj = RandomForestClassifier()
    obj.fit(self.x_train, self.y_train)
    y_pred = obj.predict(self.x_test)
    # print(y_pred)
    # print(precision_score(self.y_test, y_pred))
    # print(recall_score(self.y_test, y_pred))
    cm = confusion_matrix(self.y_test, y_pred)
    plt.figure(figsize=(10,7))
    sb.heatmap(cm, annot=True)
    plt.show()
    # for i in range(self.m):
    #   print(f"The Actual value was {self.y_test[i]} and the predicted value is {y_pred[i]}")

def main():
  df = pd.read_csv("./datasets/Dry_Bean_Dataset.csv")
  # print(df)

  x = df.drop("Class", axis=1).values
  y = df["Class"].values
  x_columns = list(df.drop("Class", axis=1).columns)

  
  Random_Forest(x, y, x_columns)

if __name__ == "__main__":
  main()