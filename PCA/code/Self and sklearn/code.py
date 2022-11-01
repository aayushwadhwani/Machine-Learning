import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from numpy.linalg import eig

class PCA:

  def __init__(self, dataset, features, k) -> None:
    
    self.dataset = dataset
    self.features = features
    self.k = k

    self.normalize()

    self.m = len(self.dataset)
    self.n = len(self.features)
    self.start()

  def start(self):
    covariance_matrix = self.get_covariance_matrix()
    eigen_values, eigen_vectors = eig(covariance_matrix)

    sorted_eigen_vectors = sorted(zip(eigen_values, eigen_vectors), key=lambda x: x[0], reverse=True)
    for i in range(len(sorted_eigen_vectors)):
      sorted_eigen_vectors[i] = list(sorted_eigen_vectors[i][1])
    sorted_eigen_vectors = sorted_eigen_vectors[:self.k]

    new_values = list()
    for i in range(self.m):
      l = list()
      for j in range(len(sorted_eigen_vectors)):
        s = 0
        for k in range(self.n):
          s += self.dataset[i][k] * sorted_eigen_vectors[j][k]
        l.append(s)
      new_values.append(l)
    for data in new_values:
      print(data)
        
  def normalize(self):
    for i in range(len(self.dataset[0])):
      s = 0
      for j in range(len(self.dataset)):
        s += self.dataset[j][i]
      s /= len(self.dataset)
      for j in range(len(self.dataset)):
        self.dataset[j][i] = self.dataset[j][i] - s

  def get_covariance_matrix(self):

    covariance_matrix = list()
    for i in range(self.n):
      l = list()
      for j in range(self.n):
        l.append(0)
      covariance_matrix.append(l)

    for i in range(self.n):
      for j in range(self.n):
        covariance_matrix[i][j] = self.calc(i,j)
    return covariance_matrix
  
  def calc(self, i, j):
    s = 0
    for row in self.dataset:
      s += (row[i]*row[j])
    s /= self.n
    return s
    


      


def main():
  df = pd.read_csv("./dataset/data.csv")
  dataset = list(df.values)
  for i in range(len(dataset)):
    dataset[i] = list(dataset[i])
  
  dataset_components = list(df.columns)

  PCA(dataset, dataset_components, 2)



if __name__ == "__main__":
  main()