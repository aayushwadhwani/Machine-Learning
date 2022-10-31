import math
import numpy as np
import pandas as pd

class Logistic_Regression:
  def __init__(self, x, y) -> None:

    self.x = x
    self.y = y
    self.m = len(self.x)
    self.n = len(self.x[0])

    self.iterations = 900
    self.learning_factor = 0.0000001
    self.thresold = 1e-6
    self.previous_cost = 0

    self.theta0 = 0
    self.theta1 = [0 for i in range(self.n)]

    self.update_values()
  
  def get_z(self, data_point):
    ans = self.theta0
    for i in range(self.n):
      ans += data_point[i]*self.theta1[i]
    return ans

  def update_values(self):
    for i in range(self.iterations):
      predicted_values = self.get_predicted_values()
      current_cost = self.get_current_cost(predicted_values)
      
      if abs(current_cost - self.previous_cost) <= self.thresold:
        break
      self.previous_cost = current_cost

      new_theta0 = 0
      
      for j in range(self.m):
        new_theta0 += (predicted_values[j] - self.y[j])
      
      self.theta0 = self.theta0 - (self.learning_factor*new_theta0)

      new_thetas1 = list()
      for j in range(self.n):
        new_theta = 0
        for k in range(self.m):
          new_theta += (predicted_values[k] - self.y[k])*self.x[k][j]
        new_theta = self.learning_factor*new_theta
        new_theta = self.theta1[j] - new_theta  
        new_thetas1.append(new_theta)
      
      self.theta1 = new_thetas1
      
    for i in range(self.m):
      ans = self.theta0
      for j in range(self.n):
        ans += self.theta1[j]*self.x[i][j]
      print(f"idx - {i} actual - {self.y[i]} got - {1/(1+np.exp(-ans))}")

  def get_current_cost(self, predicted_values):
    current_loss = 0
    for i in range(self.m):
      if self.y[i] >= 1:
        current_loss += math.log(predicted_values[i]+1e-6)
      else :
        current_loss += math.log(1-predicted_values[i]+1e-6) 
    current_loss /= self.m
    current_loss = 0-current_loss
    return current_loss

  def get_predicted_values(self):
    predicted_values = list()
    for i in range(self.m):
      predicted = 1/(1+np.exp(-self.get_z(self.x[i])))
      predicted_values.append(predicted)
    return predicted_values

def main():
  df = pd.read_csv("./datasets/diabetes.csv")
  x = df.drop("Outcome",axis=1).values
  y = df["Outcome"].values

  obj = Logistic_Regression(x, y)

if __name__ == "__main__":
  main()