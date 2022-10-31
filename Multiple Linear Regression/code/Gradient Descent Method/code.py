import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Regression:

  thresold = 1e-6
  iterations = 2000

  learning_rate = 0.01
  thetas = list()
  previous_loss = 0
  dataSet_length = 0

  def __init__(self, dependent_list, independent_list) -> None:
    self.dependent_list = dependent_list
    self.independent_list = independent_list

    self.dataSet_length = len(independent_list)
    for i in range(len(self.independent_list[0])+1):
      self.thetas.append(0.1)
    
    self.start()
  
  def get_predicted_values(self):
    y_predicted = list()
    for r in range(self.dataSet_length):
        y_predicted.append(self.calculate_new_value(self.independent_list[r]))
    return y_predicted
  
  def get_current_loss(self, predicted_values):
    current_loss = 0
    for i in range(self.dataSet_length):
      current_loss += (predicted_values[i] - self.dependent_list[i])**2
    return current_loss
    
  def get_xi_list(self, theta):
    xi_list = list()
    for cl in range(self.dataSet_length):
      xi = 0
      if theta == 0:
        xi = 1
      else:
        xi = self.independent_list[cl][theta-1]
      xi_list.append(xi)
    return xi_list
        
  def start(self):

    for i in range(self.iterations):

      y_predicted = self.get_predicted_values()
      
      current_loss = self.get_current_loss(y_predicted)

      if abs(self.previous_loss - current_loss) <= self.thresold:
        break

      self.previous_loss = current_loss

      new_thetas = list()
      for theta in range(len(self.thetas)):
        xi_list = self.get_xi_list(theta)
        new_theta = 0

        for i in range(self.dataSet_length):
          new_theta += (y_predicted[i]-self.dependent_list[i])*xi_list[i]

        new_theta =  self.learning_rate*(1/self.dataSet_length)*new_theta
        new_theta = self.thetas[theta] - new_theta
        new_thetas.append(new_theta)

      self.thetas = new_thetas
    self.new_predictions() 
  
  def new_predictions(self):
    for i in range(self.dataSet_length):
      p_v = self.thetas[0]
      for j in range(1,len(self.thetas)):
        p_v += self.thetas[j]*self.independent_list[i][j-1]
      
      print(f"The actual value was {self.dependent_list[i]} and the predicted is {p_v}")




  def calculate_new_value(self, row):
    new_val = self.thetas[0]

    for i in range(1,len(self.thetas)):
      new_val += self.thetas[i]*row[i-1]
    return new_val



  


data_frame = pd.read_csv("./dataset/data.csv")

y = data_frame["y"].values;
x = data_frame.drop(["y"], axis=1).values

# headings = data_frame.columns.drop(["Species"]).values

# for x_plot in headings:
#   for y_plot in headings:
#     if x_plot == y_plot:
#       continue
#     plt.scatter(data_frame[x_plot], data_frame[y_plot], label=f"{x_plot} vs {y_plot}")
#     plt.show()

obj = Regression(y, x)