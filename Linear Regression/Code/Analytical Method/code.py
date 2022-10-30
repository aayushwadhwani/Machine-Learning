from operator import le
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("./dataset/Fish.csv")
lengths = dataset["Length1"].values
weights = dataset["Weight"].values

mean_length = sum(lengths)/len(lengths)
mean_weight = sum(weights)/len(weights)

numerator = 0
denominator = 0
for i in range(len(lengths)):
  numerator += ((lengths[i]-mean_length) * (weights[i]-mean_weight))
  denominator += ((lengths[i]-mean_length)**2)

b1 = numerator/denominator
b0 = (mean_weight - (b1*mean_length))
print(b1)
print(b0)

length_temp2 = [(b0 + (b1*l) ) for l in lengths]


# # print(mean_length_1)
plt.scatter(lengths, weights)
plt.plot(lengths, length_temp2,'.r-')
plt.show()
