import os
from itertools import islice
from scipy import pi
import numpy as nm
import matplotlib.pyplot as plt



data_folder = "driving_dataset"
train_file = os.path.join(data_folder, 'data.txt')
LIMIT = None

split = 0.8
x=[]
y=[]

with open(train_file) as tf:
    for line in islice(tf, LIMIT):
        path, angle = line.strip().split()
        full_path = os.path.join(data_folder, path)
        x.append(full_path)

        y.append(float(angle)*pi/100)
y = nm.array(y)
print("successfull!!")

# splitting the data
print(len(y))
# since the multiplying the len(y) results in a float value, we are typecasting it to integer
split_index = int(len(y)*0.8)


train_y = y[:split_index]
test_y = y[split_index : ]

# Performing Exploratory Data analysis
plt.hist(train_y, bins = 50, color = "blue",histtype = "step")
plt.hist(test_y, bins = 50, color = "red", histtype = "step")
plt.plot()
# Assuming the predicted steering angle is the mean value of the training data
train_mean_y = nm.mean(train_y)

# Let's put a max limit error by finding the mean squared error of the test data
# Any error less than this error is acceptable
base_error = nm.mean(nm.square(test_y-train_mean_y))
print("The mean value of training data is {} degrees".format(train_mean_y))
print("The max acceptable error is {}".format(base_error))
# Using CNN for learning 
# Here we are not concerned about the sequence of image as we are just concerned about the steering angle
