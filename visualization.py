import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt("data_w3_ex1.csv", delimiter=',')
# Separating features(x) and targets(y)

# Changing 1-D arrays into 2D which helps in the code moving forward
x = np.expand_dims(data[:, 0], axis=1)
y = np.expand_dims(data[:, 1], axis=1)

plt.scatter(x, y)
plt.title("scatter plot of the data points")
plt.xlabel("Change in water level (m)")
plt.ylabel("Amount of water flowing out of the dam (m^3)")
plt.show()

